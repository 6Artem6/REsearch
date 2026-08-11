const API = "/api/v1";
const LS_ACTIVE = "ke_skill_tree_active_curriculum";

async function waitWorkJob(jobId, timeoutSec = 600) {
  const r = await fetch(
    `${API}/work-jobs/${encodeURIComponent(jobId)}/wait?timeout_sec=${timeoutSec}`,
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  const data = await r.json();
  if (data.timed_out && !data.done) {
    throw new Error(
      "Worker не завершил задачу в отведённое время. Проверьте терминал make dev (WORKER) и перезапустите dev.",
    );
  }
  const job = data.job || data;
  if (job.status === "running" || job.status === "pending") {
    throw new Error(
      "Задача всё ещё в очереди (worker не ответил). Перезапустите make dev.",
    );
  }
  if (job.error) throw new Error(job.error);
  if (job.status === "failed") throw new Error(job.error || "job failed");
  return job;
}

async function resolveMaybeJobResponse(data) {
  if (data && data.job_id && data.status === "pending") {
    const job = await waitWorkJob(data.job_id);
    return job.result;
  }
  return data;
}

export async function fetchRagStatus() {
  const r = await fetch(`${API}/rag-gateway/memory-status`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchCurriculaList() {
  const r = await fetch(`${API}/skill-tree/curricula`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchWorkspace(curriculumId) {
  const r = await fetch(
    `${API}/skill-tree/curricula/${encodeURIComponent(curriculumId)}/workspace`,
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

export async function setActiveCurriculum(curriculumId) {
  const r = await fetch(`${API}/skill-tree/curricula/active`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ curriculum_id: curriculumId }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function rememberActiveCurriculumId(id) {
  if (id) localStorage.setItem(LS_ACTIVE, id);
}

export function readActiveCurriculumId() {
  return localStorage.getItem(LS_ACTIVE) || "";
}

/** Исправляет legacy tutor→user на user→tutor (как на сервере). */
import { repairLlMText, repairLectureMarkdownLayout, postprocessTutorHtml } from "./llmTextRepair.js";

function historyItemHtml(item) {
  const raw = String(item.content_html || item.contentHtml || "").trim();
  return raw ? postprocessTutorHtml(raw) : "";
}

/** Парсит msg_id / id (включая pending-<timestamp>). */
export function dialogMsgId(item) {
  const raw = item?.msg_id ?? item?.id;
  if (raw == null || raw === "") return null;
  const s = String(raw).trim();
  const pending = /^pending-(\d+)$/i.exec(s);
  if (pending) return Number(pending[1]);
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

/** Хронология = порядок в массиве (msg_id только для ключей, не для сортировки UI). */
export function sortDialogMessages(messages) {
  return [...(messages || [])];
}

function stripModePrefix(text) {
  return String(text || "")
    .replace(/^\[mode:\w+\]\s*/i, "")
    .trim();
}

/** User text for match/dedupe — без lecture-layout repair (он ломает сравнение с optimistic pending). */
export function normalizeUserDialogContent(text) {
  return stripModePrefix(repairLlMText(String(text || "")).trim());
}

function isPendingMsgId(msgId) {
  return /^pending-\d+$/i.test(String(msgId || "").trim());
}

export { isPendingMsgId };

export function userMessageMatches(content, userMsg) {
  const a = normalizeUserDialogContent(content);
  const b = normalizeUserDialogContent(userMsg);
  if (!a || !b) return false;
  return a === b;
}

export function normalizeDialogHistory(history) {
  const cleaned = [];
  for (const item of history || []) {
    const role = (item.role || "").trim();
    const raw = (item.content || "").trim();
    const content =
      role === "user"
        ? normalizeUserDialogContent(raw)
        : repairLectureMarkdownLayout(raw);
    if (!content) continue;
    const row = {
      role: role === "user" || role === "tutor" ? role : "tutor",
      content,
    };
    const mid = String(item.msg_id ?? item.id ?? "").trim();
    if (mid) row.msg_id = mid;
    const html = historyItemHtml(item);
    if (html) row.content_html = html;
    cleaned.push(row);
  }
  return cleaned;
}

export function historyToMessages(history) {
  return normalizeDialogHistory(history).map((h, idx) => {
    const mid = String(h.msg_id || h.id || "").trim() || String(idx + 1);
    return {
      role: h.role || "tutor",
      content: h.content || "",
      contentHtml: historyItemHtml(h),
      msg_id: mid,
    };
  });
}

export function tutorMessageFromApi(res) {
  const last = Array.isArray(res.history) ? res.history[res.history.length - 1] : null;
  const mid = last?.msg_id ?? last?.id ?? "";
  const content = repairLectureMarkdownLayout(composeTutorDisplayFromApi(res));
  const contentHtml = postprocessTutorHtml(
    String(res.tutor_message_html || "").trim(),
  );
  return {
    role: "tutor",
    content,
    contentHtml: tutorHtmlMatchesContent(content, contentHtml) ? contentHtml : "",
    msg_id: mid,
  };
}

function tutorHtmlMatchesContent(content, html) {
  const c = (content || "").trim();
  const h = postprocessTutorHtml(String(html || "").trim());
  if (!c || !h) return Boolean(h);
  const tail = c.slice(-120);
  if (tail.includes("?") && !h.includes(tail.slice(-60))) return false;
  return h.replace(/<[^>]+>/g, "").length + 40 >= c.length;
}

export function tutorHtmlMatchesContentForMessage(content, contentHtml) {
  return tutorHtmlMatchesContent(content, contentHtml);
}

export function composeTutorDisplayFromApi(res) {
  const parts = [
    res?.tutor_dialogue_feedback,
    res?.tutor_dialogue_technical,
    res?.tutor_dialogue_follow_up,
  ]
    .map((p) => String(p || "").trim())
    .filter(Boolean);
  if (parts.length) return parts.join("\n\n");
  return String(res?.tutor_message || "").trim();
}

/** После complete: HTML с сервера + полный tutor_message (включая follow_up_question). */
export function patchLastTutorMessageHtml(messages, res) {
  const composed = composeTutorDisplayFromApi(res);
  const html = postprocessTutorHtml(String(res.tutor_message_html || "").trim());
  const fu = String(res?.tutor_dialogue_follow_up || "").trim();
  let text = repairLectureMarkdownLayout(composed || res.tutor_message || "");
  if (fu && text && !text.includes(fu)) {
    text = `${text}\n\n${fu}`.trim();
  }
  if (!html && !text) return messages;
  const copy = [...messages];
  for (let i = copy.length - 1; i >= 0; i -= 1) {
    if (copy[i].role !== "tutor") continue;
    const useText = text || copy[i].content;
    let useHtml = html;
    if (!tutorHtmlMatchesContent(useText, useHtml)) {
      useHtml = "";
    }
    copy[i] = {
      ...copy[i],
      content: useText || copy[i].content,
      contentHtml: useHtml || "",
    };
    break;
  }
  return copy;
}

export function mergeHistoryWithPendingUser(messages, userMsg) {
  const u = normalizeUserDialogContent(userMsg);
  if (!u) return messages;
  const copy = [...messages];
  const last = copy[copy.length - 1];
  // Same text earlier in history is fine (repeat lecture button). Only skip
  // if the tail is already this turn (pending or just-synced user).
  if (last?.role === "user" && userMessageMatches(last.content, u)) {
    return copy;
  }
  copy.push({
    role: "user",
    content: stripModePrefix((userMsg || "").trim()),
    msg_id: `pending-${Date.now()}`,
  });
  return sortDialogMessages(copy);
}

/** Collapse only pending-* into a server row for the *same completed turn*.
 *  Never collapse two historical turns that share identical button text
 *  (e.g. repeated [mode:lecture]).
 */
export function dropRedundantPendingUsers(messages, userMsg) {
  const u = normalizeUserDialogContent(userMsg);
  if (!u) return messages || [];
  const list = messages || [];
  const matchingServer = list.filter(
    (m) =>
      m.role === "user" &&
      !isPendingMsgId(m.msg_id) &&
      userMessageMatches(m.content, u),
  );
  if (!matchingServer.length) return list;
  // Prefer the latest server copy of this turn (max numeric msg_id).
  const latestServer = matchingServer.reduce((a, b) => {
    const ai = dialogMsgId(a) ?? 0;
    const bi = dialogMsgId(b) ?? 0;
    return bi >= ai ? b : a;
  });
  const latestId = dialogMsgId(latestServer);
  return list.filter((m) => {
    if (m.role !== "user" || !isPendingMsgId(m.msg_id)) return true;
    if (!userMessageMatches(m.content, u)) return true;
    // Drop pending only when a server row for this turn exists.
    return latestId == null;
  });
}

/** @deprecated Keep export name: only strips pending duplicates, not history-by-text. */
export function dedupeUserMessagesByContent(messages) {
  return dropRedundantPendingUsers(messages, "");
}

/** Dedupe by msg_id only. Identical user text across turns is allowed. */
export function dedupeDialogMessages(messages) {
  const out = [];
  const seenIds = new Set();
  for (const m of messages || []) {
    const numericId = dialogMsgId(m);
    if (numericId != null) {
      if (seenIds.has(numericId)) continue;
      seenIds.add(numericId);
    } else if (isPendingMsgId(m.msg_id)) {
      // Keep at most one pending with the same text at the tail.
      const key = normalizeUserDialogContent(m.content || "");
      const dupPending = out.some(
        (x) =>
          x.role === "user" &&
          isPendingMsgId(x.msg_id) &&
          normalizeUserDialogContent(x.content || "") === key,
      );
      if (dupPending) continue;
    }
    out.push(m);
  }
  return out;
}

export function buildMessagesAfterChatComplete(res, userMsg, oldMessages, streamMsgId) {
  const hasHistory = Array.isArray(res.history) && res.history.length > 0;
  let messages;
  if (hasHistory) {
    messages = mergeHistoryWithPendingUser(historyToMessages(res.history), userMsg);
  } else {
    const next = (oldMessages || []).filter((m) => m.msg_id !== streamMsgId);
    messages = mergeHistoryWithPendingUser(next, userMsg);
    if (res.tutor_message) {
      messages.push(tutorMessageFromApi(res));
    }
    messages = sortDialogMessages(messages);
  }
  messages = dropRedundantPendingUsers(messages, userMsg);
  return patchLastTutorMessageHtml(
    dedupeDialogMessages(sortDialogMessages(messages)),
    res,
  );
}

export function hydrateSessionsFromServer(sessions) {
  const out = {};
  for (const [nodeId, blob] of Object.entries(sessions || {})) {
    const history = blob.history || [];
    const content = blob.content || {};
    const hasMemory = Boolean(blob.memory_prepared || blob.memory);
    out[nodeId] = {
      initialized: hasMemory || history.length > 0 || Boolean((content.summary || "").trim()),
      prepared: hasMemory && history.length === 0,
      content,
      messages: historyToMessages(dedupeDialogMessages(history)),
      ragLabels: blob.rag_fact_labels || [],
      masteryDashboard: blob.mastery_dashboard || null,
      coverageSummary:
        blob.coverage_summary ||
        blob.mastery_dashboard?.coverage_summary ||
        null,
      topicMasteryScore: blob.topic_mastery_score ?? 0,
      learningPhase: blob.learning_phase,
      learningMode: blob.learning_mode,
      sourceRegistry: blob.source_registry || [],
      lectureRagInspector: blob.lecture_rag_inspector || [],
      readyForTransition: Boolean(blob.ready_for_transition),
      lastEvalDirective: String(blob.last_eval_directive || "").trim(),
      quickReplies: Array.isArray(blob.quick_replies) ? blob.quick_replies : [],
    };
  }
  return out;
}

export function mergeNodeStatuses(curriculum, serverStatuses) {
  const out = {};
  for (const n of curriculum?.nodes || []) {
    out[n.node_id] = serverStatuses?.[n.node_id] || "unexplored";
  }
  return out;
}

export async function createCurriculum(targetGoal, sourcePolicy) {
  const policy = sourcePolicy || "practical_only";
  const depth =
    policy === "hybrid" || policy === "academic_only"
      ? "Deep Mechanics"
      : "Standard";
  const r = await fetch(`${API}/curriculum/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target_goal: targetGoal,
      user_level: "Intermediate/Advanced",
      depth_level: depth,
      source_policy: policy,
      generation_mode: policy === "academic_only" ? "consensus" : "fast",
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  const data = await r.json();
  if (data.graph) return data.graph;
  if (data.job_id) {
    const job = await waitWorkJob(data.job_id);
    return job.result;
  }
  return data;
}

/** @deprecated alias — используйте createCurriculum */
export async function generateCurriculum(targetGoal, sourcePolicy) {
  return createCurriculum(targetGoal, sourcePolicy);
}

export async function expandCurriculum(
  curriculumId,
  expansionPrompt,
  sourcePolicy,
) {
  const policy = sourcePolicy || "practical_only";
  const r = await fetch(`${API}/curriculum/expand`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      curriculum_id: curriculumId,
      expansion_prompt: expansionPrompt,
      source_policy: policy,
      generation_mode: policy === "academic_only" ? "consensus" : "fast",
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  const data = await r.json();
  if (data.graph) return data.graph;
  if (data.job_id) {
    const job = await waitWorkJob(data.job_id);
    return job.result;
  }
  return data;
}

export async function fetchNodeStatuses(curriculumId) {
  const r = await fetch(
    `${API}/node/statuses/${encodeURIComponent(curriculumId)}`,
  );
  if (!r.ok) return { statuses: {} };
  return r.json();
}

export async function fetchNodeSourceRegistry(curriculumId, nodeId) {
  const r = await fetch(
    `${API}/node/source-registry/${encodeURIComponent(curriculumId)}/${encodeURIComponent(nodeId)}`,
  );
  if (!r.ok) return { source_registry: [] };
  return r.json();
}

export async function nodeInit(curriculumId, nodeData) {
  const r = await fetch(`${API}/node/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ curriculum_id: curriculumId, node_data: nodeData }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return resolveMaybeJobResponse(await r.json());
}

/** Сброс сессии ноды и повторный сбор RAG / init (как первое открытие). */
export async function nodeRestart(curriculumId, nodeData) {
  const r = await fetch(`${API}/node/restart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ curriculum_id: curriculumId, node_data: nodeData }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return resolveMaybeJobResponse(await r.json());
}

export async function nodeChat(curriculumId, nodeData, userMessage) {
  const r = await fetch(`${API}/node/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      curriculum_id: curriculumId,
      node_data: nodeData,
      user_message: userMessage,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return resolveMaybeJobResponse(await r.json());
}

/** Читает SSE POST (data: JSON lines). */
async function readNodeSsePost(url, body, onEvent, { signal } = {}) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  const reader = r.body?.getReader();
  if (!reader) throw new Error("sse: no body");
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const evt = JSON.parse(line.slice(6));
        if (onEvent) onEvent(evt);
      } catch {
        /* skip partial */
      }
    }
  }
}

/** SSE POST /node/chat-stream — onEvent({type, text?, result?, detail?}). */
export async function nodeChatStream(
  curriculumId,
  nodeData,
  userMessage,
  onEvent,
) {
  return readNodeSsePost(
    `${API}/node/chat-stream`,
    {
      curriculum_id: curriculumId,
      node_data: nodeData,
      user_message: userMessage,
    },
    onEvent,
  );
}

export async function nodeSuggestQuestions(
  curriculumId,
  nodeData,
  selectedText,
  surroundingParagraph,
  { signal } = {},
) {
  const r = await fetch(`${API}/node/suggest-questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      curriculum_id: curriculumId,
      node_data: nodeData,
      selected_text: selectedText,
      surrounding_paragraph: surroundingParagraph,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

export async function nodeExplainSelection(
  curriculumId,
  nodeData,
  selectedText,
  surroundingParagraph,
  userQuestion,
) {
  const r = await fetch(`${API}/node/explain-selection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      curriculum_id: curriculumId,
      node_data: nodeData,
      selected_text: selectedText,
      surrounding_paragraph: surroundingParagraph,
      user_question: userQuestion || "",
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

/** SSE POST /node/explain-selection-stream */
export async function nodeExplainSelectionStream(
  curriculumId,
  nodeData,
  selectedText,
  surroundingParagraph,
  userQuestion,
  onEvent,
  { signal } = {},
) {
  return readNodeSsePost(
    `${API}/node/explain-selection-stream`,
    {
      curriculum_id: curriculumId,
      node_data: nodeData,
      selected_text: selectedText,
      surrounding_paragraph: surroundingParagraph,
      user_question: userQuestion || "",
    },
    onEvent,
    { signal },
  );
}

export async function nodeVerify(curriculumId, nodeData, userMessage) {
  const r = await fetch(`${API}/node/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      curriculum_id: curriculumId,
      node_data: nodeData,
      user_message: userMessage,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || r.statusText);
  }
  return resolveMaybeJobResponse(await r.json());
}

export function toNodeDataInput(node) {
  return {
    node_id: node.node_id,
    title: node.title,
    layer: node.layer,
    core_concepts: node.core_concepts || [],
    prerequisites: node.prerequisites || [],
    brief_summary: node.brief_summary || "",
    category: node.category || "",
    learning_materials: node.learning_materials || null,
    mapped_source_ids: node.mapped_source_ids || [],
    learning_goal: node.learning_goal || "",
    source_ref: node.source_ref || null,
    node_curriculum_breakdown: node.node_curriculum_breakdown || null,
    primary_source_id: node.primary_source_id || "",
    resource_urls: node.resource_urls || [],
  };
}
