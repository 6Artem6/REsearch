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
import { repairLlMText } from "./llmTextRepair.js";

function historyItemHtml(item) {
  return String(item.content_html || item.contentHtml || "").trim();
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

export function normalizeDialogHistory(history) {
  const cleaned = [];
  for (const item of history || []) {
    const role = (item.role || "").trim();
    const content = repairLlMText((item.content || "").trim());
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
  return {
    role: "tutor",
    content: repairLlMText(res.tutor_message || ""),
    contentHtml: String(res.tutor_message_html || "").trim(),
    msg_id: mid,
  };
}

export function mergeHistoryWithPendingUser(messages, userMsg) {
  const u = (userMsg || "").trim();
  if (!u) return messages;
  const copy = [...messages];
  if (copy.some((m) => m.role === "user" && (m.content || "").trim() === u)) {
    return copy;
  }
  const last = copy[copy.length - 1];
  if (last?.role === "user" && (last.content || "").trim() === u) {
    return copy;
  }
  copy.push({ role: "user", content: u, msg_id: `pending-${Date.now()}` });
  return sortDialogMessages(copy);
}

export function hydrateSessionsFromServer(sessions) {
  const out = {};
  for (const [nodeId, blob] of Object.entries(sessions || {})) {
    const history = blob.history || [];
    const content = blob.content || {};
    out[nodeId] = {
      initialized:
        history.length > 0 || Boolean((content.summary || "").trim()),
      content,
      messages: historyToMessages(history),
      ragLabels: blob.rag_fact_labels || [],
      masteryDashboard: blob.mastery_dashboard || null,
      topicMasteryScore: blob.topic_mastery_score ?? 0,
      learningPhase: blob.learning_phase,
      learningMode: blob.learning_mode,
      sourceRegistry: blob.source_registry || [],
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
