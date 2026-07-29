const API = "/api/v1/v07/runs";
const EXPLAIN_API = "/api/v1/v08/explain";
const SUGGEST_QUESTIONS_API = "/api/v1/v08/suggest-questions";
const DEFAULT_EXPLAIN_QUESTION = "Объясни, что это значит?";
const THEME_STORAGE_KEY = "ke-theme";

const els = {
  form: document.getElementById("query-form"),
  query: document.getElementById("query-input"),
  retrievalMode: document.getElementById("retrieval-mode-select"),
  submit: document.getElementById("submit-btn"),
  statusBar: document.getElementById("status-bar"),
  tocNav: document.getElementById("toc-nav"),
  questionsNav: document.getElementById("questions-nav"),
  content: document.getElementById("content"),
  emptyState: document.getElementById("empty-state"),
  themeSelect: document.getElementById("theme-select"),
};

function applyTheme(name) {
  const theme = name || "monokai-pro";
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (_) {
    /* private mode */
  }
  if (els.themeSelect && els.themeSelect.value !== theme) {
    els.themeSelect.value = theme;
  }
}

function initTheme() {
  let saved = "monokai-pro";
  try {
    saved = localStorage.getItem(THEME_STORAGE_KEY) || saved;
  } catch (_) {
    /* ignore */
  }
  applyTheme(saved);
  if (els.themeSelect) {
    els.themeSelect.addEventListener("change", () => applyTheme(els.themeSelect.value));
  }
}

initTheme();

let pollTimer = null;
let lastPollStep = "";
let currentRunId = getRunIdFromUrl();

const explainEls = {
  toolbar: document.getElementById("explain-toolbar"),
  suggestList: document.getElementById("explain-suggest-list"),
  openBtn: document.getElementById("explain-open-btn"),
  backdrop: document.getElementById("explain-backdrop"),
  dialog: document.getElementById("explain-dialog"),
  closeBtn: document.getElementById("explain-close-btn"),
  defaultBtn: document.getElementById("explain-default-btn"),
  selection: document.getElementById("explain-selection"),
  questionInput: document.getElementById("explain-question-input"),
  submitBtn: document.getElementById("explain-submit-btn"),
  thread: document.getElementById("explain-thread"),
};

let explainSession = {
  selectedText: "",
  surroundingParagraph: "",
  busy: false,
  suggestKey: "",
  suggestQuestions: [],
  suggestSource: "",
};

const DEFAULT_SUGGEST_QUESTIONS = [
  "Что это значит на практике?",
  "Каковы причины этого?",
  "Какая есть альтернатива?",
];

let suggestAbortController = null;
let suggestRequestId = 0;
let suggestPrefetchTimer = null;

function isExplainDialogOpen() {
  return Boolean(explainEls.dialog && !explainEls.dialog.classList.contains("hidden"));
}

function getRunIdFromUrl() {
  const p = new URLSearchParams(window.location.search);
  return (p.get("run") || "").trim();
}

function setRunInUrl(runId) {
  if (!runId) return;
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState({}, "", url);
}

function runPermalink(runId) {
  const url = new URL(window.location.origin + "/app");
  url.searchParams.set("run", runId);
  return url.pathname + url.search;
}

function setStatus(text, kind = "") {
  els.statusBar.className = "status-bar" + (kind ? ` status-${kind}` : "");
  els.statusBar.innerHTML = text;
}

async function postRun(query, retrievalMode) {
  const mode = (retrievalMode || "fast").trim().toLowerCase();
  const res = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      retrieval_mode: mode === "consensus" ? "consensus" : "fast",
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function formatRetrievalModePill(mode) {
  const m = (mode || "fast").trim().toLowerCase();
  if (m === "consensus") {
    return "<span class='pill pill-mode-consensus'>Consensus</span>";
  }
  return "<span class='pill pill-mode-fast'>Fast</span>";
}

function applyRetrievalModeSelect(mode) {
  if (!els.retrievalMode) return;
  const m = (mode || "fast").trim().toLowerCase();
  els.retrievalMode.value = m === "consensus" ? "consensus" : "fast";
}

function statusModePill(run, view) {
  const fromRun = run?.retrieval_mode;
  const fromView = view?.retrieval_mode;
  const fromResult = run?.result?.retrieval_mode;
  return formatRetrievalModePill(fromRun || fromView || fromResult || "fast");
}

async function fetchRunList(limit = 30) {
  const res = await fetch(`${API}?limit=${limit}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function truncateText(text, max = 140) {
  const s = (text || "").trim();
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}

function formatQuestionTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function normalizeQuestionKeyPart(s) {
  return String(s || "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

const ANALYSIS_STATUS_RANK = {
  completed: 4,
  running: 3,
  pending: 2,
  failed: 1,
};

function pickPreferredQuestionEntry(a, b) {
  const activeRun = (getRunIdFromUrl() || currentRunId || "").trim();
  if (a.runId === activeRun && b.runId !== activeRun) return a;
  if (b.runId === activeRun && a.runId !== activeRun) return b;

  if (a.kind === "analysis" && b.kind === "analysis") {
    const ra = ANALYSIS_STATUS_RANK[a.status] || 0;
    const rb = ANALYSIS_STATUS_RANK[b.status] || 0;
    if (ra !== rb) return ra > rb ? a : b;
  }

  const ta = a.ts ? Date.parse(a.ts) : 0;
  const tb = b.ts ? Date.parse(b.ts) : 0;
  return tb >= ta ? b : a;
}

function dedupeQuestionEntries(entries) {
  const byKey = new Map();
  for (const entry of entries) {
    let key;
    if (entry.kind === "analysis") {
      key = `analysis:${normalizeQuestionKeyPart(entry.text)}`;
    } else if (entry.kind === "explain") {
      key = `explain:${normalizeQuestionKeyPart(entry.text)}:${normalizeQuestionKeyPart(
        entry.snippet
      )}`;
    } else {
      key = `other:${normalizeQuestionKeyPart(entry.text)}`;
    }
    const existing = byKey.get(key);
    byKey.set(key, existing ? pickPreferredQuestionEntry(existing, entry) : entry);
  }
  const out = [...byKey.values()];
  out.sort((a, b) => {
    const ta = a.ts ? Date.parse(a.ts) : 0;
    const tb = b.ts ? Date.parse(b.ts) : 0;
    return tb - ta;
  });
  return out.slice(0, 80);
}

function buildQuestionEntriesFromRuns(runs) {
  const entries = [];
  for (const run of runs || []) {
    if (run.query) {
      entries.push({
        kind: "analysis",
        text: run.query,
        runId: run.id,
        ts: run.created_at || null,
        status: run.status,
      });
    }
    for (const item of run.questions_log || []) {
      const t = (item.text || "").trim();
      if (!t) continue;
      entries.push({
        kind: item.type === "explain" ? "explain" : "other",
        text: t,
        snippet: item.snippet || "",
        runId: run.id,
        ts: item.ts || null,
      });
    }
  }
  entries.sort((a, b) => {
    const ta = a.ts ? Date.parse(a.ts) : 0;
    const tb = b.ts ? Date.parse(b.ts) : 0;
    return tb - ta;
  });
  return dedupeQuestionEntries(entries);
}

function renderQuestionsNav(entries) {
  if (!els.questionsNav) return;
  const activeRun = (getRunIdFromUrl() || currentRunId || "").trim();
  els.questionsNav.innerHTML = "";
  if (!entries.length) {
    const p = document.createElement("p");
    p.className = "questions-empty muted";
    p.textContent = "Здесь появятся запросы анализа и вопросы в «Пояснении».";
    els.questionsNav.appendChild(p);
    return;
  }
  for (const entry of entries) {
    const item = document.createElement("div");
    item.className = "question-item";
    if (entry.runId && entry.runId === activeRun) item.classList.add("active");

    const kind = document.createElement("span");
    kind.className = "question-kind";
    kind.textContent =
      entry.kind === "analysis"
        ? "Анализ"
        : entry.kind === "explain"
          ? "Пояснение"
          : "Вопрос";

    const link = document.createElement("a");
    link.className = "question-link";
    link.href = entry.runId ? runPermalink(entry.runId) : "#";
    link.textContent = truncateText(entry.text, 160);
    link.addEventListener("click", (e) => {
      if (!entry.runId) return;
      e.preventDefault();
      navigateToRun(entry.runId);
    });

    item.appendChild(kind);
    item.appendChild(link);

    if (entry.snippet) {
      const sn = document.createElement("span");
      sn.className = "question-snippet";
      sn.textContent = truncateText(entry.snippet, 90);
      item.appendChild(sn);
    }

    const metaParts = [];
    if (entry.runId) metaParts.push(entry.runId);
    if (entry.status && entry.kind === "analysis") metaParts.push(entry.status);
    const time = formatQuestionTime(entry.ts);
    if (time) metaParts.push(time);
    if (metaParts.length) {
      const meta = document.createElement("span");
      meta.className = "question-meta";
      meta.textContent = metaParts.join(" · ");
      item.appendChild(meta);
    }

    els.questionsNav.appendChild(item);
  }
}

async function refreshQuestionsNav() {
  if (!els.questionsNav) return;
  try {
    const runs = await fetchRunList(30);
    renderQuestionsNav(buildQuestionEntriesFromRuns(runs));
  } catch (_) {
    /* keep previous list */
  }
}

async function fetchView(runId) {
  const res = await fetch(`${API}/${runId}/view`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchRun(runId) {
  const res = await fetch(`${API}/${runId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function buildToc(toc) {
  els.tocNav.innerHTML = "";
  toc.forEach((item) => {
    const a = document.createElement("a");
    a.href = `#${item.id}`;
    a.textContent = item.title;
    a.dataset.section = item.id;
    a.addEventListener("click", (e) => {
      document.querySelectorAll(".toc-nav a").forEach((x) => x.classList.remove("active"));
      a.classList.add("active");
    });
    els.tocNav.appendChild(a);
  });
}

function fixCorruptedTextCommands(s) {
  s = s.replace(/\\t\s*ext\{/g, "\\text{");
  s = s.replace(/\\text\t+([a-zA-Z][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z][a-zA-Z0-9_]*)/g, "\\text{$1},$2");
  s = s.replace(/\\text\t+([a-zA-Z][a-zA-Z0-9_]*)/g, "\\text{$1}");
  s = s.replace(
    /\\text\\t\s+([a-zA-Z][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z][a-zA-Z0-9_]*)\s*\}/g,
    "\\text{$1},$2}"
  );
  s = s.replace(/\\text\\t\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\}/g, "\\text{$1}}");
  s = s.replace(/_\{\t+\\text\{/g, "_{\text{");
  s = s.replace(/\^\{\t+\\text\{/g, "^{\text{");
  s = s.replace(/\\text\s+([a-zA-Z][a-zA-Z0-9_]*)\s*,/g, "\\text{$1},");
  s = s.replace(/\\text\s+([a-zA-Z][a-zA-Z0-9_]*)(?![a-zA-Z0-9_{])/g, "\\text{$1}");
  s = s.replace(/\\t\s+/g, " ");
  return s;
}

function sanitizeMathDelimited(text) {
  if (!text) return text;
  let s = fixCorruptedTextCommands(text);
  s = s.replace(/\f/g, "").replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
  s = s.replace(/\\f\\frac/g, "\\frac").replace(/f\\frac/g, "\\frac");
  s = s.replace(/\\t+\\text/g, "\\text").replace(/\t+/g, " ");
  s = s.replace(/(.{6,}?)\s+\1(?=[\)\}\]\s,;]|$)/g, "$1");
  const fixes = [
    [/(?<!\\f)rac\{/g, "\\frac{"],
    [/(?<!\\t)ext\{/g, "\\text{"],
    [/mathrm\{/g, "\\mathrm{"],
    [/mathcal\{/g, "\\mathcal{"],
    [/mathbb\{/g, "\\mathbb{"],
  ];
  for (const [re, rep] of fixes) {
    s = s.replace(re, rep);
  }
  return s.trim();
}

function repairBrokenLatexInDom(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const parent = node.parentElement;
    if (parent && /^(pre|code|script|style)$/i.test(parent.tagName)) continue;
    let t = node.textContent || "";
    if (!t.includes("$") && !t.includes("\\") && !t.includes("\f") && !t.includes("\t")) continue;
    const inlineRe = /\$([^$\n]+?)\$/g;
    const displayRe = /\$\$([\s\S]+?)\$\$/g;
    let changed = false;
    if (t.includes("$")) {
      const newT = t.replace(displayRe, (_, inner) => {
        changed = true;
        return `$$${sanitizeMathDelimited(inner)}$$`;
      });
      t = newT.replace(inlineRe, (_, inner) => {
        changed = true;
        return `$${sanitizeMathDelimited(inner)}$`;
      });
      if (t !== node.textContent) changed = true;
    } else {
      const fixes = [
        [/\f/g, ""],
        [/\\f\\frac/g, "\\frac"],
        [/\\t+\\text/g, "\\text"],
      ];
      for (const [re, rep] of fixes) {
        if (re.test(t)) {
          t = t.replace(re, rep);
          changed = true;
        }
      }
    }
    if (changed) node.textContent = t;
  }
}

function renderMathInContent(root) {
  if (!root || typeof renderMathInElement !== "function") return;
  repairBrokenLatexInDom(root);
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
  if (typeof window.keHighlightCodeInRoot === "function") {
    window.keHighlightCodeInRoot(root);
  }
}

const SECTION_DEFAULT_COLLAPSED = new Set(["sources", "scholarly_papers"]);
/** L2 и финальный ответ — всегда развёрнуты по умолчанию (без вложенных свёрток). */
const SECTION_CONTENT_ALWAYS_OPEN = new Set([
  "l2a",
  "l2b",
  "l2c",
  "final_answer",
  "reasoner_pending",
]);
const sectionCollapseState = new Map();

const SOURCE_LIKE_HEADING_RE = /источник|ссылк|публикац|doi|arxiv/i;

function isSectionDefaultCollapsed(sectionId) {
  return SECTION_DEFAULT_COLLAPSED.has(sectionId);
}

function getSectionCollapsed(sectionId) {
  if (sectionCollapseState.has(sectionId)) {
    return sectionCollapseState.get(sectionId);
  }
  if (SECTION_CONTENT_ALWAYS_OPEN.has(sectionId)) {
    return false;
  }
  return isSectionDefaultCollapsed(sectionId);
}

function applySectionCollapsedState(block, sectionId) {
  const collapsed = getSectionCollapsed(sectionId);
  block.classList.toggle("section-collapsed", collapsed);
  const btn = block.querySelector(".section-toggle");
  if (btn) btn.setAttribute("aria-expanded", String(!collapsed));
}

function setupSectionCollapse(block, sectionId) {
  if (block.dataset.collapseBound === "1") return;
  const btn = block.querySelector(".section-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const collapsed = block.classList.toggle("section-collapsed");
    btn.setAttribute("aria-expanded", String(!collapsed));
    sectionCollapseState.set(sectionId, collapsed);
  });
  block.dataset.collapseBound = "1";
}

function isSourceLikeHeading(title) {
  return SOURCE_LIKE_HEADING_RE.test(title || "");
}

function enhanceCollapsibleHeadings(root) {
  if (!root || root.dataset.headingsEnhanced === "1") return;
  let heading = root.querySelector(":scope > h2, :scope > h3");
  while (heading) {
    const details = document.createElement("details");
    details.className = "ke-details";
    const title = heading.textContent || "";
    if (isSourceLikeHeading(title)) {
      details.classList.add("ke-details-closed-default");
    }
    const summary = document.createElement("summary");
    summary.innerHTML = heading.innerHTML;
    details.appendChild(summary);
    root.insertBefore(details, heading);
    heading.remove();
    let next = details.nextElementSibling;
    while (next && next.tagName !== "H2" && next.tagName !== "H3") {
      details.appendChild(next);
      next = details.nextElementSibling;
    }
    heading = root.querySelector(":scope > h2, :scope > h3");
  }
  root.dataset.headingsEnhanced = "1";
}

function wrapStandaloneSourceLists(body) {
  body.querySelectorAll(":scope > ul.source-list").forEach((ul) => {
    if (ul.closest("details")) return;
    const details = document.createElement("details");
    details.className = "ke-details ke-details-closed-default";
    const summary = document.createElement("summary");
    summary.textContent = "Список источников";
    body.insertBefore(details, ul);
    details.appendChild(summary);
    details.appendChild(ul);
  });
}

function applyDefaultClosedDetails(root, sectionId) {
  if (!root) return;
  const onlySourceBlocks = SECTION_CONTENT_ALWAYS_OPEN.has(sectionId);
  root.querySelectorAll(".ke-details-closed-default").forEach((details) => {
    if (details.dataset.defaultApplied === "1") return;
    details.open = false;
    details.dataset.defaultApplied = "1";
  });
  if (onlySourceBlocks) {
    root.querySelectorAll(".ke-details:not(.ke-details-closed-default)").forEach((details) => {
      details.open = true;
    });
  }
}

function enhanceSectionBodyCollapsibles(body, sectionId) {
  if (!body) return;
  delete body.dataset.collapsibleEnhanced;
  body.querySelectorAll("[data-headings-enhanced]").forEach((el) => {
    delete el.dataset.headingsEnhanced;
  });
  const skipInnerCollapse = SECTION_CONTENT_ALWAYS_OPEN.has(sectionId);
  if (!skipInnerCollapse) {
    const roots = [body, ...body.querySelectorAll(".md-body, .matrix-card, .contrast-card")];
    roots.forEach((root) => enhanceCollapsibleHeadings(root));
    wrapStandaloneSourceLists(body);
  }
  body.dataset.collapsibleEnhanced = "1";
  applyDefaultClosedDetails(body, sectionId);
}

function buildSectionBlockElement(sec) {
  const block = document.createElement("section");
  block.className = "section-block";
  block.id = sec.id;
  block.dataset.sectionId = sec.id;
  block.innerHTML = `
    <h2 class="section-heading">
      <button type="button" class="section-toggle" aria-expanded="true">
        <span class="section-chevron" aria-hidden="true">▾</span>
        <span class="section-title">${escapeHtml(sec.title)}</span>
      </button>
    </h2>
    <div class="section-body-wrap">
      <div class="section-body">${sec.html}</div>
    </div>`;
  applySectionCollapsedState(block, sec.id);
  setupSectionCollapse(block, sec.id);
  const body = block.querySelector(".section-body");
  renderMathInContent(body);
  enhanceSectionBodyCollapsibles(body, sec.id);
  return block;
}

function renderSectionsIncremental(sections) {
  const order = sections.map((s) => s.id);
  sections.forEach((sec) => {
    let block = document.getElementById(sec.id);
    if (block && block.classList.contains("section-block")) {
      const body = block.querySelector(".section-body");
      if (body && body.innerHTML !== sec.html) {
        body.innerHTML = sec.html;
        renderMathInContent(body);
        enhanceSectionBodyCollapsibles(body, sec.id);
      }
      const titleEl = block.querySelector(".section-title");
      if (titleEl && titleEl.textContent !== sec.title) {
        titleEl.textContent = sec.title;
      }
      setupSectionCollapse(block, sec.id);
      applySectionCollapsedState(block, sec.id);
      return;
    }
    block = buildSectionBlockElement(sec);
    const siblings = [...els.content.querySelectorAll(".section-block")];
    const insertBefore = siblings.find((el) => order.indexOf(el.id) > order.indexOf(sec.id));
    if (insertBefore) {
      els.content.insertBefore(block, insertBefore);
    } else {
      els.content.appendChild(block);
    }
  });
}

function renderSections(sections) {
  els.content.innerHTML = "";
  renderSectionsIncremental(sections);
}

function renderView(view, runId, { partial = false, run = null } = {}) {
  if (!view.partial && !view.ready) return;
  if (runId) currentRunId = runId;
  applyRetrievalModeSelect(view.retrieval_mode || run?.retrieval_mode);
  els.emptyState.classList.add("hidden");
  buildToc(view.toc || []);
  renderSectionsIncremental(view.sections || []);
  const meta = view.meta || {};
  const modePill = statusModePill(run, view);
  const link = runId
    ? `<a class="ext-link run-permalink" href="${runPermalink(runId)}">run ${runId}</a>`
    : "";
  if (partial) {
    setStatus(
      `${modePill}
       <span class="pill">running</span>
       <span class="pill">${view.current_step || "…"}</span>
       <span class="pill">${(view.sections || []).length} блоков</span>
       ${link}`,
      "running"
    );
    return;
  }
  setStatus(
    `${modePill}
     <span class="pill">completed</span>
     <span class="pill">docs ${meta.docs ?? 0}</span>
     <span class="pill">chunks ${meta.chunks ?? 0}</span>
     <span class="pill">depth ${meta.depth ?? "—"}</span>
     ${link}`,
    "done"
  );
}

async function tryRenderPartialView(runId, run) {
  try {
    const view = await fetchView(runId);
    if (view.ready || view.status === "completed" || !view.partial) {
      clearInterval(pollTimer);
      pollTimer = null;
      lastPollStep = "";
      renderView(view, runId, { partial: false, run });
      els.submit.disabled = false;
      return;
    }
    if ((view.sections && view.sections.length) || view.partial) {
      renderView(view, runId, { partial: view.partial || run.status === "running", run });
    }
  } catch (_) {
    /* view not ready yet */
  }
}

async function pollRun(runId) {
  const data = await fetchRun(runId);
  const run = data.run;
  setRunInUrl(runId);
  const link = `<a class="ext-link run-permalink" href="${runPermalink(runId)}">run ${runId}</a>`;
  const modePill = statusModePill(run);
  if (run.status === "running" || run.status === "pending") {
    if (run.current_step !== lastPollStep || run.has_partial) {
      lastPollStep = run.current_step;
      await tryRenderPartialView(runId, run);
    } else {
      setStatus(
        `${modePill}
         <span class="pill">${run.status}</span>
         <span class="pill">${run.current_step}</span>
         ${link}`,
        "running"
      );
    }
    return;
  }
  if (run.status === "completed") {
    clearInterval(pollTimer);
    pollTimer = null;
    lastPollStep = "";
    const view = await fetchView(runId);
    renderView(view, runId, { partial: false, run });
    els.submit.disabled = false;
    return;
  }
  if (run.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    lastPollStep = "";
    await tryRenderPartialView(runId, run);
    setStatus(
      `<span class="pill">failed</span> ${run.error || ""} ${link}`,
      "running"
    );
    els.submit.disabled = false;
  }
}

async function navigateToRun(runId) {
  if (!runId) return;
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  lastPollStep = "";
  currentRunId = runId;
  setRunInUrl(runId);
  els.submit.disabled = true;
  setStatus(`<span class="pill">loading</span> run ${runId}`, "running");
  try {
    const data = await fetchRun(runId);
    const run = data.run;
    if (run.query) els.query.value = run.query;
    applyRetrievalModeSelect(run.retrieval_mode);
    refreshQuestionsNav();
    if (run.status === "completed") {
      const view = await fetchView(runId);
      renderView(view, runId, { partial: false, run });
      els.submit.disabled = false;
      return;
    }
    if (run.status === "failed") {
      setStatus(
        `${statusModePill(run)} <span class="pill">failed</span> ${run.error || ""} ` +
          `<a class="ext-link" href="${runPermalink(runId)}">${runId}</a>`,
        "running"
      );
      await tryRenderPartialView(runId, run);
      els.submit.disabled = false;
      return;
    }
    els.emptyState.classList.add("hidden");
    pollTimer = setInterval(() => pollRun(runId), 2000);
    await pollRun(runId);
  } catch (err) {
    setStatus(`<span class="pill">error</span> ${err.message}`);
    els.submit.disabled = false;
  }
}

async function bootstrapFromUrl() {
  const runId = getRunIdFromUrl();
  if (runId) currentRunId = runId;
  if (!runId) return;
  await navigateToRun(runId);
}

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = els.query.value.trim();
  if (query.length < 3) return;
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  els.submit.disabled = true;
  els.emptyState.classList.remove("hidden");
  els.content.innerHTML = "";
  els.tocNav.innerHTML = "";
  sectionCollapseState.clear();
  lastPollStep = "";
  setStatus("<span class='pill'>starting</span>", "running");
  try {
    const created = await postRun(query, els.retrievalMode?.value);
    const runId = created.run.id;
    applyRetrievalModeSelect(created.run.retrieval_mode);
    currentRunId = runId;
    setRunInUrl(runId);
    refreshQuestionsNav();
    pollTimer = setInterval(() => pollRun(runId), 2000);
    pollRun(runId);
  } catch (err) {
    setStatus(`<span class="pill">error</span> ${err.message}`);
    els.submit.disabled = false;
  }
});

// Highlight TOC on scroll
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        document.querySelectorAll(".toc-nav a").forEach((a) => {
          a.classList.toggle("active", a.dataset.section === id);
        });
      }
    });
  },
  { rootMargin: "-20% 0px -60% 0px" }
);

new MutationObserver(() => {
  document.querySelectorAll(".section-block").forEach((el) => observer.observe(el));
}).observe(els.content, { childList: true });

function resolveActiveRunId() {
  let id = (getRunIdFromUrl() || currentRunId || "").trim();
  const link = document.querySelector("a.run-permalink");
  if (link?.href) {
    try {
      const u = new URL(link.href, window.location.origin);
      const fromLink = (u.searchParams.get("run") || "").trim();
      if (fromLink) id = fromLink;
    } catch (_) {
      /* ignore */
    }
  }
  if (id) currentRunId = id;
  return id;
}

function isExplainableSelectionContainer(container) {
  if (!container) return false;
  const el =
    container.nodeType === Node.ELEMENT_NODE
      ? container
      : container.parentElement;
  return Boolean(el?.closest("#content .section-body, #content .md-body"));
}

function hideExplainToolbar() {
  if (explainEls.toolbar) explainEls.toolbar.classList.add("hidden");
  if (!isExplainDialogOpen()) {
    abortSelectionPrompts();
    clearSuggestChips();
  }
}

function abortSelectionPrompts() {
  if (suggestAbortController) suggestAbortController.abort();
  suggestAbortController = null;
}

function resolveAnalysisTopic() {
  const fromQuery = (els.query?.value || "").trim();
  if (fromQuery) return fromQuery;
  const finalSec = document.getElementById("final_answer");
  if (finalSec) {
    const h2 = finalSec.querySelector(".section-title");
    if (h2?.textContent) return h2.textContent.trim();
  }
  return "";
}

function cacheSelectionPrompts(selectedText, paragraphContext, questions, source) {
  explainSession.suggestKey = `${selectedText.slice(0, 160)}::${paragraphContext.slice(0, 160)}`;
  explainSession.suggestQuestions = questions;
  explainSession.suggestSource = source || "";
}

function currentSelectionSuggestKey() {
  return `${explainSession.selectedText.slice(0, 160)}::${explainSession.surroundingParagraph.slice(0, 160)}`;
}

function clearSuggestChips() {
  if (explainEls.suggestList) explainEls.suggestList.innerHTML = "";
}

function renderSuggestChips(questions, { loading = false, source = "" } = {}) {
  if (!explainEls.suggestList) return;
  explainEls.suggestList.innerHTML = "";
  if (loading) {
    const el = document.createElement("div");
    el.className = "explain-suggest-loading";
    el.textContent = "Подсказки (Ollama)…";
    explainEls.suggestList.appendChild(el);
    return;
  }
  if (source === "default") {
    const hint = document.createElement("div");
    hint.className = "explain-suggest-loading";
    hint.textContent = "Базовые подсказки (Ollama не ответила вовремя)";
    explainEls.suggestList.appendChild(hint);
  }
  for (const q of questions || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "explain-suggest-btn";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      requestExplain(q);
    });
    explainEls.suggestList.appendChild(btn);
  }
}

function setSuggestButtonsDisabled(disabled) {
  if (!explainEls.suggestList) return;
  explainEls.suggestList.querySelectorAll(".explain-suggest-btn").forEach((btn) => {
    btn.disabled = disabled;
  });
}

async function fetchSelectionPrompts(
  selectedText,
  paragraphContext,
  { prefetchOnly = false } = {}
) {
  if (!explainEls.suggestList && !prefetchOnly) return;
  if (suggestAbortController) suggestAbortController.abort();
  suggestAbortController = new AbortController();
  const reqId = ++suggestRequestId;
  if (!prefetchOnly) {
    renderSuggestChips([], { loading: true });
  }
  try {
    const res = await fetch(SUGGEST_QUESTIONS_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: suggestAbortController.signal,
      body: JSON.stringify({
        selected_text: selectedText,
        paragraph_context: paragraphContext,
        topic: resolveAnalysisTopic(),
      }),
    });
    if (reqId !== suggestRequestId) return;
    let questions = [...DEFAULT_SUGGEST_QUESTIONS];
    let source = "default";
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.questions) && data.questions.length >= 3) {
        questions = data.questions;
        source = data.source || "ollama";
      }
    }
    cacheSelectionPrompts(selectedText, paragraphContext, questions, source);
    if (!prefetchOnly && isExplainDialogOpen()) {
      renderSuggestChips(questions, { source });
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    if (reqId !== suggestRequestId) return;
    cacheSelectionPrompts(
      selectedText,
      paragraphContext,
      DEFAULT_SUGGEST_QUESTIONS,
      "default"
    );
    if (!prefetchOnly && isExplainDialogOpen()) {
      renderSuggestChips(DEFAULT_SUGGEST_QUESTIONS, { source: "default" });
    }
  }
}

function scheduleSelectionPromptsPrefetch(selectedText, paragraphContext) {
  window.clearTimeout(suggestPrefetchTimer);
  suggestPrefetchTimer = window.setTimeout(() => {
    fetchSelectionPrompts(selectedText, paragraphContext, { prefetchOnly: true });
  }, 350);
}

function loadDialogSelectionPrompts() {
  if (!explainSession.selectedText) return;
  if (
    explainSession.suggestKey === currentSelectionSuggestKey() &&
    explainSession.suggestQuestions?.length >= 3
  ) {
    renderSuggestChips(explainSession.suggestQuestions, {
      source: explainSession.suggestSource,
    });
    return;
  }
  fetchSelectionPrompts(
    explainSession.selectedText,
    explainSession.surroundingParagraph
  );
}

function getSurroundingParagraph(node, selectedText) {
  let el = node;
  while (el && el !== document.body) {
    if (el.matches && el.matches("p, li, blockquote, h3, h4, td")) {
      return (el.textContent || "").trim();
    }
    el = el.parentElement;
  }
  const block = node?.parentElement?.closest(".section-body");
  const text = (block?.textContent || "").trim();
  if (!text) return selectedText;
  const idx = text.indexOf(selectedText);
  if (idx < 0) return text.slice(0, 1200);
  const start = Math.max(0, idx - 400);
  const end = Math.min(text.length, idx + selectedText.length + 400);
  return text.slice(start, end);
}

function positionExplainToolbar(rect) {
  if (!explainEls.toolbar || !rect) return;
  const x = Math.min(
    window.innerWidth - 16,
    Math.max(16, rect.left + rect.width / 2)
  );
  const y = Math.max(8, rect.top);
  explainEls.toolbar.style.left = `${x}px`;
  explainEls.toolbar.style.top = `${y}px`;
  explainEls.toolbar.classList.remove("hidden");
}

function onDocumentSelectionChange() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
    hideExplainToolbar();
    return;
  }
  const range = sel.getRangeAt(0);
  const node = range.commonAncestorContainer;
  const container = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  if (!isExplainableSelectionContainer(container)) {
    hideExplainToolbar();
    return;
  }
  const text = sel.toString().trim();
  if (text.length < 2) {
    hideExplainToolbar();
    return;
  }
  resolveActiveRunId();
  explainSession.selectedText = text;
  explainSession.surroundingParagraph = getSurroundingParagraph(container, text);
  scheduleSelectionPromptsPrefetch(text, explainSession.surroundingParagraph);
  let rect = range.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) {
    const rects = range.getClientRects();
    if (rects.length > 0) rect = rects[0];
  }
  if (rect.width === 0 && rect.height === 0) {
    hideExplainToolbar();
    return;
  }
  positionExplainToolbar(rect);
}

function closeExplainDialog() {
  explainEls.dialog?.classList.add("hidden");
  explainEls.backdrop?.classList.add("hidden");
  explainSession.busy = false;
  abortSelectionPrompts();
  clearSuggestChips();
}

function openExplainDialog() {
  const runId = resolveActiveRunId();
  if (!explainSession.selectedText) return;
  if (!runId) {
    explainEls.selection.textContent = explainSession.selectedText;
    explainEls.thread.innerHTML =
      '<div class="explain-msg explain-msg-a">Нужен id прогона: откройте страницу с <code>?run=&lt;id&gt;</code> в URL (ссылка run в статус-баре после анализа).</div>';
    clearSuggestChips();
    explainEls.dialog?.classList.remove("hidden");
    explainEls.backdrop?.classList.remove("hidden");
    hideExplainToolbar();
    return;
  }
  explainEls.selection.textContent = explainSession.selectedText;
  explainEls.thread.innerHTML = "";
  explainEls.questionInput.value = "";
  clearSuggestChips();
  explainEls.dialog?.classList.remove("hidden");
  explainEls.backdrop?.classList.remove("hidden");
  hideExplainToolbar();
  loadDialogSelectionPrompts();
}

function appendExplainMessage(kind, bodyHtml, { renderMath = false } = {}) {
  const div = document.createElement("div");
  div.className = `explain-msg explain-msg-${kind}`;
  div.innerHTML = bodyHtml;
  explainEls.thread.appendChild(div);
  if (renderMath) {
    renderMathInContent(div);
  }
  explainEls.thread.scrollTop = explainEls.thread.scrollHeight;
  return div;
}

async function requestExplain(userQuestion) {
  const question = (userQuestion || "").trim();
  const runId = resolveActiveRunId();
  if (!question || explainSession.busy) return;
  if (!runId) {
    appendExplainMessage("a", "Нет run id — добавьте <code>?run=…</code> в URL.");
    return;
  }
  explainSession.busy = true;
  setSuggestButtonsDisabled(true);
  appendExplainMessage("q", `<strong>Вопрос:</strong> ${escapeHtml(question)}`);
  const loading = document.createElement("div");
  loading.className = "explain-loading";
  loading.textContent = "Gemini Lite…";
  explainEls.thread.appendChild(loading);
  try {
    const res = await fetch(EXPLAIN_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        selected_text: explainSession.selectedText,
        user_question: question,
        surrounding_paragraph: explainSession.surroundingParagraph,
      }),
    });
    loading.remove();
    if (!res.ok) {
      const errText = await res.text();
      appendExplainMessage("a", `<span class="pill">error</span> ${escapeHtml(errText)}`);
      return;
    }
    const data = await res.json();
    const ref = data.source_ref || {};
    let meta = "";
    if (ref.source_id) meta += `[${ref.source_id}] `;
    if (ref.title) meta += ref.title;
    if (ref.url) {
      meta += ` — <a class="ext-link" href="${escapeHtml(ref.url)}" target="_blank" rel="noopener">источник</a>`;
    }
    const answerBody = data.explanation_html
      ? data.explanation_html
      : `<p>${escapeHtml(data.explanation || "")}</p>`;
    appendExplainMessage(
      "a",
      answerBody + (meta ? `<div class="explain-msg-meta">${meta}</div>` : ""),
      { renderMath: true }
    );
    refreshQuestionsNav();
  } catch (err) {
    loading.remove();
    appendExplainMessage("a", escapeHtml(err.message || String(err)));
  } finally {
    explainSession.busy = false;
    setSuggestButtonsDisabled(false);
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

if (explainEls.openBtn) {
  explainEls.openBtn.addEventListener("click", () => openExplainDialog());
}
if (explainEls.closeBtn) {
  explainEls.closeBtn.addEventListener("click", () => closeExplainDialog());
}
if (explainEls.backdrop) {
  explainEls.backdrop.addEventListener("click", () => closeExplainDialog());
}
if (explainEls.defaultBtn) {
  explainEls.defaultBtn.addEventListener("click", () => requestExplain(DEFAULT_EXPLAIN_QUESTION));
}
if (explainEls.submitBtn) {
  explainEls.submitBtn.addEventListener("click", () => {
    const q = explainEls.questionInput.value.trim();
    if (!q) return;
    requestExplain(q);
    explainEls.questionInput.value = "";
  });
}

document.addEventListener("mouseup", () => {
  window.setTimeout(onDocumentSelectionChange, 10);
});
document.addEventListener("selectionchange", () => {
  window.clearTimeout(onDocumentSelectionChange._debounce);
  onDocumentSelectionChange._debounce = window.setTimeout(onDocumentSelectionChange, 80);
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Escape") closeExplainDialog();
});

bootstrapFromUrl();
refreshQuestionsNav();
