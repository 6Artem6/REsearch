/** Литералы \\n из structured LLM → реальные переносы (с повтором при двойном escape). */
export function repairLlMText(text) {
  if (!text) return "";
  let t = String(text);
  for (let i = 0; i < 6; i += 1) {
    const next = t
      .replace(/\\r\\n/g, "\n")
      .replace(/\\r/g, "\n")
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t");
    if (next === t) break;
    t = next;
  }
  return t;
}

const ANALYSIS_LIST_SECTIONS = [
  ["pros", "Плюсы"],
  ["cons", "Минусы и риски"],
  ["cons_and_risks", "Минусы и риски"],
  ["takeaways", "Ключевые выводы"],
  ["failure_modes", "Типичные сбои"],
];

function looksLikeAnalysisObject(obj) {
  if (!obj || typeof obj !== "object") return false;
  const keys = Object.keys(obj);
  if (!keys.includes("title") && !keys.includes("description")) return false;
  return keys.some((k) =>
    ["pros", "cons", "cons_and_risks", "takeaways", "failure_modes"].includes(k),
  );
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnalysisObjectHtml(obj) {
  const parts = [];
  const title = String(obj.title || "").trim();
  if (title) parts.push(`<h2>${escapeHtml(title)}</h2>`);
  const desc = String(obj.description || "").trim();
  if (desc) parts.push(`<p>${escapeHtml(desc)}</p>`);
  const seen = new Set();
  for (const [key, label] of ANALYSIS_LIST_SECTIONS) {
    if (seen.has(label)) continue;
    const raw = obj[key];
    if (!raw || !Array.isArray(raw)) continue;
    const items = raw.map((x) => String(x).trim()).filter(Boolean);
    if (!items.length) continue;
    seen.add(label);
    parts.push(`<h3>${escapeHtml(label)}</h3><ul>`);
    for (const it of items) {
      parts.push(`<li>${escapeHtml(it)}</li>`);
    }
    parts.push("</ul>");
  }
  return parts.join("");
}

function tryParseJsonObjectAt(text, start) {
  let pos = start;
  while (pos < text.length && text[pos] !== "{") pos += 1;
  if (pos >= text.length) return null;
  let depth = 0;
  for (let i = pos; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const candidate = text.slice(pos, i + 1);
        try {
          const obj = JSON.parse(candidate);
          if (looksLikeAnalysisObject(obj)) {
            return { obj, start: pos, end: i + 1 };
          }
        } catch {
          /* continue */
        }
        return null;
      }
    }
  }
  return null;
}

/** Trade-off JSON в тексте → Markdown (как на сервере). */
export function repairStructuredAnalysisJson(text) {
  const raw = repairLlMText(text).trim();
  if (!raw || raw.indexOf("{") < 0) return raw;

  let fenced = raw;
  if (fenced.startsWith("```")) {
    fenced = fenced.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
    try {
      const whole = JSON.parse(fenced);
      if (looksLikeAnalysisObject(whole)) {
        return formatAnalysisObjectMarkdown(whole);
      }
    } catch {
      /* mixed text */
    }
  }

  if (raw.startsWith("{")) {
    try {
      const whole = JSON.parse(raw);
      if (looksLikeAnalysisObject(whole)) {
        return formatAnalysisObjectMarkdown(whole);
      }
    } catch {
      /* embedded */
    }
  }

  const indexed = [];
  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] !== "{") continue;
    const hit = tryParseJsonObjectAt(raw, i);
    if (hit) indexed.push({ start: hit.start, end: hit.end, obj: hit.obj });
  }
  if (!indexed.length) return raw;

  let out = raw;
  for (let s = indexed.length - 1; s >= 0; s -= 1) {
    const { start, end, obj } = indexed[s];
    const md = formatAnalysisObjectMarkdown(obj);
    const before = raw.slice(0, start).trimEnd();
    const lastLine = before.split("\n").pop()?.trim() || "";
    const t = String(obj.title || "").trim();
    let replacement = md;
    if (
      t &&
      lastLine &&
      (lastLine.toLowerCase().includes(t.toLowerCase()) ||
        t.toLowerCase().includes(lastLine.toLowerCase()))
    ) {
      replacement = md.replace(/^##[^\n]*\n?/, "").trim();
    }
    out = out.slice(0, start) + replacement + out.slice(end);
  }
  return out.trim();
}

function formatAnalysisObjectMarkdown(obj) {
  const lines = [];
  const title = String(obj.title || "").trim();
  if (title) lines.push(`## ${title}`);
  const desc = String(obj.description || "").trim();
  if (desc) lines.push(desc);
  const seen = new Set();
  for (const [key, label] of ANALYSIS_LIST_SECTIONS) {
    if (seen.has(label)) continue;
    const raw = obj[key];
    if (!raw || !Array.isArray(raw)) continue;
    const items = raw.map((x) => String(x).trim()).filter(Boolean);
    if (!items.length) continue;
    seen.add(label);
    lines.push(`### ${label}`);
    for (const it of items) lines.push(`- ${it}`);
  }
  return lines.join("\n\n");
}

/** Если текст — trade-off JSON, простой HTML для LlmHtmlBlock / drawer. */
export function structuredAnalysisToHtml(text) {
  const raw = repairLlMText(text).trim();
  if (!raw || raw.indexOf("{") < 0) return "";

  if (raw.startsWith("{") || raw.startsWith("```")) {
    let jsonText = raw;
    if (jsonText.startsWith("```")) {
      jsonText = jsonText.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
    }
    try {
      const whole = JSON.parse(jsonText);
      if (looksLikeAnalysisObject(whole)) {
        return `<div class="md-body prose">${formatAnalysisObjectHtml(whole)}</div>`;
      }
    } catch {
      /* fall through */
    }
  }

  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] !== "{") continue;
    const hit = tryParseJsonObjectAt(raw, i);
    if (hit) {
      const prefix = escapeHtml(raw.slice(0, hit.start).trim());
      const body = formatAnalysisObjectHtml(hit.obj);
      const prefixBlock = prefix
        ? `<p class="analysis-prefix">${prefix.replace(/\n/g, "<br>")}</p>`
        : "";
      return `<div class="md-body prose">${prefixBlock}${body}</div>`;
    }
  }
  return "";
}
