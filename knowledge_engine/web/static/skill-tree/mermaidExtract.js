import { repairLlMText } from "./llmTextRepair.js";

function stripOuterQuotes(s) {
  let t = (s || "").trim();
  if (
    t.length >= 2 &&
    t[0] === t[t.length - 1] &&
    (t[0] === '"' || t[0] === "'")
  ) {
    return t.slice(1, -1).trim();
  }
  return t;
}

function quoteSubgraphTitles(inner) {
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*subgraph\s+)(.+)$/i);
      if (!m) return line;
      const rest = m[2].trim();
      if (rest.startsWith('"') || rest.startsWith("'")) return line;
      if (/^\w[\w-]*\s*\[/i.test(rest)) return line;
      const safe = rest.replace(/"/g, "'");
      return `${m[1]}"${safe}"`;
    })
    .join("\n");
}

function quoteParticipantAliases(inner) {
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*participant\s+\S+\s+as\s+)(.+)$/i);
      if (!m) return line;
      const alias = m[2].trim();
      if (alias.startsWith('"') || alias.startsWith("'")) return line;
      if (/[/()]/.test(alias) || alias.includes("  ") || alias.length > 22) {
        const safe = wrapLongLabel(alias.replace(/"/g, "'"), 22).replace(/"/g, "'");
        return `${m[1]}"${safe}"`;
      }
      return line;
    })
    .join("\n");
}

function quoteLoopLabels(inner) {
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*loop\s+)(.+)$/i);
      if (!m) return line;
      let rest = m[2].trim();
      const arrow = rest.match(/\s+[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:/);
      if (arrow) {
        rest = rest.slice(0, arrow.index).trim();
      }
      if (!rest || rest.startsWith('"') || rest.startsWith("'")) return line;
      if (/[()]/.test(rest) || rest.includes(" ")) {
        const safe = rest.replace(/"/g, "'");
        const tail = arrow ? line.slice(arrow.index) : "";
        return `${m[1]}"${safe}"${tail}`;
      }
      return line;
    })
    .join("\n");
}

function wrapLongLabel(text, maxLen = 38, joiner = "\n") {
  const t = (text || "").trim();
  if (t.length <= maxLen) return t;
  const words = t.split(/\s+/);
  const lines = [];
  let line = "";
  for (const w of words) {
    if (!line) line = w;
    else if (line.length + 1 + w.length <= maxLen) line += ` ${w}`;
    else {
      lines.push(line);
      line = w;
    }
  }
  if (line) lines.push(line);
  return lines.join(joiner);
}

const MERMAID_RAW_START =
  /^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|stateDiagram(?:-v2)?\b|erDiagram\b|xychart(?:-beta)?\b)/i;

const MERMAID_INIT_RE = /^\s*%%\s*\{init:[\s\S]*?\}%%\s*/i;

function peelInitDirective(inner) {
  const raw = (inner || "").trim();
  const m = raw.match(/^(\s*%%\s*\{init:[\s\S]*?\}%%)\s*([\s\S]*)$/i);
  if (m) return { init: m[1].trim(), body: (m[2] || "").trim() };
  const broken = raw.match(/^(\s*%%\s*\{init:[\s\S]*?)(?=flowchart|graph\s|sequenceDiagram|xychart)/i);
  if (broken) {
    return { init: "", body: raw.slice(broken[0].length).trim() };
  }
  return { init: "", body: raw };
}

function hasMermaidGraphBody(s) {
  const t = (s || "").trim();
  return MERMAID_RAW_START.test(t) || /-->|---|==>/.test(t);
}

/** Разбить несколько рёбер на одной строке: "...| NMA NMA -->" → отдельные строки. */
function splitMultipleEdgesPerLine(line) {
  const t = (line || "").trim();
  if (!t || /^(graph|flowchart|subgraph|end|%%|classDef|class |linkStyle|style )/i.test(t)) {
    return line;
  }
  const arrows = t.match(/-->/g);
  if (!arrows || arrows.length < 2) return line;
  return t.replace(/\s+(?=[A-Za-z_][\w-]*\s+-->)/g, (match, offset) => {
    if (offset === 0) return match;
    const before = t.slice(0, offset);
    const pipeCount = (before.match(/\|/g) || []).length;
    if (pipeCount % 2 === 1) return match;
    const arrowCount = (before.match(/-->/g) || []).length;
    if (arrowCount < 1) return match;
    return "\n";
  });
}

/** Разбить graph TD; и цепочки A[...] --> B[...]; ... на отдельные строки. */
function splitFlowchartOneLine(inner) {
  let s = (inner || "").trim();
  if (!s) return s;
  s = s.replace(/^(graph\s+(?:TD|LR|BT|RL));?\s*/im, "$1\n");
  s = s.replace(/^(flowchart\s+(?:TD|LR|BT|RL)?);?\s*/im, "$1\n");
  s = s.replace(/\]\s*;\s*(?=[A-Za-z0-9_])/g, "]\n");
  s = s.replace(/\}\s*;\s*(?=[A-Za-z0-9_])/g, "}\n");
  s = s.replace(/\)\s*;\s*(?=[A-Za-z0-9_])/g, ")\n");
  // Следующий узел на той же строке после закрытия label — не рвём "] -->".
  s = s.replace(
    /([)\]])\s+(?=[A-Za-z_][\w-]*\s*(\[|-->|-->))/g,
    "$1\n",
  );
  s = s
    .split("\n")
    .map((ln) => splitMultipleEdgesPerLine(ln))
    .join("\n");
  return s;
}

function expandGluedFlowchartLines(inner) {
  const lines = (inner || "").split("\n");
  const out = [];
  for (const line of lines) {
    const t = line.trim();
    if (!t || t.startsWith("%%")) {
      out.push(line);
      continue;
    }
    const glued =
      (t.includes(";") && /-->|---/.test(t)) ||
      (t.startsWith("-->") && /-->|---/.test(t)) ||
      ((t.match(/-->/g) || []).length >= 2);
    if (glued && !/^(graph|flowchart)\s/i.test(t)) {
      out.push(...splitFlowchartOneLine(t).split("\n"));
      continue;
    }
    if (/^(graph|flowchart)\s/i.test(t) && t.includes(";")) {
      out.push(...splitFlowchartOneLine(t).split("\n"));
      continue;
    }
    out.push(line);
  }
  return out.join("\n");
}

function isXychartMermaid(inner) {
  return /^xychart(?:-beta)?\b/im.test((inner || "").trimStart());
}

function sanitizeNoteLines(inner) {
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(
        /^(\s*Note\s+(?:over|left of|right of)\s+[^:]+:\s*)(.*)$/i,
      );
      if (!m) return line;
      const prefix = m[1];
      let body = (m[2] || "").trim().replace(/^"|"$/g, "").replace(/"/g, "'");
      body = wrapLongLabel(body, 36);
      if (!body) return line;
      return prefix + `"${body}"`;
    })
    .join("\n");
}

function quoteArrowMessages(inner) {
  const isSeq = /^sequenceDiagram\b/im.test((inner || "").trimStart());
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*)(.*)$/);
      if (!m) return line;
      const prefix = m[1];
      let body = (m[2] || "").trim().replace(/^"|"$/g, "");
      if (!body) return line;
      body = body.replace(/"/g, "'");
      if (!isSeq) body = wrapLongLabel(body, 40);
      return prefix + `"${body}"`;
    })
    .join("\n");
}

function ensureSequenceInit(inner) {
  if (!/^sequenceDiagram/im.test(inner)) return inner;
  if (/%%\s*\{init:/i.test(inner)) return inner;
  return (
    "%%{init: {'themeVariables': {'fontSize': '10px'}, " +
    "'sequence': {'wrap': true, 'width': 240, 'messageFontSize': 10, " +
    "'noteFontSize': 10, 'actorFontSize': 11, 'messageMargin': 48, " +
    "'boxMargin': 10, 'mirrorActors': false}}}%%\n" +
    inner
  );
}

function ensureFlowchartInit(inner) {
  const peeled = peelInitDirective(inner);
  let body = peeled.body;
  const head = body.trimStart();
  if (!/^(flowchart\b|graph\s+(?:TD|LR|BT|RL)\b)/im.test(head)) {
    return inner;
  }
  if (peeled.init || /%%\s*\{init:/i.test(inner)) return inner;
  const init =
    "%%{init: {" +
    '"flowchart": {"htmlLabels": true, "useMaxWidth": false, "padding": 28, "nodeSpacing": 56, "rankSpacing": 64, "curve": "basis"}, ' +
    '"themeVariables": {"fontSize": "14px", "fontFamily": "system-ui, sans-serif"}' +
    "}}%%";
  return `${init}\n${body}`;
}

function quoteInlineFlowchartNodes(inner) {
  return inner.replace(
    /([A-Za-z0-9_]+)\[([^\]]+)\]/g,
    (match, id, rawLabel) => {
      let label = rawLabel.trim().replace(/^"|"$/g, "");
      if (label.startsWith('"')) return match;
      const needs =
        label.length > 14 || /[()]/.test(label) || label.includes("  ");
      if (!needs) return match;
      label = wrapLongLabel(label.replace(/"/g, "'"), 20, "<br/>");
      return `${id}["${label}"]`;
    },
  );
}

/** Длинные подписи в узлах flowchart → кавычки + переносы. */
function quoteFlowchartNodeLabels(inner) {
  return inner
    .split("\n")
    .map((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("%%")) return line;
      if (/^(subgraph|end|classDef|class |linkStyle|style )/i.test(trimmed)) {
        return line;
      }
      const bracket = trimmed.match(
        /^([A-Za-z0-9_]+)\[([^\]]+)\](.*)$/,
      );
      if (bracket) {
        let label = bracket[2].trim().replace(/^"|"$/g, "");
        if (label.startsWith('"')) return line;
        const needsQuote =
          label.length > 14 ||
          /[()]/.test(label) ||
          label.includes("  ");
        if (!needsQuote) return line;
        label = wrapLongLabel(label.replace(/"/g, "'"), 20, "<br/>");
        const safe = `"${label}"`;
        return line.replace(
          bracket[0],
          `${bracket[1]}[${safe}]${bracket[3] || ""}`,
        );
      }
      const round = trimmed.match(/^([A-Za-z0-9_]+)\(([^)]+)\)(.*)$/);
      if (round) {
        let label = round[2].trim().replace(/^"|"$/g, "");
        if (label.startsWith('"')) return line;
        if (label.length <= 14) return line;
        label = wrapLongLabel(label.replace(/"/g, "'"), 20, "<br/>");
        return line.replace(
          round[0],
          `${round[1]}("${label}")${round[3] || ""}`,
        );
      }
      const diamond = trimmed.match(/^([A-Za-z0-9_]+)\{([^}]+)\}(.*)$/);
      if (diamond) {
        let label = diamond[2].trim().replace(/^"|"$/g, "");
        if (label.startsWith('"')) return line;
        if (label.length <= 12) return line;
        label = wrapLongLabel(label.replace(/"/g, "'"), 18, "<br/>");
        return line.replace(
          diamond[0],
          `${diamond[1]}{"${label}"}${diamond[3] || ""}`,
        );
      }
      return line;
    })
    .join("\n");
}

/** Узел flowchart: пробелы/скобки → id + подпись при необходимости. */
function mermaidNodeId(raw) {
  const t = (raw || "").trim();
  if (!t) return "node";
  const paren = t.match(/^(.+?)\s+\(([^)]+)\)\s*$/);
  if (paren) {
    const base = paren[1].trim();
    const inner = paren[2].trim();
    const id = `${base.replace(/\s+/g, "_")}_${inner.replace(/\s+/g, "_")}`;
    return `${id}["${base.replace(/"/g, "'")} (${inner.replace(/"/g, "'")})"]`;
  }
  if (/[\s/]/.test(t)) {
    const id = t.replace(/[^\w]+/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");
    return `${id || "node"}["${t.replace(/"/g, "'")}"]`;
  }
  return t.replace(/\s+/g, "_");
}

function parseLinkChain(chain) {
  let s = chain.replace(/--\(([^)]+)\)-->/g, "-->|$1|");
  const rawParts = s.split(/\s*-->\s*/).filter((p) => p.trim());
  if (rawParts.length < 2) return null;
  const edges = [];
  const nodes = [];
  for (let i = 0; i < rawParts.length; i += 1) {
    let p = rawParts[i].trim();
    if (i === 0) {
      nodes.push(p);
      continue;
    }
    let label = "";
    const lm = p.match(/^\|([^|]+)\|\s*(.+)$/s);
    if (lm) {
      label = lm[1].trim();
      p = lm[2].trim();
    }
    const from = nodes[nodes.length - 1];
    edges.push({ from, to: p, label });
    nodes.push(p);
  }
  return edges;
}

function chainSegmentToMermaid(title, chain, subIdx) {
  const edges = parseLinkChain(chain);
  if (!edges || edges.length === 0) return [];
  const lines = [];
  const subId = `sg_${subIdx}`;
  if (title) {
    const safeTitle = title.replace(/"/g, "'").replace(/\]/g, "");
    lines.push(`subgraph ${subId} [${safeTitle}]`);
  }
  for (const e of edges) {
    const a = mermaidNodeId(e.from);
    const b = mermaidNodeId(e.to);
    const lbl = e.label ? `|${e.label.replace(/"/g, "'")}|` : "";
    lines.push(`${title ? "  " : ""}${a} -->${lbl} ${b}`);
  }
  if (title) lines.push("end");
  return lines;
}

/**
 * LLM часто шлёт без fence/header:
 * Short Polling: Client --(HTTP/GET)--> LB | WebSockets: Client --(WS)--> LB (Sticky) ...
 */
export function liftPseudoFlowchart(text) {
  const t = stripOuterQuotes(repairLlMText(text)).trim();
  if (!t) return "";
  if (/^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|xychart(?:-beta)?\b)/im.test(t)) {
    return "";
  }
  if (!/--\([^)]+\)-->|-->/.test(t)) return "";

  const segments = t.includes("|")
    ? t.split(/\s*\|\s*/).filter((seg) => /-->|--\(/i.test(seg))
    : [t];
  if (!segments.length) return "";

  const bodyLines = ["flowchart LR"];
  segments.forEach((seg, idx) => {
    let title = "";
    let chain = seg.trim();
    const m = chain.match(/^([^:]{2,56}):\s*(.+)$/s);
    if (m && /-->|--\(/i.test(m[2])) {
      title = m[1].trim();
      chain = m[2].trim();
    }
    bodyLines.push(...chainSegmentToMermaid(title, chain, idx));
  });
  if (bodyLines.length <= 1) return "";
  return bodyLines.join("\n");
}

function formatXychartBody(s) {
  let t = (s || "").trim();
  if (!isXychartMermaid(t)) return t;
  t = t.replace(/^(xychart(?:-beta)?)\s+/im, "$1\n");
  t = t.replace(/\s+(title\s)/gi, "\n$1");
  t = t.replace(/\s+(x-axis\s)/gi, "\n$1");
  t = t.replace(/\s+(y-axis\s)/gi, "\n$1");
  t = t.replace(/\s+(line\s)/gi, "\n$1");
  t = t.replace(/\s+(bar\s)/gi, "\n$1");
  return t.replace(/\n{3,}/g, "\n\n").trim();
}

/** Разбивает one-line sequenceDiagram / graph от LLM. */
export function formatMermaidInner(inner) {
  const peeled = peelInitDirective(inner);
  let s = quoteSubgraphTitles(repairLlMText(peeled.body).trim());
  const initPrefix = peeled.init;
  if (!s) return initPrefix || s;
  s = splitFlowchartOneLine(s);
  if (isXychartMermaid(s)) {
    const out = formatXychartBody(s);
    return initPrefix ? `${initPrefix}\n${out}` : out;
  }
  const hasInit = Boolean(initPrefix) || /%%\s*\{init:/i.test(s);
  if (!hasInit) {
    s = ensureSequenceInit(s);
    s = ensureFlowchartInit(s);
  }
  s = quoteFlowchartNodeLabels(s);

  s = s.replace(/^(sequenceDiagram(?:\s+autonumber)?)\s+/i, "$1\n");
  s = s.replace(/^(graph\s+(?:TD|LR|BT|RL));?\s*/im, "$1\n");
  s = s.replace(/^(flowchart\s+(?:TD|LR|BT|RL)?);?\s*/im, "$1\n");

  const isSeq = /^sequenceDiagram\b/im.test(s);
  if (isSeq) {
    const blockKw =
      /\s+(participant\s|actor\s|rect\s|loop\s|alt\s|opt\s|par\s|else\s|critical\s|break\s)/gi;
    s = s.replace(blockKw, "\n$1");

    s = s.replace(/\s+(Note\s+(?:over|left of|right of)\s)/gi, "\n$1");
    s = s.replace(/\s+(activate\s|deactivate\s)/gi, "\n$1");
    s = s.replace(
      /\s+([A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:)/g,
      "\n$1",
    );
  }

  s = s.replace(/\s+(subgraph\s)/gi, "\n$1");
  s = s.replace(/\s+(end)\b/gi, "\n$1");

  s = quoteParticipantAliases(s);
  s = quoteLoopLabels(s);
  s = sanitizeNoteLines(s);
  s = quoteArrowMessages(s);
  s = expandGluedFlowchartLines(s);
  s = quoteInlineFlowchartNodes(s);
  s = s.replace(/\n{3,}/g, "\n\n");
  s = s.trim();
  if (initPrefix) {
    const bodyOnly = peelInitDirective(s).body.trim();
    return `${initPrefix}\n${bodyOnly}`;
  }
  return s;
}

function stripFence(d) {
  let inner = d.replace(/^```(?:mermaid)?\s*/i, "").trim();
  inner = inner.replace(/```\s*$/i, "").trim();
  return inner;
}

export function repairDiagramMarkdown(diagram) {
  let d = stripOuterQuotes(repairLlMText((diagram || "").trim()));
  if (!d) return d;
  for (let i = 0; i < 4; i += 1) {
    let inner;
    if (d.startsWith("```")) {
      inner = formatMermaidInner(stripFence(d));
      const next = inner ? "```mermaid\n" + inner + "\n```" : d;
      if (next === d) break;
      d = next;
    } else if (MERMAID_RAW_START.test(d) || MERMAID_INIT_RE.test(d)) {
      const peeled = peelInitDirective(d);
      const body = peeled.body || d;
      inner = formatMermaidInner(
        peeled.init ? `${peeled.init}\n${body}` : body,
      );
      const next = inner ? "```mermaid\n" + inner + "\n```" : d;
      if (next === d) break;
      d = next;
    } else {
      const lifted = liftPseudoFlowchart(d);
      if (lifted) {
        inner = formatMermaidInner(lifted);
        d = inner ? "```mermaid\n" + inner + "\n```" : d;
        continue;
      }
      break;
    }
  }
  return d;
}

function extractOnce(text) {
  const trimmed = stripOuterQuotes(repairLlMText((text || "").trim()));
  if (!trimmed) return "";

  const peeled = peelInitDirective(trimmed);
  const core = peeled.body || trimmed;

  if (trimmed.startsWith("```")) {
    return formatMermaidInner(stripFence(trimmed));
  }

  const fenced = trimmed.match(/```(?:mermaid)?\s*([\s\S]*?)```/i);
  if (fenced) return formatMermaidInner(fenced[1]);

  if (hasMermaidGraphBody(core)) {
    if (isXychartMermaid(core)) return trimmed.trim();
    return formatMermaidInner(trimmed);
  }

  const firstLine = core.split(/\r?\n/, 1)[0].trim();
  if (MERMAID_RAW_START.test(firstLine)) {
    if (isXychartMermaid(core)) return trimmed.trim();
    return formatMermaidInner(trimmed);
  }

  const lifted = liftPseudoFlowchart(core);
  if (lifted) return formatMermaidInner(peeled.init ? `${peeled.init}\n${lifted}` : lifted);

  return "";
}

/** Извлекает отдельные xychart-блоки (VLM иногда лепит flowchart + chart в один файл). */
function extractStandaloneXychartBlocks(text) {
  const s = stripFence(repairLlMText((text || "").trim()));
  if (!s) return [];
  const blocks = [];
  const re =
    /(?:^|\n)\s*(xychart(?:-beta)?[\s\S]*?)(?=\n\s*(?:flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|sequenceDiagram|classDiagram|xychart)|$)/gi;
  let m;
  while ((m = re.exec(s)) !== null) {
    const b = formatXychartBody((m[1] || "").trim());
    if (
      b.length > 24 &&
      /x-axis/i.test(b) &&
      /y-axis/i.test(b) &&
      (/\bline\b/i.test(b) || /\bbar\b/i.test(b))
    ) {
      blocks.push(b);
    }
  }
  return blocks;
}

function isCleanXychartBlock(block) {
  const b = (block || "").trim();
  if (!b) return false;
  const first = b.split(/\r?\n/, 1)[0].trim();
  if (!/^xychart(?:-beta)?\b/i.test(first)) return false;
  if (/\bflowchart\b/i.test(b)) return false;
  if (/-->/i.test(b)) return false;
  return true;
}

function parseCommaSeparatedLabels(raw) {
  const t = (raw || "").trim();
  if (!t) return [];
  const out = [];
  let cur = "";
  let inQ = false;
  let q = "";
  for (let i = 0; i < t.length; i += 1) {
    const c = t[i];
    if ((c === "'" || c === '"') && !inQ) {
      inQ = true;
      q = c;
      continue;
    }
    if (inQ && c === q) {
      inQ = false;
      if (cur.trim()) out.push(cur.trim());
      cur = "";
      continue;
    }
    if (!inQ && c === ",") {
      if (cur.trim()) out.push(cur.trim().replace(/^['"]|['"]$/g, ""));
      cur = "";
      continue;
    }
    cur += c;
  }
  if (cur.trim()) out.push(cur.trim().replace(/^['"]|['"]$/g, ""));
  return out.filter(Boolean);
}

function readBracketContent(s, openBracketIdx) {
  let depth = 0;
  const start = openBracketIdx + 1;
  for (let i = openBracketIdx; i < s.length; i += 1) {
    if (s[i] === "[") depth += 1;
    else if (s[i] === "]") {
      depth -= 1;
      if (depth === 0) return { inner: s.slice(start, i).trim(), end: i };
    }
  }
  return null;
}

function seriesNumbersFromInner(inner) {
  const cleaned = (inner || "")
    .replace(/["']\s*$/g, "")
    .replace(/^["']/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const nums = cleaned.match(/[\d.]+/g);
  if (!nums || nums.length < 2) return null;
  return nums.map((n) => parseFloat(n)).filter((n) => Number.isFinite(n));
}

function extractSeriesLines(s) {
  const series = [];
  const re = /\b(bar|line)\s*\[/gi;
  let m;
  while ((m = re.exec(s)) !== null) {
    const bracketPos = m.index + m[0].length - 1;
    const chunk = readBracketContent(s, bracketPos);
    if (!chunk) continue;
    const nums = seriesNumbersFromInner(chunk.inner);
    if (!nums) continue;
    series.push(`${m[1].toLowerCase()} [${nums.join(", ")}]`);
    re.lastIndex = chunk.end + 1;
  }
  return series;
}

function inferYRangeFromSeries(seriesLines) {
  let max = 0;
  for (const line of seriesLines) {
    for (const n of line.match(/[\d.]+/g) || []) {
      max = Math.max(max, parseFloat(n));
    }
  }
  if (!Number.isFinite(max) || max <= 0) return "0 --> 1";
  if (max <= 1) return "0 --> 1";
  return `0 --> ${Math.ceil(max * 10) / 10}`;
}

/** VLM: xychart-beta внутри flowchart node — собрать валидный xychart-beta. */
export function reconstructXychartFromGarbage(text) {
  let s = stripFence(repairLlMText((text || "").trim()));
  s = s.replace(/%%\s*\{init:[\s\S]*?\}%%/gi, "").trim();
  if (!/x-axis/i.test(s)) return "";
  const hasSeries = /\bbar\s*\[/i.test(s) || /\bline\s*\[/i.test(s);
  if (!hasSeries) return "";
  const mixedGarbage =
    /\bflowchart\b/i.test(s) &&
    (/xychart/i.test(s) || /xychart_beta/i.test(s) || /-->/i.test(s));
  if (!mixedGarbage && !/^xychart/i.test(s)) {
    return "";
  }

  const titleM =
    s.match(/xychart-beta\s+title\s+['"]([^'"]+)['"]/i) ||
    s.match(/title\s+['"]([^'"]+)['"]/i);
  const title = titleM ? titleM[1].replace(/\s+/g, " ").trim() : "";

  const xAxisM = s.match(/x-axis\s*\[([\s\S]*?)\]/i);
  const xLabels = xAxisM ? parseCommaSeparatedLabels(xAxisM[1]) : [];

  const yAxisM = s.match(
    /y-axis\s+['"]([^'"]+)['"]\s*([\d.]+\s*(?:to|-->|--)\s*[\d.]+|[\d.]+)?/i,
  );
  const series = extractSeriesLines(s);
  if (!series.length) return "";

  let yLine = `y-axis "Метрика" ${inferYRangeFromSeries(series)}`;
  if (yAxisM) {
    const label = yAxisM[1].replace(/"/g, "'");
    let range = (yAxisM[2] || "").trim().replace(/\bto\b/i, "-->");
    if (range && !/-->/.test(range) && /^[\d.]+$/.test(range)) {
      range = `0 --> ${range}`;
    }
    if (!range) range = inferYRangeFromSeries(series);
    yLine = `y-axis "${label}" ${range}`;
    const maxInData = inferYRangeFromSeries(series).match(/([\d.]+)\s*$/);
    const maxY = maxInData ? parseFloat(maxInData[1]) : 1;
    const declared = range.match(/-->\s*([\d.]+)/);
    if (declared && parseFloat(declared[1]) < maxY) {
      yLine = `y-axis "${label}" ${inferYRangeFromSeries(series)}`;
    }
  }

  let out = "xychart-beta\n";
  if (title) out += `title "${title.replace(/"/g, "'")}"\n`;
  if (xLabels.length) {
    const allNum = xLabels.every((l) => /^\d+$/.test(l.trim()));
    if (allNum) {
      out += `x-axis [${xLabels.join(", ")}]\n`;
    } else {
      out += `x-axis [${xLabels.map((l) => `"${l.replace(/\s+/g, " ").replace(/"/g, "'")}"`).join(", ")}]\n`;
    }
  } else if (xAxisM && /^\d/.test(xAxisM[1].trim())) {
    const nums = xAxisM[1].trim().replace(/["']\s*$/g, "").replace(/[^\d,\s.]/g, "");
    out += `x-axis [${nums}]\n`;
  }
  out += `${yLine}\n`;
  out += series.join("\n");
  return formatXychartBody(out);
}

/** Для рендера: чистый xychart из «грязного» ответа VLM, иначе обычный extract. */
export function extractRenderableMermaid(text) {
  const xycharts = extractStandaloneXychartBlocks(text).filter(isCleanXychartBlock);
  if (xycharts.length) {
    return xycharts[xycharts.length - 1];
  }
  const reconstructed = reconstructXychartFromGarbage(text);
  if (reconstructed) return reconstructed;
  const src = extractMermaidSource(text);
  if (src && /\bflowchart\b/i.test(src) && /x-axis/i.test(text || "")) {
    const again = reconstructXychartFromGarbage(text);
    if (again) return again;
  }
  return src;
}

/** Извлекает исходник Mermaid для mermaid.run (без fence). */
export function extractMermaidSource(text) {
  const repaired = repairDiagramMarkdown(text);
  let src = extractOnce(repaired);
  if (!src) src = extractOnce(text);
  if (!src) {
    const stripped = stripOuterQuotes(text);
    if (stripped !== (text || "").trim()) {
      src = extractOnce(repairDiagramMarkdown(stripped));
    }
  }
  if (src) {
    const again = formatMermaidInner(src);
    if (again) src = again;
  }
  return src;
}

/** Убрать autonumber при повторной попытке render. */
export function softenMermaidSource(source) {
  let s = (source || "").trim();
  s = s.replace(/^sequenceDiagram\s+autonumber/im, "sequenceDiagram");
  const peeled = peelInitDirective(s);
  if (peeled.init) return peeled.body.trim();
  return s;
}
