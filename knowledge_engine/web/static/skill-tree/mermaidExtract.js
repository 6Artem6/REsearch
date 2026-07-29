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

function wrapLongLabel(text, maxLen = 38) {
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
  return lines.join("\n");
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
  return inner
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*)(.*)$/);
      if (!m) return line;
      const prefix = m[1];
      let body = (m[2] || "").trim().replace(/^"|"$/g, "");
      if (!body) return line;
      body = wrapLongLabel(body.replace(/"/g, "'"), 40);
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
  if (/^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b)/im.test(t)) {
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

/** Разбивает one-line sequenceDiagram / graph от LLM. */
export function formatMermaidInner(inner) {
  let s = quoteSubgraphTitles(repairLlMText(inner).trim());
  if (!s) return s;
  s = ensureSequenceInit(s);

  s = s.replace(/^(sequenceDiagram(?:\s+autonumber)?)\s+/i, "$1\n");
  s = s.replace(/^(graph\s+(?:TD|LR|BT|RL))\s+/i, "$1\n");
  s = s.replace(/^(flowchart\s+(?:TD|LR|BT|RL)?)\s+/i, "$1\n");

  const blockKw =
    /\s+(participant\s|actor\s|rect\s|loop\s|alt\s|opt\s|par\s|and\s|else\s|critical\s|break\s)/gi;
  s = s.replace(blockKw, "\n$1");

  s = s.replace(/\s+(Note\s+(?:over|left of|right of)\s)/gi, "\n$1");
  s = s.replace(/\s+(activate\s|deactivate\s)/gi, "\n$1");
  s = s.replace(/\s+(subgraph\s)/gi, "\n$1");
  s = s.replace(/\s+(end)\b/gi, "\n$1");

  s = s.replace(
    /\s+([A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:)/g,
    "\n$1",
  );

  s = quoteParticipantAliases(s);
  s = quoteLoopLabels(s);
  s = sanitizeNoteLines(s);
  s = quoteArrowMessages(s);
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
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
    } else if (
      /^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b)/i.test(d)
    ) {
      inner = formatMermaidInner(d);
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

  if (trimmed.startsWith("```")) {
    return formatMermaidInner(stripFence(trimmed));
  }

  const fenced = trimmed.match(/```(?:mermaid)?\s*([\s\S]*?)```/i);
  if (fenced) return formatMermaidInner(fenced[1]);

  const firstLine = trimmed.split(/\r?\n/, 1)[0].trim();
  const rawStart =
    /^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|stateDiagram(?:-v2)?\b|erDiagram\b)/i;
  if (rawStart.test(firstLine)) return formatMermaidInner(trimmed);

  const lifted = liftPseudoFlowchart(trimmed);
  if (lifted) return formatMermaidInner(lifted);

  return "";
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
  return (source || "")
    .replace(/^sequenceDiagram\s+autonumber/im, "sequenceDiagram")
    .trim();
}
