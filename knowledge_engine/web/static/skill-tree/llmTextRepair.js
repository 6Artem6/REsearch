/** Литералы \\n из structured LLM → реальные переносы (не ломая \\times, \\neq, …). */
const NL_NOT_LATEX = /\\n(?!eq|ot|u|abla|eg|mid|otin|rightarrow|leftarrow|warrow|earrow|i|pm|subset|cap|cup|warrow|exists|cong|sim|propto|fancy|atural|egative)/g;
const TAB_NOT_LATEX = /\\t(?!imes|ext|heta|au|an|o|op|riangleq|ilde|hicksim|o|frac|iny|bf|it|extbf|extrm|extit|exttt|ilde|woheadrightarrow)/g;
const MATH_SEG_FOR_ESC = /\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g;

const TIMES_EXP = "(?:\\^[\\d{]+|\\^\\{[^{}]+\\})?";

const UNICODE_SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const UNICODE_SUPER_MAP = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9" };

function normalizeUnicodeMathExponents(text) {
  if (!text || !/[⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(text)) return text;
  return String(text).replace(/(\d)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g, (_m, base, sup) => {
    const digits = [...sup].map((ch) => UNICODE_SUPER_MAP[ch] || ch).join("");
    return `${base}^${digits}`;
  });
}

export function healTabCorruptedTimes(text) {
  if (!text) return "";
  let s = normalizeUnicodeMathExponents(text);
  if (!/[\t]/.test(s) && !/imes/i.test(s) && !/×/.test(s) && !/\\t/.test(s)) return s;
  s = s.replace(/\\t\s+imes/gi, "\\times");
  s = s.replace(/[\t]+imes/gi, "\\times");
  s = s.replace(
    new RegExp(`k[\\t ]+imes\\s*(\\d+)${TIMES_EXP}`, "gi"),
    (_m, n, exp) => `$k \\times ${n}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`[\\t]+imes\\s*(\\d+)${TIMES_EXP}`, "gi"),
    (_m, n, exp) => `$\\times ${n}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`k[\\t\\s]*×\\s*(\\d+)${TIMES_EXP}`, "g"),
    (_m, n, exp) => `$k \\times ${n}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`[\\t]+×\\s*(\\d+)${TIMES_EXP}`, "g"),
    (_m, n, exp) => `$\\times ${n}${exp || ""}$`,
  );
  return s;
}

/** Уже испорченный imes10 / k imes 10 / $k  imes 10$ → $\\times 10$. */
export function healBrokenTimesMarkup(text) {
  if (!text) return "";
  let s = healTabCorruptedTimes(text);
  if (!/imes/i.test(s) && !/×/.test(s) && !/\\times/.test(s)) return s;
  s = s.replace(
    /\$\s*k\s+imes\s*(\d+)(\^[\d{]+|\^\{[^{}]+\})?\s*\$/gi,
    (_m, n, exp) => `$k \\times ${n}${exp || ""}$`,
  );
  s = s.replace(
    /\$\s*imes\s*(\d+)(\^[\d{]+|\^\{[^{}]+\})?\s*\$/gi,
    (_m, n, exp) => `$\\times ${n}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`(\\d)\\s*k\\s+imes\\s*(\\d+)${TIMES_EXP}`, "gi"),
    (_m, a, b, exp) => `${a} $\\times ${b}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`(\\d)\\s*imes(\\d+)${TIMES_EXP}`, "g"),
    (_m, a, b, exp) => `${a} $\\times ${b}${exp || ""}$`,
  );
  s = s.replace(
    new RegExp(`k\\s+imes\\s*(\\d+)${TIMES_EXP}`, "gi"),
    (_m, n, exp) => `$k \\times ${n}${exp || ""}$`,
  );
  s = s.replace(new RegExp(`kimes(\\d+)${TIMES_EXP}`, "g"), (_m, n, exp) => `$k \\times ${n}${exp || ""}$`);
  s = s.replace(
    new RegExp(`(?<![a-zA-Z\\\\])imes\\s*(\\d+)${TIMES_EXP}`, "gi"),
    (_m, n, exp) => `$\\times ${n}${exp || ""}$`,
  );
  s = s.replace(/(\$\\times \d+[^$]*\$)(?:\1)+/g, "$1");
  return s;
}

const BARE_FRAC_GLUE_RE = /(?<![a-zA-Z\\])frac(\d+)[·⋅]\s*10(\d)(\d)(?![0-9\w])/gi;

const BARE_FRAC_SPAN_RE =
  /(?:\\frac|(?<![a-zA-Z])frac)(?:\{[^}]+\}|\d+)?(?:\s*(?:\\cdot|·|⋅)\s*)?[^\n$]{0,120}?(?:\\approx|≈)\s*[\d.,]+(?:\s*(?:\\text\s*\{[^{}]+\}|\\text\{[^{}]+\}|[А-Яа-яёA-Za-z][А-Яа-яёA-Za-z %]{0,30})?)?/gi;

const BARE_K_TIMES_RE = new RegExp(
  `(?<![$\w/])(k[\\t ]*\\\\times\\s*\\d+${TIMES_EXP})(?![$\w])`,
  "gi",
);
const BARE_TIMES_RE = new RegExp(
  `(?<![$\w/])(?<![kK]\\s)(\\\\times\\s*\\d+${TIMES_EXP})(?![$\w])`,
  "gi",
);

function healBrokenFracInner(inner) {
  if (!inner || !/frac/i.test(inner)) return inner || "";
  let s = String(inner).replace(/·/g, "\\cdot ").replace(/⋅/g, "\\cdot ");
  s = s.replace(/(?<![a-zA-Z\\])frac/gi, "\\frac");
  s = s.replace(
    /(?<![a-zA-Z\\])frac(\d+)[·⋅]\s*10(\d)(\d)(?![0-9])/gi,
    (_m, a, pow, den) => `\\frac{${a} \\cdot 10^{${pow}}}{${den}}`,
  );
  s = s.replace(
    /\\frac\{?(\d+)\}?\\cdot\s*10(\d)(\d)(?![0-9])/gi,
    (_m, a, pow, den) => `\\frac{${a} \\cdot 10^{${pow}}}{${den}}`,
  );
  s = s.replace(
    /\\frac\{?(\d+)\}?\s*(?:\\cdot\s*)?(\d+(?:\^[\d{]+|\^\{[^{}]+\})?)\s+(\d{1,4})(\s*(?:\\approx|≈|=(?!=))\s*)?/g,
    (_m, a, b, c, tail) => `\\frac{${a} \\cdot ${b}}{${c}}${tail || " "}`,
  );
  s = s.replace(
    /\\frac\{([^}]+)\}\s+(\d{1,4})\s*(\\approx|≈)?/g,
    (_m, num, den, approx) => `\\frac{${num}}{${den}}${approx ? ` ${approx}` : " "}`,
  );
  return s;
}

function wrapBareLatexOutsideMath(text) {
  if (!text) return "";
  const chunks = [];
  let last = 0;
  const re = /\$\$[\s\S]+?\$\$|(?<!\$)\$[^$\n]+?\$(?!\$)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) chunks.push(["plain", text.slice(last, m.index)]);
    chunks.push(["math", m[0]]);
    last = m.index + m[0].length;
  }
  if (last < text.length) chunks.push(["plain", text.slice(last)]);
  if (!chunks.length) chunks.push(["plain", text]);

  function fixPlain(chunk) {
    let out = chunk;
    if (/frac/i.test(out)) {
      out = out.replace(BARE_FRAC_GLUE_RE, (seg) => {
        const body = healBrokenFracInner(healCorruptedTimesInMathInner(seg));
        return `$${body}$`;
      });
      out = out.replace(BARE_FRAC_SPAN_RE, (seg) => {
        const body = healBrokenFracInner(healCorruptedTimesInMathInner(seg));
        return `$${body}$`;
      });
    }
    if (/\\times|imes|×/i.test(out)) {
      out = out.replace(BARE_K_TIMES_RE, (_m, expr) => `$${healCorruptedTimesInMathInner(expr)}$`);
      out = out.replace(BARE_TIMES_RE, (_m, expr) => `$${healCorruptedTimesInMathInner(expr)}$`);
    }
    return out;
  }

  return chunks.map(([kind, val]) => (kind === "math" ? val : fixPlain(val))).join("");
}

export function repairBrokenLatex(text) {
  if (!text) return "";
  let s = healTabCorruptedTimes(text);
  s = healBrokenTimesMarkup(s);
  s = wrapBareLatexOutsideMath(s);
  s = String(s).replace(MATH_SEG_FOR_ESC, (seg) => {
    const isDisplay = seg.startsWith("$$");
    const inner = seg.replace(/^\$+/, "").replace(/\$+$/, "");
    const body = healBrokenFracInner(healCorruptedTimesInMathInner(inner));
    return isDisplay ? `$$${body}$$` : `$${body}$`;
  });
  return s;
}

export function healCorruptedTimesInMathInner(inner) {
  if (!inner) return inner || "";
  if (!/imes/i.test(inner) && !/[\t]/.test(inner) && !/×/.test(inner)) return inner;
  let s = String(inner);
  s = s.replace(/\\t\s+imes/gi, "\\times");
  s = s.replace(/[\t]+imes/gi, "\\times ");
  s = s.replace(/k[\t ]+imes\s*/gi, "\\times ");
  s = s.replace(/[\t]+×\s*/g, "\\times ");
  s = s.replace(/k\s+imes\s*/gi, "\\times ");
  s = s.replace(/(?<![a-zA-Z\\])imes\s*(\d+)/gi, "\\times $1");
  return s;
}

export function repairLlMText(text) {
  if (!text) return "";
  let t = healTabCorruptedTimes(text);
  const mathSlots = [];
  t = String(t).replace(MATH_SEG_FOR_ESC, (seg) => {
    mathSlots.push(seg);
    return `\uE000M${mathSlots.length - 1}\uE001`;
  });
  for (let i = 0; i < 6; i += 1) {
    const next = t
      .replace(/\\r\\n/g, "\n")
      .replace(/\\r/g, "\n")
      .replace(NL_NOT_LATEX, "\n")
      .replace(TAB_NOT_LATEX, "\t");
    if (next === t) break;
    t = next;
  }
  mathSlots.forEach((seg, idx) => {
    let fixed = seg;
    if (/imes/i.test(fixed) || /[\t]/.test(fixed) || /×/.test(fixed)) {
      const isDisplay = fixed.startsWith("$$");
      const inner = fixed.replace(/^\$+/, "").replace(/\$+$/, "");
      const body = healCorruptedTimesInMathInner(inner);
      fixed = isDisplay ? `$$${body}$$` : `$${body}$`;
    }
    t = t.replace(`\uE000M${idx}\uE001`, fixed);
  });
  return repairBrokenLatex(healBrokenTimesMarkup(t));
}

/** Подпункт «а)» / «**а)**» — в regex-литералах `\)` после `(?:` ломает парсер. */
const LETTER_SUB_ITEM_SRC = "\\*\\*[а-яёa-z]\\)\\*\\*|[а-яёa-z]\\)";
const reWrongNumLetter = new RegExp(
  `^\\d{1,2}\\.\\s*(${LETTER_SUB_ITEM_SRC})\\s*(.*)$`,
  "iu",
);
const reOrderedLine = new RegExp(
  `^\\d{1,2}\\.\\s+(?!${LETTER_SUB_ITEM_SRC})`,
  "iu",
);
const reSubLineStart = new RegExp(
  "^(?:\\*\\*[а-яёa-z]\\)\\*\\*|[а-яёa-z]\\)\\s)",
  "iu",
);
const reWrongNumLinePrefix = new RegExp(
  `^\\d{1,2}\\.\\s*(${LETTER_SUB_ITEM_SRC})\\s*`,
  "iu",
);
const reInlineWrongNumLetter = new RegExp(
  `(?<![\\d/])(\\d{1,2})\\.\\s*(${LETTER_SUB_ITEM_SRC})\\s+`,
  "giu",
);
const reSplitBeforeOrdered = new RegExp(
  `\\s+(?=\\d{1,2}\\.\\s+(?!${LETTER_SUB_ITEM_SRC}))`,
  "iu",
);
const reSplitBeforeSubitem = new RegExp(
  `\\n\\s*(?=(?:\\*\\*[а-яёa-z]\\)\\*\\*|[а-яёa-z]\\)\\s))`,
  "iu",
);
const reOlWrongLetter = new RegExp(
  `<ol>((?:(?!<\\/ol>).)*\\d{1,2}\\.\\s*(?:${LETTER_SUB_ITEM_SRC})(?:(?!<\\/ol>).)*?)<\\/ol>`,
  "giu",
);
const reLiWrongLetter = new RegExp(
  `<li>\\s*\\d{1,2}\\.\\s*(${LETTER_SUB_ITEM_SRC})\\s*(.*?)<\\/li>`,
  "giu",
);
const reLiRealOrdered = new RegExp(
  `<li>\\s*\\d{1,2}\\.\\s+(?!${LETTER_SUB_ITEM_SRC})`,
  "iu",
);
const reSubInParagraph = new RegExp(
  "<p>((?:(?!<\\/p>).)*(?:\\n|<br\\s*\\/?>)\\s*(?:\\*\\*[а-яёa-z]\\)\\*\\*|[а-яёa-z]\\)\\s)(?:(?!<\\/p>).)+)<\\/p>",
  "giu",
);

const reMathOperatorTail = new RegExp("[+*/=,]\\s*$");

function letterSubitemIsParenContinuation(prefix) {
  const prev = String(prefix || "").replace(/\s+$/u, "");
  if (!prev) return false;
  const open = (prev.match(/\(/g) || []).length;
  const close = (prev.match(/\)/g) || []).length;
  if (open > close) return true;
  if (/[([{]$/u.test(prev)) return true;
  return reMathOperatorTail.test(prev);
}

function stripParagraphInnerHtml(inner) {
  return inner
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .trim();
}

function listHtmlFromParagraphTexts(texts) {
  const segments = [];
  let curKind = null;
  let curItems = [];

  function flush() {
    if (curItems.length) segments.push([curKind || "ol", curItems]);
    curKind = null;
    curItems = [];
  }

  let prevText = "";
  for (const raw of texts) {
    const text = raw.trim();
    if (!text) return null;
    if (reSubLineStart.test(text) && letterSubitemIsParenContinuation(prevText)) {
      return null;
    }
    const wrong = reWrongNumLetter.exec(text);
    if (wrong) {
      const li = `<li>${wrong[1].trim()} ${(wrong[2] || "").trim()}</li>`.trim();
      if (curKind && curKind !== "ul") flush();
      curKind = "ul";
      curItems.push(li);
      prevText = text;
      continue;
    }
    if (reOrderedLine.test(text)) {
      const body = text.replace(/^\d{1,2}\.\s+/, "");
      if (curKind && curKind !== "ol") flush();
      curKind = "ol";
      curItems.push(`<li>${body}</li>`);
      prevText = text;
      continue;
    }
    const bullet = /^-\s+(.*)$/s.exec(text);
    if (bullet) {
      if (curKind && curKind !== "ul") flush();
      curKind = "ul";
      curItems.push(`<li>${bullet[1].trim()}</li>`);
      prevText = text;
      continue;
    }
    if (reSubLineStart.test(text)) {
      if (curKind && curKind !== "ul") flush();
      curKind = "ul";
      curItems.push(`<li>${text}</li>`);
      prevText = text;
      continue;
    }
    return null;
  }
  flush();
  if (!segments.length) return null;
  const parts = [];
  for (const [kind, items] of segments) {
    if (!items.length) continue;
    const tag = kind === "ol" ? "ol" : "ul";
    parts.push(`<${tag}>${items.join("")}</${tag}>`);
  }
  return parts.length ? parts.join("") : null;
}

function mergeAdjacentParagraphLists(html) {
  const raw = String(html || "");
  if (!raw.includes("<p>")) return raw;
  let s = raw.replace(/<p>\s*-\s*<\/p>/gi, "");
  const chunks = [];
  let pos = 0;
  while (pos < s.length) {
    const ws = s.slice(pos).match(/^\s*/);
    if (ws && ws[0]) {
      chunks.push(["raw", ws[0]]);
      pos += ws[0].length;
      if (pos >= s.length) break;
    }
    const pM = s.slice(pos).match(/^<p>([\s\S]*?)<\/p>/i);
    if (pM) {
      chunks.push(["p", stripParagraphInnerHtml(pM[1])]);
      pos += pM[0].length;
      continue;
    }
    chunks.push(["raw", s.slice(pos)]);
    break;
  }
  const out = [];
  let i = 0;
  while (i < chunks.length) {
    const [kind, val] = chunks[i];
    if (kind !== "p") {
      out.push(val);
      i += 1;
      continue;
    }
    const run = [];
    let j = i;
    while (j < chunks.length && chunks[j][0] === "p") {
      run.push(chunks[j][1]);
      j += 1;
    }
    const merged = listHtmlFromParagraphTexts(run);
    if (merged) out.push(merged);
    else for (const line of run) out.push(`<p>${line}</p>`);
    i = j;
  }
  return out.join("");
}

function collapseBlankLinesInListRuns(text) {
  const lines = text.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    const stripped = lines[i].trim();
    if (stripped) {
      out.push(lines[i]);
      continue;
    }
    if (!out.length) {
      out.push(lines[i]);
      continue;
    }
    let j = i + 1;
    while (j < lines.length && !lines[j].trim()) j += 1;
    const nextS = j < lines.length ? lines[j].trim() : "";
    const prevS = out[out.length - 1].trim();
    if (
      reOrderedLine.test(prevS) &&
      (reOrderedLine.test(nextS) ||
        reWrongNumLetter.test(nextS) ||
        nextS === "-")
    ) {
      continue;
    }
    out.push(lines[i]);
  }
  return out.join("\n");
}

function collapseTableInternalBlankLines(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const stripped = lines[i].trim();
    if (stripped.startsWith("|") && (stripped.match(/\|/g) || []).length >= 2) {
      const block = [];
      while (i < lines.length) {
        const s = lines[i].trim();
        if (!s) {
          let j = i + 1;
          while (j < lines.length && !lines[j].trim()) j += 1;
          if (j < lines.length && lines[j].trim().startsWith("|")) {
            i += 1;
            continue;
          }
          break;
        }
        if (s.startsWith("|")) {
          block.push(lines[i].replace(/\s+$/, ""));
          i += 1;
        } else break;
      }
      out.push(...block);
      continue;
    }
    out.push(lines[i]);
    i += 1;
  }
  return out.join("\n");
}

function repairMarkdownTablesLayout(text) {
  const lines = [];
  for (const line of text.split("\n")) {
    const stripped = line.trim();
    if (!stripped) {
      lines.push(line);
      continue;
    }
    let work = stripped;
    if (!work.startsWith("|") && work.includes("|")) {
      const pipeIdx = work.indexOf("|");
      const before = work.slice(0, pipeIdx).trimEnd();
      const rest = work.slice(pipeIdx);
      if (before && (rest.match(/\|/g) || []).length >= 2) {
        lines.push(before);
        lines.push("");
        work = rest;
      }
    }
    if ((work.match(/\|/g) || []).length >= 2) {
      work = work.replace(/\|\s+\|/g, "|\n|");
    }
    for (const subLine of work.split("\n")) {
      const sm = subLine.trim();
      if (!sm) {
        lines.push("");
        continue;
      }
      const rowMatch = /^(.*\|)\s+([А-ЯЁA-ZВЁ][^|]+)$/u.exec(sm);
      if (rowMatch && (rowMatch[1].match(/\|/g) || []).length >= 2) {
        lines.push(rowMatch[1].trim());
        lines.push("");
        lines.push(rowMatch[2].trim());
      } else {
        lines.push(sm);
      }
    }
  }
  return collapseTableInternalBlankLines(lines.join("\n"));
}

function normalizeListBlocksForMarkdown(text) {
  const lines = text.split("\n");
  const out = [];
  const buf = [];

  function flushBuf() {
    if (!buf.length) return;
    if (out.length && out[out.length - 1].trim()) out.push("");
    out.push(...buf);
    buf.length = 0;
  }

  function appendBullet(lineText) {
    let bullet = lineText.trim();
    if (!bullet) return;
    if (!bullet.startsWith("-")) bullet = `- ${bullet}`;
    if (out.length && out[out.length - 1].trim().startsWith("-")) {
      out.push(bullet);
    } else if (out.length && out[out.length - 1].trim()) {
      out.push("");
      out.push(bullet);
    } else {
      out.push(bullet);
    }
  }

  for (const line of lines) {
    const stripped = line.trim();
    if (!stripped || stripped === "-") {
      flushBuf();
      if (stripped === "-") continue;
      out.push(line);
      continue;
    }
    const wrong = reWrongNumLetter.exec(stripped);
    if (wrong) {
      flushBuf();
      appendBullet(`${wrong[1].trim()} ${(wrong[2] || "").trim()}`.trim());
      continue;
    }
    if (reOrderedLine.test(stripped)) {
      buf.push(stripped);
      continue;
    }
    if (reSubLineStart.test(stripped)) {
      const prevLine = buf.length ? buf[buf.length - 1] : out.length ? out[out.length - 1] : "";
      if (letterSubitemIsParenContinuation(prevLine)) {
        const joined = `${prevLine.replace(/\s+$/u, "")} ${stripped}`;
        if (buf.length) buf[buf.length - 1] = joined;
        else if (out.length) out[out.length - 1] = joined;
        else out.push(stripped);
        continue;
      }
      flushBuf();
      appendBullet(stripped);
      continue;
    }
    if (stripped.startsWith("- ")) {
      flushBuf();
      appendBullet(stripped.replace(/^-/, "").trim());
      continue;
    }
    flushBuf();
    out.push(line);
  }
  flushBuf();
  return out.join("\n");
}

function splitGluedOrderedParagraph(body) {
  let parts = body.split(/(?<=[.!?…])\s+(?=\d{1,2}\.\s+)/u);
  if (parts.length < 2) {
    parts = body.split(reSplitBeforeOrdered);
  }
  if (parts.length < 2) return null;
  const items = [];
  const subItems = [];
  for (const part of parts) {
    const chunk = part.trim();
    if (!chunk) continue;
    const wrong = reWrongNumLetter.exec(chunk);
    if (wrong) {
      subItems.push(`<li>${wrong[1]} ${(wrong[2] || "").trim()}</li>`);
      continue;
    }
    const main = chunk.replace(/^\d{1,2}\.\s+/, "");
    if (main) items.push(`<li>${main}</li>`);
  }
  if (items.length < 2 && !subItems.length) return null;
  const ol = items.length >= 2 ? `<ol>${items.join("")}</ol>` : "";
  const ul = subItems.length ? `<ul>${subItems.join("")}</ul>` : "";
  return ol || ul ? `${ol}${ul}` : null;
}

function splitGluedSubitemsParagraph(body) {
  const text = String(body || "").replace(/<br\s*\/?>/gi, "\n");
  const parts = text.split(reSplitBeforeSubitem);
  if (parts.length < 2) return null;
  const merged = [parts[0]];
  for (let i = 1; i < parts.length; i += 1) {
    const prev = merged[merged.length - 1];
    if (letterSubitemIsParenContinuation(prev)) {
      merged[merged.length - 1] = `${prev.replace(/\s+$/u, "")} ${parts[i].replace(/^\s+/u, "")}`;
    } else {
      merged.push(parts[i]);
    }
  }
  const items = merged.map((part) => part.trim()).filter(Boolean);
  return items.length >= 2
    ? `<ul>${items.map((chunk) => `<li>${chunk}</li>`).join("")}</ul>`
    : null;
}

/** Разбить <p> с «1. … 2. …» / «а) … б) …» (зеркало backend postprocess_html_glued_lists). */
export function postprocessTutorHtml(html) {
  const raw = String(html || "");
  if (!raw.includes("<p>") && !raw.includes("<ol")) return raw;
  let out = mergeAdjacentParagraphLists(raw);
  const orderedRe =
    /<p>((?:(?!<\/p>).)*\d{1,2}\.\s+(?:(?!<\/p>).)+)<\/p>/giu;
  out = out.replace(orderedRe, (m, body) => {
    const repl = splitGluedOrderedParagraph(body);
    return repl || m;
  });
  out = out.replace(reOlWrongLetter, (m, inner) => {
    const fixed = inner.replace(
      reLiWrongLetter,
      (_li, label, tail) => `<li>${label} ${(tail || "").trim()}</li>`,
    );
    if (reLiRealOrdered.test(fixed)) {
      return `<ol>${fixed}</ol>`;
    }
    return `<ul>${fixed}</ul>`;
  });
  out = out.replace(reSubInParagraph, (m, body) => {
    const repl = splitGluedSubitemsParagraph(body);
    return repl || m;
  });
  return out;
}

function repairGluedNumberedListsOnLine(line) {
  let s = line;
  if (!s.trim()) return line;
  if (s.trim().startsWith("|") && s.includes("|")) return line;
  const stripped = s.trim();
  if (reWrongNumLinePrefix.test(stripped)) {
    return line;
  }
  s = s.replace(/([:;])\s+(\d+\.\s+)/g, "$1\n$2");
  for (let i = 0; i < 12; i += 1) {
    const next = s.replace(
      /([.!?…])(\s+)(\d{1,2}\.\s+(?:\*\*)?[А-ЯЁA-ZВЁ])/gu,
      "$1\n$3",
    );
    if (next === s) break;
    s = next;
  }
  s = s.replace(
    /([а-яёa-zA-Z0-9)\]»"'№%])(\s+)(\*\*[а-яёa-z]\)\*\*)/giu,
    "$1\n$3",
  );
  s = s.replace(
    /([.!?…:;])(\s+)(\*\*[а-яёa-z]\)\*\*)/giu,
    "$1\n$3",
  );
  s = s.replace(reInlineWrongNumLetter, "\n- $2 ");
  return s;
}

function repairGluedNumberedLists(text) {
  return text
    .split("\n")
    .map((line) => repairGluedNumberedListsOnLine(line))
    .join("\n");
}

function splitMarkdownHeaderLine(line) {
  const s = line.trim();
  if (!s.startsWith("#")) return line;
  const m = /^(#{1,6}\s+)/u.exec(s);
  if (!m) return line;
  const rest = s.slice(m[0].length);
  // Keep numbered ATX headings intact (deep_analysis: ``## 3. Точки отказа…``).
  if (/^\d{1,2}\.\s+\S/u.test(rest)) return line;
  let pm = /\s+(При\s+[а-яё])/u.exec(rest);
  if (!pm) pm = /\s+([А-ЯЁ][а-яё]{2,}\s+[а-яё])/u.exec(rest);
  if (pm) {
    const title = s.slice(0, m[0].length + pm.index).trim();
    const body = rest.slice(pm.index).trim();
    return `${title}\n\n${body}`;
  }
  return line;
}

function rejoinSplitNumberedHeadings(text) {
  const lines = String(text || "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const stripped = lines[i].trim();
    const hm = /^(#{1,6})\s*(\d{1,2})\.\s*$/u.exec(stripped);
    if (hm) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j += 1;
      if (j < lines.length) {
        const nxt = lines[j].trim();
        if (
          nxt &&
          !nxt.startsWith("#") &&
          !/^\d{1,2}\.\s+/u.test(nxt) &&
          !nxt.startsWith("```") &&
          !nxt.startsWith("|")
        ) {
          out.push(`${hm[1]} ${hm[2]}. ${nxt}`);
          i = j + 1;
          continue;
        }
      }
    }
    out.push(lines[i]);
    i += 1;
  }
  return out.join("\n");
}

const CODE_FENCE_RE = /```[^\n`]*\n[\s\S]*?```/g;
const PY_STMT_START = /^(?:class |def |elif |else:|return |if |for |while |self\.|# |import |from )/i;

function applyOutsideCodeFences(text, fn) {
  const parts = [];
  let last = 0;
  let m;
  const re = new RegExp(CODE_FENCE_RE.source, "g");
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      let prefix = text.slice(last, m.index);
      if (prefix && !prefix.endsWith("\n")) prefix = `${prefix.replace(/\s+$/, "")}\n\n`;
      parts.push(fn(prefix));
    }
    let fence = m[0];
    const after = m.index + m[0].length;
    if (after < text.length && text[after] !== "\n") {
      fence = `${fence.replace(/\s+$/, "")}\n\n`;
    }
    parts.push(fence);
    last = after;
  }
  if (last < text.length) parts.push(fn(text.slice(last)));
  return parts.join("");
}

function basicPythonIndent(body) {
  const lines = body.split("\n");
  const out = [];
  let inClass = false;
  let inDef = false;
  for (const ln of lines) {
    const st = ln.trim();
    if (!st) {
      out.push("");
      continue;
    }
    if (st.startsWith("class ")) {
      inClass = true;
      inDef = false;
      out.push(st);
      continue;
    }
    if (st.startsWith("def ")) {
      inDef = true;
      out.push((inClass ? "    " : "") + st);
      continue;
    }
    if (st.startsWith("elif ") || st.startsWith("else:")) {
      const pad = inClass && inDef ? "        " : inDef || inClass ? "    " : "";
      out.push(pad + st);
      continue;
    }
    if (inClass && inDef) out.push("        " + st);
    else if (inDef || inClass) out.push("    " + st);
    else out.push(st);
  }
  return out.join("\n");
}

function reflowGluedPython(src) {
  if (!src || (!src.includes("def ") && !src.includes("class "))) return src;
  let s = src;
  if (s.split("\n").length < 2 && s.includes("\\n")) s = s.replace(/\\n/g, "\n");
  s = s.replace(/(\) -> [^:\n]+:)(\s*)(?=\S)/g, "$1\n");
  const defCount = (s.match(/\bdef /g) || []).length;
  const classCount = (s.match(/\bclass /g) || []).length;
  if (s.split("\n").length < 2 || defCount + classCount > s.split("\n").length / 2) {
    for (let i = 0; i < 24; i += 1) {
      const prev = s;
      s = s
        .replace(/:(\s*)(?=def |class )/g, ":\n")
        .replace(/\):(\s*)(?=self\.|return |if |elif |else:|def |class )/g, "):\n")
        .replace(/(?<=[\w)])\s*(?=def )/g, "\n")
        .replace(/(?<=[^\n])\s+(?=elif )/g, "\n")
        .replace(/(?<=[^\n])\s+(?=else:)/g, "\n")
        .replace(/(?<=[^\n])\s+(?=return )/g, "\n")
        .replace(/(?<=[^\n])\s+(?=# )/g, "\n")
        .replace(/(?<=[\w"'])(?=(?:self\.|elif |else:|return ))/g, "\n")
        .replace(/(?<!el)(?<=[a-z0-9_])(?=if )/g, "\n")
        .replace(/:(\s*)(?=return )/g, ":\n")
        .replace(/(?<=[^\n])(?=#)/g, "\n");
      if (s === prev) break;
    }
    s = basicPythonIndent(s);
  } else if (s.includes("def ") || s.includes("class ")) {
    s = basicPythonIndent(s);
  }
  return s;
}

function repairFenceInner(chunk) {
  const m = /^(```[^\n]*\n)([\s\S]*?)(```\s*)$/s.exec(chunk.trim());
  if (!m) return chunk;
  return `${m[1]}${reflowGluedPython(m[2]).trimEnd()}\n${m[3]}`;
}

function wrapBarePythonRegions(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const st = lines[i].trim();
    if (st.startsWith("class ") || (st.startsWith("def ") && st.includes("def ") && st.includes(":"))) {
      const block = [];
      let j = i;
      while (j < lines.length) {
        const ln = lines[j];
        const ls = ln.trim();
        if (!ls) {
          let k = j + 1;
          while (k < lines.length && !lines[k].trim()) k += 1;
          const nxt = k < lines.length ? lines[k].trim() : "";
          if (
            block.length &&
            (nxt.startsWith("def ") || PY_STMT_START.test(nxt) || nxt.startsWith("self."))
          ) {
            block.push(ln);
            j += 1;
            continue;
          }
          if (block.length) {
            j += 1;
            break;
          }
          j += 1;
          continue;
        }
        if (block.length && ls.startsWith("#")) {
          block.push(ln);
          j += 1;
          continue;
        }
        if (
          block.length &&
          !PY_STMT_START.test(ls) &&
          !ln.startsWith("    ") &&
          !ln.startsWith("\t")
        ) {
          if (ls.startsWith("#") || ls.startsWith("###")) break;
          if (!ls.startsWith("self.") && !ls.startsWith("return ") && !ls.startsWith("elif ") && !ls.startsWith("else:")) {
            break;
          }
        }
        if (ls.startsWith("###")) break;
        block.push(ln);
        j += 1;
      }
      const body = reflowGluedPython(block.join("\n"));
      if (body.includes("def ") || body.includes("class ")) {
        out.push(`\`\`\`python\n${body.trim()}\n\`\`\``);
      } else {
        out.push(...block);
      }
      i = j;
      continue;
    }
    out.push(lines[i]);
    i += 1;
  }
  return out.join("\n");
}

function detachGluedCodeFences(text) {
  const raw = text || "";
  if (!raw.includes("```")) return raw;
  const parts = [];
  let last = 0;
  let m;
  const re = new RegExp(CODE_FENCE_RE.source, "g");
  while ((m = re.exec(raw)) !== null) {
    let prefix = raw.slice(last, m.index);
    if (prefix && !prefix.endsWith("\n")) prefix = `${prefix.replace(/\s+$/, "")}\n\n`;
    let fence = m[0];
    const after = m.index + m[0].length;
    if (after < raw.length && raw[after] !== "\n") fence = `${fence.replace(/\s+$/, "")}\n\n`;
    parts.push(prefix, fence);
    last = after;
  }
  parts.push(raw.slice(last));
  return parts.join("");
}

export function repairLectureCodeBlocks(text) {
  const raw = detachGluedCodeFences((text || "").trim());
  if (!raw || (!raw.includes("def ") && !raw.includes("class "))) return raw || text || "";
  const parts = [];
  let last = 0;
  let m;
  const re = new RegExp(CODE_FENCE_RE.source, "g");
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) parts.push(wrapBarePythonRegions(raw.slice(last, m.index)));
    parts.push(repairFenceInner(m[0]));
    last = m.index + m[0].length;
  }
  if (last < raw.length) parts.push(wrapBarePythonRegions(raw.slice(last)));
  return parts.length ? parts.join("") : wrapBarePythonRegions(raw);
}

/** После стрима лекции: заголовки, таблицы, нумерованные списки (зеркало backend). */
export function repairLectureMarkdownLayout(text) {
  let t = repairLlMText(text).trim();
  if (!t) return "";
  t = repairLectureCodeBlocks(t);
  const layoutChunk = (chunk) => {
    let c = chunk.replace(/([.!?…])\s+(#{1,6}\s+)/g, "$1\n\n$2");
    c = c.replace(
      /(?<=[а-яА-ЯёЁa-zA-Z0-9)\]»"'№%])(\s+)(#{1,6}\s+)/gu,
      "\n\n$2",
    );
    c = c.replace(/([.!?…])\s+(\|)/g, "$1\n\n$2");
    const headerSplit = c.split("\n").map((line) => splitMarkdownHeaderLine(line));
    c = repairMarkdownTablesLayout(headerSplit.join("\n"));
    c = repairGluedNumberedLists(c);
    c = collapseBlankLinesInListRuns(c);
    c = normalizeListBlocksForMarkdown(c);
    c = rejoinSplitNumberedHeadings(c);
    return c;
  };
  t = applyOutsideCodeFences(t, layoutChunk);
  t = rejoinSplitNumberedHeadings(t);
  t = t.replace(/\n{3,}/g, "\n\n");
  return t.trim();
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
          /* skip */
        }
        return null;
      }
    }
  }
  return null;
}

export function structuredAnalysisToHtml(text) {
  const raw = repairLlMText(text).trim();
  if (!raw || !raw.includes("{")) return "";
  const hit = tryParseJsonObjectAt(raw, 0);
  if (!hit) return "";
  return formatAnalysisObjectHtml(hit.obj);
}

function renderMarkdownInline(text) {
  const codeSlots = [];
  let html = escapeHtml(text).replace(/`([^`\n]+)`/g, (_m, code) => {
    const key = `\uE010${codeSlots.length}\uE011`;
    codeSlots.push(`<code>${code}</code>`);
    return key;
  });
  html = html.replace(
    /\[([^\]]+)]\((https?:\/\/[^ )]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  html = html
    .replace(/(\*\*|__)(.+?)\1/g, "<strong>$2</strong>")
    .replace(/~~(.+?)~~/g, "<del>$1</del>")
    .replace(/(^|[^\w*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^\w_])_([^_\n]+)_/g, "$1<em>$2</em>");
  return html.replace(/\uE010(\d+)\uE011/g, (_m, index) => codeSlots[Number(index)]);
}

/**
 * Безопасный минимальный Markdown-рендер для стримящихся ответов тьютора,
 * когда API ещё не предоставил поле contentHtml.
 */
export function tutorMarkdownToHtml(text) {
  const source = repairLectureMarkdownLayout(text || "");
  if (!source) return "";
  const out = [];
  let paragraph = [];
  let listType = "";
  let inCodeFence = false;
  let codeLines = [];

  function closeParagraph() {
    if (!paragraph.length) return;
    out.push(`<p>${paragraph.map(renderMarkdownInline).join("<br>")}</p>`);
    paragraph = [];
  }
  function closeList() {
    if (!listType) return;
    out.push(`</${listType}>`);
    listType = "";
  }
  function openList(type) {
    if (listType === type) return;
    closeList();
    listType = type;
    out.push(`<${type}>`);
  }

  for (const line of source.split("\n")) {
    const fence = /^```([^`]*)$/.exec(line.trim());
    if (fence) {
      closeParagraph();
      closeList();
      if (inCodeFence) {
        out.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }
      inCodeFence = !inCodeFence;
      continue;
    }
    if (inCodeFence) {
      codeLines.push(line);
      continue;
    }
    if (!line.trim()) {
      closeParagraph();
      closeList();
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      closeParagraph();
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${renderMarkdownInline(heading[2])}</h${level}>`);
      continue;
    }
    const unordered = /^\s*[-*+]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      closeParagraph();
      openList(ordered ? "ol" : "ul");
      out.push(`<li>${renderMarkdownInline((unordered || ordered)[1])}</li>`);
      continue;
    }
    const quote = /^>\s?(.*)$/.exec(line);
    if (quote) {
      closeParagraph();
      closeList();
      out.push(`<blockquote><p>${renderMarkdownInline(quote[1])}</p></blockquote>`);
      continue;
    }
    closeList();
    paragraph.push(line);
  }
  if (inCodeFence) out.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  else {
    closeParagraph();
    closeList();
  }
  return out.join("");
}
