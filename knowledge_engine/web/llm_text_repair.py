"""Починка типичного «сырого» текста от LLM (литералы \\n, inline mermaid)."""

from __future__ import annotations

import json
import re

from knowledge_engine.web.linkify import (
    _normalize_unicode_math_exponents,
    heal_broken_times_markup,
    heal_tab_corrupted_times,
    repair_broken_latex,
)

_RAW_MERMAID_INLINE = re.compile(
    r"^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b)",
    re.IGNORECASE,
)

_SEQ_BLOCK_KW = re.compile(
    r"\s+(participant\s|actor\s|rect\s|loop\s|alt\s|opt\s|par\s|else\s|"
    r"critical\s|break\s)",
    re.IGNORECASE,
)
_SEQ_NOTE = re.compile(r"\s+(Note\s+(?:over|left of|right of)\s)", re.IGNORECASE)
_SEQ_ACT = re.compile(r"\s+(activate\s|deactivate\s)", re.IGNORECASE)
_SEQ_ARROW = re.compile(r"\s+([A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:)")

# \n / \t как escape — но не префиксы TeX (\times, \text, \neq, \nu, …)
_NL_LITERAL_ESC = re.compile(
    r"\\n(?!eq|ot|u|abla|eg|mid|otin|rightarrow|leftarrow|warrow|earrow|i|pm|"
    r"subset|cap|cup|warrow|exists|cong|sim|propto|fancy|atural|egative)",
)
_TAB_LITERAL_ESC = re.compile(
    r"\\t(?!imes|ext|heta|au|an|o|op|riangleq|ilde|hicksim|o|frac|iny|bf|it|"
    r"extbf|extrm|extit|exttt|ilde|woheadrightarrow)",
)
_MATH_SEG_FOR_ESC = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$")


def repair_llm_literal_escapes(text: str) -> str:
    """Литералы \\n / \\t → символы; TeX-команды (\\times, \\neq, …) не ломаем."""
    if not text:
        return ""
    math_slots: list[str] = []

    def _shield_math(m: re.Match[str]) -> str:
        math_slots.append(m.group(0))
        return f"\ue000M{len(math_slots) - 1}\ue001"

    t = _MATH_SEG_FOR_ESC.sub(_shield_math, text)
    for _ in range(6):
        t2 = t.replace("\\r\\n", "\n").replace("\\r", "\n")
        t2 = _NL_LITERAL_ESC.sub("\n", t2)
        t2 = _TAB_LITERAL_ESC.sub("\t", t2)
        if t2 == t:
            break
        t = t2
    for i, seg in enumerate(math_slots):
        t = t.replace(f"\ue000M{i}\ue001", seg)
    return t


_LIST_SECTIONS: tuple[tuple[str, str], ...] = (
    ("pros", "Плюсы"),
    ("cons", "Минусы и риски"),
    ("cons_and_risks", "Минусы и риски"),
    ("takeaways", "Ключевые выводы"),
    ("failure_modes", "Типичные сбои"),
)


def _looks_like_analysis_object(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    if "title" not in keys and "description" not in keys:
        return False
    return bool(keys & {"pros", "cons", "cons_and_risks", "takeaways", "failure_modes"})


def _format_analysis_object(obj: dict) -> str:
    parts: list[str] = []
    title = str(obj.get("title") or "").strip()
    if title:
        parts.append(f"## {title}")
    desc = str(obj.get("description") or "").strip()
    if desc:
        parts.append(desc)
    seen_labels: set[str] = set()
    for key, label in _LIST_SECTIONS:
        if label in seen_labels:
            continue
        raw = obj.get(key)
        if not raw:
            continue
        items = [str(x).strip() for x in raw if str(x).strip()]
        if not items:
            continue
        seen_labels.add(label)
        parts.append(f"### {label}")
        parts.extend(f"- {it}" for it in items)
    return "\n\n".join(parts).strip()


def _iter_json_objects(text: str):
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if _looks_like_analysis_object(obj):
            yield i, end, obj
        i = end if end > i else i + 1


def repair_structured_analysis_json(text: str) -> str:
    """
    LLM иногда вставляет trade-off JSON (title/pros/cons/takeaways/failure_modes)
    в tutor_message или summary — превращаем в Markdown для UI.
    """
    raw = (text or "").strip()
    if not raw or "{" not in raw:
        return text or ""

    if raw.startswith("{") or raw.startswith("```"):
        fenced = raw
        if fenced.startswith("```"):
            fenced = re.sub(r"^```(?:json)?\s*", "", fenced, flags=re.IGNORECASE)
            fenced = re.sub(r"```\s*$", "", fenced.strip())
        try:
            whole = json.loads(fenced.strip())
            if _looks_like_analysis_object(whole):
                return _format_analysis_object(whole)
        except json.JSONDecodeError:
            pass

    spans = list(_iter_json_objects(raw))
    if not spans:
        return text

    out = raw
    for start, end, obj in reversed(spans):
        md = _format_analysis_object(obj)
        before = raw[:start].rstrip()
        last_line = before.split("\n")[-1].strip() if before else ""
        title = str(obj.get("title") or "").strip()
        if last_line and title:
            if title.lower() in last_line.lower() or last_line.lower() in title.lower():
                md_lines = md.split("\n", 1)
                if md_lines and md_lines[0].startswith("## "):
                    md = md_lines[1].strip() if len(md_lines) > 1 else ""
        replacement = md if md else raw[start:end]
        out = out[:start] + replacement + out[end:]

    return out.strip()


_INLINE_HEADER_RE = re.compile(r"(?<=[а-яА-ЯёЁa-zA-Z0-9\)\]»\"'№%])(\s+)(#{1,6}\s+)")


def _split_markdown_header_line(line: str) -> str:
    s = line.strip()
    if not s.startswith("#"):
        return line
    m = re.match(r"^(#{1,6}\s+)", s)
    if not m:
        return line
    rest = s[m.end() :]
    pm = re.search(r"\s+(При\s+[а-яё])", rest)
    if not pm:
        pm = re.search(r"\s+([А-ЯЁ][а-яё]{2,}\s+[а-яё])", rest)
    if pm:
        title = s[: m.end() + pm.start()].strip()
        body = rest[pm.start() :].strip()
        return f"{title}\n\n{body}"
    return line


_TABLE_ROW_GLUE_RE = re.compile(r"\|\s+\|")
_NUMBERED_LIST_GLUE_RE = re.compile(r"([:;])\s+(\d+\.\s+)")
_GLUE_ORDERED_AFTER_PERIOD_RE = re.compile(
    r"([.!?…])(\s+)(\d{1,2}\.\s+(?:\*\*)?[А-ЯЁA-ZВЁ])"
)
_LETTER_SUB_LABEL = r"(?:\*\*[а-яёa-z][)]\*\*|[а-яёa-z][)])"
_GLUE_LETTER_BOLD_SUB_RE = re.compile(
    r"([а-яёa-zA-Z0-9\)\]»\"'№%])(\s+)(\*\*[а-яёa-z]\)\*\*)",
    re.IGNORECASE,
)
_GLUE_LETTER_PLAIN_AFTER_CLOSE_RE = re.compile(
    r"(\))\s+([а-яёa-z]\)\s)",
    re.IGNORECASE,
)
_GLUE_LETTER_PLAIN_AFTER_PUNCT_RE = re.compile(
    r"([.!?…:;])(\s+)([а-яёa-z]\)\s)",
    re.IGNORECASE,
)
_GLUE_BOLD_SUB_AFTER_PUNCT_RE = re.compile(
    r"([.!?…:;])(\s+)(\*\*[а-яёa-z]\)\*\*)",
    re.IGNORECASE,
)
_WRONG_NUMBERED_LETTER_LINE_RE = re.compile(
    rf"^\d{{1,2}}\.\s*({_LETTER_SUB_LABEL})\s*(.*)$",
    re.IGNORECASE,
)
_ORDERED_LIST_LINE_RE = re.compile(
    rf"^\d{{1,2}}\.\s+(?!{_LETTER_SUB_LABEL})",
    re.IGNORECASE,
)
_INLINE_WRONG_NUMBERED_LETTER_RE = re.compile(
    rf"(?<![\d/])(\d{{1,2}})\.\s*({_LETTER_SUB_LABEL})\s+",
    re.IGNORECASE,
)
_GLUED_PIPE_AFTER_PERIOD_RE = re.compile(r"([.!?…])\s+(\|)")
_TABLE_ROW_TRAILING_PROSE_RE = re.compile(
    r"^(?P<table>\|.+\|)\s+(?P<prose>[А-ЯЁA-ZВЁ][^\|]+)$"
)


def _collapse_table_internal_blank_lines(t: str) -> str:
    """Markdown tables: без пустых строк между header, :--- и data rows."""
    lines = t.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block: list[str] = []
            while i < n:
                s = lines[i].strip()
                if not s:
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and lines[j].strip().startswith("|"):
                        i += 1
                        continue
                    break
                if s.startswith("|"):
                    block.append(lines[i].rstrip())
                    i += 1
                else:
                    break
            out.extend(block)
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _repair_markdown_tables_layout(t: str) -> str:
    """Отделяет markdown-таблицы от текста и разбивает склеенные строки."""
    lines_out: list[str] = []
    for line in t.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines_out.append(line)
            continue

        work = stripped
        if not work.startswith("|") and "|" in work:
            pipe_idx = work.find("|")
            before = work[:pipe_idx].rstrip()
            rest = work[pipe_idx:]
            if before and rest.count("|") >= 2:
                lines_out.append(before)
                lines_out.append("")
                work = rest

        if work.count("|") >= 2:
            work = _TABLE_ROW_GLUE_RE.sub("|\n|", work)

        for sub_line in work.split("\n"):
            sm = sub_line.strip()
            if not sm:
                lines_out.append("")
                continue
            m = _TABLE_ROW_TRAILING_PROSE_RE.match(sm)
            if m and m.group("table").count("|") >= 2:
                lines_out.append(m.group("table").strip())
                lines_out.append("")
                lines_out.append(m.group("prose").strip())
            else:
                lines_out.append(sm)
    return _collapse_table_internal_blank_lines("\n".join(lines_out))


def _repair_glued_numbered_lists_on_line(line: str) -> str:
    s = line
    if not s.strip():
        return line
    if s.strip().startswith("|") and "|" in s:
        return line
    stripped = s.strip()
    if _WRONG_NUMBERED_LETTER_LINE_RE.match(stripped):
        return line
    s = _NUMBERED_LIST_GLUE_RE.sub(r"\1\n\2", s)
    for _ in range(12):
        s2 = _GLUE_ORDERED_AFTER_PERIOD_RE.sub(r"\1\n\3", s)
        if s2 == s:
            break
        s = s2
    s = _GLUE_LETTER_BOLD_SUB_RE.sub(r"\1\n\3", s)
    s = _GLUE_BOLD_SUB_AFTER_PUNCT_RE.sub(r"\1\n\3", s)
    s = _GLUE_LETTER_PLAIN_AFTER_CLOSE_RE.sub(r"\1\n\2", s)
    s = _GLUE_LETTER_PLAIN_AFTER_PUNCT_RE.sub(r"\1\n\3", s)
    s = _INLINE_WRONG_NUMBERED_LETTER_RE.sub(r"\n- \2 ", s)
    return s


def _repair_glued_numbered_lists(t: str) -> str:
    return "\n".join(
        _repair_glued_numbered_lists_on_line(line) for line in t.split("\n")
    )


def _normalize_list_blocks_for_markdown(t: str) -> str:
    """Склеить последовательные «1. … 2. …»; «3. а)» → маркированный подпункт."""
    lines = t.split("\n")
    out: list[str] = []
    buf: list[str] = []
    sub_re = re.compile(r"^(?:\*\*[а-яёa-z]\)\*\*|[а-яёa-z]\)\s)", re.I)

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        if out and out[-1].strip():
            out.append("")
        out.extend(buf)
        buf = []

    def append_bullet(line_text: str) -> None:
        bullet = line_text.strip()
        if not bullet:
            return
        if not bullet.startswith("-"):
            bullet = f"- {bullet}"
        if out and out[-1].strip().startswith("-"):
            out.append(bullet)
        elif out and out[-1].strip():
            out.append("")
            out.append(bullet)
        else:
            out.append(bullet)

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "-":
            flush_buf()
            if stripped == "-":
                continue
            out.append(line)
            continue
        wrong = _WRONG_NUMBERED_LETTER_LINE_RE.match(stripped)
        if wrong:
            flush_buf()
            label = wrong.group(1).strip()
            tail = (wrong.group(2) or "").strip()
            append_bullet(f"{label} {tail}".strip())
            continue
        if _ORDERED_LIST_LINE_RE.match(stripped):
            buf.append(stripped)
            continue
        if sub_re.match(stripped) or stripped.startswith("- "):
            flush_buf()
            append_bullet(
                stripped.lstrip("-").strip() if stripped.startswith("-") else stripped
            )
            continue
        flush_buf()
        out.append(line)
    flush_buf()
    return "\n".join(out)


def _collapse_blank_lines_in_list_runs(t: str) -> str:
    """Пустые строки между «1.» / «2.» / «-» / «3. а)» не разрывают список."""
    lines = t.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            out.append(line)
            continue
        if not out:
            out.append(line)
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        next_s = lines[j].strip() if j < len(lines) else ""
        prev_s = out[-1].strip()
        if _ORDERED_LIST_LINE_RE.match(prev_s) and (
            _ORDERED_LIST_LINE_RE.match(next_s)
            or _WRONG_NUMBERED_LETTER_LINE_RE.match(next_s)
            or next_s == "-"
        ):
            continue
        out.append(line)
    return "\n".join(out)


_CODE_FENCE_RE = re.compile(r"```[^\n`]*\n[\s\S]*?```", re.MULTILINE)
_PYTHON_STMT_START = re.compile(
    r"^(?:class |def |elif |else:|return |if |for |while |self\.|# |import |from )",
    re.IGNORECASE,
)


def _apply_outside_code_fences(text: str, fn) -> str:
    parts: list[str] = []
    last = 0
    for m in _CODE_FENCE_RE.finditer(text):
        if m.start() > last:
            parts.append(fn(text[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()
    if last < len(text):
        parts.append(fn(text[last:]))
    return "".join(parts)


def _reflow_glued_python(src: str) -> str:
    """Вставляет переносы в «однострочный» Python из lecture_body."""
    if not src or ("def " not in src and "class " not in src):
        return src
    s = src
    if s.count("\n") < 2 and "\\n" in s:
        s = s.replace("\\n", "\n")
    s = re.sub(r"(\) -> [^:\n]+:)(\s*)(?=\S)", r"\1\n", s)
    if s.count("\n") < 2 or (s.count("def ") + s.count("class ") > s.count("\n") // 2):
        for _ in range(24):
            prev = s
            s = re.sub(r":(\s*)(?=def |class )", ":\n", s)
            s = re.sub(
                r"\):(\s*)(?=self\.|return |if |elif |else:|def |class )",
                "):\n",
                s,
            )
            s = re.sub(r"(?<=[\w\)])\s*(?=def )", "\n", s)
            s = re.sub(r"(?<=[^\n])\s+(?=elif )", "\n", s)
            s = re.sub(r"(?<=[^\n])\s+(?=else:)", "\n", s)
            s = re.sub(r"(?<=[^\n])\s+(?=return )", "\n", s)
            s = re.sub(r"(?<=[^\n])\s+(?=# )", "\n", s)
            s = re.sub(
                r"(?<=[\w\"'])(?=(?:self\.|elif |else:|return ))",
                "\n",
                s,
            )
            s = re.sub(r"(?<!el)(?<=[a-z0-9_])(?=if )", "\n", s)
            s = re.sub(r":(\s*)(?=return )", ":\n", s)
            s = re.sub(r"(?<=[^\n])(?=#)", "\n", s)
            if s == prev:
                break
        s = _basic_python_indent(s)
    elif "def " in s or "class " in s:
        s = _basic_python_indent(s)
    return s


def _basic_python_indent(body: str) -> str:
    lines = body.split("\n")
    out: list[str] = []
    in_class = False
    in_def = False
    for ln in lines:
        st = ln.strip()
        if not st:
            out.append("")
            continue
        if st.startswith("class "):
            in_class = True
            in_def = False
            out.append(st)
            continue
        if st.startswith("def "):
            in_def = True
            out.append(("    " if in_class else "") + st)
            continue
        if st.startswith(("elif ", "else:")):
            pad = (
                "        "
                if in_class and in_def
                else ("    " if in_def or in_class else "")
            )
            out.append(pad + st)
            continue
        if in_class and in_def:
            out.append("        " + st)
        elif in_def or (in_class and not st.startswith("class ")):
            out.append("    " + st)
        else:
            out.append(st)
    return "\n".join(out)


def _repair_fence_inner(chunk: str) -> str:
    m = re.match(r"(```[^\n]*\n)([\s\S]*?)(```\s*)$", chunk.strip(), re.DOTALL)
    if not m:
        return chunk
    inner = _reflow_glued_python(m.group(2))
    return f"{m.group(1)}{inner.rstrip()}\n{m.group(3)}"


def _wrap_bare_python_regions(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        st = lines[i].strip()
        if st.startswith("class ") or (
            st.startswith("def ") and "def " in st and st.count(":") >= 1
        ):
            block: list[str] = []
            j = i
            while j < len(lines):
                ln = lines[j]
                ls = ln.strip()
                if not ls:
                    k = j + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    nxt = lines[k].strip() if k < len(lines) else ""
                    if block and (
                        nxt.startswith("def ")
                        or _PYTHON_STMT_START.match(nxt)
                        or nxt.startswith("self.")
                    ):
                        block.append(ln)
                        j += 1
                        continue
                    if block:
                        j += 1
                        break
                    j += 1
                    continue
                if block and ls.startswith("#"):
                    block.append(ln)
                    j += 1
                    continue
                if (
                    block
                    and not _PYTHON_STMT_START.match(ls)
                    and not ln.startswith(("    ", "\t"))
                ):
                    if ls.startswith(("#", "###")):
                        break
                    if not ls.startswith(("self.", "return ", "elif ", "else:")):
                        break
                if ls.startswith("###"):
                    break
                block.append(ln)
                j += 1
            body = _reflow_glued_python("\n".join(block))
            if "def " in body or "class " in body:
                if not body.lstrip().startswith("```"):
                    out.append(f"```python\n{body.strip()}\n```")
                else:
                    out.append(body.strip())
            else:
                out.extend(block)
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def repair_lecture_code_blocks(text: str) -> str:
    """Склеенный Python в lecture_body → переносы + ```python fences."""
    raw = (text or "").strip()
    if not raw or ("def " not in raw and "class " not in raw):
        return text or ""
    parts: list[str] = []
    last = 0
    for m in _CODE_FENCE_RE.finditer(raw):
        if m.start() > last:
            chunk = _wrap_bare_python_regions(raw[last : m.start()])
            parts.append(chunk)
        parts.append(_repair_fence_inner(m.group(0)))
        last = m.end()
    if last < len(raw):
        parts.append(_wrap_bare_python_regions(raw[last:]))
    return "".join(parts) if parts else _wrap_bare_python_regions(raw)


def repair_lecture_markdown_layout(text: str) -> str:
    """
    Починка «стены текста» из lecture_body: заголовки ### без переноса,
    таблицы в одну строку, списки после двоеточия.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = heal_tab_corrupted_times(raw)
    raw = repair_llm_literal_escapes(raw)
    if (
        "\\times" in raw
        or "imes" in raw
        or "×" in raw
        or "\t" in raw
        or "\\t" in raw
        or "frac" in raw.lower()
        or "\\approx" in raw
        or "≈" in raw
    ):
        raw = repair_broken_latex(heal_broken_times_markup(raw))
    t = raw

    def _layout_chunk(chunk: str) -> str:
        c = re.sub(r"([.!?…])\s+(#{1,6}\s+)", r"\1\n\n\2", chunk)
        c = _INLINE_HEADER_RE.sub(r"\n\n\2", c)
        header_lines: list[str] = []
        for line in c.split("\n"):
            header_lines.append(_split_markdown_header_line(line))
        c = "\n".join(header_lines)
        c = _GLUED_PIPE_AFTER_PERIOD_RE.sub(r"\1\n\n\2", c)
        c = _repair_markdown_tables_layout(c)
        c = _repair_glued_numbered_lists(c)
        c = _collapse_blank_lines_in_list_runs(c)
        c = _normalize_list_blocks_for_markdown(c)
        return c

    t = _apply_outside_code_fences(t, _layout_chunk)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def repair_llm_display_text(text: str) -> str:
    """
    Полная починка текста тьютора/лекции для хранения и HTML:
    escapes → heal imes10/kimes10 → LaTeX → markdown layout.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    t = _normalize_unicode_math_exponents(raw)
    t = heal_tab_corrupted_times(t)
    t = repair_llm_literal_escapes(t)
    t = repair_lecture_code_blocks(t)
    t = repair_broken_latex(t)
    t = repair_structured_analysis_json(t)
    return repair_lecture_markdown_layout(t)


def _strip_outer_quotes(s: str) -> str:
    t = (s or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1].strip()
    return t


def _strip_fence(d: str) -> str:
    inner = re.sub(r"^```(?:mermaid)?\s*", "", d.strip(), count=1, flags=re.IGNORECASE)
    inner = re.sub(r"```\s*$", "", inner.strip(), flags=re.IGNORECASE)
    return inner.strip()


def _quote_subgraph_titles(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*subgraph\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        rest = m.group(2).strip()
        if rest.startswith('"') or rest.startswith("'"):
            out.append(line)
            continue
        if re.match(r"^\w[\w-]*\s*\[", rest):
            out.append(line)
            continue
        safe = rest.replace('"', "'")
        out.append(f'{m.group(1)}"{safe}"')
    return "\n".join(out)


def _quote_participant_aliases(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*participant\s+\S+\s+as\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        alias = m.group(2).strip()
        if alias.startswith('"') or alias.startswith("'"):
            out.append(line)
            continue
        if "/" in alias or "(" in alias or "  " in alias:
            safe = alias.replace('"', "'")
            out.append(f'{m.group(1)}"{safe}"')
        else:
            out.append(line)
    return "\n".join(out)


def _quote_loop_labels(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*loop\s+)(.+)$", line, re.IGNORECASE)
        if not m:
            out.append(line)
            continue
        rest = m.group(2)
        arrow = _SEQ_ARROW.search(rest)
        label = rest[: arrow.start()] if arrow else rest
        label = label.strip()
        tail = rest[arrow.start() :] if arrow else ""
        if not label or label.startswith('"') or label.startswith("'"):
            out.append(line)
            continue
        if "(" in label or " " in label:
            safe = label.replace('"', "'")
            out.append(f'{m.group(1)}"{safe}"{tail}')
        else:
            out.append(line)
    return "\n".join(out)


def _wrap_long_label(text: str, max_len: int = 38) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    words = t.split()
    lines: list[str] = []
    line = ""
    for w in words:
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= max_len:
            line += f" {w}"
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


def _ensure_sequence_init(inner: str) -> str:
    if not re.search(r"^sequenceDiagram", inner, re.I | re.M):
        return inner
    if re.search(r"%%\s*\{init:", inner, re.I):
        return inner
    init = (
        "%%{init: {'themeVariables': {'fontSize': '10px'}, "
        "'sequence': {'wrap': true, 'width': 240, 'messageFontSize': 10, "
        "'noteFontSize': 10, 'actorFontSize': 11, 'messageMargin': 48, "
        "'boxMargin': 10, 'mirrorActors': false}}}%%\n"
    )
    return init + inner


def _sanitize_note_lines(inner: str) -> str:
    lines = inner.split("\n")
    out: list[str] = []
    note_re = re.compile(
        r"^(\s*Note\s+(?:over|left of|right of)\s+[^:]+:\s*)(.*)$",
        re.IGNORECASE,
    )
    for line in lines:
        m = note_re.match(line)
        if not m:
            out.append(line)
            continue
        prefix = m.group(1)
        body = (m.group(2) or "").strip().strip('"').replace('"', "'")
        body = _wrap_long_label(body, 36)
        if not body:
            out.append(line)
            continue
        out.append(prefix + f'"{body}"')
    return "\n".join(out)


def _quote_arrow_messages(inner: str) -> str:
    is_seq = bool(re.search(r"^sequenceDiagram\b", inner, re.I | re.M))
    lines = inner.split("\n")
    out: list[str] = []
    arrow_re = re.compile(r"^(\s*[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*)(.*)$")
    for line in lines:
        m = arrow_re.match(line)
        if not m:
            out.append(line)
            continue
        prefix = m.group(1)
        body = (m.group(2) or "").strip().strip('"')
        if not body:
            out.append(line)
            continue
        body = body.replace('"', "'")
        if not is_seq:
            body = _wrap_long_label(body, 40)
        out.append(f'{prefix}"{body}"')
    return "\n".join(out)


def _format_mermaid_inner(inner: str) -> str:
    from knowledge_engine.services.mermaid_validate import sanitize_mermaid_syntax

    s = sanitize_mermaid_syntax(repair_llm_literal_escapes(inner).strip())
    s = _quote_subgraph_titles(s)
    if not s:
        return s

    from knowledge_engine.services.mermaid_validate import strip_mermaid_init_directive

    s = strip_mermaid_init_directive(s)
    has_init = bool(re.search(r"%%\s*\{init:", s, re.I))
    if not has_init:
        s = _ensure_sequence_init(s)

    s = re.sub(
        r"^(sequenceDiagram(?:\s+autonumber)?)\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(graph\s+(?:TD|LR|BT|RL))\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(flowchart\s+(?:TD|LR|BT|RL)?)\s+",
        r"\1\n",
        s,
        count=1,
        flags=re.IGNORECASE,
    )

    is_seq = bool(re.search(r"^sequenceDiagram\b", s, re.I | re.M))
    if is_seq:
        s = _SEQ_BLOCK_KW.sub(r"\n\1", s)
        s = _SEQ_NOTE.sub(r"\n\1", s)
        s = _SEQ_ACT.sub(r"\n\1", s)
        s = _SEQ_ARROW.sub(r"\n\1", s)
    s = re.sub(r"\s+(subgraph\s)", r"\n\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(end)\b", r"\n\1", s, flags=re.IGNORECASE)
    if not is_seq:
        s = _split_flowchart_lines(s)

    s = _quote_participant_aliases(s)
    s = _quote_loop_labels(s)
    s = _sanitize_note_lines(s)
    s = _quote_arrow_messages(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_multiple_edges_per_line(line: str) -> str:
    t = (line or "").strip()
    if not t or re.match(
        r"^(graph|flowchart|subgraph|end|%%|classDef|class |linkStyle|style )",
        t,
        re.I,
    ):
        return line
    if len(re.findall(r"-->", t)) < 2:
        return line

    def repl(m: re.Match[str]) -> str:
        offset = m.start()
        if offset == 0:
            return m.group(0)
        before = t[:offset]
        if before.count("|") % 2 == 1:
            return m.group(0)
        if len(re.findall(r"-->", before)) < 1:
            return m.group(0)
        return "\n"

    return re.sub(r"\s+(?=[A-Za-z_][\w-]*\s+-->)", repl, t)


def _split_flowchart_lines(s: str) -> str:
    lines: list[str] = []
    for line in s.split("\n"):
        ln = line
        if not re.search(r"^sequenceDiagram\b", s, re.I | re.M):
            ln = re.sub(
                r"([)\]])\s+(?=[A-Za-z_][\w-]*\s*(\[|-->|-->))",
                r"\1\n",
                ln,
            )
            ln = _split_multiple_edges_per_line(ln)
        lines.append(ln)
    return "\n".join(lines)


def _mermaid_node_id(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return "node"
    m = re.match(r"^(.+?)\s+\(([^)]+)\)\s*$", t)
    if m:
        base = m.group(1).strip()
        inner = m.group(2).strip()
        id_ = f"{base.replace(' ', '_')}_{inner.replace(' ', '_')}"
        return f'{id_}["{base.replace(chr(34), chr(39))} ({inner.replace(chr(34), chr(39))})"]'
    if re.search(r"[\s/]", t):
        id_ = re.sub(r"[^\w]+", "_", t).strip("_") or "node"
        return f'{id_}["{t.replace(chr(34), chr(39))}"]'
    return t.replace(" ", "_")


def _parse_link_chain(chain: str) -> list[tuple[str, str, str]] | None:
    s = re.sub(r"--\(([^)]+)\)-->", r"-->|\1|", chain)
    parts = [p.strip() for p in re.split(r"\s*-->\s*", s) if p.strip()]
    if len(parts) < 2:
        return None
    edges: list[tuple[str, str, str]] = []
    nodes: list[str] = []
    for i, part in enumerate(parts):
        if i == 0:
            nodes.append(part)
            continue
        label = ""
        lm = re.match(r"^\|([^|]+)\|\s*(.+)$", part, re.DOTALL)
        if lm:
            label = lm.group(1).strip()
            part = lm.group(2).strip()
        from_n = nodes[-1]
        edges.append((from_n, part, label))
        nodes.append(part)
    return edges


def _lift_pseudo_flowchart(text: str) -> str:
    t = _strip_outer_quotes(repair_llm_literal_escapes(text)).strip()
    if not t or _RAW_MERMAID_INLINE.search(t.split("\n", 1)[0]):
        return ""
    if not re.search(r"--\([^)]+\)-->|-->", t):
        return ""
    segments = (
        [s for s in re.split(r"\s*\|\s*", t) if re.search(r"-->|--\(", s)]
        if "|" in t
        else [t]
    )
    if not segments:
        return ""
    lines = ["flowchart LR"]
    for idx, seg in enumerate(segments):
        title = ""
        chain = seg.strip()
        m = re.match(r"^([^:]{2,56}):\s*(.+)$", chain, re.DOTALL)
        if m and re.search(r"-->|--\(", m.group(2)):
            title = m.group(1).strip()
            chain = m.group(2).strip()
        edges = _parse_link_chain(chain)
        if not edges:
            continue
        sub_id = f"sg_{idx}"
        if title:
            safe = title.replace('"', "'").replace("]", "")
            lines.append(f"subgraph {sub_id} [{safe}]")
        prefix = "  " if title else ""
        for from_n, to_n, label in edges:
            a = _mermaid_node_id(from_n)
            b = _mermaid_node_id(to_n)
            lbl = f"|{label.replace(chr(34), chr(39))}|" if label else ""
            lines.append(f"{prefix}{a} -->{lbl} {b}")
        if title:
            lines.append("end")
    return "\n".join(lines) if len(lines) > 1 else ""


def _repair_diagram_once(diagram: str) -> str:
    d = _strip_outer_quotes(repair_llm_literal_escapes((diagram or "").strip()))
    if not d:
        return d

    if d.startswith("```"):
        inner = _format_mermaid_inner(_strip_fence(d))
        return f"```mermaid\n{inner}\n```" if inner else d

    if _RAW_MERMAID_INLINE.search(d):
        inner = _format_mermaid_inner(d)
        return f"```mermaid\n{inner}\n```" if inner else d

    lifted = _lift_pseudo_flowchart(d)
    if lifted:
        inner = _format_mermaid_inner(lifted)
        return f"```mermaid\n{inner}\n```" if inner else d
    return d


def repair_diagram_markdown(diagram: str) -> str:
    """Mermaid в content.diagram: unescape, fence, переносы, sequenceDiagram."""
    d = (diagram or "").strip()
    for _ in range(4):
        d2 = _repair_diagram_once(d)
        if d2 == d:
            break
        d = d2
    return d
