"""Валидация и детерминированная санитизация Mermaid."""

from __future__ import annotations

import asyncio
import re

from knowledge_engine.ui.run_log import trace

_MERMAID_HEAD = re.compile(
    r"^(sequenceDiagram|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b|xychart(?:-beta)?\b)",
    re.IGNORECASE | re.MULTILINE,
)

_FORBIDDEN = re.compile(r"```|<script", re.IGNORECASE)

_BENCHMARK_TEXT_HINTS = re.compile(
    r"\b(qps|recall@?k?|latency|throughput|benchmark|performance\s+vs|vs\s+recall)\b",
    re.IGNORECASE,
)

_BROKEN_INIT = re.compile(r"%%\s*\{init:", re.I)
_INIT_WELL_FORMED = re.compile(
    r"%%\s*\{init:\s*\{[\s\S]*?\}\s*\}%%\s*",
    re.I,
)

_LINE_START_OK = re.compile(
    r"^(?:"
    r"sequenceDiagram\b|flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b|xychart(?:-beta)?\b|"
    r"participant\s|actor\s|autonumber\b|"
    r"Note\s+(?:over|left of|right of)\s|"
    r"rect\s|loop\s|alt\s|opt\s|par\s|else\b|end\b|and\s*$|"
    r"activate\s|deactivate\s|critical\s|break\s|"
    r"subgraph\s|title\s|x-axis\s|y-axis\s|line\s|bar\s|"
    r"class\s|style\s|linkStyle\s|direction\s|"
    r"[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:|"
    r"%%"
    r")",
    re.IGNORECASE,
)

_ARROW_LINE = re.compile(
    r"^[A-Za-z0-9_]+[-]+>>?[A-Za-z0-9_]+:\s*(.*)$",
    re.IGNORECASE,
)

# --- AST / tokenizer lint -------------------------------------------------
_INIT_BLOCK = re.compile(r"%%\s*\{init:\s*\{[\s\S]*?\}\s*\}%%", re.I)
_SUBGRAPH_OPEN_RE = re.compile(r"^subgraph\b\s*(.*)$", re.I)
_BLOCK_END_STANDALONE = re.compile(r"^end\s*(?:%%.*)?$", re.I)
_BLOCK_END_MERGED = re.compile(r"^end\b", re.I)
_DIRECTION_WORD = re.compile(r"\bdirection\b", re.I)
_HTML_OR_SPECIAL_IN_LABEL = re.compile(
    r"<\s*/?\s*(?:br|b|i|div|span)\b|<|>|\"",
    re.I,
)
# Arrows used to detect spliced multi-statements on one line.
_CONN_ARROW = re.compile(
    r"(?:<-->|-->|---|-\.->|-\.-|==>|-->>|->>|o-->|x-->|<\-\-)",
)
# After a closed node label, another node/edge starts on the same line.
_AFTER_LABEL_STATEMENT = re.compile(
    r"(?:\]|\)|\})\s+[A-Za-z_][\w]*",
)
# Edge completed (… --> target), then a NEW edge starts: `B B -->` (not chain `B -->`).
_SPLICED_SECOND_EDGE = re.compile(
    r"(?:<-->|-->|---|-\.->|-\.-|==>|-->>|->>)\s*"
    r"(?:\|[^|\n]*\|\s*)?"
    r"[A-Za-z_][\w]*"
    r"(?:\s*(?:\[[^\]]*\]|\([^)]*\)|\{[^}]*\}))?"
    r"\s+[A-Za-z_][\w]*\s*"
    r"(?:<-->|-->|---|-\.->|-\.-|==>|-->>|->>)",
)


def _strip_init_and_fence(code: str) -> str:
    inner = strip_mermaid_fences(code or "")
    peeled = _INIT_BLOCK.sub("", inner)
    while _INIT_BLOCK.search(peeled):
        peeled = _INIT_BLOCK.sub("", peeled, count=1)
    return peeled.strip()


def _subgraph_id_from_tail(tail: str) -> str:
    t = (tail or "").strip()
    if not t:
        return "(anonymous)"
    m = re.match(r'^([A-Za-z_][\w]*)', t)
    if m:
        return m.group(1)
    m = re.match(r'^["\']([^"\']+)["\']', t)
    if m:
        return m.group(1)[:40]
    return t[:40]


def _iter_shape_labels(line: str) -> list[tuple[str, str]]:
    """
    Extract shape labels: [...], (...), {...} with quote awareness.
    Returns list of (raw_inside_including_optional_quotes, opener_char).
    """
    out: list[tuple[str, str]] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch not in "[({":
            i += 1
            continue
        close = {"[": "]", "(": ")", "{": "}"}[ch]
        # Avoid matching arrow fragments like `-->` (no open bracket there).
        j = i + 1
        depth = 1
        in_dq = False
        while j < n and depth > 0:
            c = line[j]
            if c == '"' and (j == 0 or line[j - 1] != "\\"):
                in_dq = not in_dq
            elif not in_dq:
                if c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0:
            out.append((line[i + 1 : j], ch))
            i = j + 1
        else:
            i += 1
    return out


def _label_needs_quotes(inner: str) -> bool:
    """True when label content has HTML/specials and is not already quoted."""
    s = (inner or "").strip()
    if not s:
        return False
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return False
    return bool(_HTML_OR_SPECIAL_IN_LABEL.search(s))


def lint_mermaid_ast(code: str) -> tuple[bool, list[str]]:
    """
    Strict line-oriented Mermaid tokenizer + block-stack linter.

    Returns (is_valid, error_messages) with ``Line N:`` prefixes where possible.
    """
    errors: list[str] = []
    raw = _strip_init_and_fence(code).replace("\r\n", "\n")
    if not raw.strip():
        return False, ["Line 1: empty diagram"]

    stack: list[tuple[int, str]] = []

    for line_no, line in enumerate(raw.split("\n"), 1):
        stripped = line.strip()
        if not stripped:
            continue
        # Comments / init leftovers
        if stripped.startswith("%%"):
            continue

        # Bare `subgraph` without ID/title on the same line
        if _BARE_SUBGRAPH.match(stripped):
            errors.append(
                f"Line {line_no}: 'subgraph' statement is missing its ID/Title "
                f"on the same line."
            )

        # Rule 3: subgraph + direction on one line
        if re.search(r"\bsubgraph\b", stripped, re.I) and _DIRECTION_WORD.search(
            stripped
        ):
            errors.append(
                f"Line {line_no}: 'subgraph' and 'direction' must be on separate lines."
            )

        # Rule 1: spliced end + connections/nodes
        if _BLOCK_END_MERGED.match(stripped) and not _BLOCK_END_STANDALONE.match(
            stripped
        ):
            preview = stripped if len(stripped) <= 80 else stripped[:77] + "..."
            errors.append(
                f"Line {line_no}: Syntax Error - 'end' statement is merged with "
                f"connections/nodes ('{preview}'). "
                f"'end' MUST be on its own dedicated line."
            )
            # Do not pop stack — this is not a valid BLOCK_END.
        elif _BLOCK_END_STANDALONE.match(stripped):
            if not stack:
                errors.append(
                    f"Line {line_no}: Unexpected 'end' without matching 'subgraph'."
                )
            else:
                stack.pop()
            continue

        sub_m = _SUBGRAPH_OPEN_RE.match(stripped)
        if sub_m:
            sid = _subgraph_id_from_tail(sub_m.group(1) or "")
            stack.append((line_no, sid))

        # Rule 4: node labels with HTML/specials must use ["..."]
        for inner, _opener in _iter_shape_labels(stripped):
            if _label_needs_quotes(inner):
                errors.append(
                    f'Line {line_no}: Node label with HTML or special characters '
                    f'must be enclosed in double quotes ["text"].'
                )
                break

        # Rule 5: multiple statements spliced on one line
        # e.g. `N["lab"] A --> B` or `A --> B B --> C` (not a chain `A --> B --> C`)
        if _AFTER_LABEL_STATEMENT.search(stripped) or _SPLICED_SECOND_EDGE.search(
            stripped
        ):
            preview = stripped if len(stripped) <= 80 else stripped[:77] + "..."
            errors.append(
                f"Line {line_no}: Syntax Error - multiple Mermaid statements on one "
                f"line ('{preview}'). Put each node declaration and each edge on its "
                f"own line (chains like A --> B --> C are OK)."
            )

    # Rule 2: leftover open subgraphs
    for open_line, sid in stack:
        errors.append(
            f"Line {open_line}: Unclosed 'subgraph' ('{sid}') block (missing 'end')."
        )

    return len(errors) == 0, errors


def format_mermaid_lint_report(errors: list[str]) -> str:
    """Format lint errors for Gemma / logs."""
    if not errors:
        return ""
    return "Validation/Lint Errors Found:\n" + "\n".join(f"- {e}" for e in errors)


_SPLIT_ARROW = r"(?:<-->|-->|-\.->|==>)"
_SPLIT_ID = r"[A-Za-z_][\w-]*"
# 1a) Node label close then a new edge starts on the same line.
_SPLIT_AFTER_LABEL = re.compile(
    rf'([\]\)"])\s+({_SPLIT_ID}\s*{_SPLIT_ARROW})'
)
# 1b) Node label close then another node declaration.
_SPLIT_AFTER_LABEL_NODE = re.compile(
    rf'([\]\)"])\s+({_SPLIT_ID}\s*[\[\(\{{])'
)
# 1c) Full subgraph header (ID and/or title) then a node on the same line.
# Never match bare `subgraph` — ID/title must stay on the same line as `subgraph`.
_SPLIT_SUBGRAPH_NODE = re.compile(
    rf"(subgraph\b\s+(?:{_SPLIT_ID}\s*(?:\[[^\]]*\])?|\[[^\]]*\]))\s+"
    rf"({_SPLIT_ID}\s*[\[\(\{{])",
    re.I,
)
# 2) Independent edges glued: `A --> B C --> D` (not chain `A --> B --> C`).
_SPLIT_SEQ_EDGE = re.compile(
    rf"({_SPLIT_ID}\s*{_SPLIT_ARROW}\s*(?:\|[^|\n]*\|\s*)?{_SPLIT_ID})"
    rf"\s+({_SPLIT_ID}\s*{_SPLIT_ARROW})"
)
_SPLIT_END_LEAD = re.compile(r"^(\s*)end\b\s+(\S.*)$", re.I)
_SPLIT_END_TRAIL = re.compile(r"^(?P<body>\s*\S.*?)\s+end\s*$", re.I)
_BARE_SUBGRAPH = re.compile(r"^subgraph\s*$", re.I)
_SUBGRAPH_ID_ONLY = re.compile(
    rf"^(?:{_SPLIT_ID}\s*(?:\[[^\]]*\])?|\[[^\]]*\])\s*$"
)


def _rejoin_broken_subgraph_headers(lines: list[str]) -> list[str]:
    """Reattach `subgraph\\nID[\"label\"]` → `subgraph ID[\"label\"]`."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if _BARE_SUBGRAPH.match(cur.strip()) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if _SUBGRAPH_ID_ONLY.match(nxt.strip()):
                indent = re.match(r"^(\s*)", cur)
                prefix = indent.group(1) if indent else ""
                out.append(f"{prefix}subgraph {nxt.strip()}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def _explode_spliced_line(line: str) -> list[str]:
    """Recursively split one physical line into statement lines."""
    if not (line or "").strip():
        return [line]

    m = _SPLIT_END_LEAD.match(line)
    if m:
        lead = m.group(1) + "end"
        rest = m.group(1) + m.group(2)
        return _explode_spliced_line(lead) + _explode_spliced_line(rest)

    m = _SPLIT_END_TRAIL.match(line)
    if m and not re.match(r"^\s*end\s*$", line, re.I):
        body = m.group("body")
        indent = re.match(r"^(\s*)", line)
        prefix = indent.group(1) if indent else ""
        return _explode_spliced_line(body) + _explode_spliced_line(prefix + "end")

    for pat in (
        _SPLIT_SUBGRAPH_NODE,
        _SPLIT_AFTER_LABEL,
        _SPLIT_AFTER_LABEL_NODE,
        _SPLIT_SEQ_EDGE,
    ):
        m = pat.search(line)
        if not m:
            continue
        left = line[: m.start()] + m.group(1)
        indent = re.match(r"^(\s*)", line)
        prefix = indent.group(1) if indent else ""
        right = prefix + m.group(2) + line[m.end() :]
        return _explode_spliced_line(left) + _explode_spliced_line(right)

    return [line]


def split_spliced_mermaid_lines(code: str) -> str:
    """
    Deterministic newline recovery for spliced Mermaid statements.

    Fixes patterns like:
    - ``N["lab"] A --> B`` / ``N["lab"] M["lab2"]``
    - ``subgraph X["t"] N["lab"]`` (keeps ``subgraph X["t"]`` intact)
    - ``A --> B C --> D`` (not chains ``A --> B --> C``)
    - ``end A --> B`` / ``A --> B end``
    Also repairs a previously broken ``subgraph\\nID["lab"]`` header.
    """
    raw = (code or "").replace("\r\n", "\n")
    if not raw.strip():
        return raw
    fenced = raw.strip().startswith("```")
    inner = strip_mermaid_fences(raw) if fenced else raw
    out_lines: list[str] = []
    for line in inner.split("\n"):
        out_lines.extend(_explode_spliced_line(line))
    out_lines = _rejoin_broken_subgraph_headers(out_lines)
    fixed = "\n".join(out_lines)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip()
    if fenced:
        return f"```mermaid\n{fixed}\n```"
    return fixed


def has_broken_init_directive(code: str) -> bool:
    inner = strip_mermaid_fences(code)
    if not _BROKEN_INIT.search(inner):
        return False
    if _INIT_WELL_FORMED.search(inner):
        return False
    return True


def strip_mermaid_init_directive(code: str) -> str:
    inner = strip_mermaid_fences(code)
    if not inner:
        return ""
    peeled = _INIT_WELL_FORMED.sub("", inner).strip()
    while _INIT_WELL_FORMED.search(peeled):
        peeled = _INIT_WELL_FORMED.sub("", peeled, count=1).strip()
    if has_broken_init_directive(peeled):
        peeled = _BROKEN_INIT.sub("", peeled, count=1).strip()
    return peeled


def _dedupe_init_directives(inner: str) -> str:
    """Удалить все %%{init}%% — сервер/UI добавят один при необходимости."""
    s = (inner or "").strip()
    while _INIT_WELL_FORMED.search(s):
        s = _INIT_WELL_FORMED.sub("", s, count=1).strip()
    s = re.sub(r"%%\s*\{init:[\s\S]*?\}%%\s*", "", s, flags=re.I).strip()
    return s


def _fix_unbalanced_quotes(line: str) -> str:
    t = line.rstrip()
    if t.count('"') % 2 == 1:
        return t + '"'
    return t


_CONTINUATION_LINE = re.compile(
    r"^(?:and\s+[a-z]|coherency\b|output\s|queries\b|[a-z].*activity)",
    re.I,
)


def _merge_orphan_lines(inner: str) -> str:
    lines = (inner or "").split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if out and out[-1].strip():
                out.append("")
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        merge = False
        if _CONTINUATION_LINE.match(stripped):
            merge = True
        elif not _LINE_START_OK.match(stripped):
            merge = True
        if merge:
            joined = (prev.rstrip() + " " + stripped).strip()
            out[-1] = joined
            continue
        out.append(line)
    lines = [_fix_unbalanced_quotes(ln) for ln in out]
    return "\n".join(lines)


def _lift_pseudo_xychart(inner: str) -> str:
    """VLM иногда кладёт xychart-beta внутрь flowchart LR — восстановить нормальный синтаксис."""
    if not re.search(r"flowchart\s", inner, re.I):
        return inner
    if "xychart" not in inner.lower():
        return inner
    embed = re.search(r'\["xychart-beta\s+(.+?)"\s*\]', inner, re.I | re.S)
    blob = f"xychart-beta {embed.group(1)}" if embed else inner.replace("_", " ")
    title_m = re.search(r"title\s+'([^']+)'", blob, re.I)
    x_m = re.search(r"x-axis\s+'([^']+)'\s*(\[[^\]]+\])", blob, re.I)
    y_m = re.search(r"y-axis\s+'([^']+)'\s*([\d.]+)", blob, re.I)
    bar_m = re.search(r"bar\s*(\[[0-9.,\s]+\])", inner, re.I)
    if not (title_m and x_m and bar_m):
        return inner
    title = title_m.group(1).replace('"', "'")
    xlab = x_m.group(1)
    labels = x_m.group(2)
    ylab = (y_m.group(1) if y_m else "Y").replace('"', "'")
    ymax = "100000"
    ym = re.search(r"(\d{3,7})_bar", inner.replace(" ", ""))
    if ym:
        ymax = ym.group(1)
    return (
        "xychart-beta\n"
        f'    title "{title}"\n'
        f'    x-axis "{xlab}" {labels}\n'
        f'    y-axis "{ylab}" 0 --> {ymax}\n'
        f"    bar {bar_m.group(1)}"
    )


def sanitize_mermaid_syntax(code: str) -> str:
    """Детерминированная починка без LLM."""
    from knowledge_engine.utils.mermaid_sanitizer import sanitize_mermaid_code

    inner = strip_mermaid_fences(sanitize_mermaid_code((code or "").strip()))
    if not inner:
        return ""
    inner = _dedupe_init_directives(inner)
    lifted = _lift_pseudo_xychart(inner)
    if lifted != inner:
        inner = lifted
    else:
        # Orphan-line merge helps sequence/VLM prose, but glues flowchart
        # node declarations onto `subgraph` headers — skip for flow/graph.
        head = next((ln.strip() for ln in inner.split("\n") if ln.strip()), "")
        if not re.match(r"(?:flowchart|graph)\b", head, re.I):
            inner = _merge_orphan_lines(inner)
    inner = re.sub(r"\blineplot\b", "line", inner, flags=re.I)
    inner = inner.replace('activity""', 'activity"')
    inner = re.sub(r"\boutput buff\b", 'output buffer"', inner)
    lines = [_fix_unbalanced_quotes(ln) for ln in inner.split("\n")]
    inner = "\n".join(lines)
    inner = re.sub(r"\n{3,}", "\n\n", inner).strip()
    # Deterministic splice fix BEFORE lint / Gemma.
    split = split_spliced_mermaid_lines(inner)
    if split != inner:
        trace("MERMAID_SPLIT spliced lines recovered")
        inner = split
    ok, lint_errors = lint_mermaid_ast(inner)
    if not ok:
        trace(
            "MERMAID_LINT post-sanitize | "
            + "; ".join(lint_errors[:8])
            + (f" …+{len(lint_errors) - 8}" if len(lint_errors) > 8 else "")
        )
    return inner


def normalize_stored_mermaid(code: str) -> str:
    """Чтение из БД / session: sanitize + форматирование без Gemma."""
    from knowledge_engine.web.llm_text_repair import repair_diagram_markdown

    raw = (code or "").strip()
    if not raw:
        return ""
    sanitized = sanitize_mermaid_syntax(raw)
    if not sanitized:
        return repair_diagram_markdown(raw)
    return repair_diagram_markdown(sanitized)


def _needs_ui_repair(inner: str) -> bool:
    """Эвристика: слабый синтаксис, который ломает Mermaid.js."""
    low = (inner or "").lower()
    if re.search(r"^flowchart\b", inner.strip(), re.I) and (
        "xychart" in low or "x-axis" in low.replace("_", " ")
    ):
        return True
    if is_xychart_mermaid(inner) and re.search(r"\blineplot\b", low):
        return True
    return False


def _validate_sequence_lines(inner: str) -> bool:
    if not re.search(r"^sequenceDiagram\b", inner, re.I | re.M):
        return True
    for line in inner.split("\n"):
        t = line.strip()
        if not t or t.startswith("%%"):
            continue
        if not _LINE_START_OK.match(t):
            return False
        m = _ARROW_LINE.match(t)
        if m:
            body = (m.group(1) or "").strip()
            if body and not (
                body.startswith('"') or re.match(r"^[A-Za-z0-9_]+\s*:", body)
            ):
                if " " in body and not body.startswith('"'):
                    return False
    return True


def _validate_xychart_body(inner: str) -> bool:
    if not is_xychart_mermaid(inner):
        return True
    low = inner.lower()
    if "flowchart" in low or re.match(r"graph\s", inner.strip(), re.I):
        return False
    if "xychart_beta" in low or re.search(r"flowchart\b[\s\S]+xychart", low):
        return False
    if re.search(r"\blineplot\b", low):
        return False
    if "x-axis" not in low and "x-axis" not in inner:
        if "x-axis" not in low.replace("_", "-"):
            return False
    if "y-axis" not in low and "y-axis" not in inner:
        if "y-axis" not in low.replace("_", "-"):
            return False
    if "line" not in low and "bar" not in low:
        return False
    return True


def strip_mermaid_fences(code: str) -> str:
    inner = (code or "").strip()
    if inner.startswith("```"):
        inner = re.sub(r"^```(?:mermaid)?\s*", "", inner, flags=re.I).strip()
        inner = re.sub(r"```\s*$", "", inner).strip()
    return inner


def is_xychart_mermaid(code: str) -> bool:
    inner = strip_mermaid_fences(code).lstrip()
    return bool(re.match(r"xychart(?:-beta)?\b", inner, re.I))


def is_misclassified_benchmark_flowchart(code: str, caption: str = "") -> bool:
    inner = strip_mermaid_fences(code)
    if not inner:
        return False
    if is_xychart_mermaid(inner):
        return False
    low = inner.lower()
    combined = f"{caption} {inner}"
    if not _BENCHMARK_TEXT_HINTS.search(combined):
        return False
    if "flowchart" not in low and not re.match(r"graph\s", low):
        return False
    if re.search(
        r"\[[^\]]*(ось\s*[xy]|x-axis|y-axis|axis)\b",
        inner,
        re.IGNORECASE,
    ):
        return True
    nodes = re.findall(r"\[([^\]]+)\]", inner)
    bench_nodes = sum(
        1 for n in nodes if re.search(r"qps|recall|ось\s*[xy]", n, re.IGNORECASE)
    )
    return bench_nodes >= 2


def validate_mermaid_syntax(code: str) -> bool:
    raw = (code or "").strip()
    if len(raw) < 8 or len(raw) > 12000:
        return False
    if _FORBIDDEN.search(raw):
        return False
    inner = strip_mermaid_fences(raw)
    from knowledge_engine.utils.mermaid_sanitizer import is_mermaid_syntax_valid

    # Fast gate: flowchart+xychart mix / unmatched quotes → Gemma repair path.
    if not is_mermaid_syntax_valid(inner):
        return False
    if has_broken_init_directive(inner):
        return False
    if not _MERMAID_HEAD.search(inner):
        return False
    if inner.count("\n") > 200:
        return False
    if not _validate_xychart_body(inner):
        return False
    low = inner.lower()
    if is_xychart_mermaid(inner) and (
        "flowchart" in low or re.match(r"graph\s", inner.strip(), re.I)
    ):
        return False
    if not _validate_sequence_lines(inner):
        return False
    if _needs_ui_repair(inner):
        return False
    # Strict AST/block lint (spliced end, unclosed subgraph, …).
    ok, _errs = lint_mermaid_ast(inner)
    if not ok:
        return False
    return True


def format_mermaid_for_storage(inner: str) -> str:
    """Fenced mermaid: sanitize → validate; без llm_text_repair если уже валидно."""
    from knowledge_engine.web.llm_text_repair import (
        _ensure_sequence_init,
        repair_diagram_markdown,
    )

    sanitized = sanitize_mermaid_syntax((inner or "").strip())
    if sanitized and validate_mermaid_syntax(sanitized):
        body = _ensure_sequence_init(sanitized)
        return f"```mermaid\n{body}\n```"
    return repair_diagram_markdown(sanitize_mermaid_syntax(inner)).strip()


def _format_for_storage(inner: str) -> str:
    return format_mermaid_for_storage(inner).strip()


def process_mermaid_for_ingest(
    raw_vlm: str,
    *,
    allow_gemma_repair: bool = True,
) -> str:
    """
    Этапы: sanitize → format → validate → (опционально) Gemma repair → validate.
    Возвращает fenced mermaid или "" если сохранять нельзя.
    """
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                process_mermaid_for_ingest_async(
                    raw_vlm, allow_gemma_repair=allow_gemma_repair
                ),
            ).result()
    except RuntimeError:
        return asyncio.run(
            process_mermaid_for_ingest_async(
                raw_vlm, allow_gemma_repair=allow_gemma_repair
            )
        )


async def process_mermaid_for_ingest_async(
    raw_vlm: str,
    *,
    allow_gemma_repair: bool = True,
    client: object | None = None,
    rl: object | None = None,
) -> str:
    """Async ingest path; optional shared httpx + RateLimitedLLMClient for batches."""
    from knowledge_engine.config import gemma_cloud_api_key_available
    from knowledge_engine.services.mermaid_gemma_repair import (
        _repair_invalid_mermaid_async,
    )

    raw = (raw_vlm or "").strip()
    if not raw:
        return ""

    def _try_candidate(source: str) -> str:
        formatted = _format_for_storage(source)
        inner = strip_mermaid_fences(formatted)
        if validate_mermaid_syntax(inner):
            return formatted
        return ""

    hit = _try_candidate(raw)
    if hit:
        return hit

    if allow_gemma_repair and gemma_cloud_api_key_available():
        from knowledge_engine.utils.mermaid_sanitizer import (
            has_mixed_flowchart_xychart,
            sanitize_mermaid_raw_text,
        )

        pre = sanitize_mermaid_raw_text(sanitize_mermaid_syntax(raw) or raw)
        pre_inner = strip_mermaid_fences(pre)
        report_parts: list[str] = []
        _ok, lint_errs = lint_mermaid_ast(pre_inner)
        if not _ok:
            report_parts.extend(lint_errs)
        if has_mixed_flowchart_xychart(pre_inner):
            report_parts.append(
                "flowchart mixed with xychart-beta "
                "(separate diagram types; extract pure xychart-beta or flowchart)"
            )
        if not report_parts:
            report_parts.append("validate_mermaid_syntax failed after sanitize")
        err = format_mermaid_lint_report(report_parts)
        trace(f"MERMAID_INGEST → gemma | lint_errors={len(report_parts)}")
        gemma_out = await _repair_invalid_mermaid_async(
            pre,
            err,
            client=client,  # type: ignore[arg-type]
            rl=rl,  # type: ignore[arg-type]
        )
        if gemma_out:
            hit = _try_candidate(gemma_out)
            if hit:
                trace("MERMAID_INGEST ✓ | gemma repair accepted")
                return hit
            trace("MERMAID_INGEST ⊘ | gemma repair still invalid")

    trace(
        "MERMAID_INGEST ⊘ | rejected after sanitize"
        + ("+gemma" if allow_gemma_repair else "")
    )
    return ""


def sanitize_mermaid_from_vlm(code: str) -> str:
    """Лёгкая санитизация сразу после VLM (без Gemma, без жёсткого reject)."""
    return sanitize_mermaid_syntax(code)
