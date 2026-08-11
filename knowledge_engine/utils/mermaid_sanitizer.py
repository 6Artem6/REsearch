"""Пост-обработка Mermaid от VLM (экранирование, subgraph, br, quotes)."""

from __future__ import annotations

import re

_ALLOWED_INLINE_TAGS = re.compile(r"<\/?(br|b|i|div|span)\b", re.I)
_SUBGRAPH_DIRECTION_SPLIT = re.compile(
    r"(subgraph\s+[^\n]+?)\s+direction\s+(TB|LR|TD|RL|BT)",
    re.IGNORECASE,
)
_DIRECTION_LINE_TAIL = re.compile(
    r"^(\s*)direction\s+(TB|LR|TD|RL|BT)\s+(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

# Duplicate closing brackets/quotes at node ends (VLM noise).
_DUP_CLOSE_BRACKET_QUOTE = re.compile(r'"\]"\]"+')
_DUP_CLOSE_BRACKET = re.compile(r'"\]"\]+')
# Hanging single quote before node close: label'"] → label"]
_HANGING_SQUOTE_BEFORE_CLOSE = re.compile(r"'\"\]")
# Spaces inside node bracket declarations: [ "Text"] / ["Text" ]
_SPACE_AFTER_OPEN_BRACKET = re.compile(r"\[\s+\"")
_SPACE_BEFORE_CLOSE_BRACKET = re.compile(r"\"\s+\]")

_XYCHART_TOKEN = re.compile(r"\bxychart(?:-beta)?\b", re.I)
# xychart embedded inside a flowchart node label/body.
_XYCHART_IN_NODE = re.compile(
    r"\[[^\]]*\bxychart(?:-beta)?\b",
    re.I,
)


def sanitize_mermaid_raw_text(code: str) -> str:
    """
    Fast regex cleanup before renderer / Gemma.

    Fixes duplicate closers, hanging quotes, and spaced bracket labels.
    """
    if not code:
        return ""
    text = code.replace("\r\n", "\n")

    # 1) Duplicate brackets/quotes at ends: "]"]" / "]"] → "]"
    text = _DUP_CLOSE_BRACKET_QUOTE.sub('"]', text)
    text = _DUP_CLOSE_BRACKET.sub('"]', text)
    # Hanging single quote before close: label'"] → label"]
    text = _HANGING_SQUOTE_BEFORE_CLOSE.sub('"]', text)

    # 2) Spaces inside node brackets: [ "… / …" ] → ["… / …"]
    text = _SPACE_AFTER_OPEN_BRACKET.sub('["', text)
    text = _SPACE_BEFORE_CLOSE_BRACKET.sub('"]', text)

    return text.strip()


def has_mixed_flowchart_xychart(code: str) -> bool:
    """
    True when flowchart/graph coexists with xychart-beta (esp. inside a node).

    Such diagrams must be treated as broken and sent to Gemma repair.
    """
    raw = (code or "").strip()
    if not raw:
        return False
    has_flow = bool(
        re.search(r"\bflowchart\b", raw, re.I)
        or re.search(r"\bgraph\s+(?:TD|LR|BT|RL)\b", raw, re.I)
    )
    if not has_flow:
        return False
    if not _XYCHART_TOKEN.search(raw):
        return False
    # Common VLM failure: xychart-beta pasted inside a flowchart node label.
    if _XYCHART_IN_NODE.search(raw):
        return True
    # Any coexistence of both diagram types in one body is invalid.
    return True


def is_mermaid_syntax_valid(code: str) -> bool:
    """
    Fast structural validity gate (not a full Mermaid parser).

    Returns False when flowchart mixes with xychart-beta (triggers Gemma).
    """
    raw = (code or "").strip()
    if not raw:
        return False
    if has_mixed_flowchart_xychart(raw):
        return False
    # Rough unmatched double-quote check on non-comment lines.
    for line in raw.split("\n"):
        t = line.strip()
        if not t or t.startswith("%%"):
            continue
        if t.count('"') % 2 == 1:
            return False
    return True


def _escape_lt_in_bracket_labels(code: str) -> str:
    """Экранирует `<` внутри `[...]` (подписи нод), не трогая `-->` снаружи."""

    def fix_inner(inner: str) -> str:
        inner = re.sub(r"<br/>\s*<\s*<br/>", "<br/>&lt;<br/>", inner, flags=re.I)
        parts: list[str] = []
        pos = 0
        for m in _ALLOWED_INLINE_TAGS.finditer(inner):
            parts.append(inner[pos : m.start()])
            parts.append(m.group(0))
            pos = m.end()
        tail = inner[pos:]
        tail = re.sub(r"<", "&lt;", tail)
        parts.append(tail)
        return "".join(parts)

    def repl(match: re.Match[str]) -> str:
        return f"[{fix_inner(match.group(1))}]"

    return re.sub(r"\[([^\]]*)\]", repl, code)


def sanitize_mermaid_code(code: str) -> str:
    """
    Очищает частые синтаксические ошибки Mermaid из VLM:
    regex raw cleanup, неэкранированный `<`, слитые subgraph+direction, дубли `<br/>`.
    """
    if not code:
        return ""

    text = sanitize_mermaid_raw_text(code.replace("\r\n", "\n"))

    text = re.sub(r"<br/>\s*<\s*<br/>", "<br/>&lt;<br/>", text, flags=re.I)
    text = re.sub(r"(<br/>\s*){2,}", "<br/>", text, flags=re.I)
    text = _escape_lt_in_bracket_labels(text)

    text = _SUBGRAPH_DIRECTION_SPLIT.sub(r"\1\n    direction \2", text)
    text = _DIRECTION_LINE_TAIL.sub(r"\1direction \2\n\1\3", text)

    return text.strip()
