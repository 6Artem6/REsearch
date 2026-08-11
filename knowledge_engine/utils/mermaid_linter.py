"""Pure-Python синтаксическая проверка Mermaid (без Node/npm)."""

from __future__ import annotations

import re

_DIAGRAM_HEAD = re.compile(
    r"^(?:"
    r"flowchart\b|graph\s+(?:TD|LR|BT|RL)\b|"
    r"sequenceDiagram\b|classDiagram\b|"
    r"stateDiagram(?:-v2)?\b|erDiagram\b|"
    r"xychart(?:-beta)?\b"
    r")",
    re.IGNORECASE,
)

_INIT_WELL_FORMED = re.compile(r"%%\s*\{init:\s*\{[\s\S]*?\}\s*\}%%", re.I)
_INIT_BROKEN = re.compile(r"%%\s*\{init:", re.I)

_SUBGRAPH_ON_LINE = re.compile(r"subgraph\b.*\bdirection\b", re.I)
_DIRECTION_WITH_TAIL = re.compile(
    r"^\s*direction\s+(?:TB|LR|TD|RL|BT)\s+(.+)$",
    re.I,
)

_ALLOWED_TAG = re.compile(r"<\/?(?:br|b|i|div|span)\b", re.I)
_ARROW_FRAG = re.compile(
    r"(?:-->|---|----|-.->|==>|o-->|x-->|-->>|->>|-\)|\)-->|<\-\-|<\-\->)",
    re.I,
)


def _strip_init_blocks(text: str) -> str:
    s = (text or "").strip()
    while _INIT_WELL_FORMED.search(s):
        s = _INIT_WELL_FORMED.sub("", s, count=1).strip()
    return s


def _is_skipped_comment_line(stripped: str) -> bool:
    if not stripped.startswith("%%"):
        return False
    if _INIT_BROKEN.search(stripped):
        return True
    return True


def _check_unescaped_lt_gt(line_no: int, line: str) -> list[str]:
    errors: list[str] = []
    i = 0
    n = len(line)
    lt_err = False
    gt_err = False
    while i < n:
        ch = line[i]
        if ch == "<":
            if line[i : i + 4].lower() == "&lt;":
                i += 4
                continue
            tail = line[i:]
            if _ALLOWED_TAG.match(tail):
                close = tail.find(">")
                i += (close + 1) if close >= 0 else len(tail)
                continue
            if tail.startswith("<!--"):
                end = tail.find("-->")
                i += (end + 3) if end >= 0 else n
                continue
            if not lt_err:
                errors.append(
                    f"line {line_no}: unescaped '<' (use &lt; or allowed tags like <br/>)"
                )
                lt_err = True
            i += 1
            continue
        if ch == ">":
            if i >= 3 and line[i - 3 : i + 1].lower() == "&gt;":
                i += 1
                continue
            if i > 0 and line[i - 1] == "/":
                i += 1
                continue
            window = line[max(0, i - 4) : min(n, i + 4)]
            if _ARROW_FRAG.search(window):
                i += 1
                continue
            if not gt_err:
                errors.append(f"line {line_no}: unescaped '>' outside arrows/tags")
                gt_err = True
            i += 1
            continue
        i += 1
    return errors


def lint_mermaid_code(code: str) -> tuple[bool, list[str]]:
    """
    Синтаксическая проверка Mermaid без внешних CLI.
    Returns (is_valid, error_messages).
    """
    errors: list[str] = []
    raw = (code or "").replace("\r\n", "\n")
    if not raw.strip():
        return False, ["empty diagram"]

    _strip_init_blocks(raw)
    meaningful: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(raw.split("\n"), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_skipped_comment_line(stripped):
            continue
        meaningful.append((line_no, line, stripped))

    if not meaningful:
        errors.append("no diagram body after comments/init")
        return False, errors

    head_line = _strip_init_blocks(meaningful[0][2])
    if not _DIAGRAM_HEAD.match(head_line):
        errors.append(
            f"line {meaningful[0][0]}: missing or invalid diagram type header"
        )

    stack: list[str] = []
    for line_no, _original, stripped in meaningful:
        if re.match(r"subgraph\b", stripped, re.I):
            stack.append("subgraph")
            if _SUBGRAPH_ON_LINE.search(stripped):
                errors.append(
                    f"line {line_no}: subgraph and direction on the same line"
                )

        dir_tail = _DIRECTION_WITH_TAIL.match(stripped)
        if dir_tail and (dir_tail.group(1) or "").strip():
            errors.append(
                f"line {line_no}: direction and other declarations on the same line"
            )

        if re.match(r"end\s*$", stripped, re.I):
            if not stack:
                errors.append(f"line {line_no}: unmatched end")
            else:
                stack.pop()

        for bracket in re.findall(r"\[([^\]]*)\]", stripped):
            if bracket.count('"') % 2 == 1:
                errors.append(
                    f"line {line_no}: unbalanced double quotes in '[...]' label"
                )

        errors.extend(_check_unescaped_lt_gt(line_no, stripped))

    if stack:
        errors.append(f"unclosed subgraph blocks: {len(stack)}")

    return len(errors) == 0, errors
