"""Post-process dense lecture markdown (code fences in lecture_body)."""

from __future__ import annotations

import re

_PY_FENCE_RE = re.compile(
    r"```[Pp]ython\s*\n?(.*?)```",
    re.DOTALL,
)

# Heuristic splits for model-glued Python lines (no newline before keyword).
_GLUE_PATTERNS: list[tuple[str, str]] = [
    (r"import ([a-zA-Z0-9_]+)import ", r"import \1\nimport "),
    (r"import ([a-zA-Z0-9_]+)from ", r"import \1\nfrom "),
    (r"from ([^\n]+?)import ", r"from \1\nimport "),
    (r"([^\n])def ", r"\1\ndef "),
    (r"([^\n])class ", r"\1\nclass "),
    (r"\)def ", r")\ndef "),
    (r"\)class ", r")\nclass "),
    (r"return ([^\n]+?)def ", r"return \1\ndef "),
    (r"import hmacimport ", r"import hmac\nimport "),
    (r"import hashlibimport ", r"import hashlib\nimport "),
    (r"([a-zA-Z0-9_]+)class ", r"\1\n\nclass "),
    (r"Fieldclass ", r"Field\n\nclass "),
]


def _normalize_python_fence_body(code: str) -> str:
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in text and text.count("\n") <= 1:
        text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    for pat, repl in _GLUE_PATTERNS:
        text = re.sub(pat, repl, text)
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        stripped = ln.lstrip(" ")
        if stripped and not stripped.startswith("#"):
            pad = len(ln) - len(stripped)
            if pad > 0 and pad % 4 != 0 and pad < 8:
                ln = " " * (pad // 4 * 4) + stripped
        out.append(ln)
    return "\n".join(out).strip("\n") + "\n"


def sanitize_lecture_body_markdown(body: str) -> str:
    """Restore newlines inside ```python``` blocks in lecture_body."""
    raw = body or ""

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1) or ""
        fixed = _normalize_python_fence_body(inner)
        return f"\n\n```python\n{fixed}```\n\n"

    return _PY_FENCE_RE.sub(_repl, raw)


_CREDIT_SCOREBOARD_RE = re.compile(
    r"(?is)^(?:---\s*)?"
    r"\*\*[^\n]*Что уже зачтено[^\n]*\*\*[^\n]*\n?"
    r"(?:\*\*[^\n]*Чего не хватило[^\n]*\*\*[^\n]*\n?)?"
    r"(?:---\s*)?"
)

_SELF_CHECK_MARKER = "**Самопроверка:**"
_MIN_TRAILING_QUESTION_LEN = 20
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Trailing quiz headings. «Самопроверка» is distinctive; «Вопрос:» only at line start
# so mid-sentence «рассмотрим вопрос:» is not treated as a checkpoint header.
_SELF_CHECK_HEADER_RE = re.compile(
    r"(?is)(?:\*{1,2}\s*)?самопроверка\s*(?:\*{1,2})?\s*:"
)
_LEADING_QUIZ_HEADER_RE = re.compile(
    r"(?is)^\s*(?:\*{1,2}\s*)?(?:самопроверка|вопрос)\s*(?:\*{1,2})?\s*:"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _normalize_question_text(text: str) -> str:
    t = (text or "").strip().lower().replace("ё", "е")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _questions_equivalent(a: str, b: str) -> bool:
    na = _normalize_question_text(a)
    nb = _normalize_question_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ca, cb = na.replace(" ", ""), nb.replace(" ", "")
    if ca == cb:
        return True
    if len(na) >= _MIN_TRAILING_QUESTION_LEN and (na in nb or nb in na):
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(shorter) / max(len(longer), 1) >= 0.85:
            return True
    return False


def _strip_inline_checkpoint_tail(paragraph: str, checkpoint: str) -> str:
    """Remove a checkpoint question glued to the end of one paragraph."""
    text = (paragraph or "").strip()
    q = (checkpoint or "").strip()
    if not text or not q:
        return text
    if text.rstrip().endswith(q.rstrip()):
        prefix = text[: -len(q.rstrip())].rstrip()
        return prefix.rstrip(" .")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2 and _questions_equivalent(lines[-1], q):
        return "\n".join(lines[:-1]).strip()
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) >= 2 and _questions_equivalent(sentences[-1], q):
        return " ".join(sentences[:-1]).rstrip(" .")
    if _questions_equivalent(text, q):
        return ""
    return text


def _extract_self_check_tail(body: str) -> tuple[str, str]:
    """Return (head without trailing quiz header, text after that header)."""
    text = (body or "").strip()
    if not text:
        return "", ""
    last_break = text.rfind("\n\n")
    region_start = 0 if last_break < 0 else last_break + 2
    last_block = text[region_start:]
    leading = _LEADING_QUIZ_HEADER_RE.match(last_block)
    if leading:
        return (
            text[: region_start + leading.start()].rstrip(),
            last_block[leading.end() :].strip(),
        )
    last_sc = None
    for match in _SELF_CHECK_HEADER_RE.finditer(last_block):
        last_sc = match
    if last_sc is None:
        return text, ""
    return (
        text[: region_start + last_sc.start()].rstrip(),
        last_block[last_sc.end() :].strip(),
    )


def strip_trailing_checkpoint_from_lecture_body(
    body: str,
    checkpoint: str = "",
) -> str:
    """Remove PART 2 question duplicated at the end of lecture_body (plain or marked)."""
    head, marker_q = _extract_self_check_tail(body)
    q = (checkpoint or "").strip() or marker_q
    if not head:
        return ""

    paras = [p.strip() for p in re.split(r"\n\s*\n", head) if p.strip()]
    while paras:
        last = paras[-1]
        if _LEADING_QUIZ_HEADER_RE.match(last):
            paras.pop()
            continue
        header_match = None
        for match in _SELF_CHECK_HEADER_RE.finditer(last):
            header_match = match
        if header_match is not None:
            prefix = last[: header_match.start()].rstrip()
            if prefix:
                paras[-1] = prefix
            else:
                paras.pop()
            continue
        if q and _questions_equivalent(last, q):
            if _normalize_question_text(last) == _normalize_question_text(q):
                paras.pop()
                continue
            trimmed = _strip_inline_checkpoint_tail(last, q)
            if trimmed != last:
                if trimmed:
                    paras[-1] = trimmed
                else:
                    paras.pop()
                continue
            paras.pop()
            continue
        if (
            q
            and last.rstrip().endswith("?")
            and _questions_equivalent(last.split("\n")[-1].strip(), q)
        ):
            lines = [ln.strip() for ln in last.split("\n") if ln.strip()]
            if len(lines) >= 2 and _questions_equivalent(lines[-1], q):
                rebuilt = "\n".join(lines[:-1]).strip()
                if rebuilt:
                    paras[-1] = rebuilt
                else:
                    paras.pop()
                continue
        if (
            not q
            and len(last) >= _MIN_TRAILING_QUESTION_LEN
            and last.rstrip().endswith(("?", "？"))
        ):
            paras.pop()
            continue
        break

    cleaned = "\n\n".join(paras).strip()
    if q:
        inline_trimmed = _strip_inline_checkpoint_tail(cleaned, q)
        if inline_trimmed != cleaned:
            cleaned = inline_trimmed
        elif cleaned.rstrip().endswith(q.rstrip()):
            cleaned = cleaned[: -len(q.rstrip())].rstrip().rstrip(" .")
        elif cleaned.endswith(q):
            cleaned = cleaned[: -len(q)].rstrip().rstrip(" .")
    return cleaned


def format_self_check_block(checkpoint: str) -> str:
    q = (checkpoint or "").strip()
    if not q:
        return ""
    return f"{_SELF_CHECK_MARKER} {q}"


def append_checkpoint_to_lecture_body(body: str, checkpoint: str) -> str:
    """Append exactly one **Самопроверка:** block; dedupe plain tail duplicates."""
    q = (checkpoint or "").strip()
    if not q:
        return (body or "").strip()

    head, marker_q = _extract_self_check_tail(body)
    canonical = q or marker_q
    head = strip_trailing_checkpoint_from_lecture_body(head, canonical)
    block = format_self_check_block(canonical)
    if not head:
        return block
    return f"{head}\n\n{block}".strip()


def strip_lecture_credit_scoreboard(body: str) -> str:
    """Remove dialogue credit/scoreboard headers if the model glued them onto a lecture."""
    text = (body or "").strip()
    if not text:
        return ""
    if "Что уже зачтено" not in text and "Чего не хватило" not in text:
        return text
    cleaned = _CREDIT_SCOREBOARD_RE.sub("", text, count=1).strip()
    return cleaned or text


from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    SCHEMA_TUTOR_MESSAGE_MAX,
)

_TUTOR_MESSAGE_CHAR_LIMIT = SCHEMA_TUTOR_MESSAGE_MAX


def clip_lecture_keeping_checkpoint(
    body: str,
    checkpoint: str = "",
    limit: int = _TUTOR_MESSAGE_CHAR_LIMIT,
) -> str:
    """Fit lecture + closing question into ``tutor_message`` max_length.

    Host must not slice ``body[:limit]`` after appending the checkpoint — that
    drops PART 2 when ``lecture_body`` is long.
    """
    text = strip_trailing_checkpoint_from_lecture_body(body, checkpoint)
    q = (checkpoint or "").strip()
    merged = append_checkpoint_to_lecture_body(text, q) if q else text
    if len(merged) <= limit:
        return merged

    closing = format_self_check_block(q)
    if not closing:
        return merged[:limit]

    reserve = min(len(closing) + 2, max(0, limit - 1))
    head_budget = max(0, limit - reserve)
    head = text[:head_budget].rstrip()
    if not head:
        return closing[:limit]
    return f"{head}\n\n{closing}"[:limit]
