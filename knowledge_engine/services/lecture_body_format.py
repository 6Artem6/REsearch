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
        return f"```python\n{fixed}```"

    return _PY_FENCE_RE.sub(_repl, raw)


_CREDIT_SCOREBOARD_RE = re.compile(
    r"(?is)^(?:---\s*)?"
    r"\*\*[^\n]*Что уже зачтено[^\n]*\*\*[^\n]*\n?"
    r"(?:\*\*[^\n]*Чего не хватило[^\n]*\*\*[^\n]*\n?)?"
    r"(?:---\s*)?"
)


def strip_lecture_credit_scoreboard(body: str) -> str:
    """Remove dialogue credit/scoreboard headers if the model glued them onto a lecture."""
    text = (body or "").strip()
    if not text:
        return ""
    if "Что уже зачтено" not in text and "Чего не хватило" not in text:
        return text
    cleaned = _CREDIT_SCOREBOARD_RE.sub("", text, count=1).strip()
    return cleaned or text


def append_checkpoint_to_lecture_body(body: str, checkpoint: str) -> str:
    """Ensure closing technical question is part of the streamed lecture text."""
    base = (body or "").strip()
    q = (checkpoint or "").strip()
    if not q:
        return base
    if q in base:
        return base
    if not base:
        return q
    return f"{base}\n\n{q}".strip()
