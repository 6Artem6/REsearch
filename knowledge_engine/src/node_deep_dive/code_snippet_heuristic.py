"""Эвристика: отличить код от заголовков и plain-text в code_snippets."""

from __future__ import annotations

import re

_DEEP_DIVE_PREFIX = re.compile(r"^deep\s*dive\s*:", re.I)
_FENCE_OPEN = re.compile(r"^```", re.M)
_CODE_KEYWORDS = re.compile(
    r"\b("
    r"def|class|import|from|return|elif|else|async|await|"
    r"function|const|let|var|interface|struct|enum|namespace|"
    r"SELECT|INSERT|UPDATE|DELETE|CREATE|WITH|"
    r"public|private|protected|static|void|int|float|bool"
    r")\b",
    re.I,
)
_CONTROL_LINE = re.compile(
    r"^\s*(if|for|while|switch|case|try|catch)\s*[\(:]", re.I | re.M
)


def is_likely_code_snippet(text: str) -> bool:
    raw = (text or "").strip()
    if len(raw) < 4:
        return False
    if _DEEP_DIVE_PREFIX.match(raw):
        return False
    if _FENCE_OPEN.search(raw):
        return True
    inner = re.sub(r"^```[\w-]*\s*\n?", "", raw, count=1, flags=re.I)
    inner = re.sub(r"\n?```\s*$", "", inner.strip())
    if _CODE_KEYWORDS.search(inner):
        return True
    if _CONTROL_LINE.search(inner):
        return True
    lines = [ln for ln in inner.splitlines() if ln.strip()]
    if len(lines) >= 2:
        indented = sum(
            1
            for ln in lines
            if ln.startswith(("    ", "\t"))
            or ln.strip().startswith(("def ", "class ", "#", "//", "/*"))
        )
        if indented >= 1:
            return True
        if any(";" in ln or "{" in ln or "}" in ln for ln in lines):
            return True
    if ";" in inner and ("=" in inner or "(" in inner):
        return True
    if inner.count("(") >= 2 and inner.count(")") >= 2 and "=" in inner:
        return True
    if re.search(r"=>|::|\+\+|--|\|\||&&", inner):
        return True
    return False


def filter_code_snippets(snippets: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in snippets or []:
        t = (s or "").strip()
        if not t or t in seen:
            continue
        if not is_likely_code_snippet(t):
            continue
        seen.add(t)
        out.append(t)
    return out
