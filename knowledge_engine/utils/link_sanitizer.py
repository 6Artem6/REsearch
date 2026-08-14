"""Post-processing: только allow-list URL в Markdown-ссылках лекции."""

from __future__ import annotations

import re

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
_URL_EXTRACT_RE = re.compile(r"https?://[^\s\]<\"')]+", re.I)


def normalize_lecture_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def extract_urls_from_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in _URL_EXTRACT_RE.finditer(text or ""):
        u = normalize_lecture_url(m.group(0).rstrip(".,);]"))
        if u:
            out.add(u)
    return out


def sanitize_lecture_output(generated_text: str, allowed_urls: set[str]) -> str:
    """
    Markdown [label](url): если url не в allow-list — оставить **label** без гиперссылки.
    """
    if not generated_text:
        return generated_text
    allowed = {normalize_lecture_url(u) for u in allowed_urls if (u or "").strip()}

    def replace_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if normalize_lecture_url(url) in allowed:
            return f"[{label}]({url})"
        return f"**{label}**"

    return _MD_LINK_RE.sub(replace_link, generated_text)
