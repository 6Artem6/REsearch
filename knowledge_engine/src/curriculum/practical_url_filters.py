"""Post-filters для practical сбора: не arXiv/DOI на веб-ветке."""

from __future__ import annotations

from urllib.parse import urlparse

from knowledge_engine.ui.run_log import trace

_PRACTICAL_BLOCKED_HOST_SUFFIXES = (
    "arxiv.org",
    "doi.org",
    "semanticscholar.org",
    "export.arxiv.org",
    "dictionary.com",
    "merriam-webster.com",
    "cambridge.org",
    "wikipedia.org",
    "books.google.com",
)


_PRACTICAL_API_DOC_PATH_MARKERS: tuple[str, ...] = (
    "/docs/",
    "/reference/",
    "/api/",
    "v1_operations",
    "v2_operations",
    "/sdk/",
    "/swagger/",
    "/openapi/",
    "/apidocs/",
)


def practical_url_reject_reason(url: str) -> str | None:
    """Причина отсева для practical; None = URL допустим."""
    u = (url or "").strip()
    if not u.startswith("http"):
        return "not_http"
    low = u.lower()
    for marker in _PRACTICAL_API_DOC_PATH_MARKERS:
        if marker in low:
            return f"practical_api_doc_path:{marker}"
    try:
        host = (urlparse(u).netloc or "").lower()
    except Exception:
        return "parse_error"
    if host.startswith("www."):
        host = host[4:]
    for suffix in _PRACTICAL_BLOCKED_HOST_SUFFIXES:
        if host == suffix or host.endswith(f".{suffix}"):
            return f"practical_blocked_host:{suffix}"
    if host.endswith(".arxiv.org"):
        return "practical_blocked_host:arxiv"
    return None


def filter_practical_search_row(row: dict[str, str]) -> bool:
    url = str(row.get("url") or "")
    reason = practical_url_reject_reason(url)
    if reason:
        trace(f"CURRICULUM practical filter ⊘ | {url[:70]} | {reason}")
        return False
    return True


def filter_practical_curriculum_hit(hit) -> bool:
    from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit

    if not isinstance(hit, CurriculumSearchHit):
        url = str(hit.get("url") or "")
    else:
        url = hit.url
    reason = practical_url_reject_reason(url)
    if reason:
        trace(f"CURRICULUM practical filter ⊘ | {url[:70]} | {reason}")
        return False
    return True
