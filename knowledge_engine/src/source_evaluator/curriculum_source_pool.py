"""Динамический пул источников: статический whitelist + архив после Lite-валидации."""

from __future__ import annotations

from typing import Iterable, TypeVar

from urllib.parse import urlparse

from knowledge_engine.config import SOURCE_ARCHIVE_ENABLED
from knowledge_engine.services.search.url_filter import is_blocked_url
from knowledge_engine.src.source_evaluator.evaluator import match_whitelist
from knowledge_engine.ui.run_log import trace

_ARCHIVE_TRUST_REUSE = 0.72
_ARCHIVE_TRUST_REGISTER = 0.86

_ACADEMIC_OPEN_HOST_SUFFIXES = (
    "arxiv.org",
    "export.arxiv.org",
    "semanticscholar.org",
    "aclanthology.org",
    "doi.org",
    "openreview.net",
)


def _host_from_url(url: str) -> str:
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def is_academic_open_host(url: str) -> bool:
    """Открытые академические домены — всегда whitelisted provenance."""
    host = _host_from_url(url)
    if not host:
        return False
    for suffix in _ACADEMIC_OPEN_HOST_SUFFIXES:
        if host == suffix or host.endswith(f".{suffix}"):
            return True
    return False


def is_collectible_article_url(url: str) -> bool:
    """Открытый сбор: HTTP(S), не blocklist, не голый homepage."""
    u = (url or "").strip()
    if not u.startswith("http"):
        return False
    if is_blocked_url(u):
        return False
    parsed = urlparse(u)
    host = (parsed.netloc or "").strip()
    if not host or "." not in host:
        return False
    path = (parsed.path or "").strip("/")
    # Домашняя страница без пути — слабый сигнал
    if not path and not (parsed.query or "").strip():
        return False
    return True


def normalize_site_host(raw: str) -> str:
    """Hostname из URL / site: строки (для Lite site-suggest + SearXNG)."""
    s = (raw or "").strip().lower()
    for prefix in ("https://", "http://", "www."):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.split("/")[0].strip()


def cap_collectible_urls(urls: Iterable[str], cap: int) -> list[str]:
    out: list[str] = []
    for u in urls:
        if len(out) >= cap:
            break
        if is_collectible_article_url(u):
            out.append(u)
    return out


_T = TypeVar("_T")


def cap_collectible_items(
    items: Iterable[_T],
    cap: int,
    url_attr: str = "url",
) -> list[_T]:
    out: list[_T] = []
    for item in items:
        if len(out) >= cap:
            break
        url = str(getattr(item, url_attr, "") or "")
        if is_collectible_article_url(url):
            out.append(item)
    return out


def archive_trust_for_url(url: str) -> float:
    if not SOURCE_ARCHIVE_ENABLED:
        return 0.0
    try:
        from knowledge_engine.db.source_links import get_source_link_archive

        return get_source_link_archive().get_url_trust(url)
    except Exception:
        return 0.0


def resolve_source_provenance(url: str) -> tuple[str, str]:
    """
    (category_label, origin) — origin: static_whitelist | archive | open_candidate
    """
    if is_academic_open_host(url):
        return "academic_open", "static_whitelist"
    matched, cat = match_whitelist(url)
    if matched:
        return cat or "whitelist", "static_whitelist"
    trust = archive_trust_for_url(url)
    if trust >= _ARCHIVE_TRUST_REUSE:
        return "archive_trusted", "archive"
    return "open_candidate", "open_candidate"


def is_fast_trusted_source(url: str) -> bool:
    """Уже в статическом whitelist, академический open host или архив."""
    if is_academic_open_host(url):
        return True
    matched, _ = match_whitelist(url)
    if matched:
        return True
    return archive_trust_for_url(url) >= _ARCHIVE_TRUST_REUSE


def register_curriculum_source(
    url: str,
    discovery_query: str,
    *,
    category: str = "lite_approved",
    trust_score: float | None = None,
    status: str = "lite_approved",
    reason: str = "",
) -> None:
    """Пополнение .source_archive после Lite APPROVED."""
    if not SOURCE_ARCHIVE_ENABLED:
        return
    u = (url or "").strip()
    if not u.startswith("http"):
        return
    matched, static_cat = match_whitelist(u)
    if matched:
        category = static_cat or category
        trust = trust_score if trust_score is not None else 0.92
        status = "accepted"
    else:
        trust = trust_score if trust_score is not None else _ARCHIVE_TRUST_REGISTER
    try:
        from knowledge_engine.config import SOURCE_ARCHIVE_DB_PATH
        from knowledge_engine.db.source_links import SourceLinkArchive
        from knowledge_engine.services.domain_profiler import normalize_domain

        archive = SourceLinkArchive(SOURCE_ARCHIVE_DB_PATH)
        archive.upsert(
            url=u,
            domain=normalize_domain(u),
            trust_score=trust,
            category=(category or "lite_approved")[:120],
            status=status,
            rejection_reason=(reason or "")[:400] or None,
            discovery_query=(discovery_query or "")[:400],
        )
        trace(
            f"CURRICULUM source pool + | {u[:70]} | "
            f"cat={category} trust={trust:.2f}"
        )
    except Exception as exc:
        trace(f"CURRICULUM source pool upsert skip | {exc}")
