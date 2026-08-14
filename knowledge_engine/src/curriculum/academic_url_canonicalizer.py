"""Каноникализация академических URL/DOI до blocklist и url_validate."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

import httpx

from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

_DEFAULT_HEAD_TIMEOUT = 2.5

_DOI_RESOLVER_HOSTS = frozenset(
    {
        "doi.org",
        "www.doi.org",
        "dx.doi.org",
        "www.dx.doi.org",
    }
)

_CANONICAL_TARGET_SUFFIXES = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "zenodo.org",
    "semanticscholar.org",
    "huggingface.co",
    "openreview.net",
    "aclanthology.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "wiley.com",
    "acs.org",
    "cambridge.org",
    "oup.com",
    "plos.org",
)

_ARXIV_DOI_NEW = re.compile(
    r"10\.48550/arxiv\.(\d{4}\.\d{4,5})(v\d+)?",
    re.IGNORECASE,
)
_ARXIV_DOI_OLD = re.compile(
    r"10\.48550/arxiv\.([a-z][a-z\-]*/\d{4,7})(v\d+)?",
    re.IGNORECASE,
)

_BIORXIV_MEDRXIV_DOI = re.compile(
    r"10\.1101/(\d{4}\.\d{2}\.\d{2}\.\d{6,9})(v\d+)?",
    re.IGNORECASE,
)

_ZENODO_DOI = re.compile(
    r"10\.5281/zenodo\.(\d+)",
    re.IGNORECASE,
)

_ARXIV_ABS_PATH = re.compile(r"arxiv\.org/abs/(.+)$", re.IGNORECASE)
_ARXIV_PDF_PATH = re.compile(r"arxiv\.org/pdf/(.+)$", re.IGNORECASE)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; REsearch/1.0; +https://github.com/) "
    "AcademicUrlCanonicalizer"
)


def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    if h.startswith("www."):
        return h[4:]
    return h


def arxiv_id_from_url(url: str) -> str | None:
    u = (url or "").strip()
    if not u:
        return None
    m = _ARXIV_ABS_PATH.search(u)
    if m:
        return m.group(1).strip().rstrip("/")
    m = _ARXIV_PDF_PATH.search(u)
    if m:
        tail = m.group(1).strip().rstrip("/")
        if tail.lower().endswith(".pdf"):
            tail = tail[:-4]
        return tail
    return None


def arxiv_pdf_url_for_id(arxiv_id: str) -> str:
    aid = (arxiv_id or "").strip().rstrip("/")
    if aid.lower().endswith(".pdf"):
        aid = aid[:-4]
    return f"https://arxiv.org/pdf/{aid}.pdf"


def arxiv_abs_url_to_pdf_url(url: str) -> str | None:
    """https://arxiv.org/abs/ID → https://arxiv.org/pdf/ID.pdf"""
    aid = arxiv_id_from_url(url)
    if not aid or "arxiv.org/abs/" not in (url or "").lower():
        return None
    return arxiv_pdf_url_for_id(aid)


def coerce_arxiv_url_to_pdf(url: str) -> str:
    """Единый ingest/registry URL для arXiv: только /pdf/, не /abs/."""
    u = (url or "").strip()
    if not u:
        return u
    pdf = arxiv_abs_url_to_pdf_url(u)
    if pdf:
        return pdf
    aid = arxiv_id_from_url(u)
    if aid and "arxiv.org" in u.lower():
        return arxiv_pdf_url_for_id(aid)
    return u


def academic_source_dedupe_key(url: str) -> str:
    """
    Ключ для seen/dedup: одна статья — один слот (abs/pdf/doi не дублируют).
    """
    u = coerce_arxiv_url_to_pdf((url or "").strip())
    aid = arxiv_id_from_url(u)
    if aid:
        norm = aid.lower().rstrip("/")
        if norm.endswith(".pdf"):
            norm = norm[:-4]
        return f"arxiv:{norm}"
    tail = _doi_tail_from_url(u)
    if tail:
        m = _ARXIV_DOI_NEW.search(tail) or _ARXIV_DOI_OLD.search(tail)
        if m:
            aid_part = m.group(1)
            ver = (m.group(2) or "").lower()
            return f"arxiv:{aid_part.lower()}{ver}"
        return f"doi:{tail.lower()}"
    return u.rstrip("/").lower()


def _is_doi_resolver_url(url: str) -> bool:
    try:
        host = _normalize_host(urlparse(url.strip()).netloc)
    except ValueError:
        return False
    return host in _DOI_RESOLVER_HOSTS


def _doi_tail_from_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        path = unquote(urlparse(raw).path or "").strip("/")
    except ValueError:
        return ""
    if path.lower().startswith("doi/"):
        path = path[4:]
    return path.strip()


def _is_allowed_canonical_target(url: str) -> bool:
    try:
        host = _normalize_host(urlparse(url.strip()).netloc)
    except ValueError:
        return False
    if not host:
        return False
    for suffix in _CANONICAL_TARGET_SUFFIXES:
        if host == suffix or host.endswith(f".{suffix}"):
            return True
    if host.endswith(".arxiv.org"):
        return True
    return False


def _arxiv_pdf_from_doi_tail(tail: str) -> str | None:
    m = _ARXIV_DOI_NEW.search(tail)
    if m:
        aid = m.group(1) + (m.group(2) or "")
        return arxiv_pdf_url_for_id(aid)
    m = _ARXIV_DOI_OLD.search(tail)
    if m:
        aid = m.group(1) + (m.group(2) or "")
        return arxiv_pdf_url_for_id(aid)
    return None


def canonicalize_academic_url_pure(url: str) -> str | None:
    """
    Этап 1: RegEx, без сети. arXiv DOI → PDF URL (не /abs/).
    """
    tail = _doi_tail_from_url(url)
    if not tail:
        aid = arxiv_id_from_url(url)
        if aid:
            return arxiv_pdf_url_for_id(aid)
        return None

    arxiv_pdf = _arxiv_pdf_from_doi_tail(tail)
    if arxiv_pdf:
        return arxiv_pdf

    m = _BIORXIV_MEDRXIV_DOI.search(tail)
    if m:
        body = m.group(1)
        ver = m.group(2) or ""
        return f"https://www.biorxiv.org/content/10.1101/{body}{ver}"

    m = _ZENODO_DOI.search(tail)
    if m:
        rec = m.group(1)
        return f"https://zenodo.org/record/{rec}"

    return None


async def resolve_doi_resolver_via_head(
    url: str,
    *,
    timeout: float = _DEFAULT_HEAD_TIMEOUT,
) -> str | None:
    """Этап 2: HEAD с follow_redirects для doi.org / dx.doi.org."""
    raw = (url or "").strip()
    if not raw.startswith("http") or not _is_doi_resolver_url(raw):
        return None
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.head(raw)
            final = str(resp.url).strip()
            if (
                final
                and final.startswith("http")
                and _is_allowed_canonical_target(final)
            ):
                return coerce_arxiv_url_to_pdf(final)
    except Exception as exc:
        trace(f"ACADEMIC_URL canon HEAD skip | {raw[:70]} | {exc}")
    return None


async def canonicalize_academic_url(
    url: str,
    *,
    head_timeout: float = _DEFAULT_HEAD_TIMEOUT,
) -> str:
    """Pure RegEx, HEAD fallback; arXiv всегда → /pdf/."""
    raw = (url or "").strip()
    if not raw.startswith("http"):
        return raw

    pure = canonicalize_academic_url_pure(raw)
    if pure:
        return coerce_arxiv_url_to_pdf(pure)

    if _is_doi_resolver_url(raw):
        resolved = await resolve_doi_resolver_via_head(raw, timeout=head_timeout)
        if resolved:
            return coerce_arxiv_url_to_pdf(resolved)

    return coerce_arxiv_url_to_pdf(raw)


async def canonicalize_curriculum_hit(
    hit: CurriculumSearchHit,
    *,
    head_timeout: float = _DEFAULT_HEAD_TIMEOUT,
) -> CurriculumSearchHit:
    new_url = await canonicalize_academic_url(hit.url, head_timeout=head_timeout)
    if new_url != (hit.url or "").strip():
        trace(f"ACADEMIC_URL canon ✓ | {hit.url[:55]} → {new_url[:55]}")
        return hit.model_copy(update={"url": new_url[:2000]})
    return hit


async def canonicalize_curriculum_hits(
    hits: list[CurriculumSearchHit],
    *,
    head_timeout: float = _DEFAULT_HEAD_TIMEOUT,
) -> list[CurriculumSearchHit]:
    out: list[CurriculumSearchHit] = []
    for h in hits:
        out.append(await canonicalize_curriculum_hit(h, head_timeout=head_timeout))
    return out
