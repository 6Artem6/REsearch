"""Рекурсивный поиск PDF-URL в HTML (ACM epdf → /doi/pdf/?download=true)."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from knowledge_engine.services.parsers.pdf_bytes import (
    acm_doi_pdf_download_variants,
    acm_epdf_to_pdf_url,
    is_acm_doi_pdf_url,
)
from knowledge_engine.services.web_extract import smart_fetch_page_html
from knowledge_engine.ui.run_log import trace

_ABSOLUTE_PDF_RE = re.compile(
    r"https?://[^\s\"'<>]+(?:/doi/pdf/[^\s\"'<>]*|\.pdf(?:\?[^\s\"'<>]*)?)",
    re.I,
)
_ACM_EPDF_RE = re.compile(r"/doi/epdf/", re.I)
_READER_PATH_HINTS = ("/epdf/", "/reader/", "/viewer/", "pdf-reader", "readcube")
_SAME_HOST_READER_HINTS = ("/doi/", "/epdf/", "/pdf/")


def _normalize_key(url: str) -> str:
    return (url or "").split("#", 1)[0].rstrip("/").lower()


def _expand_acm_variants(url: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    u = (url or "").strip()
    if not u.startswith("http"):
        return out
    candidates = [u, acm_epdf_to_pdf_url(u)]
    if is_acm_doi_pdf_url(u) or _ACM_EPDF_RE.search(u):
        candidates.extend(acm_doi_pdf_download_variants(u))
    for c in candidates:
        key = _normalize_key(c)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def extract_pdf_urls_from_html(html: str, base_url: str) -> list[str]:
    """Все прямые PDF-URL из разметки + ACM-варианты (?download=true)."""
    if not (html or "").strip():
        return []
    base = (base_url or "").strip()
    found: list[str] = []
    seen: set[str] = set()

    def ingest(raw: str) -> None:
        u = (raw or "").strip()
        if not u or u.startswith("#") or u.lower().startswith("javascript:"):
            return
        full = urljoin(base, u)
        if not full.startswith("http"):
            return
        for variant in _expand_acm_variants(full):
            key = _normalize_key(variant)
            if key in seen:
                continue
            low = variant.lower()
            if (
                low.endswith(".pdf")
                or "/doi/pdf/" in low
                or "download=true" in low
                or ("/pdf/" in low and "type=pdf" in low)
            ):
                seen.add(key)
                found.append(variant)

    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all("a", href=True):
        ingest(el.get("href") or "")
    for el in soup.find_all("link", href=True):
        ingest(el.get("href") or "")
    for tag, attr in (("iframe", "src"), ("embed", "src"), ("object", "data")):
        for el in soup.find_all(tag):
            ingest(el.get(attr) or "")
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and (meta.get("content") or "").strip():
        ingest(meta.get("content") or "")

    for m in _ABSOLUTE_PDF_RE.findall(html):
        ingest(m)

    return found


def extract_reader_follow_urls(html: str, base_url: str) -> list[str]:
    """Ссылки на просмотрщики того же хоста (для рекурсии epdf → landing)."""
    if not (html or "").strip():
        return []
    base = (base_url or "").strip()
    parsed_base = urlparse(base)
    host = (parsed_base.netloc or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all("a", href=True):
        href = (el.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        full = urljoin(base, href)
        if not full.startswith("http"):
            continue
        low = full.lower()
        if urlparse(full).netloc.lower() != host:
            continue
        if not any(h in low for h in _SAME_HOST_READER_HINTS):
            continue
        if not (
            any(h in low for h in _READER_PATH_HINTS)
            or "/doi/pdf/" in low
            or low.endswith(".pdf")
        ):
            continue
        key = _normalize_key(full)
        if key not in seen:
            seen.add(key)
            out.append(full)
    return out


def discover_pdf_urls_recursive(
    seed_urls: list[str],
    *,
    max_depth: int = 2,
    max_pages: int = 8,
    skip_page_keys: set[str] | None = None,
) -> list[str]:
    """
    Обход страниц-просмотрщиков: из HTML извлекаются PDF-URL и (ограниченно) follow.
    """
    queue: list[tuple[str, int]] = []
    seen_pages: set[str] = set()
    pdf_urls: list[str] = []
    pdf_seen: set[str] = set()
    skip = skip_page_keys or set()

    for u in seed_urls:
        u = (u or "").strip()
        if u.startswith("http"):
            queue.append((u, 0))

    while queue and len(seen_pages) < max_pages:
        page_url, depth = queue.pop(0)
        page_key = _normalize_key(page_url)
        if page_key in seen_pages or page_key in skip:
            continue
        seen_pages.add(page_key)

        html, method = smart_fetch_page_html(page_url)
        if not html:
            continue
        trace(f"PDF_LINK_RECURSE ▶ | depth={depth} via={method} | {page_url[:65]}")

        for pdf_u in extract_pdf_urls_from_html(html, page_url):
            pk = _normalize_key(pdf_u)
            if pk not in pdf_seen:
                pdf_seen.add(pk)
                pdf_urls.append(pdf_u)

        if depth >= max_depth:
            continue
        for follow in extract_reader_follow_urls(html, page_url):
            fk = _normalize_key(follow)
            if fk not in seen_pages:
                queue.append((follow, depth + 1))

    if pdf_urls:
        trace(f"PDF_LINK_RECURSE ✓ | pages={len(seen_pages)} pdf_urls={len(pdf_urls)}")
    return pdf_urls
