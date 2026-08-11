"""Универсальный discovery PDF-ресурсов (Pre-Ingest Manifest)."""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Literal
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from knowledge_engine.services.parsers.article_manifest import (
    ArticleResourceManifest,
    PDFCandidate,
)
from knowledge_engine.services.parsers.html_attr import coerce_html_attr
from knowledge_engine.services.parsers.llm_link_extractor import LLMShortlinkResolver
from knowledge_engine.services.parsers.pdf_bytes import (
    acm_epdf_to_pdf_url,
    prefer_acm_pdf_endpoint,
)
from knowledge_engine.services.web_extract import (
    resolve_doi_landing,
    smart_fetch_page_html,
)
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_academic_open_host,
)
from knowledge_engine.ui.run_log import trace

_MANIFEST_CACHE: dict[str, ArticleResourceManifest] = {}

_DIRECT_HINTS = (".pdf", "/pdf/", "download=true", "type=pdf")
_READER_HINTS = (
    "/epdf/",
    "epdf",
    "viewer",
    "readcube",
    "reader",
    "/pdfviewer",
    "pdf-reader",
)
_HREF_SCORE_HINTS = ("/pdf/", "/epdf/", "download", "reader", "viewer")
_LABEL_TEXT_RE = re.compile(r"(pdf|ereader|full\s*text|download)", re.I)
_ATTR_HINT_RE = re.compile(r"(pdf|ereader|reader|viewer|download)", re.I)
_MIN_DOM_SCORE = 2


def _score_link_element(el, url_low: str) -> int:
    score = 0
    for hint in _HREF_SCORE_HINTS:
        if hint in url_low:
            score += 2
    text = el.get_text(" ", strip=True)
    if _LABEL_TEXT_RE.search(text):
        score += 2
    for attr in ("aria-label", "title", "class"):
        raw = el.get(attr)
        if not raw:
            continue
        val = coerce_html_attr(raw)
        if _ATTR_HINT_RE.search(val):
            score += 1
    return score


def _kind_from_dom_anchor(el, url_low: str) -> Literal["direct_pdf", "html_reader"]:
    cls_s = coerce_html_attr(el.get("class")).lower()
    if (
        "/epdf/" in url_low
        or "/reader/" in url_low
        or "/viewer/" in url_low
        or "ereader" in cls_s
    ):
        return "html_reader"
    if any(h in url_low for h in _READER_HINTS):
        return "html_reader"
    return "direct_pdf"


def _should_run_pdf_manifest_discovery(
    canonical: str,
    html: str,
    doi: str | None,
) -> bool:
    """HTML-only blogs/docs: не гонять PDF/LLM manifest (spatial annotate из HTML)."""
    if doi:
        return True
    if is_academic_open_host(canonical):
        return True
    low = (canonical or "").lower()
    if low.endswith(".pdf") or "/doi/" in low or "/epdf/" in low:
        return True
    path = low.split("?", 1)[0]
    if "/pdf/" in path:
        return True
    return False


def _cache_key(url: str) -> str:
    u = (url or "").strip().split("#", 1)[0].rstrip("/").lower()
    return hashlib.sha256(u.encode("utf-8")).hexdigest()[:32]


def _cache_key_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    return hashlib.sha256(f"doi:{d}".encode("utf-8")).hexdigest()[:32]


def _normalize_page_key(url: str) -> str:
    return (url or "").split("#", 1)[0].rstrip("/").lower()


def get_cached_manifest(url: str) -> ArticleResourceManifest | None:
    cached = _MANIFEST_CACHE.get(_cache_key(url))
    if cached is not None and _manifest_usable(cached):
        return cached
    from knowledge_engine.src.fetcher.academic import extract_doi

    doi = extract_doi(url)
    if doi:
        by_doi = _MANIFEST_CACHE.get(_cache_key_doi(doi))
        if by_doi is not None and _manifest_usable(by_doi):
            return by_doi
    return None


def _manifest_usable(manifest: ArticleResourceManifest) -> bool:
    from knowledge_engine.services.parsers.pdf_bytes import is_parseable_pdf

    if is_parseable_pdf(manifest.fetched_pdf_bytes):
        return True
    return bool(manifest.pdf_candidates)


def store_manifest(manifest: ArticleResourceManifest) -> None:
    _MANIFEST_CACHE[_cache_key(manifest.canonical_url)] = manifest
    if manifest.doi:
        _MANIFEST_CACHE[_cache_key_doi(manifest.doi)] = manifest


def discover_article_resources(
    url: str,
    source_id: str = "",
    html_content: str | None = None,
    *,
    use_cache: bool = True,
) -> ArticleResourceManifest:
    cached = get_cached_manifest(url) if use_cache else None
    if cached is not None:
        return cached
    manifest = ArticleResourceDiscoverer().discover(
        url,
        source_id=source_id,
        html_content=html_content,
    )
    if _manifest_usable(manifest) or manifest.fetched_pdf_bytes:
        store_manifest(manifest)
    return manifest


class ArticleResourceDiscoverer:
    def discover(
        self,
        url: str,
        source_id: str = "",
        html_content: str | None = None,
    ) -> ArticleResourceManifest:
        input_url = (url or "").strip()
        canonical = prefer_acm_pdf_endpoint(input_url)
        from knowledge_engine.services.parsers.pdf_bytes import is_acm_doi_pdf_url
        from knowledge_engine.src.fetcher.academic import extract_doi

        doi = extract_doi(canonical) or extract_doi(input_url)
        pdf_canonical = is_acm_doi_pdf_url(canonical)
        is_epdf_input = "/doi/epdf/" in input_url.lower()
        candidates: list[PDFCandidate] = []
        seen: set[str] = set()

        def add(
            raw_url: str,
            kind: Literal["direct_pdf", "html_reader"],
            source_type: Literal[
                "meta_tag",
                "dom_anchor",
                "unpaywall",
                "scihub",
                "llm_validated",
            ],
            priority: int,
        ) -> None:
            u = (raw_url or "").strip()
            if not u.startswith("http"):
                return
            key = u.split("#", 1)[0].rstrip("/").lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                PDFCandidate(
                    url=u,
                    kind=kind,
                    source_type=source_type,
                    priority=priority,
                )
            )

        html = (html_content or "").strip()
        base_url = canonical
        if not html:
            if is_epdf_input:
                html, method = smart_fetch_page_html(input_url)
                base_url = input_url
                trace(
                    f"RESOURCE_DISCOVERY epdf html ✓ | via={method} | "
                    f"{input_url[:60]}"
                )
            elif not pdf_canonical:
                html, method = smart_fetch_page_html(canonical)
                trace(f"RESOURCE_DISCOVERY html ✓ | via={method} | {canonical[:60]}")
            else:
                trace(f"RESOURCE_DISCOVERY pdf-first ⊘ html | {canonical[:60]}")
        elif pdf_canonical and not is_epdf_input:
            trace(f"RESOURCE_DISCOVERY pdf-first ⊘ html | {canonical[:60]}")

        if doi and len(html) < 400 and not pdf_canonical:
            land_url, land_html = resolve_doi_landing(doi)
            if land_url:
                base_url = land_url
            if land_html and len(land_html) > len(html):
                html = land_html
                trace(f"RESOURCE_DISCOVERY doi landing ✓ | {land_url[:60]}")

        manifest = ArticleResourceManifest(
            source_id=(source_id or "").strip(),
            canonical_url=canonical,
            doi=doi,
            html_snapshot=html or None,
        )

        if html and not _should_run_pdf_manifest_discovery(canonical, html, doi):
            trace(f"RESOURCE_DISCOVERY html-only ⊘ pdf hunt | {canonical[:60]}")
            return manifest

        self._from_canonical_url(canonical, add)
        if base_url != canonical:
            self._from_canonical_url(base_url, add)

        if html:
            self._from_meta(html, base_url, add)
            self._from_dom(html, base_url, add)
            from knowledge_engine.services.parsers.pdf_link_discovery import (
                extract_pdf_urls_from_html,
            )

            for pdf_u in extract_pdf_urls_from_html(html, base_url):
                add(pdf_u, "direct_pdf", "dom_anchor", 1)
            if doi is None:
                soup = BeautifulSoup(html, "html.parser")
                meta_doi = soup.find("meta", attrs={"name": "citation_doi"})
                doi_content = (
                    coerce_html_attr(meta_doi.get("content")) if meta_doi else ""
                )
                if doi_content:
                    manifest.doi = extract_doi(doi_content) or doi

        self._recursive_pdf_discovery(
            input_url,
            canonical,
            add,
            skip_page_keys=(
                {_normalize_page_key(input_url)} if is_epdf_input and html else set()
            ),
        )

        if manifest.doi:
            self._from_unpaywall(manifest.doi, add)

        has_high = any(c.priority <= 2 for c in candidates)
        if html and not has_high:
            llm = LLMShortlinkResolver().resolve(base_url, html)
            if llm and llm.best_pdf_url:
                pri = 2 if llm.kind == "direct_pdf" else 3
                add(llm.best_pdf_url, llm.kind, "llm_validated", pri)

        if manifest.doi and not any(c.priority <= 3 for c in candidates):
            self._from_scihub(manifest.doi, canonical, add)

        candidates.sort(key=lambda c: (c.priority, c.url))
        manifest.pdf_candidates = candidates
        trace(
            f"RESOURCE_DISCOVERY manifest ✓ | candidates={len(candidates)} "
            f"doi={manifest.doi or '-'} | {canonical[:55]}"
        )
        return manifest

    def _recursive_pdf_discovery(
        self,
        input_url: str,
        canonical: str,
        add: Callable[..., None],
        *,
        skip_page_keys: set[str] | None = None,
    ) -> None:
        from knowledge_engine.services.parsers.pdf_bytes import is_acm_doi_pdf_url
        from knowledge_engine.services.parsers.pdf_link_discovery import (
            discover_pdf_urls_recursive,
        )

        seeds: list[str] = []
        low_in = (input_url or "").lower()
        if "/doi/epdf/" in low_in:
            seeds.append(input_url)
        if is_acm_doi_pdf_url(canonical):
            epdf = re.sub(
                r"/doi/pdf/",
                "/doi/epdf/",
                canonical.split("?", 1)[0],
                count=1,
                flags=re.I,
            )
            if epdf not in seeds:
                seeds.append(epdf)
        for c in discover_pdf_urls_recursive(
            seeds,
            max_depth=2,
            max_pages=6,
            skip_page_keys=skip_page_keys,
        ):
            add(c, "direct_pdf", "dom_anchor", 1)

    def _from_canonical_url(
        self,
        canonical: str,
        add: Callable[..., None],
    ) -> None:
        """URL сам является reader или direct PDF (без парсинга HTML)."""
        low = (canonical or "").lower().split("?", 1)[0]
        if not low.startswith("http"):
            return
        if low.endswith(".pdf"):
            add(canonical, "direct_pdf", "dom_anchor", 2)
            return
        if "/doi/pdf/" in low and "/epdf/" not in low:
            add(canonical, "direct_pdf", "dom_anchor", 1)
            pdf_base = canonical.split("?", 1)[0].rstrip("/")
            add(f"{pdf_base}?download=true", "direct_pdf", "dom_anchor", 1)
            return
        if "/doi/epdf/" in low:
            pdf_u = re.sub(r"/doi/epdf/", "/doi/pdf/", canonical, count=1, flags=re.I)
            pdf_base = pdf_u.split("?", 1)[0]
            add(pdf_base, "direct_pdf", "dom_anchor", 1)
            add(f"{pdf_base}?download=true", "direct_pdf", "dom_anchor", 1)
            add(canonical, "html_reader", "dom_anchor", 3)
            return
        if any(h in low for h in _READER_HINTS):
            add(canonical, "html_reader", "dom_anchor", 2)
            return
        if any(h in low for h in _DIRECT_HINTS):
            add(canonical, "direct_pdf", "dom_anchor", 2)

    def _from_meta(
        self,
        html: str,
        base: str,
        add: Callable[..., None],
    ) -> None:
        soup = BeautifulSoup(html, "html.parser")
        meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        content = coerce_html_attr(meta.get("content")) if meta else ""
        if content:
            from knowledge_engine.services.parsers.pdf_bytes import acm_epdf_to_pdf_url

            pdf_u = acm_epdf_to_pdf_url(urljoin(base, content))
            add(pdf_u, "direct_pdf", "meta_tag", 1)

    def _from_dom(self, html: str, base: str, add: Callable[..., None]) -> None:
        from knowledge_engine.services.parsers.pdf_bytes import is_acm_doi_pdf_url

        skip_epdf = is_acm_doi_pdf_url(base)
        soup = BeautifulSoup(html, "html.parser")
        low_base = base.lower()
        if "arxiv.org/abs/" in low_base:
            pdf = re.sub(r"/abs/", "/pdf/", base, count=1)
            if not pdf.lower().endswith(".pdf"):
                pdf = pdf.rstrip("/") + ".pdf"
            add(pdf, "direct_pdf", "dom_anchor", 2)

        for el in soup.find_all("a", href=True):
            href = coerce_html_attr(el.get("href"))
            if (
                not href
                or href.startswith("#")
                or href.lower().startswith("javascript:")
            ):
                continue
            full = urljoin(base, href)
            low = full.lower()
            if skip_epdf and "/epdf/" in low:
                pdf_u = acm_epdf_to_pdf_url(full)
                add(pdf_u, "direct_pdf", "dom_anchor", 1)
                continue
            score = _score_link_element(el, low)
            if score >= _MIN_DOM_SCORE:
                kind = _kind_from_dom_anchor(el, low)
                if kind == "html_reader" and skip_epdf:
                    pdf_u = acm_epdf_to_pdf_url(full)
                    if "/doi/pdf/" in pdf_u.lower():
                        add(pdf_u, "direct_pdf", "dom_anchor", 1)
                        continue
                priority = 2 if score >= 4 else 3
                add(full, kind, "dom_anchor", priority)
                continue
            if any(h in low for h in _READER_HINTS):
                if skip_epdf:
                    continue
                add(full, "html_reader", "dom_anchor", 4)
                continue
            if any(h in low for h in _DIRECT_HINTS) or low.endswith(".pdf"):
                add(full, "direct_pdf", "dom_anchor", 2)

        for el in soup.find_all("button"):
            href = coerce_html_attr(el.get("href") or el.get("data-href"))
            if not href or href.startswith("#"):
                continue
            full = urljoin(base, href)
            low = full.lower()
            score = _score_link_element(el, low)
            if score >= _MIN_DOM_SCORE:
                kind = _kind_from_dom_anchor(el, low)
                priority = 2 if score >= 4 else 3
                add(full, kind, "dom_anchor", priority)

        for tag, attr in (("iframe", "src"), ("embed", "src")):
            for el in soup.find_all(tag):
                href = coerce_html_attr(el.get(attr))
                if not href or href.startswith("#"):
                    continue
                full = urljoin(base, href)
                low = full.lower()
                if any(h in low for h in _READER_HINTS):
                    add(full, "html_reader", "dom_anchor", 3)
                    continue
                if any(h in low for h in _DIRECT_HINTS) or low.endswith(".pdf"):
                    add(full, "direct_pdf", "dom_anchor", 2)

    def _from_unpaywall(self, doi: str, add: Callable[..., None]) -> None:
        from knowledge_engine.src.fetcher.academic import _unpaywall_pdf_url

        pdf_url = _unpaywall_pdf_url(doi)
        if not pdf_url or pdf_url == "__bytes__":
            return
        low = pdf_url.lower()
        if any(h in low for h in _DIRECT_HINTS) or low.endswith(".pdf"):
            kind: Literal["direct_pdf", "html_reader"] = "direct_pdf"
        elif any(h in low for h in _READER_HINTS):
            kind = "html_reader"
        else:
            kind = "html_reader"
        add(pdf_url, kind, "unpaywall", 4)

    def _from_scihub(self, doi: str, page_url: str, add: Callable[..., None]) -> None:
        from knowledge_engine.src.fetcher.academic import _scihub_pdf_url
        from knowledge_engine.src.fetcher.context import fast_academic_fetch_enabled

        if fast_academic_fetch_enabled():
            return
        pdf_url = _scihub_pdf_url(doi or page_url)
        if pdf_url:
            add(pdf_url, "direct_pdf", "scihub", 5)
