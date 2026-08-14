"""Скачивание PDF по ArticleResourceManifest (httpx + Playwright)."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from knowledge_engine.config import GEMINI_BROWSER_HEADLESS
from knowledge_engine.services.parsers.article_manifest import (
    ArticleResourceManifest,
    PDFCandidate,
)
from knowledge_engine.services.parsers.pdf_bytes import (
    doi_pdf_url_variants,
    is_acm_doi_pdf_url,
    is_parseable_pdf,
    prefer_acm_pdf_endpoint,
)
from knowledge_engine.services.search.browser_search import persistent_browser
from knowledge_engine.services.web_extract import fetch_url_bytes
from knowledge_engine.ui.run_log import trace

_PDF_URL_RE = re.compile(r"https?://[^\s\"'<>]+\.pdf[^\s\"'<>]*", re.I)


def _accept_pdf_bytes(data: bytes | None, label: str) -> bytes | None:
    if is_parseable_pdf(data):
        return data
    if data and data[:5] == b"%PDF-":
        trace(f"RESOURCE_FETCH ⊘ | corrupt pdf (0 pages?) | {label}")
    return None


def boost_manifest_pdf_candidates(manifest: ArticleResourceManifest) -> None:
    """ACM: /doi/pdf/ в очереди; epdf не добавляем если источник уже pdf."""
    if is_acm_doi_pdf_url(manifest.canonical_url):
        return
    seen = {c.url.split("#", 1)[0].rstrip("/").lower() for c in manifest.pdf_candidates}
    for u in doi_pdf_url_variants(manifest.canonical_url):
        key = u.split("#", 1)[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        low = key
        if "/doi/pdf/" in low or low.endswith(".pdf"):
            manifest.pdf_candidates.append(
                PDFCandidate(
                    url=u,
                    kind="direct_pdf",
                    source_type="dom_anchor",
                    priority=1,
                )
            )
        elif "/doi/epdf/" in low:
            manifest.pdf_candidates.append(
                PDFCandidate(
                    url=u,
                    kind="html_reader",
                    source_type="dom_anchor",
                    priority=4,
                )
            )


def _candidate_order(
    cand: PDFCandidate,
    manifest: ArticleResourceManifest,
) -> tuple[int, int, str]:
    pri = cand.priority
    pdf_bias = 0
    if is_acm_doi_pdf_url(cand.url):
        pdf_bias = 0
    elif "/doi/epdf/" in cand.url.lower():
        pdf_bias = 1
    else:
        pdf_bias = 0
    if is_acm_doi_pdf_url(manifest.canonical_url) and "/epdf/" in cand.url.lower():
        pri += 10
    return (pri, pdf_bias, cand.url)


def _fetch_direct_pdf(url: str) -> bytes | None:
    url = prefer_acm_pdf_endpoint(url) if "/doi/" in (url or "").lower() else url
    want_pdf = is_acm_doi_pdf_url(url)
    data = fetch_url_bytes(url)
    accepted = _accept_pdf_bytes(data, f"httpx {url[:55]}")
    if accepted:
        trace(f"RESOURCE_FETCH direct ✓ httpx | {url[:70]}")
        return accepted
    try:
        with persistent_browser(headless=GEMINI_BROWSER_HEADLESS) as (_, context):
            try:
                req = context.request.get(url, timeout=90000)
                if req.ok:
                    accepted = _accept_pdf_bytes(
                        req.body(), f"playwright-request {url[:45]}"
                    )
                    if accepted:
                        trace(
                            f"RESOURCE_FETCH direct ✓ playwright-request | {url[:70]}"
                        )
                        return accepted
            except Exception as exc:
                trace(f"RESOURCE_FETCH playwright-request ⊘ | {url[:45]} | {exc}")

            page = context.new_page()
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=90000)
                if resp and want_pdf and "/epdf/" in (page.url or "").lower():
                    trace(
                        f"RESOURCE_FETCH ⊘ | ACM redirected pdf→epdf | "
                        f"retry {url[:60]}"
                    )
                    resp = page.goto(url, wait_until="networkidle", timeout=90000)
                if resp:
                    accepted = _accept_pdf_bytes(resp.body(), f"playwright {url[:50]}")
                    if accepted:
                        trace(f"RESOURCE_FETCH direct ✓ playwright | {url[:70]}")
                        return accepted
                page.wait_for_timeout(2000)
                if want_pdf:
                    accepted = _accept_pdf_bytes(
                        _httpx_pdf_no_epdf_redirect(url),
                        f"httpx-noredirect {url[:45]}",
                    )
                    if accepted:
                        return accepted
            except Exception as exc:
                trace(f"RESOURCE_FETCH direct ⊘ playwright | {url[:50]} | {exc}")
            finally:
                page.close()
    except Exception as exc:
        trace(f"RESOURCE_FETCH direct ⊘ | {url[:50]} | {exc}")
    return None


def _httpx_pdf_no_epdf_redirect(url: str) -> bytes | None:
    """GET /doi/pdf/ без следования редиректу на epdf (если сервер отдаёт PDF)."""
    import httpx

    from knowledge_engine.services.web_extract import _USER_AGENT

    try:
        with httpx.Client(
            timeout=45.0,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = client.get(url)
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = (resp.headers.get("location") or "").lower()
                if "/epdf/" in loc:
                    trace(f"RESOURCE_FETCH ⊘ | pdf redirect to epdf | {loc[:70]}")
                    return None
            if resp.status_code < 400:
                return resp.content
    except Exception:
        return None
    return None


def _fetch_html_reader_pdf(reader_url: str) -> bytes | None:
    for alt in doi_pdf_url_variants(reader_url):
        if "/doi/pdf/" in alt.lower() and "/epdf/" not in alt.lower():
            data = _fetch_direct_pdf(alt)
            if data:
                return data

    trace(f"RESOURCE_FETCH reader ▶ | {reader_url[:75]}")
    with persistent_browser(headless=GEMINI_BROWSER_HEADLESS) as (_, context):
        page = context.new_page()
        best: bytes | None = None
        try:

            def on_response(response) -> None:
                nonlocal best
                try:
                    ct = (response.headers.get("content-type") or "").lower()
                    rurl = (response.url or "").lower()
                    if (
                        "pdf" not in ct
                        and not rurl.endswith(".pdf")
                        and "/pdf/" not in rurl
                        and "/epdf/" not in rurl
                    ):
                        return
                    body = response.body()
                    accepted = _accept_pdf_bytes(
                        body,
                        f"intercept {response.url[:50]}",
                    )
                    if accepted and (best is None or len(accepted) > len(best)):
                        best = accepted
                        trace(
                            f"RESOURCE_FETCH reader ✓ intercept | "
                            f"{response.url[:70]} | {len(accepted)}"
                        )
                except Exception:
                    pass

            page.on("response", on_response)
            page.goto(reader_url, wait_until="domcontentloaded", timeout=90000)
            try:
                page.wait_for_load_state("networkidle", timeout=45000)
            except Exception:
                pass
            page.wait_for_timeout(4000)
            if best:
                return best

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            from knowledge_engine.services.parsers.pdf_link_discovery import (
                extract_pdf_urls_from_html,
            )

            for pdf_u in extract_pdf_urls_from_html(html, reader_url):
                data = _fetch_direct_pdf(pdf_u)
                if data:
                    return data
            for iframe in soup.find_all("iframe", src=True):
                src = urljoin(reader_url, iframe.get("src") or "")
                if ".pdf" in src.lower() or "pdf" in src.lower():
                    data = _fetch_direct_pdf(src)
                    if data:
                        return data
            for m in _PDF_URL_RE.findall(html):
                data = _fetch_direct_pdf(m)
                if data:
                    return data
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and (meta.get("content") or "").strip():
                data = _fetch_direct_pdf(
                    urljoin(reader_url, meta.get("content").strip())
                )
                if data:
                    return data
        except Exception as exc:
            trace(f"RESOURCE_FETCH reader ✗ | {reader_url[:50]} | {exc}")
        finally:
            page.close()
    return None


def fetch_pdf_from_manifest(
    manifest: ArticleResourceManifest,
) -> bytes | None:
    boost_manifest_pdf_candidates(manifest)
    cached = _accept_pdf_bytes(manifest.fetched_pdf_bytes, "manifest cache")
    if cached:
        return cached
    if manifest.fetched_pdf_bytes:
        manifest.fetched_pdf_bytes = None

    for cand in sorted(
        manifest.pdf_candidates,
        key=lambda c: _candidate_order(c, manifest),
    ):
        data: bytes | None = None
        if cand.kind == "direct_pdf":
            data = _fetch_direct_pdf(cand.url)
        else:
            data = _fetch_html_reader_pdf(cand.url)
        if data:
            manifest.selected_pdf_url = cand.url
            manifest.fetched_pdf_bytes = data
            store_manifest(manifest)
            trace(f"RESOURCE_FETCH ✓ | {len(data)} B pages-valid | " f"{cand.url[:70]}")
            return data

    trace(
        f"RESOURCE_FETCH ⊘ | no parseable pdf | candidates={len(manifest.pdf_candidates)} "
        f"| {manifest.canonical_url[:55]}"
    )
    return None


def store_manifest(manifest: ArticleResourceManifest) -> None:
    from knowledge_engine.services.parsers.article_resource_discoverer import (
        store_manifest as _store,
    )

    _store(manifest)
