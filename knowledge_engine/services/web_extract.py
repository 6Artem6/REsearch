"""Гибридный fetch: httpx → Playwright; PDF resolve для ingest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from knowledge_engine.config import GEMINI_BROWSER_HEADLESS
from knowledge_engine.services.search.browser_search import fetch_page_html
from knowledge_engine.ui.run_log import trace

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MIN_TEXT_CHARS = 300
_HTTP_TIMEOUT = 10.0
_FETCH_BYTES_TIMEOUT = 45.0
_DOI_LANDING_TIMEOUT = 35.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_CHALLENGE_MARKERS = (
    "just a moment",
    "cloudflare",
    "cf-browser-verification",
    "checking your browser",
    "verifying your browser",
    "enable javascript",
    "access denied",
    "captcha",
    "bot detection",
    "please turn javascript on",
    "robot check",
)


def is_anti_bot_html(html: str) -> bool:
    return _looks_like_challenge(html)


def is_anti_bot_fetch_result(
    text: str,
    method: str,
    *,
    html: str | None = None,
    http_status: int | None = None,
) -> bool:
    """True when fetch likely hit anti-bot / access wall (no usable article body)."""
    if http_status in (403, 401, 429, 503):
        return True
    if (method or "").strip().lower() == "failed":
        return True
    if html and is_anti_bot_html(html):
        return True
    t = (text or "").strip()
    if not t and (html or "").strip():
        return is_anti_bot_html(html)
    return False


def _clean_html(html: str, max_chars: int = 12000) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _looks_like_challenge(html: str) -> bool:
    low = (html or "").lower()[:8000]
    return any(m in low for m in _CHALLENGE_MARKERS)


def _httpx_get(url: str, timeout: float = _HTTP_TIMEOUT) -> httpx.Response | None:
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "ru,en;q=0.9"},
        ) as client:
            resp = client.get(url)
            return resp
    except httpx.HTTPError as exc:
        trace(f"WEB httpx ✗ {url[:60]} | {type(exc).__name__}")
        return None


def smart_fetch_page_html(url: str) -> tuple[str, str]:
    """Возвращает (raw_html, метод: httpx | playwright | failed)."""
    resp = _httpx_get(url)
    if resp is not None:
        if resp.status_code in (403, 503):
            trace(f"WEB httpx ✗ {url[:60]} | status={resp.status_code}")
        elif resp.status_code < 400:
            ctype = (
                (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            )
            body = resp.content or b""
            if body[:5] == b"%PDF-" or "pdf" in ctype:
                trace(f"WEB httpx pdf body → skip html | {url[:50]}")
            else:
                html = resp.text
                if _looks_like_challenge(html):
                    trace(f"WEB httpx challenge → Playwright | {url[:50]}")
                else:
                    text = _clean_html(html)
                    if len(text) >= _MIN_TEXT_CHARS:
                        trace(f"WEB httpx html ✓ {url[:60]} | {len(html)} bytes")
                        return html, "httpx"
                    trace(
                        f"WEB httpx thin {len(text)} sym < {_MIN_TEXT_CHARS} → Playwright"
                    )

    try:
        html = fetch_page_html(url, headless=GEMINI_BROWSER_HEADLESS)
        if html and _looks_like_challenge(html):
            trace(f"WEB playwright challenge html | {url[:50]}")
            return "", "failed"
        trace(f"WEB playwright html ✓ {url[:60]} | {len(html)} bytes")
        return html, "playwright"
    except Exception as exc:
        trace(f"WEB playwright ✗ {url[:60]} | {exc}")
        return "", "failed"


def resolve_doi_landing(doi: str) -> tuple[str, str]:
    """
    doi.org → финальный publisher URL + HTML (для epdf-ссылок на landing).
    """
    raw = (doi or "").strip()
    if not raw:
        return "", ""
    if raw.lower().startswith("http"):
        from knowledge_engine.src.fetcher.academic import extract_doi

        extracted = extract_doi(raw)
        if not extracted:
            return "", ""
        raw = extracted
    start = f"https://doi.org/{raw}"
    resp = _httpx_get(start, timeout=_DOI_LANDING_TIMEOUT)
    if resp is not None and resp.text:
        final = str(resp.url).strip()
        body = resp.content or b""
        html = resp.text
        if body[:5] != b"%PDF-" and resp.status_code < 500:
            if not _looks_like_challenge(html) and len(_clean_html(html)) >= 80:
                trace(
                    f"WEB doi landing ✓ httpx | status={resp.status_code} "
                    f"| {final[:70]} | {len(html)} bytes"
                )
                return final, html
    try:
        from knowledge_engine.services.search.browser_search import persistent_browser

        with persistent_browser(headless=GEMINI_BROWSER_HEADLESS) as (_, context):
            page = context.new_page()
            try:
                page.goto(start, wait_until="domcontentloaded", timeout=90000)
                page.wait_for_timeout(2500)
                final = (page.url or start).strip()
                html = page.content()
                if html and not _looks_like_challenge(html):
                    trace(
                        f"WEB doi landing ✓ playwright | {final[:70]} | "
                        f"{len(html)} bytes"
                    )
                    return final, html
            finally:
                page.close()
    except Exception as exc:
        trace(f"WEB doi landing ✗ | {raw[:45]} | {exc}")
    return start, ""


def smart_fetch_page_text(url: str) -> tuple[str, str]:
    html, method = smart_fetch_page_html(url)
    if not html:
        return "", "failed"
    return _clean_html(html), method


def resolve_embedded_pdf_url(html: str, page_url: str) -> str | None:
    """citation_pdf_url, явные PDF-ссылки, arXiv abs→pdf."""
    base = (page_url or "").strip()
    soup = BeautifulSoup(html or "", "html.parser")
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and (meta.get("content") or "").strip():
        return urljoin(base, meta["content"].strip())

    low_base = base.lower()
    if "arxiv.org/abs/" in low_base:
        pdf = re.sub(r"/abs/", "/pdf/", base, count=1)
        if not pdf.lower().endswith(".pdf"):
            pdf = pdf.rstrip("/") + ".pdf"
        return pdf

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        text = (a.get_text(" ", strip=True) or "").lower()
        href_low = href.lower()
        is_pdf = (
            href_low.endswith(".pdf") or ".pdf?" in href_low or "/epdf/" in href_low
        )
        label_hit = any(
            k in text
            for k in (
                "download pdf",
                "full text",
                "pdf",
                "download",
                "view pdf",
            )
        )
        if is_pdf and (label_hit or len(text) < 40):
            return urljoin(base, href)
    return None


def fetch_url_bytes(url: str, timeout: float = _FETCH_BYTES_TIMEOUT) -> bytes | None:
    u = (url or "").strip()
    if not u.startswith("http"):
        return None
    resp = _httpx_get(u, timeout=timeout)
    if resp is None or resp.status_code >= 400:
        return None
    data = resp.content
    return data if len(data) > 100 else None


@dataclass
class FetchedArticleDocument:
    data: bytes
    content_kind: str  # html | pdf
    final_url: str
    fetch_method: str


def smart_fetch_article_document(url: str) -> FetchedArticleDocument | None:
    """
    Pre-Ingest Manifest: discovery + smart PDF fetch (httpx / Playwright).
    Fallback: HTML snapshot из манифеста.
    """
    from knowledge_engine.services.parsers.article_resource_discoverer import (
        discover_article_resources,
    )
    from knowledge_engine.services.parsers.smart_fetcher import fetch_pdf_from_manifest

    u = (url or "").strip()
    if not u.startswith("http"):
        return None
    low = u.lower().split("?", 1)[0]
    if low.endswith(".pdf"):
        data = fetch_url_bytes(u)
        if data and data[:5] == b"%PDF-":
            return FetchedArticleDocument(data, "pdf", u, "httpx")

    manifest = discover_article_resources(u)
    html = (manifest.html_snapshot or "").strip()
    pdf_bytes = manifest.fetched_pdf_bytes
    if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
        pdf_bytes = fetch_pdf_from_manifest(manifest)

    if pdf_bytes and pdf_bytes[:5] == b"%PDF-":
        final = (manifest.selected_pdf_url or u).strip()
        trace(f"WEB manifest pdf ✓ | {final[:90]}")
        return FetchedArticleDocument(
            pdf_bytes,
            "pdf",
            final,
            "resource_manifest",
        )

    if html:
        trace(f"WEB manifest html ✓ | {u[:60]} | {len(html)} bytes")
        return FetchedArticleDocument(
            html.encode("utf-8", errors="replace"),
            "html",
            u,
            "resource_manifest_html",
        )
    return None
