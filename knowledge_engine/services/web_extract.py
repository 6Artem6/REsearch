"""Гибридный fetch: httpx (быстро) → Playwright при 403/мало текста."""

from __future__ import annotations

import re

import httpx

from knowledge_engine.config import GEMINI_BROWSER_HEADLESS
from knowledge_engine.services.search.browser_search import fetch_page_html
from knowledge_engine.ui.run_log import trace

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MIN_TEXT_CHARS = 300
_HTTP_TIMEOUT = 10.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _clean_html(html: str, max_chars: int = 12000) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def smart_fetch_page_html(url: str) -> tuple[str, str]:
    """Возвращает (raw_html, метод: httpx | playwright | failed)."""
    try:
        with httpx.Client(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "ru,en;q=0.9"},
        ) as client:
            resp = client.get(url)
            if resp.status_code in (403, 404):
                trace(f"WEB httpx ✗ {url[:60]} | status={resp.status_code}")
            else:
                resp.raise_for_status()
                html = resp.text
                text = _clean_html(html)
                if len(text) >= _MIN_TEXT_CHARS:
                    trace(f"WEB httpx html ✓ {url[:60]} | {len(html)} bytes")
                    return html, "httpx"
                trace(
                    f"WEB httpx thin {len(text)} sym < {_MIN_TEXT_CHARS} → Playwright | {url[:50]}"
                )
    except httpx.HTTPError as exc:
        trace(f"WEB httpx ✗ {url[:60]} | {type(exc).__name__}")

    try:
        html = fetch_page_html(url, headless=GEMINI_BROWSER_HEADLESS)
        trace(f"WEB playwright html ✓ {url[:60]} | {len(html)} bytes")
        return html, "playwright"
    except Exception as exc:
        trace(f"WEB playwright ✗ {url[:60]} | {exc}")
        return "", "failed"


def smart_fetch_page_text(url: str) -> tuple[str, str]:
    """
    Сначала httpx; при ошибке или text < 300 символов — Playwright.
    Возвращает (очищенный текст, метод: httpx | playwright).
    """
    httpx_failed = False
    try:
        with httpx.Client(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "ru,en;q=0.9"},
        ) as client:
            resp = client.get(url)
            if resp.status_code in (403, 404):
                httpx_failed = True
                trace(f"WEB httpx ✗ {url[:60]} | status={resp.status_code}")
            else:
                resp.raise_for_status()
                text = _clean_html(resp.text)
                if len(text) >= _MIN_TEXT_CHARS:
                    trace(f"WEB httpx ✓ {url[:60]} | {len(text)} sym")
                    return text, "httpx"
                trace(
                    f"WEB httpx thin {len(text)} sym < {_MIN_TEXT_CHARS} → Playwright | {url[:50]}"
                )
    except httpx.HTTPError as exc:
        httpx_failed = True
        trace(f"WEB httpx ✗ {url[:60]} | {type(exc).__name__}")

    try:
        html = fetch_page_html(url, headless=GEMINI_BROWSER_HEADLESS)
        text = _clean_html(html)
        trace(
            f"WEB playwright ✓ {url[:60]} | {len(text)} sym (httpx_failed={httpx_failed})"
        )
        return text, "playwright"
    except Exception as exc:
        trace(f"WEB playwright ✗ {url[:60]} | {exc}")
        return "", "failed"
