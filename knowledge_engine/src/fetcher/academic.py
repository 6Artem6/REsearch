"""Academic cascade: Unpaywall → Sci-Hub mirrors → HTTP fallback."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from knowledge_engine.config import UNPAYWALL_EMAIL
from knowledge_engine.src.fetcher.cleaner import (
    CleanedDocument,
    clean_pdf_bytes,
    clean_text_document,
)
from knowledge_engine.ui.run_log import trace

_USER_AGENT = "KnowledgeEngine/0.7 AcademicCascade (+local research)"
_TIMEOUT = httpx.Timeout(45.0, connect=12.0)

_DOI_URL_RE = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", re.I)
_DOI_RAW_RE = re.compile(r"(10\.\d{4,9}/[^\s]+)", re.I)
_ARXIV_RE = re.compile(r"arxiv\.org", re.I)

_SCIHUB_MIRRORS = (
    "https://sci-hub.se",
    "https://sci-hub.ru",
    "https://sci-hub.st",
)

_CHALLENGE_MARKERS = (
    "aguarde, estamos validando",
    "just a moment",
    "cloudflare",
    "cf-browser-verification",
    "checking your browser",
    "access denied",
    "captcha",
    "enable javascript",
)


def extract_doi(url: str) -> str | None:
    url = (url or "").strip()
    m = _DOI_URL_RE.search(url)
    if m:
        return m.group(1).rstrip("/.")
    m = _DOI_RAW_RE.search(url)
    if m:
        doi = m.group(1).rstrip("/.")
        return doi.split("?")[0]
    return None


def is_academic_url(url: str) -> bool:
    low = (url or "").lower()
    if "doi.org" in low or extract_doi(url):
        return True
    if _ARXIV_RE.search(low):
        return True
    if any(
        x in low
        for x in (
            "unicamp.br",
            "usp.br",
            "springer.com",
            "ieee.org",
            "acm.org",
            "sciencedirect.com",
            "wiley.com",
        )
    ):
        return True
    return False


def is_challenge_or_empty(text: str, min_len: int = 120) -> bool:
    t = (text or "").strip()
    if len(t) < min_len:
        return True
    low = t.lower()
    return any(marker in low for marker in _CHALLENGE_MARKERS)


def _http_get_text(url: str) -> tuple[str, str, bytes | None]:
    with httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return "", ctype, resp.content
        return resp.text, ctype, None


def _http_get_bytes(url: str) -> bytes:
    with httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _unpaywall_pdf_url(doi: str) -> str | None:
    api = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    trace(f"ACADEMIC tier1 ▶ Unpaywall | doi={doi}")
    try:
        with httpx.Client(
            timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        ) as client:
            resp = client.get(api)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        trace(f"ACADEMIC tier1 ✗ Unpaywall | {exc}")
        return None

    if not data.get("is_oa"):
        trace("ACADEMIC tier1 ⊘ Unpaywall non-OA")
        return None

    best = data.get("best_oa_location") or {}
    pdf_url = (best.get("url_for_pdf") or "").strip()
    if pdf_url:
        trace(f"ACADEMIC tier1 ✓ Unpaywall PDF | {pdf_url[:90]}")
        return pdf_url

    landing = (best.get("url") or "").strip()
    if landing:
        trace(f"ACADEMIC tier1 ✓ Unpaywall landing | {landing[:90]}")
        return landing
    return None


def _scihub_pdf_url(target: str) -> str | None:
    for mirror in _SCIHUB_MIRRORS:
        url = f"{mirror}/{target}"
        trace(f"ACADEMIC tier2 ▶ Sci-Hub | {mirror}")
        try:
            html, _, pdf_bytes = _http_get_text(url)
            if pdf_bytes:
                trace("ACADEMIC tier2 ✓ Sci-Hub direct PDF bytes")
                return "__bytes__"
            soup = BeautifulSoup(html, "html.parser")
            iframe = soup.find("iframe", id="pdf")
            if iframe and iframe.get("src"):
                src = str(iframe["src"]).strip()
                if src.startswith("//"):
                    src = "https:" + src
                trace(f"ACADEMIC tier2 ✓ Sci-Hub iframe | {src[:90]}")
                return src
            embed = soup.find("embed", attrs={"type": "application/pdf"})
            if embed and embed.get("src"):
                src = str(embed["src"]).strip()
                if src.startswith("//"):
                    src = "https:" + src
                trace(f"ACADEMIC tier2 ✓ Sci-Hub embed | {src[:90]}")
                return src
        except Exception as exc:
            trace(f"ACADEMIC tier2 ✗ {mirror} | {exc}")
            continue
    return None


def _fetch_scihub_pdf_bytes(target: str) -> bytes | None:
    for mirror in _SCIHUB_MIRRORS:
        url = f"{mirror}/{target}"
        try:
            html, _, direct = _http_get_text(url)
            if direct:
                return direct
            soup = BeautifulSoup(html, "html.parser")
            iframe = soup.find("iframe", id="pdf")
            src = iframe.get("src") if iframe else None
            if not src:
                embed = soup.find("embed", attrs={"type": "application/pdf"})
                src = embed.get("src") if embed else None
            if not src:
                continue
            src = str(src).strip()
            if src.startswith("//"):
                src = "https:" + src
            return _http_get_bytes(src)
        except Exception:
            continue
    return None


def resolve_academic_document(url: str) -> CleanedDocument | None:
    """
      Academic cascade for doi / arxiv / challenge-heavy publisher pages.
    Returns CleanedDocument or None.
    """
    url = unquote((url or "").strip())
    if not url.startswith("http"):
        return None

    doi = extract_doi(url)
    title_hint = urlparse(url).path.rsplit("/", 1)[-1]

    # Tier 1: Unpaywall
    if doi:
        pdf_url = _unpaywall_pdf_url(doi)
        if pdf_url and pdf_url != "__bytes__":
            try:
                if pdf_url.lower().endswith(".pdf") or "pdf" in pdf_url.lower():
                    pdf_bytes = _http_get_bytes(pdf_url)
                    cleaned = clean_pdf_bytes(
                        pdf_bytes,
                        source_url=url,
                        title=title_hint,
                    )
                    if cleaned and len(cleaned.clean_text) >= 80:
                        trace(
                            f"ACADEMIC ✓ PDF via Unpaywall | "
                            f"{len(cleaned.clean_text)} chars"
                        )
                        return cleaned
                html, _, embedded = _http_get_text(pdf_url)
                if embedded:
                    cleaned = clean_pdf_bytes(
                        embedded, source_url=url, title=title_hint
                    )
                    if cleaned and len(cleaned.clean_text) >= 80:
                        return cleaned
                if html and not is_challenge_or_empty(html, 200):
                    cleaned = clean_text_document(
                        html, source_url=url, title=title_hint, is_pdf=False
                    )
                    if cleaned and len(cleaned.clean_text) >= 80:
                        trace("ACADEMIC ✓ HTML via Unpaywall landing")
                        return cleaned
            except Exception as exc:
                trace(f"ACADEMIC tier1 fetch ✗ | {exc}")

    # Tier 2: Sci-Hub
    scihub_target = doi or url
    pdf_bytes = _fetch_scihub_pdf_bytes(scihub_target)
    if pdf_bytes:
        cleaned = clean_pdf_bytes(pdf_bytes, source_url=url, title=title_hint)
        if cleaned and len(cleaned.clean_text) >= 80:
            trace(f"ACADEMIC ✓ PDF via Sci-Hub | {len(cleaned.clean_text)} chars")
            return cleaned

    trace("ACADEMIC tier3 ⊘ cascade incomplete — caller may use HTTP/trafilatura")
    return None
