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
from knowledge_engine.src.fetcher.context import fast_academic_fetch_enabled
from knowledge_engine.ui.run_log import trace

_USER_AGENT = "KnowledgeEngine/0.7 AcademicCascade (+local research)"
_TIMEOUT = httpx.Timeout(45.0, connect=12.0)
_FAST_TIMEOUT = httpx.Timeout(8.0, connect=3.0)


def _active_http_timeout() -> httpx.Timeout:
    return _FAST_TIMEOUT if fast_academic_fetch_enabled() else _TIMEOUT


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
        timeout=_active_http_timeout(),
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
        timeout=_active_http_timeout(),
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
            timeout=_active_http_timeout(), headers={"User-Agent": _USER_AGENT}
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
    if fast_academic_fetch_enabled():
        trace("ACADEMIC tier2 ⊘ Sci-Hub skipped (fast academic / node init)")
        return None
    from knowledge_engine.config import ACADEMIC_SCIHUB_TIMEOUT_SEC

    scihub_timeout = httpx.Timeout(
        max(0.5, ACADEMIC_SCIHUB_TIMEOUT_SEC),
        connect=min(1.0, ACADEMIC_SCIHUB_TIMEOUT_SEC),
    )
    for mirror in _SCIHUB_MIRRORS:
        url = f"{mirror}/{target}"
        trace(f"ACADEMIC tier2 ▶ Sci-Hub | {mirror}")
        try:
            with httpx.Client(
                timeout=scihub_timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                ctype = (
                    (resp.headers.get("content-type") or "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                if "pdf" in ctype or url.lower().endswith(".pdf"):
                    trace("ACADEMIC tier2 ✓ Sci-Hub direct PDF bytes")
                    return "__bytes__"
                html = resp.text
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


_PUBLISHER_PDF_HOST_SUFFIXES = (
    "acm.org",
    "ieee.org",
    "springer.com",
    "sciencedirect.com",
    "wiley.com",
)


def _is_publisher_pdf_host(url: str) -> bool:
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        return False
    return any(
        host == s or host.endswith(f".{s}") for s in _PUBLISHER_PDF_HOST_SUFFIXES
    )


def _fetch_scihub_pdf_bytes(target: str) -> bytes | None:
    if fast_academic_fetch_enabled():
        trace("ACADEMIC tier2 ⊘ Sci-Hub bytes skipped (fast academic / node init)")
        return None
    from knowledge_engine.config import ACADEMIC_SCIHUB_TIMEOUT_SEC

    scihub_timeout = httpx.Timeout(
        max(0.5, ACADEMIC_SCIHUB_TIMEOUT_SEC),
        connect=min(1.0, ACADEMIC_SCIHUB_TIMEOUT_SEC),
    )
    for mirror in _SCIHUB_MIRRORS:
        url = f"{mirror}/{target}"
        trace(f"ACADEMIC tier2 ▶ Sci-Hub | {mirror} | {target[:80]}")
        try:
            with httpx.Client(
                timeout=scihub_timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                ctype = (
                    (resp.headers.get("content-type") or "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                if "pdf" in ctype:
                    trace("ACADEMIC tier2 ✓ Sci-Hub direct PDF bytes")
                    return resp.content
                html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            iframe = soup.find("iframe", id="pdf")
            src = iframe.get("src") if iframe else None
            if not src:
                embed = soup.find("embed", attrs={"type": "application/pdf"})
                src = embed.get("src") if embed else None
            if not src:
                trace(f"ACADEMIC tier2 ⊘ Sci-Hub no embed | {mirror}")
                continue
            src = str(src).strip()
            if src.startswith("//"):
                src = "https:" + src
            trace(f"ACADEMIC tier2 ✓ Sci-Hub fetch embed | {src[:90]}")
            return _http_get_bytes(src)
        except Exception as exc:
            trace(f"ACADEMIC tier2 ✗ {mirror} | {exc}")
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
    pdf_url: str | None = None

    # Tier 1: Unpaywall
    tier1_publisher_pdf_failed = False
    if doi:
        pdf_url = _unpaywall_pdf_url(doi)
        if pdf_url and pdf_url != "__bytes__":
            skip_publisher_pdf = _is_publisher_pdf_host(pdf_url)
            if skip_publisher_pdf:
                trace(
                    "ACADEMIC tier1 ⊘ skip publisher PDF fetch "
                    f"(Sci-Hub first) | {pdf_url[:90]}"
                )
            elif pdf_url.lower().endswith(".pdf") or "pdf" in pdf_url.lower():
                try:
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
                except Exception as exc:
                    tier1_publisher_pdf_failed = True
                    trace(f"ACADEMIC tier1 fetch ✗ | {exc}")
            if not skip_publisher_pdf and not tier1_publisher_pdf_failed:
                try:
                    if not (
                        pdf_url.lower().endswith(".pdf") or "pdf" in pdf_url.lower()
                    ):
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
                    trace(f"ACADEMIC tier1 landing ✗ | {exc}")

    # Tier 2: Sci-Hub (doi или URL; после 403 на ACM/IEEE PDF)
    if not fast_academic_fetch_enabled():
        scihub_target = doi or url
        if tier1_publisher_pdf_failed or (
            doi and pdf_url and _is_publisher_pdf_host(pdf_url)
        ):
            trace(f"ACADEMIC tier2 ▶ Sci-Hub cascade | target={scihub_target[:80]}")
        pdf_bytes = _fetch_scihub_pdf_bytes(scihub_target)
        if pdf_bytes:
            cleaned = clean_pdf_bytes(pdf_bytes, source_url=url, title=title_hint)
            if cleaned and len(cleaned.clean_text) >= 80:
                trace(f"ACADEMIC ✓ PDF via Sci-Hub | {len(cleaned.clean_text)} chars")
                return cleaned
    else:
        trace("ACADEMIC tier2 ⊘ Sci-Hub cascade skipped (fast academic / node init)")

    trace("ACADEMIC tier3 ⊘ cascade incomplete — caller may use HTTP/trafilatura")
    return None
