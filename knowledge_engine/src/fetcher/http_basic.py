"""Standard HTTP fetch: arXiv, domain masks, trafilatura."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from knowledge_engine.src.fetcher.academic import is_challenge_or_empty
from knowledge_engine.src.state import ScrapedDocument, SourceType

_USER_AGENT = "KnowledgeEngine/0.7 (+local research)"
_TIMEOUT = httpx.Timeout(25.0, connect=10.0)

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([\d]{4}\.[\d]{4,5}(?:v\d+)?)",
    re.I,
)
_AR5IV_HOST = "ar5iv.org"

_DOMAIN_SELECTORS: dict[str, tuple[str, ...]] = {
    "github.com": ("article", "main", ".markdown-body", "[data-testid='article-body']"),
    "habr.com": ("article", ".article-body", ".post__text"),
    "stackoverflow.com": ("article", "#mainbar", ".s-prose"),
    "medium.com": ("article", "main", ".postArticle-content"),
    "dev.to": ("article", ".crayons-article__main", "main"),
}


def _doc_id(url: str, prefix: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _normalize_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _arxiv_id(url: str) -> str | None:
    m = _ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def _fetch_bytes(url: str) -> tuple[str, str]:
    with httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
        return resp.text, ctype


def _html_to_markdown_simple(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
        text = el.get_text("\n", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            level = int(el.name[1])
            lines.append("#" * level + " " + text)
        elif el.name == "pre" or el.name == "code":
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text)
    body = "\n\n".join(lines).strip()
    return body[:120_000]


def _extract_with_selectors(html: str, selectors: tuple[str, ...]) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in selectors:
        if sel.startswith(".") or sel.startswith("#") or sel.startswith("["):
            nodes = soup.select(sel)
        else:
            nodes = soup.find_all(sel)
        for node in nodes:
            text = node.get_text("\n", strip=True)
            if len(text) > 200:
                return text[:120_000]
    return ""


def _trafilatura_extract(html: str) -> str:
    try:
        import trafilatura

        out = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            output_format="markdown",
        )
        if out and len(out.strip()) > 80:
            return out.strip()[:120_000]
    except Exception:
        pass
    return _html_to_markdown_simple(html)


def fetch_arxiv_document(url: str) -> ScrapedDocument | None:
    aid = _arxiv_id(url)
    if not aid:
        return None
    ar5iv_url = f"https://{_AR5IV_HOST}/html/{aid}"
    try:
        html, _ = _fetch_bytes(ar5iv_url)
    except httpx.HTTPError:
        export_url = f"https://export.arxiv.org/abs/{aid}"
        try:
            html, _ = _fetch_bytes(export_url)
        except httpx.HTTPError:
            return None
    text = _html_to_markdown_simple(html)
    if len(text) < 120:
        text = _trafilatura_extract(html)
    if is_challenge_or_empty(text, 80):
        return None
    return ScrapedDocument(
        doc_id=_doc_id(url, "arxiv"),
        source_url=url,
        source_type="arxiv_html5",
        raw_markdown=text,
        cosine_dedup_passed=False,
    )


def fetch_dom_masked_document(url: str) -> ScrapedDocument | None:
    try:
        html, _ = _fetch_bytes(url)
    except httpx.HTTPError:
        return None

    host = _normalize_host(url)
    matched_selectors: tuple[str, ...] = ()
    for domain, selectors in _DOMAIN_SELECTORS.items():
        if host == domain or host.endswith("." + domain):
            matched_selectors = selectors
            break

    text = ""
    source_type: SourceType = "trafilatura"
    if matched_selectors:
        text = _extract_with_selectors(html, matched_selectors)
        if text:
            source_type = "github_dom" if "github" in host else "dom_mask"

    if len(text) < 120:
        text = _trafilatura_extract(html)
        source_type = "trafilatura"

    if is_challenge_or_empty(text, 80):
        return None

    return ScrapedDocument(
        doc_id=_doc_id(url, source_type.split("_")[0]),
        source_url=url,
        source_type=source_type,
        raw_markdown=text,
        cosine_dedup_passed=False,
    )


def fetch_http_document(url: str) -> ScrapedDocument | None:
    """Tier 3: arXiv HTML5 → domain masks → trafilatura."""
    url = (url or "").strip()
    if not url.startswith("http"):
        return None
    if "arxiv.org" in url.lower() or _arxiv_id(url):
        doc = fetch_arxiv_document(url)
        if doc is not None:
            return doc
    return fetch_dom_masked_document(url)
