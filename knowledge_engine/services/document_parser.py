"""Structure-aware парсинг HTML → DocumentStructure."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from knowledge_engine.schemas import (
    CodeArtifact,
    DocumentMetaSummary,
    DocumentStructure,
    MediaArtifact,
    TocEntry,
)
from knowledge_engine.ui.run_log import trace

_MEDIA_KEYWORDS = (
    "architecture",
    "benchmark",
    "diagram",
    "chart",
    "graph",
    "schema",
    "flow",
    "latency",
    "throughput",
)
_ICON_HINTS = ("avatar", "logo", "icon", "favicon", "emoji", "sprite", "badge")
_MIN_MEDIA_DIM = 80


def _meta_from_soup(soup: BeautifulSoup) -> DocumentMetaSummary:
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip() or title
    desc = ""
    md = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", property="og:description"
    )
    if md and md.get("content"):
        desc = md["content"].strip()
    keywords: list[str] = []
    kw = soup.find("meta", attrs={"name": "keywords"})
    if kw and kw.get("content"):
        keywords = [k.strip() for k in kw["content"].split(",") if k.strip()][:12]
    return DocumentMetaSummary(title=title, description=desc, keywords=keywords)


def _abstract_from_soup(soup: BeautifulSoup, sections: dict[str, str]) -> str:
    try:
        import trafilatura

        downloaded = trafilatura.extract(
            str(soup),
            include_comments=False,
            include_tables=True,
        )
        if downloaded and len(downloaded.strip()) > 80:
            return downloaded.strip()[:4000]
    except Exception:
        pass
    for key in ("abstract", "аннотация", "summary", "введение", "introduction"):
        for heading, body in sections.items():
            if key in heading.lower():
                return body[:2500]
    paras = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 60
    ]
    return "\n\n".join(paras[:2])[:2500]


def _sections_and_toc(soup: BeautifulSoup) -> tuple[list[TocEntry], dict[str, str]]:
    toc: list[TocEntry] = []
    sections: dict[str, str] = {}
    headings = soup.find_all(re.compile(r"^h[1-3]$", re.I))
    for h in headings:
        level = int(h.name[1]) if h.name and len(h.name) == 2 else 2
        title = h.get_text(" ", strip=True)
        if not title:
            continue
        toc.append(TocEntry(level=level, title=title))
        parts: list[str] = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) and re.match(r"^h[1-3]$", sib.name, re.I):
                break
            text = (
                sib.get_text(" ", strip=True)
                if hasattr(sib, "get_text")
                else str(sib).strip()
            )
            if text:
                parts.append(text)
        body = "\n".join(parts).strip()
        if body:
            sections[title] = body[:6000]
    return toc, sections


def _code_artifacts(
    soup: BeautifulSoup, sections: dict[str, str]
) -> list[CodeArtifact]:
    artifacts: list[CodeArtifact] = []
    section_titles = list(sections.keys())
    for pre in soup.find_all("pre"):
        code_el = pre.find("code")
        code = (code_el or pre).get_text("\n", strip=True)
        if len(code) < 12:
            continue
        lang = ""
        if code_el:
            classes = code_el.get("class") or []
            for c in classes:
                if c.startswith("language-"):
                    lang = c.replace("language-", "")
        context = ""
        prev = pre.find_previous(re.compile(r"^h[1-4]$", re.I))
        if prev:
            context = prev.get_text(" ", strip=True)
        elif section_titles:
            context = section_titles[0]
        artifacts.append(CodeArtifact(language=lang, context=context, code=code[:8000]))
    return artifacts[:20]


def _parse_dim(val: str | None) -> int | None:
    if not val:
        return None
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else None


def _is_important_media(
    src: str,
    alt: str,
    width: int | None,
    height: int | None,
) -> tuple[bool, str]:
    blob = f"{src} {alt}".lower()
    if any(h in blob for h in _ICON_HINTS):
        return False, ""
    w, h = width or 0, height or 0
    if w and h and w < _MIN_MEDIA_DIM and h < _MIN_MEDIA_DIM:
        return False, ""
    for kw in _MEDIA_KEYWORDS:
        if kw in blob:
            return True, f"keyword:{kw}"
    if w >= 200 or h >= 200:
        return True, "large_dimensions"
    if alt.strip() and len(alt.strip()) > 12:
        return True, "descriptive_alt"
    return False, ""


def _media_artifacts(soup: BeautifulSoup, page_url: str) -> list[MediaArtifact]:
    picked: list[MediaArtifact] = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        full = urljoin(page_url, src)
        alt = (img.get("alt") or "").strip()
        w = _parse_dim(img.get("width"))
        h = _parse_dim(img.get("height"))
        ok, reason = _is_important_media(src, alt, w, h)
        if not ok:
            continue
        picked.append(
            MediaArtifact(url=full, alt=alt, width=w, height=h, reason=reason)
        )
        if len(picked) >= 8:
            break
    for svg in soup.find_all("svg")[:3]:
        parent = svg.find_parent(["figure", "div"])
        cap = ""
        if parent:
            cap = parent.get_text(" ", strip=True)[:200]
        ok, reason = _is_important_media("svg", cap, 400, 300)
        if ok:
            picked.append(
                MediaArtifact(
                    url=page_url + "#svg", alt=cap or "inline-svg", reason=reason
                )
            )
    return picked[:10]


def parse_html_to_structure(html: str, page_url: str) -> DocumentStructure:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    meta = _meta_from_soup(soup)
    toc, sections = _sections_and_toc(soup)
    abstract = _abstract_from_soup(soup, sections)
    code = _code_artifacts(soup, sections)
    media = _media_artifacts(soup, page_url)
    trace(
        f"PARSER ✓ {page_url[:50]} | sections={len(sections)} code={len(code)} media={len(media)}"
    )
    return DocumentStructure(
        source_url=page_url,
        meta_summary=meta,
        abstract=abstract,
        toc=toc,
        sections=sections,
        code_artifacts=code,
        media_artifacts=media,
    )
