"""HTML → annotated Markdown с координатами [P_n] и [FIG_n]."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import NavigableString, Tag

from knowledge_engine.services.parsers.article_content import article_content_soup
from knowledge_engine.services.parsers.html_attr import coerce_html_attr
from knowledge_engine.services.parsers.image_bytes import pick_img_src
from knowledge_engine.ui.run_log import trace

_BLOCK_TAGS = frozenset(
    {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "pre",
        "td",
        "th",
        "dt",
        "dd",
    }
)
_SKIP_TAGS = frozenset({"script", "style", "svg"})


def _attr_text(value: object) -> str:
    return coerce_html_attr(value)


def _escape_attr(value: str) -> str:
    return _attr_text(value).replace('"', "'").replace("\n", " ").strip()[:400]


@dataclass
class AnnotatedArticle:
    annotated_markdown: str = ""
    fig_map: dict[str, str] = field(default_factory=dict)
    paragraph_map: dict[str, str] = field(default_factory=dict)
    page_url: str = ""
    fig_bytes: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    fig_extract_source: dict[str, str] = field(default_factory=dict)
    fig_extract_topology: dict[str, dict] = field(default_factory=dict)
    paragraph_page: dict[str, int] = field(default_factory=dict)
    source_pdf_bytes: bytes = field(default_factory=bytes)


def build_annotated_article(
    raw_html: bytes | str, page_url: str = ""
) -> AnnotatedArticle:
    """Разметить основной текст статьи и плейсхолдеры FIG без загрузки картинок."""
    if isinstance(raw_html, bytes):
        html = raw_html.decode("utf-8", errors="replace")
    else:
        html = raw_html or ""
    url = (page_url or "").strip()
    soup = article_content_soup(html, url)

    lines: list[str] = []
    fig_map: dict[str, str] = {}
    paragraph_map: dict[str, str] = {}
    p_idx = 0
    fig_idx = 0
    seen_img_src: set[str] = set()

    def emit_paragraph(text: str) -> None:
        nonlocal p_idx
        t = re.sub(r"\s+", " ", (text or "").strip())
        if len(t) < 2:
            return
        p_idx += 1
        pid = f"P_{p_idx}"
        paragraph_map[pid] = t[:4000]
        lines.append(f"[{pid}] {t}")

    def emit_figure(img: Tag, figcaption: Tag | None) -> None:
        nonlocal fig_idx
        src = pick_img_src(img, url)
        if not src or src in seen_img_src:
            return
        seen_img_src.add(src)
        fig_idx += 1
        fid = f"FIG_{fig_idx}"
        fig_map[fid] = src
        alt = _escape_attr(img.get("alt"))
        cap_parts: list[str] = []
        if figcaption is not None:
            cap_parts.append(figcaption.get_text(" ", strip=True))
        title = _escape_attr(img.get("title"))
        if title:
            cap_parts.append(title)
        caption = _escape_attr(" — ".join(cap_parts))
        lines.append(f'[{fid}: alt="{alt}" | caption="{caption}"]')

    def walk(node: Tag) -> None:
        if node.name in _SKIP_TAGS:
            return
        if node.name == "noscript":
            for child in node.children:
                if isinstance(child, Tag):
                    walk(child)
            return
        if node.name == "figure":
            img = node.find("img")
            if img is not None:
                emit_figure(img, node.find("figcaption"))
            return
        if node.name == "img":
            if node.find_parent("figure") is not None:
                return
            emit_figure(node, None)
            return
        if node.name in _BLOCK_TAGS:
            emit_paragraph(node.get_text(" ", strip=True))
            return
        for child in node.children:
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t and node.name in ("div", "section", "article", "main", "body"):
                    emit_paragraph(t)
            elif isinstance(child, Tag):
                walk(child)

    root = soup.body or soup
    walk(root)

    for img in soup.find_all("img"):
        emit_figure(img, None)

    annotated = "\n\n".join(lines).strip()
    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    pipeline_audit(
        "Annotate",
        url,
        annotated,
        extra=f"P={len(paragraph_map)} FIG={len(fig_map)}",
    )
    trace(
        f"BLOG_SPATIAL annotate ✓ | P={len(paragraph_map)} FIG={len(fig_map)} "
        f"chars={len(annotated)} | {url[:70]}"
    )
    return AnnotatedArticle(
        annotated_markdown=annotated,
        fig_map=fig_map,
        paragraph_map=paragraph_map,
        page_url=url,
    )
