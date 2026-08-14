"""HTML → ExtractedImage из основного текста статьи (agnostic)."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import Tag

from knowledge_engine.services.image_filter import LayoutHint
from knowledge_engine.services.parsers.article_content import article_content_soup
from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.services.parsers.html_annotator import (
    AnnotatedArticle,
    build_annotated_article,
)
from knowledge_engine.services.parsers.html_attr import coerce_html_attr
from knowledge_engine.services.parsers.image_bytes import (
    load_image_bytes,
    pick_img_src,
    resolve_image_url,
)

_MD_IMG_IN_HTML = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class HtmlArticleParser:
    def __init__(
        self,
        base_path: Path | None = None,
        *,
        page_url: str = "",
    ) -> None:
        self._base_path = base_path
        self._page_url = (page_url or "").strip()

    def build_annotated_document(self, data: bytes) -> AnnotatedArticle:
        """Размеченный Markdown [P_n] / [FIG_n] + fig_map (без download img)."""
        return build_annotated_article(data, self._page_url)

    def parse(self, data: bytes) -> list[ExtractedImage]:
        raw_html = data.decode("utf-8", errors="replace")
        soup = article_content_soup(raw_html, self._page_url)
        soup_source = str(soup)
        text_lines = soup.get_text("\n", strip=True).splitlines()
        out: list[ExtractedImage] = []
        pos = 0
        seen_src: set[str] = set()

        for fig in soup.find_all("figure"):
            img = fig.find("img")
            if img is None:
                continue
            pos += 1
            item = self._from_img(
                img,
                pos,
                text_lines,
                figcaption=fig.find("figcaption"),
                seen_src=seen_src,
            )
            if item is not None:
                out.append(item)

        for img in soup.find_all("img"):
            if img.find_parent("figure") is not None:
                continue
            pos += 1
            item = self._from_img(img, pos, text_lines, seen_src=seen_src)
            if item is not None:
                out.append(item)

        for match in _MD_IMG_IN_HTML.finditer(soup_source):
            alt = (match.group(1) or "").strip()
            src = resolve_image_url(match.group(2), self._page_url)
            if not src or src in seen_src:
                continue
            loaded = load_image_bytes(src, base_path=self._base_path)
            if loaded is None:
                continue
            seen_src.add(src)
            pos += 1
            line_no = soup_source[: match.start()].count("\n")
            context = _paragraph_context(text_lines, line_no)
            image_bytes, mime = loaded
            out.append(
                ExtractedImage(
                    image_bytes=image_bytes,
                    caption=(alt or context[:400])[:500],
                    context_text=context[:2000],
                    page_or_pos=pos,
                    mime=mime,
                )
            )

        return out

    def _from_img(
        self,
        img: Tag,
        pos: int,
        text_lines: list[str],
        *,
        figcaption: Tag | None = None,
        seen_src: set[str],
    ) -> ExtractedImage | None:
        raw_src = pick_img_src(img, self._page_url)
        if not raw_src:
            return None
        src = raw_src
        loaded = load_image_bytes(src, base_path=self._base_path)
        if loaded is None:
            return None
        seen_src.add(src)
        image_bytes, mime = loaded
        caption_parts: list[str] = []
        if figcaption is not None:
            caption_parts.append(figcaption.get_text(" ", strip=True))
        alt = coerce_html_attr(img.get("alt"))
        title = coerce_html_attr(img.get("title"))
        if alt:
            caption_parts.append(alt)
        if title:
            caption_parts.append(title)
        caption = " — ".join(caption_parts)[:500]
        line_hint = _line_index_for_img(img, text_lines)
        context = _paragraph_context(text_lines, line_hint)
        return ExtractedImage(
            image_bytes=image_bytes,
            caption=caption,
            context_text=context[:2000],
            page_or_pos=pos,
            layout=LayoutHint(),
            mime=mime,
        )


def _line_index_for_img(img: Tag, text_lines: list[str]) -> int:
    alt = coerce_html_attr(img.get("alt"))
    if alt and len(alt) > 3:
        for i, ln in enumerate(text_lines):
            if alt[:40] in ln:
                return i
    return max(0, len(text_lines) // 2)


def _paragraph_context(lines: list[str], line_index: int) -> str:
    chunks: list[str] = []
    for i in range(max(0, line_index - 2), min(len(lines), line_index + 3)):
        t = (lines[i] or "").strip()
        if t:
            chunks.append(t)
    return " ".join(chunks)[:2000]
