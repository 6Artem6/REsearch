"""Markdown → ExtractedImage + inline Mermaid."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from knowledge_engine.services.parsers.base import ExtractedDiagram, ExtractedImage
from knowledge_engine.services.parsers.html_parser import HtmlArticleParser
from knowledge_engine.services.parsers.image_bytes import load_image_bytes

_MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_MERMAID_FENCE_RE = re.compile(r"```\s*mermaid\s*\n([\s\S]*?)```", re.IGNORECASE)


class MarkdownArticleParser:
    def __init__(self, base_path: Path | None = None) -> None:
        self._base_path = base_path

    def parse_inline_diagrams(self, data: bytes) -> list[ExtractedDiagram]:
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        out: list[ExtractedDiagram] = []
        for match in _MERMAID_FENCE_RE.finditer(text):
            mermaid = (match.group(1) or "").strip()
            if not mermaid:
                continue
            line_no = text[: match.start()].count("\n")
            caption = _heading_above(lines, line_no) or _paragraph_context(
                lines, line_no
            )
            out.append(
                ExtractedDiagram(
                    mermaid_code=mermaid,
                    caption=(caption or "")[:500],
                    is_inline_mermaid=True,
                    page_or_pos=line_no + 1,
                )
            )
        return out

    def parse(self, data: bytes) -> list[ExtractedImage]:
        text = data.decode("utf-8", errors="replace")
        out: list[ExtractedImage] = []
        lines = text.splitlines()
        for match in _MD_IMG_RE.finditer(text):
            alt = (match.group(1) or "").strip()
            src = (match.group(2) or "").strip()
            line_no = text[: match.start()].count("\n")
            context = _paragraph_context(lines, line_no)
            loaded = load_image_bytes(src, base_path=self._base_path)
            if loaded is None:
                continue
            image_bytes, mime = loaded
            caption = alt or context[:400]
            out.append(
                ExtractedImage(
                    image_bytes=image_bytes,
                    caption=caption[:500],
                    page_or_pos=line_no + 1,
                    mime=mime,
                )
            )
        for match in _HTML_IMG_RE.finditer(text):
            snippet = match.group(0)
            soup = BeautifulSoup(snippet, "html.parser")
            img = soup.find("img")
            if img is None:
                continue
            src = (img.get("src") or "").strip()
            if not src:
                continue
            line_no = text[: match.start()].count("\n")
            loaded = load_image_bytes(src, base_path=self._base_path)
            if loaded is None:
                continue
            image_bytes, mime = loaded
            alt = (img.get("alt") or "").strip()
            context = _paragraph_context(lines, line_no)
            caption = (alt or context)[:500]
            out.append(
                ExtractedImage(
                    image_bytes=image_bytes,
                    caption=caption,
                    page_or_pos=line_no + 1,
                    mime=mime,
                )
            )
        if "<html" in text.lower() or "<body" in text.lower():
            out.extend(HtmlArticleParser(self._base_path).parse(data))
        return out

    def parse_all(
        self, data: bytes
    ) -> tuple[list[ExtractedImage], list[ExtractedDiagram]]:
        inline = self.parse_inline_diagrams(data)
        images = self.parse(data)
        return images, inline


def _heading_above(lines: list[str], line_index: int) -> str:
    for i in range(line_index - 1, -1, -1):
        t = lines[i].strip()
        if t.startswith("#"):
            return t.lstrip("#").strip()[:500]
    return ""


def _paragraph_context(lines: list[str], line_index: int) -> str:
    chunks: list[str] = []
    for i in range(max(0, line_index - 2), min(len(lines), line_index + 3)):
        t = lines[i].strip()
        if t and not t.startswith("!["):
            chunks.append(t)
    return " ".join(chunks)[:400]
