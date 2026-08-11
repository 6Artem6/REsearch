"""TOC из HTML / PDF / Markdown + виртуальное оглавление."""

from __future__ import annotations

import re
from typing import Literal

import fitz
from bs4 import BeautifulSoup, Tag

from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    _norm_p,
    p_index_map,
    sorted_p_ids,
)
from knowledge_engine.services.article_ingestion.triage_schemas import (
    DocumentStructureTree,
    TOCNode,
)
from knowledge_engine.services.parsers.article_content import article_content_soup
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

SourceFormat = Literal["html", "pdf", "markdown"]

_TOC_TITLE_RE = re.compile(
    r"^(table of contents|contents|оглавление|содержание)\s*$",
    re.I,
)
_PAGE_FOOTER_RE = re.compile(
    r"^(page\s+\d+\s+of\s+\d+|\d+\s*/\s*\d+)$",
    re.I,
)
_HEADING_NUM_RE = re.compile(r"^\d+([\.\)]\s+|\s+)")


class UniversalTOCExtractor:
    def extract(
        self,
        annotated: AnnotatedArticle,
        source_format: SourceFormat,
        raw: bytes | str | None = None,
    ) -> DocumentStructureTree:
        if source_format == "pdf" and isinstance(raw, bytes):
            tree = self._from_pdf(raw, annotated)
            if tree.nodes:
                return tree
        if source_format == "html" and raw is not None:
            html = (
                raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            )
            tree = self._from_html(html, annotated.page_url, annotated)
            if tree.nodes:
                return tree
        if source_format == "markdown" and raw is not None:
            text = (
                raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            )
            tree = self._from_markdown_text(text, annotated)
            if tree.nodes:
                return tree
        return self._virtual_from_paragraphs(annotated, explicit=False)

    def _from_pdf(
        self, data: bytes, annotated: AnnotatedArticle
    ) -> DocumentStructureTree:
        doc = fitz.open(stream=data, filetype="pdf")
        nodes: list[TOCNode] = []
        explicit = False
        try:
            toc = doc.get_toc(simple=True) or []
            if toc:
                explicit = True
                for level, title, page in toc:
                    title = (title or "").strip()
                    if not title:
                        continue
                    start = self._p_id_for_page(annotated, int(page))
                    if not start:
                        start = self._match_p_id_for_text(annotated, title)
                    if start:
                        nodes.append(
                            TOCNode(
                                title=title[:300],
                                level=max(1, int(level)),
                                start_p_id=start,
                                page_number=int(page),
                            )
                        )
            if not nodes:
                text_toc = self._pdf_text_toc_block(doc, annotated)
                if text_toc:
                    explicit = True
                    nodes = text_toc
            if not nodes:
                nodes = self._pdf_font_headings(doc, annotated)
        finally:
            doc.close()
        nodes = self._finalize_nodes(nodes, annotated)
        return DocumentStructureTree(has_explicit_toc=explicit, nodes=nodes)

    def _pdf_text_toc_block(
        self, doc: fitz.Document, annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        lines: list[tuple[int, str]] = []
        max_pages = min(3, doc.page_count)
        for pi in range(max_pages):
            page = doc[pi]
            for line in (page.get_text("text") or "").splitlines():
                t = line.strip()
                if len(t) < 2:
                    continue
                lines.append((pi + 1, t))
        start_i = None
        for i, (_p, t) in enumerate(lines):
            if _TOC_TITLE_RE.match(_HEADING_NUM_RE.sub("", t).strip()):
                start_i = i + 1
                break
        if start_i is None:
            return []
        nodes: list[TOCNode] = []
        for page_no, t in lines[start_i : start_i + 40]:
            if _TOC_TITLE_RE.match(t):
                break
            if _PAGE_FOOTER_RE.match(t):
                continue
            title = re.sub(r"\s+\d{1,4}\s*$", "", t).strip()
            if len(title) < 3:
                continue
            start = self._match_p_id_for_text(annotated, title)
            if not start:
                start = self._p_id_for_page(annotated, page_no)
            if start:
                nodes.append(
                    TOCNode(
                        title=title[:300],
                        level=1,
                        start_p_id=start,
                        page_number=page_no,
                    )
                )
        return nodes

    def _pdf_font_headings(
        self, doc: fitz.Document, annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        nodes: list[TOCNode] = []
        sizes: list[float] = []
        for page in doc:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = float(span.get("size") or 0)
                        if sz > 0:
                            sizes.append(sz)
        base = sorted(sizes)[len(sizes) // 2] if sizes else 11.0
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(str(s.get("text", "")) for s in spans).strip()
                    if len(text) < 3 or len(text) > 200:
                        continue
                    max_sz = max(float(s.get("size") or 0) for s in spans)
                    flags = int(spans[0].get("flags") or 0)
                    bold = bool(flags & 2**4)
                    if max_sz < base * 1.08 and not bold:
                        continue
                    level = 1 if max_sz >= base * 1.35 or bold else 2
                    start = self._match_p_id_for_text(annotated, text)
                    if not start:
                        start = self._p_id_for_page(annotated, page_no)
                    if start:
                        nodes.append(
                            TOCNode(
                                title=text[:300],
                                level=level,
                                start_p_id=start,
                                page_number=page_no,
                            )
                        )
        return nodes

    def _from_html(
        self,
        html: str,
        page_url: str,
        annotated: AnnotatedArticle,
    ) -> DocumentStructureTree:
        soup = article_content_soup(html, page_url)
        nodes: list[TOCNode] = []
        explicit = False

        nav_nodes = self._toc_from_nav(soup, annotated)
        if nav_nodes:
            explicit = True
            nodes = nav_nodes

        if not nodes:
            for tag in soup.find_all(["nav", "aside"]):
                cls = " ".join(tag.get("class") or []).lower()
                if "toc" in cls or tag.name == "nav":
                    nav_nodes = self._toc_from_link_list(tag, annotated)
                    if nav_nodes:
                        explicit = True
                        nodes = nav_nodes
                        break

        if not nodes:
            heading_nodes = self._toc_from_headings(soup, annotated)
            if heading_nodes:
                nodes = heading_nodes

        nodes = self._finalize_nodes(nodes, annotated)
        return DocumentStructureTree(has_explicit_toc=explicit, nodes=nodes)

    def _toc_from_nav(
        self, soup: BeautifulSoup, annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        nodes: list[TOCNode] = []
        for nav in soup.find_all("nav"):
            part = self._toc_from_link_list(nav, annotated)
            if part:
                return part
        return nodes

    def _toc_from_link_list(
        self, root: Tag, annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        nodes: list[TOCNode] = []
        for a in root.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            title = a.get_text(" ", strip=True)
            if not title or len(title) < 2:
                continue
            start: str | None = None
            if href.startswith("#"):
                anchor = href[1:].split("?", 1)[0]
                el = root.find(id=anchor)
                if el is None and root.parent:
                    el = root.parent.find(id=anchor)
                if el is not None:
                    el_text = el.get_text(" ", strip=True)[:400]
                    start = self._match_p_id_for_text(annotated, el_text) or (
                        self._match_p_id_for_text(annotated, title)
                    )
            if not start:
                start = self._match_p_id_for_text(annotated, title)
            if not start:
                continue
            depth = 1
            for parent in a.parents:
                if parent.name in ("ol", "ul"):
                    depth += 1
                if parent is root:
                    break
            nodes.append(
                TOCNode(
                    title=title[:300],
                    level=min(depth, 6),
                    start_p_id=start,
                )
            )
        return nodes

    def _toc_from_headings(
        self, soup: BeautifulSoup, annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        nodes: list[TOCNode] = []
        for tag in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
            title = tag.get_text(" ", strip=True)
            if len(title) < 2:
                continue
            level = int(tag.name[1]) if tag.name and len(tag.name) == 2 else 1
            start = self._match_p_id_for_text(annotated, title)
            if start:
                nodes.append(TOCNode(title=title[:300], level=level, start_p_id=start))
        return nodes

    def _from_markdown_text(
        self, text: str, annotated: AnnotatedArticle
    ) -> DocumentStructureTree:
        nodes: list[TOCNode] = []
        explicit = False
        in_toc = False
        for line in text.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if _TOC_TITLE_RE.match(stripped):
                in_toc = True
                explicit = True
                continue
            if in_toc:
                if not stripped:
                    if nodes:
                        break
                    continue
                if stripped.startswith("#"):
                    break
                title = re.sub(r"\s+\d{1,4}\s*$", "", stripped).strip()
                if len(title) < 3:
                    continue
                start = self._match_p_id_for_text(annotated, title)
                if start:
                    nodes.append(TOCNode(title=title[:300], level=1, start_p_id=start))
            hm = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if hm:
                level = len(hm.group(1))
                title = hm.group(2).strip()
                start = self._match_p_id_for_text(annotated, title)
                if start:
                    nodes.append(
                        TOCNode(
                            title=title[:300],
                            level=level,
                            start_p_id=start,
                        )
                    )
        nodes = self._finalize_nodes(nodes, annotated)
        return DocumentStructureTree(has_explicit_toc=explicit, nodes=nodes)

    def _virtual_from_paragraphs(
        self, annotated: AnnotatedArticle, *, explicit: bool
    ) -> DocumentStructureTree:
        nodes: list[TOCNode] = []
        for pid in sorted_p_ids(annotated.paragraph_map):
            text = (annotated.paragraph_map.get(pid) or "").strip()
            if len(text) < 3 or len(text) > 220:
                continue
            if text.endswith(".") and len(text.split()) > 12:
                continue
            if _PAGE_FOOTER_RE.match(text):
                continue
            title = _HEADING_NUM_RE.sub("", text).strip()
            nodes.append(TOCNode(title=title[:300], level=1, start_p_id=pid))
        if not nodes and annotated.paragraph_map:
            first = sorted_p_ids(annotated.paragraph_map)[0]
            nodes.append(TOCNode(title="Document body", level=1, start_p_id=first))
        nodes = self._finalize_nodes(nodes, annotated)
        return DocumentStructureTree(has_explicit_toc=explicit, nodes=nodes)

    def _p_id_for_page(self, annotated: AnnotatedArticle, page_no: int) -> str | None:
        pages = annotated.paragraph_page or {}
        order = sorted_p_ids(annotated.paragraph_map)
        for pid in order:
            if pages.get(pid) == page_no:
                return pid
        for pid in order:
            if pages.get(pid, 0) >= page_no:
                return pid
        return order[0] if order else None

    def _match_p_id_for_text(
        self, annotated: AnnotatedArticle, needle: str
    ) -> str | None:
        n = re.sub(r"\s+", " ", (needle or "").strip().lower())
        if len(n) < 2:
            return None
        n_short = n[:80]
        best: str | None = None
        best_len = 0
        for pid, text in annotated.paragraph_map.items():
            t = re.sub(r"\s+", " ", (text or "").strip().lower())
            if t == n or t.startswith(n_short) or n_short.startswith(t[:80]):
                if len(t) >= best_len:
                    best = pid
                    best_len = len(t)
            elif n in t and len(n) > 10:
                if len(t) >= best_len:
                    best = pid
                    best_len = len(t)
        return best

    def _finalize_nodes(
        self, nodes: list[TOCNode], annotated: AnnotatedArticle
    ) -> list[TOCNode]:
        order = sorted_p_ids(annotated.paragraph_map)
        if not order or not nodes:
            return []
        idx = p_index_map(annotated.paragraph_map)
        cleaned: list[TOCNode] = []
        seen_start: set[str] = set()
        for node in nodes:
            sp = _norm_p(node.start_p_id)
            if sp not in idx or sp in seen_start:
                continue
            seen_start.add(sp)
            cleaned.append(node)
        cleaned.sort(key=lambda n: idx[_norm_p(n.start_p_id)])
        out: list[TOCNode] = []
        for i, node in enumerate(cleaned):
            si = idx[_norm_p(node.start_p_id)]
            if i + 1 < len(cleaned):
                next_si = idx[_norm_p(cleaned[i + 1].start_p_id)]
                end_i = max(si, next_si - 1)
            else:
                end_i = len(order) - 1
            end_p = order[end_i]
            out.append(node.model_copy(update={"end_p_id": end_p}))
        return out
