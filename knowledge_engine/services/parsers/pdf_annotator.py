"""PDF → AnnotatedArticle ([P_n], [FIG_n], fig_map / fig_bytes)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz

from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle
from knowledge_engine.services.parsers.pdf_parser import _caption_near_rect
from knowledge_engine.services.parsers.vector_pdf_cropper import (
    FigureBinary,
    VectorPDFCropper,
)
from knowledge_engine.ui.run_log import trace


def _escape_attr(value: str) -> str:
    return (value or "").replace('"', "'").replace("\n", " ").strip()[:400]


@dataclass
class _PageEvent:
    y0: float
    x0: float
    kind: str
    text: str = ""
    fig_id: str = ""
    alt: str = ""
    page_no: int = 0
    image_bytes: bytes = b""
    mime: str = "image/png"


class PdfArticleParser:
    """Парсинг PDF в тот же AnnotatedArticle, что HTML annotator."""

    def build_annotated_document(self, data: bytes) -> AnnotatedArticle:
        return build_annotated_pdf(data)

    def parse(self, data: bytes) -> list:
        from knowledge_engine.services.parsers.pdf_parser import (
            PdfArticleParser as Legacy,
        )

        return Legacy().parse(data)


def build_annotated_pdf(data: bytes) -> AnnotatedArticle:
    from knowledge_engine.services.parsers.pdf_bytes import is_parseable_pdf

    if not is_parseable_pdf(data):
        trace(
            f"PDF_ANNOTATE ⊘ | unparsable pdf bytes ({len(data)} B) — "
            "need /doi/pdf/ or Sci-Hub"
        )
        return AnnotatedArticle(
            annotated_markdown="",
            page_url="",
            source_pdf_bytes=data,
        )

    doc = fitz.open(stream=data, filetype="pdf")
    lines: list[str] = []
    fig_map: dict[str, str] = {}
    fig_bytes: dict[str, tuple[bytes, str]] = {}
    fig_extract_source: dict[str, str] = {}
    fig_extract_topology: dict[str, dict] = {}
    paragraph_map: dict[str, str] = {}
    paragraph_page: dict[str, int] = {}
    p_idx = 0
    fig_idx = 0
    try:
        for page_index, page in enumerate(doc):
            events = _page_events(page, page_index + 1)
            events.sort(key=lambda e: (_column_bucket(e.x0, page.rect.width), e.y0))
            for ev in events:
                if ev.kind == "text" and ev.text.strip():
                    p_idx += 1
                    pid = f"P_{p_idx}"
                    t = re.sub(r"\s+", " ", ev.text.strip())
                    paragraph_map[pid] = t[:4000]
                    paragraph_page[pid] = ev.page_no
                    lines.append(f"[{pid}] {t}")
                elif ev.kind == "figure" and ev.image_bytes:
                    fig_idx += 1
                    fid = f"FIG_{fig_idx}"
                    fig_map[fid] = f"embedded:{fid}"
                    fig_bytes[fid] = (ev.image_bytes, ev.mime)
                    fig_extract_source[fid] = "xref"
                    alt = _escape_attr(ev.alt)
                    lines.append(f'[{fid}: alt="{alt}" | page="{ev.page_no}"]')

        _merge_vector_crops(
            doc,
            lines,
            fig_map,
            fig_bytes,
            fig_extract_source,
            fig_extract_topology,
            fig_idx,
        )
        if not fig_bytes:
            from knowledge_engine.services.parsers.pymupdf_figure_extract import (
                extract_figures_pymupdf,
            )

            pymupdf_figs = extract_figures_pymupdf(data)
            for key, payload in pymupdf_figs.items():
                if key in fig_bytes:
                    continue
                fig_map[key] = f"embedded:{key}"
                fig_bytes[key] = payload
                fig_extract_source.setdefault(key, "caption_clip_fallback")
                if key.startswith("FIG_") and not key.startswith("FIG_SEQ_"):
                    lines.append(f'[{key}: alt="pymupdf" | page="0"]')
            trace(
                f"PDF_ANNOTATE pymupdf pass | figs={len(fig_bytes)} "
                f"pages={doc.page_count}"
            )
    finally:
        doc.close()

    article = AnnotatedArticle(
        annotated_markdown="\n\n".join(lines).strip(),
        fig_map=fig_map,
        paragraph_map=paragraph_map,
        page_url="",
        fig_bytes=fig_bytes,
        fig_extract_source=fig_extract_source,
        fig_extract_topology=fig_extract_topology,
        paragraph_page=paragraph_page,
        source_pdf_bytes=data,
    )
    from knowledge_engine.services.parsers.pdf_text_cleaner import (
        clean_annotated_article,
    )

    return clean_annotated_article(article)


def _merge_vector_crops(
    doc: fitz.Document,
    lines: list[str],
    fig_map: dict[str, str],
    fig_bytes: dict[str, tuple[bytes, str]],
    fig_extract_source: dict[str, str],
    fig_extract_topology: dict[str, dict],
    fig_idx: int,
) -> None:
    cropper = VectorPDFCropper()
    discovered = cropper.discover_from_document_detailed(doc)
    if not discovered:
        return

    def note_fig(key: str, fig: FigureBinary) -> None:
        fig_extract_source[key] = fig.source
        if fig.topology:
            fig_extract_topology[key] = dict(fig.topology)
        if not fig.is_renderable:
            return
        fig_map[key] = f"embedded:{key}"
        fig_bytes[key] = fig.as_payload()

    seq_keys = sorted(
        [k for k in discovered if k.startswith("FIG_SEQ_")],
        key=lambda k: int(k.split("_", 2)[-1]),
    )
    for key in seq_keys:
        fig = discovered.get(key)
        if fig is None:
            continue
        note_fig(key, fig)
        fig_idx += 1
        fid = f"FIG_{fig_idx}"
        if fid not in fig_extract_source:
            note_fig(fid, fig)
            if fig.is_renderable:
                lines.append(f'[{fid}: alt="vector Fig. crop" | page="0"]')

    for key, fig in discovered.items():
        if key.startswith("FIG_SEQ_"):
            continue
        if key in fig_extract_source:
            continue
        note_fig(key, fig)
        if fig.is_renderable and f"[{key}:" not in "\n".join(lines):
            lines.append(f'[{key}: alt="vector Fig. {key[4:]}" | page="0"]')


def _column_bucket(x0: float, page_width: float) -> int:
    if page_width <= 0:
        return 0
    mid = page_width * 0.5
    return 0 if x0 < mid else 1


def _page_events(page: fitz.Page, page_no: int) -> list[_PageEvent]:
    events: list[_PageEvent] = []
    page_area = page.rect.width * page.rect.height

    d = page.get_text("dict")
    for block in d.get("blocks", []):
        btype = block.get("type")
        bbox = block.get("bbox") or (0, 0, 0, 0)
        x0, y0 = float(bbox[0]), float(bbox[1])
        if btype == 0:
            parts: list[str] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(str(s.get("text", "")) for s in spans).strip()
                if line_text:
                    parts.append(line_text)
            text = " ".join(parts).strip()
            if len(text) >= 2:
                events.append(
                    _PageEvent(y0=y0, x0=x0, kind="text", text=text, page_no=page_no)
                )
        elif btype == 1:
            br = fitz.Rect(bbox)
            if br.width * br.height > page_area * 0.42:
                continue
            img_bytes = block.get("image") or b""
            if img_bytes:
                ext = (block.get("ext") or "png").lower()
                mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                events.append(
                    _PageEvent(
                        y0=y0,
                        x0=x0,
                        kind="figure",
                        alt="",
                        page_no=page_no,
                        image_bytes=img_bytes,
                        mime=mime,
                    )
                )

    doc = page.parent
    for img_info in page.get_images(full=True):
        xref = int(img_info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        if not rects:
            continue
        rect = rects[0]
        if rect.width * rect.height > page_area * 0.42:
            continue
        try:
            extracted = doc.extract_image(xref)
            image_bytes = extracted.get("image") or b""
            ext = (extracted.get("ext") or "png").lower()
        except Exception:
            continue
        if not image_bytes:
            continue
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        caption = _caption_near_rect(page, rect)
        events.append(
            _PageEvent(
                y0=float(rect.y0),
                x0=float(rect.x0),
                kind="figure",
                alt=caption[:200],
                page_no=page_no,
                image_bytes=image_bytes,
                mime=mime,
            )
        )

    dedup: list[_PageEvent] = []
    seen_bbox: set[tuple[int, int]] = set()
    for ev in events:
        key = (int(ev.y0 // 8), int(ev.x0 // 8))
        if ev.kind == "figure" and key in seen_bbox:
            continue
        if ev.kind == "figure":
            seen_bbox.add(key)
        dedup.append(ev)
    return dedup
