"""PyMuPDF: растровые xref (не full-page) + caption crops → FIG_n PNG."""

from __future__ import annotations

import fitz

from knowledge_engine.services.parsers.pdf_parser import _caption_near_rect
from knowledge_engine.services.parsers.raster_pdf_layout import (
    discover_raster_column_figures,
)
from knowledge_engine.services.parsers.vector_pdf_cropper import VectorPDFCropper
from knowledge_engine.ui.run_log import trace

_MAX_RASTER_PAGE_FRACTION = 0.42


def extract_figures_pymupdf(pdf_bytes: bytes) -> dict[str, tuple[bytes, str]]:
    from knowledge_engine.services.parsers.pdf_bytes import is_parseable_pdf

    if not is_parseable_pdf(pdf_bytes):
        return {}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: dict[str, tuple[bytes, str]] = {}
    seen_hash: set[int] = set()

    def put(key: str, png: bytes, source: str) -> None:
        if len(png) < 400:
            return
        h = hash(png[:2000])
        if h in seen_hash:
            return
        seen_hash.add(h)
        if key in out:
            return
        out[key] = (png, "image/png")
        trace(f"PYMUPDF_FIG ✓ | {key} | {source} | {len(png)} B")

    try:
        cropper = VectorPDFCropper()
        for key, fig in cropper.discover_from_document_detailed(doc).items():
            if fig and len(fig.data) >= 400:
                put(key, fig.data, fig.source)

        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            page_area = page.rect.width * page.rect.height

            for clip, png in discover_raster_column_figures(page):
                put(
                    f"FIG_RASTER_P{page_no}_B{int(clip.y0)}",
                    png,
                    f"p{page_no} layout band",
                )

            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 1:
                    continue
                bbox = block.get("bbox") or (0, 0, 0, 0)
                br = fitz.Rect(bbox)
                if br.width * br.height > page_area * _MAX_RASTER_PAGE_FRACTION:
                    continue
                raw = block.get("image") or b""
                if len(raw) < 400:
                    continue
                put(
                    f"FIG_RASTER_P{page_no}_B{int(br.y0)}",
                    _as_png_bytes(raw, block.get("ext")),
                    f"p{page_no} block",
                )

            for img_info in page.get_images(full=True):
                xref = int(img_info[0])
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    continue
                rect = rects[0]
                if rect.width * rect.height > page_area * _MAX_RASTER_PAGE_FRACTION:
                    trace(
                        f"PYMUPDF_FIG ⊘ | skip full-page xref p{page_no} "
                        f"| {int(rect.width)}×{int(rect.height)}"
                    )
                    continue
                try:
                    extracted = doc.extract_image(xref)
                    raw = extracted.get("image") or b""
                except Exception:
                    continue
                if len(raw) < 400:
                    continue
                cap = _caption_near_rect(page, rect)[:80]
                put(
                    f"FIG_RASTER_P{page_no}_X{xref}",
                    _as_png_bytes(raw, extracted.get("ext")),
                    f"p{page_no} xref {cap}",
                )

        seq = 0
        for key in sorted(
            [k for k in out if k.startswith("FIG_") and not k.startswith("FIG_SEQ")],
            key=lambda k: k,
        ):
            if key.startswith("FIG_RASTER"):
                seq += 1
                put(f"FIG_SEQ_{seq}", out[key][0], "seq")
        for i, k in enumerate(
            sorted(
                [k for k in out if k.startswith("FIG_SEQ_")],
                key=lambda x: int(x.split("_", 2)[-1]),
            ),
            start=1,
        ):
            if f"FIG_{i}" not in out:
                put(f"FIG_{i}", out[k][0], "seq-alias")
    finally:
        doc.close()

    trace(f"PYMUPDF_FIG summary | total_keys={len(out)}")
    return out


def _as_png_bytes(raw: bytes, ext: str | None) -> bytes:
    if raw[:4] == b"\x89PNG" or raw[:3] == b"\xff\xd8\xff":
        return raw
    try:
        pix = fitz.Pixmap(raw)
        return pix.tobytes("png")
    except Exception:
        return raw
