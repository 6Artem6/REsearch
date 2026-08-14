"""PDF → ExtractedImage (PyMuPDF)."""

from __future__ import annotations

import fitz

from knowledge_engine.services.image_filter import LayoutHint
from knowledge_engine.services.parsers.base import ExtractedImage


class PdfArticleParser:
    def parse(self, data: bytes) -> list[ExtractedImage]:
        out: list[ExtractedImage] = []
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page_index, page in enumerate(doc):
                page_h = float(page.rect.height)
                for img_info in page.get_images(full=True):
                    xref = int(img_info[0])
                    try:
                        rects = page.get_image_rects(xref)
                    except Exception:
                        rects = []
                    if not rects:
                        continue
                    rect = rects[0]
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
                    layout = LayoutHint(
                        page_height=page_h,
                        rect_top=float(rect.y0),
                        rect_bottom=float(rect.y1),
                    )
                    out.append(
                        ExtractedImage(
                            image_bytes=image_bytes,
                            caption=caption,
                            page_or_pos=page_index + 1,
                            layout=layout,
                            mime=mime,
                        )
                    )
        finally:
            doc.close()
        return out


def _caption_near_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    expanded = fitz.Rect(
        rect.x0,
        max(0, rect.y0 - 80),
        rect.x1,
        min(page.rect.height, rect.y1 + 120),
    )
    try:
        text = (page.get_textbox(expanded) or "").strip()
    except Exception:
        text = ""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if low.startswith("figure") or low.startswith("fig.") or low.startswith("рис"):
            return ln[:500]
    return lines[0][:500] if lines else ""
