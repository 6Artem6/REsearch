"""Векторные схемы в PDF: якорь Fig. N + drawings/raster union → PNG 300 DPI."""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any

import fitz

from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)

_RENDER_DPI = 300
_MARGIN_X_PT = 8.0
_MARGIN_Y_PT = 12.0
_DRAWING_LABEL_PROXIMITY_PT = 15.0
_PAGE_TOP_MARGIN_PT = 36.0
_CAPTION_GAP_PT = 2.0
_MIN_DRAW_W = 4.0
_MIN_DRAW_H = 4.0
_MIN_PNG_BYTES = 800
_MAX_FIGURE_HEIGHT_PT = 320.0
_MAX_FIGURE_HEIGHT_VECTOR_PT = 720.0
_MAX_FIGURE_PAGE_FRACTION = 0.52
_MIN_CROP_PT = 40.0
_MAX_ASPECT_RATIO = 8.0
_HEADER_FONT_PT = 11.5
_BODY_FONT_MAX_PT = 12.0
_SECTION_HEADING_RE = re.compile(
    r"^\d+(?:\.\d+)*\s+[A-Z\u0410-\u042f]",
)
_TEXT_DENSITY_DEFAULT = 0.35
_TEXT_DENSITY_VECTOR_BOUND = 0.50
_WORDS_PER_LINE_DENSE = 6.0
_AVG_WORD_LENGTH_DENSE = 4.5
_TIGHT_LABEL_PADDING_PT = 10.0
_TIGHT_MARGIN_X_PT = 4.0
_TIGHT_MARGIN_Y_PT = 4.0
_TIGHT_AXIS_X_SLACK_PT = 15.0
_TIGHT_BODY_LINE_WIDTH_PT = 150.0
_TIGHT_WORD_Y_BAND_PT = 12.0
_TIGHT_BODY_GAP_CUT_PT = 4.0
_RESIDUE_GRID_STEP = 4.0
_TEXT_DECORATION_TOL_PT = 2.5
_DUPLICATE_SIMILARITY = 0.95

_FIG_NUM_RE = re.compile(
    r"(?:figure|fig\.?|рис\.?)\s*[:.]?\s*(\d{1,3})",
    re.IGNORECASE,
)


def _is_figure_caption_line(line_text: str, fig_num: int) -> bool:
    """Строка-подпись (Fig. N.), не упоминание «Figure N» в теле абзаца."""
    t = line_text.strip()
    if re.match(rf"^\s*Fig\.\s*{fig_num}\b", t, re.IGNORECASE):
        return True
    if re.match(rf"^\s*Figure\s+{fig_num}\s*[\.\):,]", t, re.IGNORECASE):
        return True
    if re.match(rf"^\s*(?:рис\.?)\s*{fig_num}\s*[\.\):,]", t, re.IGNORECASE):
        return True
    return False


def _sanitize_visual_rect(page: fitz.Page, rect: fitz.Rect) -> bool:
    if rect.is_empty:
        return False
    pr = page.rect
    page_area = max(1.0, pr.width * pr.height)
    area = rect.width * rect.height
    if area > page_area * _MAX_FIGURE_PAGE_FRACTION:
        return False
    if rect.width > pr.width * 0.62 or rect.height > pr.height * 0.94:
        return False
    cx = (rect.x0 + rect.x1) * 0.5
    cy = (rect.y0 + rect.y1) * 0.5
    if cx < pr.x0 - 48 or cx > pr.x1 + 48:
        return False
    if cy < pr.y0 - 48 or cy > pr.y1 + 48:
        return False
    return True


def _scheme_column_rect(page: fitz.Page, residue_bbox: fitz.Rect) -> fitz.Rect:
    """Колонка страницы, в которой лежит визуальный остаток схемы."""
    pr = page.rect
    mid = pr.x0 + pr.width * 0.5
    cx = (residue_bbox.x0 + residue_bbox.x1) * 0.5
    gutter = 10.0
    if cx >= mid - 8.0:
        return fitz.Rect(mid + gutter, pr.y0, pr.x1, pr.y1)
    return fitz.Rect(pr.x0, pr.y0, mid - gutter, pr.y1)


@dataclass(frozen=True)
class FigureBinary:
    data: bytes
    mime: str
    source: str
    topology: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> tuple[bytes, str]:
        return (self.data, self.mime)

    @property
    def is_renderable(self) -> bool:
        return (
            not self.source.startswith("invalid:") and len(self.data) >= _MIN_PNG_BYTES
        )


@dataclass(frozen=True)
class _CaptionAnchor:
    figure_num: int
    rect: fitz.Rect
    page_index: int


def analyze_crop_text_topology(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    source: str,
) -> dict[str, Any]:
    """Относительные метрики текстового слоя внутри crop_bbox (без OCR)."""
    crop_area = max(1.0, crop_bbox.width * crop_bbox.height)
    try:
        words = page.get_text("words", clip=crop_bbox)
    except Exception as exc:
        logger.debug("VECTOR_PDF words clip failed | %s", exc)
        words = []

    words_area = 0.0
    char_total = 0
    word_count = 0
    line_buckets: dict[tuple[int, int], int] = {}

    for w in words:
        if len(w) < 5:
            continue
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        token = str(w[4])
        if not token.strip():
            continue
        words_area += max(0.0, (x1 - x0) * (y1 - y0))
        char_total += len(token)
        word_count += 1
        block_no = int(w[5]) if len(w) > 5 else 0
        line_no = int(w[6]) if len(w) > 6 else 0
        key = (block_no, line_no)
        line_buckets[key] = line_buckets.get(key, 0) + 1

    text_density = words_area / crop_area
    avg_word_length = (char_total / word_count) if word_count else 0.0
    if line_buckets:
        words_per_line = sum(line_buckets.values()) / float(len(line_buckets))
    else:
        words_per_line = 0.0

    density_threshold = (
        _TEXT_DENSITY_VECTOR_BOUND
        if source == "vector_bound"
        else _TEXT_DENSITY_DEFAULT
    )
    is_dense_paragraph = text_density > density_threshold or (
        words_per_line > _WORDS_PER_LINE_DENSE
        and avg_word_length > _AVG_WORD_LENGTH_DENSE
    )

    return {
        "text_density": round(text_density, 4),
        "words_area": round(words_area, 2),
        "crop_area": round(crop_area, 2),
        "avg_word_length": round(avg_word_length, 3),
        "words_per_line": round(words_per_line, 3),
        "word_count": word_count,
        "is_dense_paragraph": is_dense_paragraph,
        "density_threshold": density_threshold,
    }


def _rect_intersects_crop(rect: fitz.Rect, crop: fitz.Rect) -> bool:
    return (
        rect.x0 < crop.x1
        and rect.x1 > crop.x0
        and rect.y0 < crop.y1
        and rect.y1 > crop.y0
    )


def _word_rects_in_crop(page: fitz.Page, crop_bbox: fitz.Rect) -> list[fitz.Rect]:
    try:
        words = page.get_text("words", clip=crop_bbox)
    except Exception:
        return []
    rects: list[fitz.Rect] = []
    for w in words:
        if len(w) < 4:
            continue
        rects.append(fitz.Rect(float(w[0]), float(w[1]), float(w[2]), float(w[3])))
    return rects


def _drawing_matches_text_decoration(
    rect: fitz.Rect, word_rects: list[fitz.Rect]
) -> bool:
    if rect.is_empty or not word_rects:
        return False
    tol = _TEXT_DECORATION_TOL_PT
    for wr in word_rects:
        if not _rect_intersects_crop(rect, wr):
            continue
        inter = rect & wr
        if inter.is_empty:
            continue
        inter_area = inter.width * inter.height
        rect_area = max(1e-6, rect.width * rect.height)
        wr_area = max(1e-6, wr.width * wr.height)
        # Подчёркивание / rule под строкой
        if rect.height <= 3.5 and rect.width >= 8.0:
            if inter.width >= rect.width * 0.82:
                if wr.y1 - tol <= rect.y0 <= wr.y1 + tol * 2:
                    return True
        # Заливка/рамка вокруг глифа
        if inter_area / rect_area >= 0.92 and inter_area / wr_area >= 0.55:
            return True
        if (
            abs(rect.x0 - wr.x0) <= tol
            and abs(rect.x1 - wr.x1) <= tol
            and abs(rect.y0 - wr.y0) <= tol
            and abs(rect.y1 - wr.y1) <= tol
        ):
            return True
    return False


def _count_drawing_path_items(dr: dict[str, Any]) -> int:
    items = dr.get("items") or []
    n = 0
    for it in items:
        if not it:
            continue
        op = it[0]
        if op in ("l", "c", "qu", "re", "m", "v", "y", "h"):
            n += 1
    return n if n > 0 else 1


def _raster_xrefs_in_crop(page: fitz.Page, crop_bbox: fitz.Rect) -> list[int]:
    xrefs: list[int] = []
    page_area = page.rect.width * page.rect.height
    for img_info in page.get_images(full=True):
        xref = int(img_info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects or []:
            r = fitz.Rect(rect)
            if r.width * r.height > page_area * _MAX_FIGURE_PAGE_FRACTION:
                continue
            if _rect_intersects_crop(r, crop_bbox):
                xrefs.append(xref)
                break
    return sorted(set(xrefs))


def _grid_signature(
    rects: list[fitz.Rect], step: float = _RESIDUE_GRID_STEP
) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if step <= 0:
        step = 4.0
    for r in rects:
        if r.is_empty:
            continue
        x0 = int(r.x0 // step)
        x1 = int(r.x1 // step)
        y0 = int(r.y0 // step)
        y1 = int(r.y1 // step)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                cells.add((gx, gy))
    return frozenset(cells)


def _grid_similarity(
    a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]
) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / float(union) if union else 0.0


def _structural_signature_hash(
    drawing_cells: frozenset[tuple[int, int]],
    raster_xrefs: tuple[int, ...],
) -> str:
    payload = f"{sorted(drawing_cells)!r}|{list(raster_xrefs)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _raster_rects_in_crop(page: fitz.Page, crop_bbox: fitz.Rect) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    page_area = page.rect.width * page.rect.height
    for img_info in page.get_images(full=True):
        xref = int(img_info[0])
        try:
            placements = page.get_image_rects(xref)
        except Exception:
            placements = []
        for rect in placements or []:
            r = fitz.Rect(rect)
            if r.width * r.height > page_area * _MAX_FIGURE_PAGE_FRACTION:
                continue
            if _rect_intersects_crop(r, crop_bbox):
                rects.append(r)
    return rects


def _collect_non_text_drawing_rects(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    word_rects: list[fitz.Rect],
) -> list[fitz.Rect]:
    non_text_rects: list[fitz.Rect] = []
    try:
        all_drawings = page.get_drawings()
    except Exception as exc:
        logger.debug("VECTOR_PDF residue drawings failed | %s", exc)
        return non_text_rects
    for dr in all_drawings:
        rect = dr.get("rect")
        if not rect:
            continue
        r = fitz.Rect(rect)
        if r.is_empty or not _rect_intersects_crop(r, crop_bbox):
            continue
        if _drawing_matches_text_decoration(r, word_rects):
            continue
        if not _sanitize_visual_rect(page, r):
            continue
        non_text_rects.append(r)
    return non_text_rects


def _union_rects_list(rects: list[fitz.Rect]) -> fitz.Rect | None:
    if not rects:
        return None
    merged = rects[0]
    for r in rects[1:]:
        merged |= r
    return merged


@dataclass(frozen=True)
class _WordOnLine:
    rect: fitz.Rect
    block: int
    line: int
    token: str


def _crop_words_grouped(
    page: fitz.Page, clip: fitz.Rect
) -> dict[tuple[int, int], list[_WordOnLine]]:
    if clip.is_empty:
        return {}
    try:
        words = page.get_text("words", clip=clip)
    except Exception:
        return {}
    by_line: dict[tuple[int, int], list[_WordOnLine]] = {}
    for w in words:
        if len(w) < 7:
            continue
        rect = fitz.Rect(float(w[0]), float(w[1]), float(w[2]), float(w[3]))
        if rect.is_empty:
            continue
        key = (int(w[5]), int(w[6]))
        by_line.setdefault(key, []).append(
            _WordOnLine(
                rect=rect,
                block=int(w[5]),
                line=int(w[6]),
                token=str(w[4]).strip(),
            )
        )
    return by_line


def _line_union_rect(words: list[_WordOnLine]) -> fitz.Rect:
    merged = words[0].rect
    for item in words[1:]:
        merged |= item.rect
    return merged


def _is_full_width_paragraph_line(
    line_rect: fitz.Rect,
    word_count: int,
    column_width: float,
) -> bool:
    if line_rect.width > _TIGHT_BODY_LINE_WIDTH_PT:
        return True
    if (
        word_count >= int(_WORDS_PER_LINE_DENSE)
        and line_rect.width > column_width * 0.48
    ):
        return True
    return False


def _word_horizontally_attached_to_residue(
    word_rect: fitz.Rect,
    residue_bbox: fitz.Rect,
    *,
    axis_x_slack: float,
) -> bool:
    overlap = min(word_rect.x1, residue_bbox.x1) - max(word_rect.x0, residue_bbox.x0)
    if overlap > 0.5:
        return True
    slack = max(0.0, axis_x_slack)
    if (
        word_rect.x1 <= residue_bbox.x0 + 0.5
        and residue_bbox.x0 - word_rect.x1 <= slack
    ):
        if (
            word_rect.y1 >= residue_bbox.y0 - slack
            and word_rect.y0 <= residue_bbox.y1 + slack
        ):
            return True
    if (
        word_rect.x0 >= residue_bbox.x1 - 0.5
        and word_rect.x0 - residue_bbox.x1 <= slack
    ):
        if (
            word_rect.y1 >= residue_bbox.y0 - slack
            and word_rect.y0 <= residue_bbox.y1 + slack
        ):
            return True
    return False


def _estimate_body_font_size(page: fitz.Page, clip: fitz.Rect) -> float | None:
    try:
        d = page.get_text("dict", clip=clip)
    except Exception:
        return None
    sizes: list[float] = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            x0 = min(float(s["bbox"][0]) for s in spans)
            x1 = max(float(s["bbox"][2]) for s in spans)
            if x1 - x0 <= _TIGHT_BODY_LINE_WIDTH_PT:
                continue
            for span in spans:
                sz = float(span.get("size", 0) or 0)
                if _BODY_FONT_MAX_PT - 2.0 <= sz <= _BODY_FONT_MAX_PT + 1.5:
                    sizes.append(sz)
    if not sizes:
        return None
    sizes.sort()
    return sizes[len(sizes) // 2]


def _line_uses_body_font(
    page: fitz.Page,
    line_key: tuple[int, int],
    body_size: float | None,
) -> bool:
    if body_size is None:
        return False
    block_no, line_no = line_key
    try:
        d = page.get_text("dict")
    except Exception:
        return False
    text_blocks = [b for b in d.get("blocks", []) if b.get("type") == 0]
    if block_no < 0 or block_no >= len(text_blocks):
        return False
    lines = text_blocks[block_no].get("lines", [])
    if line_no < 0 or line_no >= len(lines):
        return False
    for span in lines[line_no].get("spans", []):
        sz = float(span.get("size", 0) or 0)
        if abs(sz - body_size) <= 0.6:
            return True
    return False


def _line_is_figure_caption_paragraph(line_text: str, line_rect: fitz.Rect) -> bool:
    t = line_text.strip()
    if re.match(r"^\s*Fig\.\s*\d", t, re.IGNORECASE) and line_rect.width > 120.0:
        return True
    if re.match(r"^\s*Figure\s+\d+\s*\.", t, re.IGNORECASE) and len(t) > 40:
        return True
    return False


def _line_is_article_text(
    page: fitz.Page,
    line_words: list[_WordOnLine],
    line_rect: fitz.Rect,
    column_width: float,
    body_font: float | None,
) -> bool:
    joined = " ".join(w.token for w in line_words if w.token)
    if _line_is_figure_caption_paragraph(joined, line_rect):
        return True
    if _is_full_width_paragraph_line(line_rect, len(line_words), column_width):
        return True
    if body_font is not None and _line_uses_body_font(
        page, (line_words[0].block, line_words[0].line), body_font
    ):
        if line_rect.width > 90.0 and len(line_words) >= 4:
            return True
    return False


def _visual_residue_geometry(
    page: fitz.Page, crop_bbox: fitz.Rect
) -> tuple[list[fitz.Rect], list[fitz.Rect], list[fitz.Rect]]:
    word_rects = _word_rects_in_crop(page, crop_bbox)
    drawings = _collect_non_text_drawing_rects(page, crop_bbox, word_rects)
    rasters = _raster_rects_in_crop(page, crop_bbox)
    return drawings, rasters, word_rects


def refine_bbox_by_residue(
    page: fitz.Page,
    initial_bbox: fitz.Rect,
    padding_pt: float = _TIGHT_LABEL_PADDING_PT,
) -> fitz.Rect:
    """Shrink-wrap вокруг визуального остатка + метки схемы (без абзацев статьи)."""
    _ = padding_pt  # ось Y: _TIGHT_AXIS_X_SLACK_PT
    drawings, rasters, _word_rects = _visual_residue_geometry(page, initial_bbox)
    drawings = [r for r in drawings if _sanitize_visual_rect(page, r)]
    rasters = [r for r in rasters if _sanitize_visual_rect(page, r)]
    residue_bbox = _union_rects_list(drawings + rasters)
    if residue_bbox is None or residue_bbox.is_empty:
        return initial_bbox
    if residue_bbox.width < 2.0 or residue_bbox.height < 2.0:
        return initial_bbox

    y_min = residue_bbox.y0
    y_max = residue_bbox.y1
    column_width = max(1.0, initial_bbox.width)
    body_font = _estimate_body_font_size(page, initial_bbox)

    y_band = fitz.Rect(
        initial_bbox.x0,
        y_min - _TIGHT_WORD_Y_BAND_PT,
        initial_bbox.x1,
        y_max + _TIGHT_WORD_Y_BAND_PT,
    )
    word_clip = initial_bbox & y_band
    lines = _crop_words_grouped(page, word_clip)

    expanded = fitz.Rect(residue_bbox)
    trim_top = False
    trim_bottom = False
    axis_slack = _TIGHT_AXIS_X_SLACK_PT

    for _key, line_words in lines.items():
        if not line_words:
            continue
        line_rect = _line_union_rect(line_words)
        is_article = _line_is_article_text(
            page, line_words, line_rect, column_width, body_font
        )

        if is_article:
            if line_rect.y1 < y_min + 0.5:
                trim_top = True
            elif line_rect.y0 > y_max - 0.5:
                trim_bottom = True
            continue

        for item in line_words:
            if not item.token or len(item.token) > 28:
                continue
            if not _word_horizontally_attached_to_residue(
                item.rect, residue_bbox, axis_x_slack=axis_slack
            ):
                continue
            if item.rect.y1 < y_band.y0 - 0.5 or item.rect.y0 > y_band.y1 + 0.5:
                continue
            expanded |= item.rect

    if trim_top:
        expanded.y0 = y_min - _TIGHT_BODY_GAP_CUT_PT
    if trim_bottom:
        expanded.y1 = y_max + _TIGHT_BODY_GAP_CUT_PT

    scheme_col = _scheme_column_rect(page, residue_bbox)
    expanded.x0 = max(expanded.x0, scheme_col.x0 + 2.0)
    expanded.x1 = min(expanded.x1, scheme_col.x1 - 2.0)

    final = fitz.Rect(
        expanded.x0 - _TIGHT_MARGIN_X_PT,
        expanded.y0 - _TIGHT_MARGIN_Y_PT,
        expanded.x1 + _TIGHT_MARGIN_X_PT,
        expanded.y1 + _TIGHT_MARGIN_Y_PT,
    )
    page_margin = fitz.Rect(
        page.rect.x0 + 6.0,
        page.rect.y0 + 6.0,
        page.rect.x1 - 6.0,
        page.rect.y1 - 6.0,
    )
    clipped = final & page_margin & scheme_col
    if clipped.is_empty or clipped.width < 2 or clipped.height < 2:
        return initial_bbox

    shrink = 1.0 - (clipped.width * clipped.height) / max(
        1.0, initial_bbox.width * initial_bbox.height
    )
    logger.debug(
        "VECTOR_PDF tight_crop | shrink=%.1f%% trim_top=%s trim_bottom=%s | "
        "%.0f×%.0f → %.0f×%.0f pt",
        shrink * 100.0,
        trim_top,
        trim_bottom,
        initial_bbox.width,
        initial_bbox.height,
        clipped.width,
        clipped.height,
    )
    return clipped


def check_visual_residue(page: fitz.Page, crop_bbox: fitz.Rect) -> dict[str, Any]:
    """Маскирование текстовых декораций; остаток = нетекстовая графика + растры."""
    word_rects = _word_rects_in_crop(page, crop_bbox)
    drawings, rasters, _words = _visual_residue_geometry(page, crop_bbox)
    raster_xrefs = tuple(_raster_xrefs_in_crop(page, crop_bbox))
    raster_count = len(raster_xrefs)

    non_text_rects = drawings
    non_text_primitive_count = 0
    try:
        all_drawings = page.get_drawings()
    except Exception as exc:
        logger.debug("VECTOR_PDF residue drawings failed | %s", exc)
        all_drawings = []
    for dr in all_drawings:
        rect = dr.get("rect")
        if not rect:
            continue
        r = fitz.Rect(rect)
        if r.is_empty or not _rect_intersects_crop(r, crop_bbox):
            continue
        if _drawing_matches_text_decoration(r, word_rects):
            continue
        non_text_primitive_count += _count_drawing_path_items(dr)

    drawing_cells = _grid_signature(non_text_rects)
    crop_area = max(1.0, crop_bbox.width * crop_bbox.height)
    residue_area = sum(max(0.0, r.width * r.height) for r in non_text_rects)
    residue_area_fraction = round(residue_area / crop_area, 4)

    has_visual_residue = non_text_primitive_count > 0 or raster_count > 0
    zero_residue = not has_visual_residue

    if zero_residue:
        logger.debug(
            "VECTOR_PDF residue | zero visual | words=%s primitives=0 rasters=0",
            len(word_rects),
        )

    return {
        "non_text_primitive_count": non_text_primitive_count,
        "raster_count": raster_count,
        "has_visual_residue": has_visual_residue,
        "zero_residue": zero_residue,
        "residue_area_fraction": residue_area_fraction,
        "word_box_count": len(word_rects),
        "drawing_signature": drawing_cells,
        "raster_xrefs": raster_xrefs,
        "structural_hash": _structural_signature_hash(drawing_cells, raster_xrefs),
    }


def _visual_residue_duplicate(
    residue: dict[str, Any],
    history: list[tuple[frozenset[tuple[int, int]], tuple[int, ...]]],
) -> bool:
    sig_d = residue.get("drawing_signature")
    sig_x = residue.get("raster_xrefs") or ()
    if not isinstance(sig_d, frozenset):
        sig_d = frozenset()
    for prev_d, prev_x in history:
        sim_d = _grid_similarity(sig_d, prev_d)
        if sig_x and prev_x:
            sim_x = 1.0 if sig_x == prev_x else 0.0
            combined = (sim_d + sim_x) * 0.5
        elif not sig_x and not prev_x:
            combined = sim_d
        else:
            combined = sim_d * 0.85
        if combined >= _DUPLICATE_SIMILARITY:
            logger.debug(
                "VECTOR_PDF duplicate visual | sim=%.3f hash=%s",
                combined,
                residue.get("structural_hash"),
            )
            return True
    return False


def classify_invalid_crop(
    page: fitz.Page,
    crop_bbox: fitz.Rect,
    source: str,
    *,
    residue_history: (
        list[tuple[frozenset[tuple[int, int]], tuple[int, ...]]] | None
    ) = None,
) -> tuple[str | None, dict[str, Any]]:
    """Геометрия + топология + visual residue; invalid:* без рендера PNG."""
    topology = analyze_crop_text_topology(page, crop_bbox, source)

    if crop_bbox.height < _MIN_CROP_PT:
        return "invalid:too_thin", topology
    if crop_bbox.width < _MIN_CROP_PT:
        return "invalid:too_narrow", topology

    if crop_bbox.height > 0 and crop_bbox.width > 0:
        aspect = crop_bbox.width / crop_bbox.height
        if aspect > _MAX_ASPECT_RATIO or aspect < (1.0 / _MAX_ASPECT_RATIO):
            return "invalid:aspect_ratio", topology

    if source == "caption_clip_fallback" and topology.get("is_dense_paragraph"):
        return "invalid:dense_text", topology

    residue = check_visual_residue(page, crop_bbox)
    topology["visual_residue"] = {
        k: v for k, v in residue.items() if k not in ("drawing_signature",)
    }

    if residue.get("zero_residue"):
        logger.warning(
            "VECTOR_PDF reject | zero_visual_residue | words=%s area_frac=%s",
            residue.get("word_box_count"),
            residue.get("residue_area_fraction"),
        )
        return "invalid:zero_visual_residue", topology

    if residue_history is not None and _visual_residue_duplicate(
        residue, residue_history
    ):
        logger.warning(
            "VECTOR_PDF reject | duplicate_visual | hash=%s",
            residue.get("structural_hash"),
        )
        return "invalid:duplicate_visual", topology

    if residue_history is not None:
        residue_history.append(
            (
                residue.get("drawing_signature") or frozenset(),
                residue.get("raster_xrefs") or (),
            )
        )

    return None, topology


class VectorPDFCropper:
    """Кроп диаграмм над подписью Fig. N (union визуалов или caption clip)."""

    def discover(self, pdf_bytes: bytes) -> dict[str, tuple[bytes, str]]:
        discovered = self.discover_detailed(pdf_bytes)
        return {k: v.as_payload() for k, v in discovered.items() if v.is_renderable}

    def discover_detailed(self, pdf_bytes: bytes) -> dict[str, FigureBinary]:
        if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
            return {}
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return self._discover_doc_detailed(doc)
        finally:
            doc.close()

    def discover_from_document(
        self, doc: fitz.Document
    ) -> dict[str, tuple[bytes, str]]:
        detailed = self.discover_from_document_detailed(doc)
        return {k: v.as_payload() for k, v in detailed.items() if v.is_renderable}

    def discover_from_document_detailed(
        self,
        doc: fitz.Document,
    ) -> dict[str, FigureBinary]:
        return self._discover_doc_detailed(doc)

    def resolve_figure(self, pdf_bytes: bytes, fig_id: str) -> tuple[bytes, str] | None:
        detailed = self.resolve_figure_detailed(pdf_bytes, fig_id)
        if detailed is None or not detailed.is_renderable:
            return None
        return detailed.as_payload()

    def resolve_figure_detailed(
        self, pdf_bytes: bytes, fig_id: str
    ) -> FigureBinary | None:
        discovered = self.discover_detailed(pdf_bytes)
        fid = (fig_id or "").strip().upper()
        if not fid.startswith("FIG_"):
            return None
        if fid in discovered:
            return discovered[fid]
        tail = fid[4:].lstrip("_")
        if tail.isdigit():
            alt = f"FIG_{int(tail)}"
            if alt in discovered:
                return discovered[alt]
            seq_key = f"FIG_SEQ_{int(tail)}"
            if seq_key in discovered:
                return discovered[seq_key]
        return None

    def resolve_figure_expanded(
        self, pdf_bytes: bytes, fig_id: str
    ) -> FigureBinary | None:
        """Расширенное окно вверх (только если кроп не invalid:*)."""
        if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
            return None
        fid = (fig_id or "").strip().upper()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            anchor = self._find_anchor_for_fig_id(doc, fid)
            if anchor is None:
                return self.resolve_figure_detailed(pdf_bytes, fid)
            page = doc[anchor.page_index]
            page_anchors = self._sorted_page_anchors(doc, anchor.page_index)
            idx = next(
                (
                    i
                    for i, a in enumerate(page_anchors)
                    if a.figure_num == anchor.figure_num
                    and abs(a.rect.y0 - anchor.rect.y0) < 2
                ),
                0,
            )
            return self._crop_near_caption(
                page,
                anchor,
                page_anchors=page_anchors,
                anchor_idx=idx,
                expand_band_up_pt=120.0,
            )
        finally:
            doc.close()

    def _find_anchor_for_fig_id(
        self, doc: fitz.Document, fid: str
    ) -> _CaptionAnchor | None:
        tail = fid[4:].lstrip("_") if fid.startswith("FIG_") else ""
        want_num: int | None = int(tail) if tail.isdigit() else None
        if fid.startswith("FIG_SEQ_"):
            seq = int(fid.split("_", 2)[-1])
            all_anchors: list[_CaptionAnchor] = []
            for page_index in range(doc.page_count):
                all_anchors.extend(self._caption_anchors(doc[page_index], page_index))
            all_anchors.sort(key=lambda a: (a.page_index, a.rect.y0, a.rect.x0))
            if 1 <= seq <= len(all_anchors):
                return all_anchors[seq - 1]
            return None
        for page_index in range(doc.page_count):
            for a in self._caption_anchors(doc[page_index], page_index):
                if want_num is not None and a.figure_num == want_num:
                    return a
        return None

    def _sorted_page_anchors(
        self, doc: fitz.Document, page_index: int
    ) -> list[_CaptionAnchor]:
        anchors = self._caption_anchors(doc[page_index], page_index)
        anchors.sort(key=lambda a: a.rect.y0)
        return anchors

    def _discover_doc_detailed(
        self,
        doc: fitz.Document,
    ) -> dict[str, FigureBinary]:
        anchors: list[_CaptionAnchor] = []
        for page_index, page in enumerate(doc):
            anchors.extend(self._caption_anchors(page, page_index))

        if not anchors:
            logger.debug("VECTOR_PDF ⊘ | no caption anchors")
            trace("VECTOR_PDF ⊘ | no caption anchors")
            return {}

        anchors.sort(key=lambda a: (a.page_index, a.rect.y0, a.rect.x0))
        out: dict[str, FigureBinary] = {}
        seen_nums: set[int] = set()
        seq = 0

        by_page: dict[int, list[_CaptionAnchor]] = {}
        for a in anchors:
            by_page.setdefault(a.page_index, []).append(a)

        residue_history: list[tuple[frozenset[tuple[int, int]], tuple[int, ...]]] = []

        for page_index, page_anchors in by_page.items():
            page = doc[page_index]
            page_anchors.sort(key=lambda a: a.rect.y0)

            def _anchor_sort_key(
                a: _CaptionAnchor,
                *,
                _page: fitz.Page,
            ) -> tuple[int, float, float]:
                line = _page.get_text("text", clip=a.rect).strip().split("\n")[0]
                formal = 0 if _is_figure_caption_line(line, a.figure_num) else 1
                return (formal, a.rect.y0, a.rect.x0)

            page_anchors.sort(key=lambda a, p=page: _anchor_sort_key(a, _page=p))
            for i, anchor in enumerate(page_anchors):
                fig = self._crop_near_caption(
                    page,
                    anchor,
                    page_anchors=page_anchors,
                    anchor_idx=i,
                    residue_history=residue_history,
                )
                if fig is None:
                    continue
                seq += 1
                out[f"FIG_SEQ_{seq}"] = fig
                if anchor.figure_num not in seen_nums:
                    seen_nums.add(anchor.figure_num)
                    out[f"FIG_{anchor.figure_num}"] = fig
                elif (
                    fig.is_renderable
                    and not out[f"FIG_{anchor.figure_num}"].is_renderable
                ):
                    out[f"FIG_{anchor.figure_num}"] = fig

        renderable = sum(1 for f in out.values() if f.is_renderable)
        if out:
            trace(
                f"VECTOR_PDF ✓ | figures={len(out)} renderable={renderable} "
                f"keys={sorted(out.keys())[:12]}"
            )
        return out

    def _caption_in_column(self, col: fitz.Rect, rect: fitz.Rect) -> bool:
        cx = (rect.x0 + rect.x1) * 0.5
        return col.x0 - 4 <= cx <= col.x1 + 4

    def _drawing_count_in_band(
        self,
        page: fitz.Page,
        col: fitz.Rect,
        y_top: float,
        y_bottom: float,
    ) -> int:
        if y_bottom <= y_top:
            return 0
        return len(self._collect_drawings(page, y_top, y_bottom, col))

    def _caption_below_figure_layout(
        self,
        page: fitz.Page,
        caption: fitz.Rect,
        col: fitz.Rect,
    ) -> bool:
        above_h = max(40.0, caption.y0 - col.y0 - 20.0)
        below_h = max(40.0, col.y1 - caption.y1 - 20.0)
        above_top = max(col.y0 + 8.0, caption.y0 - above_h)
        below_bottom = min(col.y1 - 8.0, caption.y1 + below_h)
        above_n = self._drawing_count_in_band(
            page, col, above_top, caption.y0 - _CAPTION_GAP_PT
        )
        below_n = self._drawing_count_in_band(
            page, col, caption.y1 + _CAPTION_GAP_PT, below_bottom
        )
        if below_n >= 3 and below_n > above_n + 2:
            return False
        if above_n >= 3 and above_n >= below_n:
            return True
        return caption.y0 > page.rect.y0 + page.rect.height * 0.38

    def _y_bottom_below_caption(
        self,
        page: fitz.Page,
        anchor: _CaptionAnchor,
        page_anchors: list[_CaptionAnchor],
        anchor_idx: int,
        col: fitz.Rect,
    ) -> float:
        y_bottom = min(col.y1, page.rect.y1) - 12.0
        for j in range(anchor_idx + 1, len(page_anchors)):
            nxt = page_anchors[j]
            if not self._caption_in_column(col, nxt.rect):
                continue
            if nxt.figure_num == anchor.figure_num:
                continue
            if nxt.rect.y0 < anchor.rect.y1 + 48.0:
                continue
            y_bottom = min(y_bottom, nxt.rect.y0 - _CAPTION_GAP_PT)
            break
        return y_bottom

    def _compute_vertical_window(
        self,
        page: fitz.Page,
        anchor: _CaptionAnchor,
        page_anchors: list[_CaptionAnchor],
        anchor_idx: int,
    ) -> tuple[float, float, fitz.Rect]:
        caption = anchor.rect
        col = self._column_rect(page, caption)

        if not self._caption_below_figure_layout(page, caption, col):
            y_top = caption.y1 + _CAPTION_GAP_PT
            y_bottom = self._y_bottom_below_caption(
                page, anchor, page_anchors, anchor_idx, col
            )
            y_bottom = min(
                y_bottom,
                caption.y1 + _MAX_FIGURE_HEIGHT_VECTOR_PT,
            )
            y_top = self._refine_y_top_headers_and_body(
                page,
                col,
                fitz.Rect(caption.x0, y_top, caption.x1, y_bottom),
                y_top,
                y_bottom,
            )
            return y_top, y_bottom, col

        y_bottom = caption.y0 - _CAPTION_GAP_PT

        y_top = col.y0 + 8.0
        y_top = max(y_top, page.rect.y0 + _PAGE_TOP_MARGIN_PT)

        for j in range(anchor_idx - 1, -1, -1):
            prev = page_anchors[j]
            if not self._caption_in_column(col, prev.rect):
                continue
            if prev.rect.y1 > caption.y0 - 6.0:
                continue
            y_top = max(y_top, prev.rect.y1 + 2.0)
            break

        y_top = self._bump_y_top_below_body_above_caption(page, col, caption, y_top)
        y_top = self._refine_y_top_headers_and_body(page, col, caption, y_top, y_bottom)
        max_band = min(520.0, max(180.0, col.height * 0.72))
        y_top = max(y_top, caption.y0 - max_band)
        if y_top >= y_bottom - 10.0:
            y_top = max(col.y0 + 8.0, caption.y0 - max_band)
        return y_top, y_bottom, col

    def _bump_y_top_below_body_above_caption(
        self,
        page: fitz.Page,
        col: fitz.Rect,
        caption: fitz.Rect,
        y_top: float,
    ) -> float:
        """Начало окна — ниже последнего абзаца над схемой (не верх колонки)."""
        last_para_y1: float | None = None
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox") or (0, 0, 0, 0)
            block_rect = fitz.Rect(bbox)
            if block_rect.y1 >= caption.y0 - 12.0:
                continue
            if block_rect.x1 < col.x0 + 4 or block_rect.x0 > col.x1 - 4:
                continue
            text_parts: list[str] = []
            max_font = 0.0
            for line in block.get("lines", []):
                for s in line.get("spans", []):
                    max_font = max(max_font, float(s.get("size", 0) or 0))
                    text_parts.append(str(s.get("text", "")))
            text = " ".join(text_parts).strip()
            if len(text) < 24 or max_font > _BODY_FONT_MAX_PT:
                continue
            if last_para_y1 is None or block_rect.y1 > last_para_y1:
                last_para_y1 = block_rect.y1
        if last_para_y1 is not None:
            y_top = max(y_top, last_para_y1 + 6.0)
        return y_top

    def _refine_y_top_headers_and_body(
        self,
        page: fitz.Page,
        col: fitz.Rect,
        caption: fitz.Rect,
        y_top: float,
        y_bottom: float,
    ) -> float:
        """Сдвинуть y_top ниже секционных заголовков; не поднимать выше первого абзаца."""
        first_body_y0: float | None = None
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox") or (0, 0, 0, 0)
            block_rect = fitz.Rect(bbox)
            if block_rect.y1 <= y_top or block_rect.y0 >= y_bottom:
                continue
            if block_rect.x1 < col.x0 + 4 or block_rect.x0 > col.x1 - 4:
                continue
            if block_rect.y0 >= caption.y0 - 4:
                continue

            line_texts: list[str] = []
            max_font = 0.0
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                parts = [str(s.get("text", "")) for s in spans]
                line_text = "".join(parts).strip()
                if line_text:
                    line_texts.append(line_text)
                for s in spans:
                    max_font = max(max_font, float(s.get("size", 0) or 0))

            text = " ".join(line_texts).strip()
            if not text:
                continue

            is_header = max_font >= _HEADER_FONT_PT or (
                max_font >= _BODY_FONT_MAX_PT - 0.5 and _SECTION_HEADING_RE.match(text)
            )
            if is_header:
                y_top = max(y_top, block_rect.y1 + 4.0)
                logger.debug(
                    "VECTOR_PDF y_top below header | y=%.1f | %s",
                    y_top,
                    text[:60],
                )
                continue

            if max_font <= _BODY_FONT_MAX_PT and len(text) >= 18:
                if first_body_y0 is None or block_rect.y0 < first_body_y0:
                    first_body_y0 = block_rect.y0

        if first_body_y0 is not None and first_body_y0 < caption.y0 - 6.0:
            y_top = max(y_top, first_body_y0)
        return min(y_top, caption.y0 - 40.0)

    @staticmethod
    def _expand_rect(rect: fitz.Rect, pad: float) -> fitz.Rect:
        return fitz.Rect(
            rect.x0 - pad,
            rect.y0 - pad,
            rect.x1 + pad,
            rect.y1 + pad,
        )

    @staticmethod
    def _rect_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
        return a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0

    def _expand_drawings_bbox_with_nearby_words(
        self,
        page: fitz.Page,
        drawings_bbox: fitz.Rect,
    ) -> fitz.Rect:
        probe = self._expand_rect(drawings_bbox, _DRAWING_LABEL_PROXIMITY_PT)
        merged = fitz.Rect(drawings_bbox)
        clip = self._expand_rect(drawings_bbox, _DRAWING_LABEL_PROXIMITY_PT + 6.0)
        lines = _crop_words_grouped(page, clip)
        column_width = max(1.0, drawings_bbox.width)
        body_font = _estimate_body_font_size(page, clip)
        for line_words in lines.values():
            if not line_words:
                continue
            line_rect = _line_union_rect(line_words)
            if _line_is_article_text(
                page, line_words, line_rect, column_width, body_font
            ):
                continue
            for item in line_words:
                if not item.token or len(item.token) > 28:
                    continue
                if not self._rect_intersects(item.rect, probe):
                    continue
                if not _word_horizontally_attached_to_residue(
                    item.rect,
                    drawings_bbox,
                    axis_x_slack=_TIGHT_AXIS_X_SLACK_PT,
                ):
                    continue
                merged |= item.rect
        return merged

    @staticmethod
    def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect | None:
        if not rects:
            return None
        merged = rects[0]
        for r in rects[1:]:
            merged |= r
        return merged

    def _caption_anchors(
        self, page: fitz.Page, page_index: int
    ) -> list[_CaptionAnchor]:
        anchors: list[_CaptionAnchor] = []
        seen: set[tuple[int, int, int]] = set()

        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            parts: list[str] = []
            line_rects: list[fitz.Rect] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(str(s.get("text", "")) for s in spans).strip()
                if not line_text:
                    continue
                parts.append(line_text)
                bbox = line.get("bbox")
                if bbox:
                    line_rects.append(fitz.Rect(bbox))
            text = " ".join(parts).strip()
            m = _FIG_NUM_RE.search(text)
            if not m or not line_rects:
                continue
            formal: list[tuple[int, fitz.Rect]] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(str(s.get("text", "")) for s in spans).strip()
                if not line_text:
                    continue
                lm = _FIG_NUM_RE.search(line_text)
                if not lm:
                    continue
                fig_n = int(lm.group(1))
                if not _is_figure_caption_line(line_text, fig_n):
                    continue
                bbox = line.get("bbox")
                if bbox:
                    formal.append((fig_n, fitz.Rect(bbox)))
            if not formal:
                continue
            fig_num = formal[0][0]
            caption_rect = fitz.Rect(formal[0][1])
            for fn, r in formal[1:]:
                if fn == fig_num:
                    caption_rect |= r
            merged = caption_rect
            key = (page_index, fig_num, int(merged.y0 // 3))
            if key in seen:
                continue
            seen.add(key)
            anchors.append(_CaptionAnchor(fig_num, merged, page_index))

        for fig_num in range(1, 40):
            for label in (f"Fig. {fig_num}", f"Fig.{fig_num}"):
                try:
                    rects = page.search_for(label)
                except Exception:
                    rects = []
                for rect in rects or []:
                    key = (page_index, fig_num, int(rect.y0 // 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    anchors.append(_CaptionAnchor(fig_num, fitz.Rect(rect), page_index))
        return anchors

    def _column_rect(self, page: fitz.Page, caption: fitz.Rect) -> fitz.Rect:
        pr = page.rect
        mid = pr.x0 + pr.width * 0.5
        cx = (caption.x0 + caption.x1) * 0.5
        margin = 6.0
        if cx >= mid - 12:
            return fitz.Rect(mid - margin, pr.y0, pr.x1, pr.y1)
        return fitz.Rect(pr.x0, pr.y0, mid + margin, pr.y1)

    @staticmethod
    def _intersects_vertical_band(
        rect: fitz.Rect, y_top: float, y_bottom: float
    ) -> bool:
        if y_bottom <= y_top:
            return False
        return rect.y0 < y_bottom and rect.y1 > y_top

    def _collect_drawings(
        self,
        page: fitz.Page,
        y_top: float,
        y_bottom: float,
        col: fitz.Rect,
    ) -> list[fitz.Rect]:
        try:
            drawings = page.get_drawings()
        except Exception as exc:
            logger.debug("VECTOR_PDF drawings failed | %s", exc)
            drawings = []
        selected: list[fitz.Rect] = []
        for dr in drawings:
            rect = dr.get("rect")
            if not rect:
                continue
            r = fitz.Rect(rect)
            if r.width < _MIN_DRAW_W or r.height < _MIN_DRAW_H:
                continue
            if not _sanitize_visual_rect(page, r):
                continue
            if not self._intersects_vertical_band(r, y_top, y_bottom):
                continue
            if r.x1 < col.x0 + 6 or r.x0 > col.x1 - 6:
                continue
            selected.append(r)

        if not selected:
            return selected

        span_y0 = min(r.y0 for r in selected)
        span_y0 = max(page.rect.y0 + 4.0, span_y0 - 120.0)
        extended: list[fitz.Rect] = list(selected)
        for dr in drawings:
            rect = dr.get("rect")
            if not rect:
                continue
            r = fitz.Rect(rect)
            if r.width < _MIN_DRAW_W or r.height < _MIN_DRAW_H:
                continue
            if not _sanitize_visual_rect(page, r):
                continue
            if r.y1 < span_y0 or r.y0 > y_bottom:
                continue
            if r.x1 < col.x0 + 6 or r.x0 > col.x1 - 6:
                continue
            if r not in extended:
                extended.append(r)
        return extended

    @staticmethod
    def _rect_close(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
        expanded = fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap)
        return expanded.intersects(b)

    def _merge_connected_drawings(
        self,
        page: fitz.Page,
        col: fitz.Rect,
        seed: list[fitz.Rect],
        y_bottom: float,
        *,
        max_gap: float = 36.0,
    ) -> list[fitz.Rect]:
        if not seed:
            return seed
        try:
            drawings = page.get_drawings()
        except Exception:
            return seed
        pool: list[fitz.Rect] = list(seed)
        merged = self._union_rects(pool)
        if merged is None:
            return seed
        changed = True
        while changed:
            changed = False
            for dr in drawings:
                rect = dr.get("rect")
                if not rect:
                    continue
                r = fitz.Rect(rect)
                if r.width < _MIN_DRAW_W or r.height < _MIN_DRAW_H:
                    continue
                if not _sanitize_visual_rect(page, r):
                    continue
                if r.y1 > y_bottom + 2 or r.y0 < merged.y0 - 160:
                    continue
                if r.x1 < col.x0 + 6 or r.x0 > col.x1 - 6:
                    continue
                if r in pool:
                    continue
                if self._rect_close(merged, r, max_gap):
                    pool.append(r)
                    merged |= r
                    changed = True
        return pool

    def _collect_diagram_label_word_rects(
        self,
        page: fitz.Page,
        y_top: float,
        y_bottom: float,
        col: fitz.Rect,
    ) -> list[fitz.Rect]:
        """Текстовые оси/метки схемы, когда get_drawings() пуст."""
        clip = fitz.Rect(page.rect.x0 + 8.0, y_top, page.rect.x1 - 8.0, y_bottom)
        try:
            words = page.get_text("words", clip=clip)
        except Exception:
            return []
        by_line: dict[tuple[int, int], list[tuple]] = {}
        for w in words:
            if len(w) < 5:
                continue
            key = (int(w[5]), int(w[6]))
            by_line.setdefault(key, []).append(w)

        rects: list[fitz.Rect] = []
        for _key, line_words in by_line.items():
            if len(line_words) > 9:
                continue
            avg_len = sum(len(str(w[4])) for w in line_words) / float(len(line_words))
            if avg_len > 14:
                continue
            for w in line_words:
                token = str(w[4]).strip()
                if not token or len(token) > 20:
                    continue
                rects.append(
                    fitz.Rect(float(w[0]), float(w[1]), float(w[2]), float(w[3]))
                )
        return rects

    def _collect_raster_rects(
        self,
        page: fitz.Page,
        y_top: float,
        y_bottom: float,
        col: fitz.Rect,
    ) -> list[fitz.Rect]:
        found: list[fitz.Rect] = []
        page_area = page.rect.width * page.rect.height

        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 1:
                continue
            bbox = block.get("bbox") or (0, 0, 0, 0)
            r = fitz.Rect(bbox)
            if r.width * r.height > page_area * _MAX_FIGURE_PAGE_FRACTION:
                continue
            if not self._intersects_vertical_band(r, y_top, y_bottom):
                continue
            if r.x1 < col.x0 + 6 or r.x0 > col.x1 - 6:
                continue
            found.append(r)

        for img_info in page.get_images(full=True):
            xref = int(img_info[0])
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects or []:
                r = fitz.Rect(rect)
                if r.width * r.height > page_area * _MAX_FIGURE_PAGE_FRACTION:
                    continue
                if not self._intersects_vertical_band(r, y_top, y_bottom):
                    continue
                if r.x1 < col.x0 + 6 or r.x0 > col.x1 - 6:
                    continue
                found.append(r)
        return found

    def _build_crop_bbox(
        self,
        page: fitz.Page,
        *,
        col: fitz.Rect,
        y_top: float,
        y_bottom: float,
        drawing_rects: list[fitz.Rect],
        raster_rects: list[fitz.Rect],
        label_rects: list[fitz.Rect] | None = None,
    ) -> tuple[fitz.Rect, str]:
        drawings_bbox = self._union_rects(drawing_rects)
        if drawings_bbox is None and label_rects:
            drawings_bbox = self._union_rects(label_rects)
        if drawings_bbox is not None:
            merged = self._expand_drawings_bbox_with_nearby_words(page, drawings_bbox)
            for r in raster_rects:
                if self._rect_intersects(r, merged) or self._intersects_vertical_band(
                    r, merged.y0, merged.y1
                ):
                    merged |= r
            crop = fitz.Rect(
                max(page.rect.x0 + 6.0, merged.x0 - _MARGIN_X_PT),
                max(page.rect.y0 + 6.0, merged.y0 - _MARGIN_Y_PT),
                min(page.rect.x1 - 6.0, merged.x1 + _MARGIN_X_PT),
                min(y_bottom, merged.y1 + _MARGIN_Y_PT),
            )
            return crop, "vector_bound"

        visual_rects = raster_rects
        if visual_rects:
            merged = self._union_rects(visual_rects)
            assert merged is not None
            crop = fitz.Rect(
                max(page.rect.x0 + 6.0, merged.x0 - _MARGIN_X_PT),
                max(page.rect.y0 + 6.0, merged.y0 - _MARGIN_Y_PT),
                min(page.rect.x1 - 6.0, merged.x1 + _MARGIN_X_PT),
                min(y_bottom, merged.y1 + _MARGIN_Y_PT),
            )
            return crop, "vector_bound"

        return fitz.Rect(col.x0, y_top, col.x1, y_bottom), "caption_clip_fallback"

    def _render_clip(
        self,
        page: fitz.Page,
        clip: fitz.Rect,
        *,
        figure_num: int,
        page_index: int,
        source: str,
        topology: dict[str, Any],
    ) -> FigureBinary | None:
        page_rect = page.rect
        max_h = (
            _MAX_FIGURE_HEIGHT_VECTOR_PT
            if source == "vector_bound"
            else min(
                _MAX_FIGURE_HEIGHT_PT,
                page_rect.height * _MAX_FIGURE_PAGE_FRACTION,
            )
        )
        if source != "vector_bound" and clip.height > max_h:
            clip = fitz.Rect(clip.x0, clip.y1 - max_h, clip.x1, clip.y1)
        elif source == "vector_bound" and clip.height > max_h:
            logger.debug(
                "VECTOR_PDF tall vector_bound | Fig.%s p%s | %.0f pt (no top trim)",
                figure_num,
                page_index + 1,
                clip.height,
            )

        scale = _RENDER_DPI / 72.0
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                clip=clip,
                alpha=False,
            )
            png = pix.tobytes("png")
        except Exception as exc:
            logger.warning(
                "VECTOR_PDF ✗ | render Fig.%s p%s | %s",
                figure_num,
                page_index + 1,
                exc,
            )
            trace(f"VECTOR_PDF ✗ | render Fig.{figure_num} | {exc}")
            return None
        if len(png) < _MIN_PNG_BYTES:
            return None
        trace(
            f"VECTOR_PDF ✓ | Fig.{figure_num} p{page_index + 1} "
            f"| {source} | {len(png)} bytes | "
            f"{int(clip.width)}×{int(clip.height)} pt"
        )
        return FigureBinary(png, "image/png", source, topology)

    def _crop_near_caption(
        self,
        page: fitz.Page,
        anchor: _CaptionAnchor,
        *,
        page_anchors: list[_CaptionAnchor],
        anchor_idx: int,
        expand_band_up_pt: float = 0.0,
        residue_history: (
            list[tuple[frozenset[tuple[int, int]], tuple[int, ...]]] | None
        ) = None,
    ) -> FigureBinary | None:
        y_top, y_bottom, col = self._compute_vertical_window(
            page, anchor, page_anchors, anchor_idx
        )
        if expand_band_up_pt > 0:
            y_top = max(page.rect.y0 + 8.0, y_top - expand_band_up_pt)

        if not self._caption_below_figure_layout(page, anchor.rect, col):
            probe_bottom = min(
                col.y1 - 12.0,
                anchor.rect.y1 + _MAX_FIGURE_HEIGHT_VECTOR_PT,
            )
            probe_dr = self._collect_drawings(page, y_top, probe_bottom, col)
            probe_dr = self._merge_connected_drawings(page, col, probe_dr, probe_bottom)
            if probe_dr:
                extent = max(r.y1 for r in probe_dr) + _MARGIN_Y_PT
                y_bottom = min(y_bottom, extent)
                y_bottom = max(y_bottom, y_top + 48.0)
            else:
                y_bottom = min(y_bottom, anchor.rect.y1 + 240.0)
                y_bottom = max(y_bottom, y_top + 48.0)

        if y_bottom <= y_top + 8:
            logger.debug(
                "VECTOR_PDF ⊘ | Fig.%s p%s | empty vertical band",
                anchor.figure_num,
                anchor.page_index + 1,
            )
            return None

        drawings = self._collect_drawings(page, y_top, y_bottom, col)
        drawings = self._merge_connected_drawings(page, col, drawings, y_bottom)
        raster_rects = self._collect_raster_rects(page, y_top, y_bottom, col)
        label_rects: list[fitz.Rect] = []
        if not drawings and not raster_rects:
            label_rects = self._collect_diagram_label_word_rects(
                page, y_top, y_bottom, col
            )

        crop_bbox, source = self._build_crop_bbox(
            page,
            col=col,
            y_top=y_top,
            y_bottom=y_bottom,
            drawing_rects=drawings,
            raster_rects=raster_rects,
            label_rects=label_rects,
        )

        if source == "vector_bound" and drawings:
            draw_max = max(r.y1 for r in drawings)
            if draw_max + _MARGIN_Y_PT > crop_bbox.y1:
                crop_bbox = fitz.Rect(
                    crop_bbox.x0,
                    crop_bbox.y0,
                    crop_bbox.x1,
                    min(y_bottom, draw_max + _MARGIN_Y_PT),
                )

        invalid, topology = classify_invalid_crop(
            page,
            crop_bbox,
            source,
            residue_history=residue_history,
        )
        if invalid:
            logger.warning(
                "VECTOR_PDF reject | Fig.%s p%s | %s | density=%.3f "
                "avg_len=%.2f wpl=%.2f",
                anchor.figure_num,
                anchor.page_index + 1,
                invalid,
                topology.get("text_density", 0),
                topology.get("avg_word_length", 0),
                topology.get("words_per_line", 0),
            )
            trace(
                f"VECTOR_PDF reject | Fig.{anchor.figure_num} "
                f"p{anchor.page_index + 1} | {invalid}"
            )
            return FigureBinary(b"", "", invalid, topology)

        vr = topology.get("visual_residue") or {}
        if not vr.get("zero_residue"):
            tight = refine_bbox_by_residue(page, crop_bbox)
            if tight.width >= _MIN_CROP_PT and tight.height >= _MIN_CROP_PT:
                crop_bbox = tight
                topology["tight_crop"] = True
                topo_refresh = analyze_crop_text_topology(page, crop_bbox, source)
                topology.update(topo_refresh)

        if source == "vector_bound" and topology.get("is_dense_paragraph"):
            density = float(topology.get("text_density", 0) or 0)
            wpl = float(topology.get("words_per_line", 0) or 0)
            if wpl >= _WORDS_PER_LINE_DENSE and density > _TEXT_DENSITY_VECTOR_BOUND:
                logger.warning(
                    "VECTOR_PDF reject | Fig.%s p%s | invalid:dense_text | "
                    "density=%.3f wpl=%.2f",
                    anchor.figure_num,
                    anchor.page_index + 1,
                    density,
                    wpl,
                )
                return FigureBinary(
                    b"",
                    "",
                    "invalid:dense_text",
                    topology,
                )

        if source == "vector_bound":
            logger.debug(
                "VECTOR_PDF vector_bound | Fig.%s p%s | paths=%s rasters=%s",
                anchor.figure_num,
                anchor.page_index + 1,
                len(drawings),
                len(raster_rects),
            )
        else:
            logger.debug(
                "VECTOR_PDF caption_clip_fallback | Fig.%s p%s | " "band=%.0f×%.0f pt",
                anchor.figure_num,
                anchor.page_index + 1,
                crop_bbox.width,
                crop_bbox.height,
            )

        return self._render_clip(
            page,
            crop_bbox,
            figure_num=anchor.figure_num,
            page_index=anchor.page_index,
            source=source,
            topology=topology,
        )


def export_status_label(source: str, topology: dict[str, Any] | None) -> str:
    topo = topology or {}
    density = topo.get("text_density")
    avg_len = topo.get("avg_word_length")
    wpl = topo.get("words_per_line")

    if source.startswith("invalid:"):
        reason = source.split(":", 1)[-1]
        if reason in ("zero_visual_residue", "duplicate_visual"):
            vr = topo.get("visual_residue") or {}
            return f"REJECTED: {reason} (hash={vr.get('structural_hash', '—')})"
        return f"REJECTED: {reason} (density={density}, avg_len={avg_len})"

    if source == "xref":
        return "OK:xref (no topology pass)"

    if not topo:
        return "OK:unscored"

    if topo.get("is_dense_paragraph"):
        return (
            f"WARN:dense_paragraph (density={density}, wpl={wpl}, "
            f"avg_len={avg_len})"
        )

    if source == "vector_bound" and isinstance(density, (int, float)):
        if density > _TEXT_DENSITY_DEFAULT:
            return f"WARN:text_density (density={density}, wpl={wpl})"

    return "OK"


def _as_png_bytes(raw: bytes, ext: str | None) -> bytes:
    if raw[:4] == b"\x89PNG" or raw[:3] == b"\xff\xd8\xff":
        return raw
    try:
        pix = fitz.Pixmap(raw)
        return pix.tobytes("png")
    except Exception:
        return raw


def png_pixel_size(png: bytes) -> tuple[int, int] | None:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width, height = struct.unpack(">II", png[16:24])
    except struct.error:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def is_invalid_vlm_crop(png: bytes, *, max_aspect: float = 8.0) -> bool:
    dims = png_pixel_size(png)
    if dims is None:
        return False
    width, height = dims
    if height < 40:
        return True
    aspect = width / float(height)
    if aspect > max_aspect:
        return True
    inv = height / float(width)
    return inv > max_aspect
