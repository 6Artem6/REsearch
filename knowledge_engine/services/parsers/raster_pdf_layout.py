"""Растровые страницы без текстового слоя: колонки → вертикальные полосы схем."""

from __future__ import annotations

import fitz
import numpy as np

from knowledge_engine.ui.run_log import trace

_RENDER_DPI = 132
_MIN_BAND_HEIGHT_PX = 55
_MIN_GAP_PX = 10
_INK_THRESHOLD = 0.018
_GAP_THRESHOLD = 0.008
_COLUMN_MARGIN_PX = 8
_MAX_BAND_PAGE_FRACTION = 0.38
_MIN_BAND_PAGE_FRACTION = 0.06


def discover_raster_column_figures(page: fitz.Page) -> list[tuple[fitz.Rect, bytes]]:
    """PNG-кропы схем в колонке, если в PDF нет extractable text."""
    text = (page.get_text() or "").strip()
    if len(text) > 120:
        return []

    page_area = page.rect.width * page.rect.height
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(int(img[0]))
        except Exception:
            rects = []
        if rects and rects[0].width * rects[0].height < page_area * 0.55:
            return []

    scale = _RENDER_DPI / 72.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    except Exception:
        return []
    if pix.width < 80 or pix.height < 80:
        return []

    samples = np.frombuffer(pix.samples, dtype=np.uint8)
    h, w = pix.height, pix.width
    rgb = samples.reshape(h, w, pix.n)
    if pix.n >= 3:
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )
    else:
        gray = rgb[:, :, 0].astype(np.float32)
    ink = gray < 235.0

    mid = w // 2
    columns = [
        (0, max(1, mid - _COLUMN_MARGIN_PX)),
        (min(w - 1, mid + _COLUMN_MARGIN_PX), w),
    ]

    out: list[tuple[fitz.Rect, bytes]] = []
    for x0, x1 in columns:
        if x1 - x0 < 40:
            continue
        col_ink = ink[:, x0:x1].mean(axis=1)
        window = 11
        kernel = np.ones(window) / window
        smooth = np.convolve(col_ink, kernel, mode="same")
        bands = _vertical_bands(smooth, h)
        for y0_px, y1_px in bands:
            if y1_px - y0_px < _MIN_BAND_HEIGHT_PX:
                continue
            if (y1_px - y0_px) > h * 0.88:
                sub = _vertical_bands(
                    smooth[y0_px:y1_px], y1_px - y0_px, aggressive=True
                )
                if len(sub) > 1:
                    for sy0, sy1 in sub:
                        gy0 = y0_px + sy0
                        gy1 = y0_px + sy1
                        if (gy1 - gy0) < h * _MIN_BAND_PAGE_FRACTION:
                            continue
                        clip = _px_rect_to_page(page, x0, gy0, x1, gy1, scale)
                        png = _clip_png(page, clip, scale)
                        if png:
                            out.append((clip, png))
                    continue
            if (y1_px - y0_px) > h * _MAX_BAND_PAGE_FRACTION:
                continue
            if (y1_px - y0_px) < h * _MIN_BAND_PAGE_FRACTION:
                continue
            clip = _px_rect_to_page(page, x0, y0_px, x1, y1_px, scale)
            png = _clip_png(page, clip, scale)
            if png:
                out.append((clip, png))

    if out:
        trace(
            f"RASTER_LAYOUT ✓ | p{page.number + 1} bands={len(out)} " f"(no text layer)"
        )
    return out


def _vertical_bands(
    smooth: np.ndarray,
    height: int,
    *,
    aggressive: bool = False,
) -> list[tuple[int, int]]:
    gap_thr = _GAP_THRESHOLD * (0.55 if aggressive else 1.0)
    ink_thr = _INK_THRESHOLD * (0.8 if aggressive else 1.0)
    min_band = 40 if aggressive else _MIN_BAND_HEIGHT_PX

    bands: list[tuple[int, int]] = []
    i = 0
    n = len(smooth)
    while i < n:
        if smooth[i] < gap_thr:
            i += 1
            continue
        start = i
        while i < n and smooth[i] >= gap_thr:
            i += 1
        end = i
        if end - start < min_band or smooth[start:end].mean() < ink_thr:
            continue
        bands.append((start, end))
    return bands


def _px_rect_to_page(
    page: fitz.Page,
    x0_px: int,
    y0_px: int,
    x1_px: int,
    y1_px: int,
    scale: float,
) -> fitz.Rect:
    inv = 72.0 / _RENDER_DPI
    pr = page.rect
    return fitz.Rect(
        max(pr.x0, x0_px * inv),
        max(pr.y0, y0_px * inv),
        min(pr.x1, x1_px * inv),
        min(pr.y1, y1_px * inv),
    )


def _clip_png(page: fitz.Page, clip: fitz.Rect, scale: float) -> bytes | None:
    if clip.width < 28 or clip.height < 28:
        return None
    page_area = page.rect.width * page.rect.height
    if clip.width * clip.height > page_area * _MAX_BAND_PAGE_FRACTION:
        return None
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=clip,
            alpha=False,
        )
        png = pix.tobytes("png")
    except Exception:
        return None
    if len(png) < 1200:
        return None
    return png
