"""Фильтрация визуального мусора из извлечённых изображений."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import imagehash
from PIL import Image

MIN_WIDTH = 150
MIN_HEIGHT = 150
MAX_ASPECT_RATIO = 4.0
PHASH_DUP_THRESHOLD = 4

_HEADER_FRAC = 0.08
_FOOTER_FRAC = 0.92


@dataclass
class LayoutHint:
    """Подсказки для layout-фильтра (PDF rect). HTML layout не используется."""

    page_height: float | None = None
    rect_top: float | None = None
    rect_bottom: float | None = None
    html_ancestors: list[str] = field(default_factory=list)
    html_classes: list[str] = field(default_factory=list)


def get_image_phash(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    return str(imagehash.phash(img))


def is_duplicate(
    new_hash: str, seen_hashes: set[str], threshold: int = PHASH_DUP_THRESHOLD
) -> bool:
    new_h = imagehash.hex_to_hash(new_hash)
    for h_str in seen_hashes:
        existing_h = imagehash.hex_to_hash(h_str)
        if (new_h - existing_h) <= threshold:
            return True
    return False


def _passes_size_aspect(image_bytes: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
    except Exception:
        return False
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    if h == 0:
        return False
    ratio = w / h
    if ratio > MAX_ASPECT_RATIO or ratio < (1.0 / MAX_ASPECT_RATIO):
        return False
    return True


def _passes_pdf_layout(hint: LayoutHint) -> bool:
    if hint.page_height is None or hint.page_height <= 0:
        return True
    if hint.rect_top is None:
        return True
    ph = hint.page_height
    top_frac = hint.rect_top / ph
    bottom_frac = (hint.rect_bottom or hint.rect_top) / ph
    if top_frac < _HEADER_FRAC:
        return False
    if bottom_frac > _FOOTER_FRAC:
        return False
    return True


class ImageSanitizer:
    """Размер/aspect, PDF header/footer, pHash dedup (без CSS-классов)."""

    def __init__(self, phash_threshold: int = PHASH_DUP_THRESHOLD) -> None:
        self._seen: set[str] = set()
        self._threshold = phash_threshold

    def reset_session_hashes(self) -> None:
        self._seen.clear()

    def accept(
        self,
        image_bytes: bytes,
        layout: LayoutHint | None = None,
        *,
        phash: str | None = None,
    ) -> tuple[bool, str]:
        if not image_bytes or len(image_bytes) < 64:
            return False, ""
        if not _passes_size_aspect(image_bytes):
            return False, ""
        hint = layout or LayoutHint()
        if not _passes_pdf_layout(hint):
            return False, ""
        try:
            ph = (phash or "").strip() or get_image_phash(image_bytes)
        except Exception:
            return False, ""
        if is_duplicate(ph, self._seen, threshold=self._threshold):
            return False, ph
        self._seen.add(ph)
        return True, ph
