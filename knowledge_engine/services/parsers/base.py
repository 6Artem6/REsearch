"""Базовый тип извлечённого изображения."""

from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_engine.services.image_filter import LayoutHint


@dataclass
class ExtractedImage:
    image_bytes: bytes
    caption: str = ""
    context_text: str = ""
    page_or_pos: int = 0
    phash: str = ""
    layout: LayoutHint = field(default_factory=LayoutHint)
    mime: str = "image/png"


@dataclass
class ExtractedDiagram:
    """Inline Mermaid из текста статьи (без VLM)."""

    mermaid_code: str
    caption: str = ""
    is_inline_mermaid: bool = True
    page_or_pos: int = 0
