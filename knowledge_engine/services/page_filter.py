"""Эвристики отсечения бесполезного текста до облачного L2."""

from __future__ import annotations

import re

from knowledge_engine.config import MIN_PAGE_CHARS_FOR_EXTRACTION

_BOILERPLATE_MARKERS = (
    "captcha",
    "access denied",
    "enable javascript",
    "sign in to continue",
    "robot or human",
    "403 forbidden",
    "404 not found",
)


def is_obviously_useless_page_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < MIN_PAGE_CHARS_FOR_EXTRACTION:
        return True
    lower = stripped.lower()
    if any(m in lower for m in _BOILERPLATE_MARKERS):
        return True
    # почти нет «слов» — навигация / JSON-LD шум
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]{3,}", stripped)
    if len(words) < 12:
        return True
    return False
