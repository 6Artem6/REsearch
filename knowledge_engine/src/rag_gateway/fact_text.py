"""Лимиты и аварийная обрезка текста фактов RAG Gateway."""

from __future__ import annotations

FACT_MAX_CHARS = 2000
GEMMA_SUMMARY_MAX_CHARS = 1500
_FALLBACK_SUFFIX = "..."


def truncate_fact_at_word_boundary(text: str, max_len: int = FACT_MAX_CHARS) -> str:
    """Last-resort: обрезка по границе слова с «...» (не режет посередине токена)."""
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    if max_len <= len(_FALLBACK_SUFFIX):
        return s[:max_len]
    budget = max_len - len(_FALLBACK_SUFFIX)
    cut = s[:budget]
    if cut and not cut[-1].isspace():
        last_space = cut.rfind(" ")
        if last_space >= max(8, int(budget * 0.55)):
            cut = cut[:last_space]
    return cut.rstrip() + _FALLBACK_SUFFIX
