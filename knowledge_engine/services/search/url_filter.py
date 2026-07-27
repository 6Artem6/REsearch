"""Фильтрация и ранжирование URL для multi_search."""

from __future__ import annotations

from knowledge_engine.config import URL_BLOCKLIST_SUBSTR, URL_PRIORITY_SUBSTR


def is_blocked_url(url: str) -> bool:
    low = url.lower()
    return any(b in low for b in URL_BLOCKLIST_SUBSTR)


def url_priority_score(url: str) -> int:
    low = url.lower()
    for i, sub in enumerate(URL_PRIORITY_SUBSTR):
        if sub in low:
            return i
    return len(URL_PRIORITY_SUBSTR) + 1


def rank_and_cap_urls(urls: list[str], cap: int) -> list[str]:
    unique = list(dict.fromkeys(urls))
    filtered = [u for u in unique if not is_blocked_url(u)]
    filtered.sort(key=lambda u: (url_priority_score(u), u))
    return filtered[:cap]
