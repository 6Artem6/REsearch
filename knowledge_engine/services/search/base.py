"""Базовые типы и async-провайдеры поиска."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SearchResult:
    """Нормализованный результат для узлов графа."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    horizon: str = ""  # sota | infra | prod
    engine: str = ""  # searxng engine name (hackernews, github, …)


def search_result_from_dict(item: Dict[str, Any]) -> SearchResult | None:
    if item.get("error") or not item.get("url"):
        return None
    return SearchResult(
        title=str(item.get("title") or item.get("url")),
        url=str(item["url"]),
        snippet=str(item.get("snippet") or item.get("content") or ""),
        source=str(item.get("source") or ""),
        engine=str(item.get("engine") or ""),
    )


class BaseSearchProvider(ABC):
    """Единый async-интерфейс: регистрация в SearchRegistry за ~1 минуту."""

    name: str = "base"

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Формат: title, url, snippet, source, engine (+ опционально error)."""
