"""Базовые типы и async-провайдеры поиска."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    published_date: str = ""
    key_extracts: list[str] = field(default_factory=list)
    skip_ollama_summary: bool = False


def search_result_from_dict(item: Dict[str, Any]) -> SearchResult | None:
    if item.get("error") or not item.get("url"):
        return None
    raw_extracts = item.get("key_extracts")
    extracts: list[str] = []
    if isinstance(raw_extracts, list):
        extracts = [str(e).strip() for e in raw_extracts if str(e).strip()][:12]
    return SearchResult(
        title=str(item.get("title") or item.get("url")),
        url=str(item["url"]),
        snippet=str(item.get("snippet") or item.get("content") or ""),
        source=str(item.get("source") or ""),
        engine=str(item.get("engine") or ""),
        published_date=str(item.get("published_date") or "")[:32],
        key_extracts=extracts,
        skip_ollama_summary=bool(item.get("skip_ollama_summary")),
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
