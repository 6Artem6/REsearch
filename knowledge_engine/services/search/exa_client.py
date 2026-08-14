"""Изолированный клиент Exa API (exa-py) для whitelist-поиска."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge_engine.config import (
    EXA_API_KEY,
    EXA_EXCLUDE_TEXT,
    EXCLUDED_SOURCES_BLACKLIST,
)
from knowledge_engine.services.search.exa_domains import get_clean_exa_domains
from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST

DEFAULT_HIGHLIGHT_QUERY = (
    "Extract key technical architecture takeaways, algorithms, "
    "memory optimizations, and trade-offs."
)


def build_exa_contents_dict(
    *,
    highlight_query: str = DEFAULT_HIGHLIGHT_QUERY,
    highlight_max_characters: int = 2000,
    highlight_num_sentences: int = 5,
) -> dict[str, Any]:
    """Highlights only — no Exa AI summary."""
    highlights: dict[str, Any] = {
        "num_sentences": max(1, min(int(highlight_num_sentences), 12)),
    }
    q = (highlight_query or DEFAULT_HIGHLIGHT_QUERY).strip()
    if q:
        highlights["query"] = q
    highlights["max_characters"] = max(200, min(highlight_max_characters, 4000))
    return {"highlights": highlights}


def merge_exa_exclude_domains(extra: list[str] | None = None) -> list[str]:
    """Static blacklist + SQLite anti-bot blocklist (unique, lowercased)."""
    from knowledge_engine.db.domain_blocklist import get_blocked_domains

    seen: set[str] = set()
    out: list[str] = []
    for raw in (
        list(EXCLUDED_SOURCES_BLACKLIST) + list(extra or []) + get_blocked_domains()
    ):
        k = (raw or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def normalize_exa_exclude_text(
    raw: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Exa excludeText: одна фраза, максимум 5 слов, без запятых."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
    else:
        text = " ".join(str(x).strip() for x in raw if str(x).strip())
    text = text.replace(",", " ").strip()
    words = [w for w in text.split() if w]
    phrase = " ".join(words[:5])
    return [phrase] if phrase else []


class ExaNotConfiguredError(RuntimeError):
    """Нет EXA_API_KEY или пакет exa-py."""


@dataclass(frozen=True)
class ExaSearchHit:
    url: str
    title: str
    highlights: list[str] = field(default_factory=list)
    published_date: str = ""
    score: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExaSearchResponse:
    query: str
    hits: list[ExaSearchHit]
    include_domains: list[str]
    exclude_domains: list[str]


def _parse_hit(item: Any) -> ExaSearchHit:
    url = str(getattr(item, "url", "") or "").strip()
    title = str(getattr(item, "title", "") or url).strip()
    highlights_raw = getattr(item, "highlights", None) or []
    highlights: list[str] = []
    if isinstance(highlights_raw, list):
        for h in highlights_raw:
            if isinstance(h, str) and h.strip():
                highlights.append(h.strip())
            elif h is not None:
                highlights.append(str(h).strip())
    score = getattr(item, "score", None)
    score_f = float(score) if score is not None else None
    raw: dict[str, Any] = {}
    if hasattr(item, "model_dump"):
        try:
            raw = item.model_dump()
        except Exception:
            pass
    elif isinstance(item, dict):
        raw = dict(item)
    pub = ""
    for key in ("published_date", "publishedDate", "published"):
        val = raw.get(key) if raw else getattr(item, key, None)
        if val is not None and str(val).strip():
            pub = str(val).strip()[:32]
            break
    return ExaSearchHit(
        url=url,
        title=title[:400],
        highlights=highlights[:12],
        published_date=pub,
        score=score_f,
        raw=raw,
    )


class ExaSearchClient:
    """Обёртка над exa-py: поиск только по очищенным доменам whitelist."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (api_key or EXA_API_KEY or "").strip()

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def whitelist_include_domains(
        self,
        whitelist_dict: dict[str, list[str]] | None = None,
    ) -> list[str]:
        wl = (
            whitelist_dict if whitelist_dict is not None else APPROVED_SOURCES_WHITELIST
        )
        return get_clean_exa_domains(wl)

    def search(
        self,
        query: str,
        *,
        num_results: int = 15,
        search_type: str = "auto",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        exclude_text: list[str] | None = None,
        highlight_query: str = DEFAULT_HIGHLIGHT_QUERY,
        highlight_max_characters: int = 2000,
        highlight_num_sentences: int = 5,
        whitelist_dict: dict[str, list[str]] | None = None,
    ) -> ExaSearchResponse:
        q = (query or "").strip()
        if not q:
            raise ValueError("Exa search: пустой query")
        if not self._api_key:
            raise ExaNotConfiguredError(
                "Задайте EXA_API_KEY в .env и установите пакет: pip install exa-py"
            )

        try:
            from exa_py import Exa
        except ImportError as exc:
            raise ExaNotConfiguredError(
                "Пакет exa-py не установлен (pip install exa-py)"
            ) from exc

        inc = include_domains
        if inc is None:
            inc = self.whitelist_include_domains(whitelist_dict)
        exc = merge_exa_exclude_domains(exclude_domains)
        if exclude_text is not None:
            excl_text = normalize_exa_exclude_text(exclude_text)
        else:
            excl_text = normalize_exa_exclude_text(EXA_EXCLUDE_TEXT)

        exa = Exa(api_key=self._api_key)
        contents = build_exa_contents_dict(
            highlight_query=highlight_query,
            highlight_max_characters=highlight_max_characters,
            highlight_num_sentences=highlight_num_sentences,
        )
        search_kwargs: dict[str, Any] = {
            "include_domains": inc,
            "exclude_domains": exc,
            "num_results": num_results,
            "type": search_type,
            "contents": contents,
        }
        if excl_text:
            search_kwargs["exclude_text"] = excl_text
        response = exa.search(q, **search_kwargs)

        results = getattr(response, "results", None) or []
        hits = [_parse_hit(r) for r in results if r is not None]
        return ExaSearchResponse(
            query=q,
            hits=hits,
            include_domains=list(inc),
            exclude_domains=exc,
        )


def exa_search_whitelist(
    query: str,
    *,
    num_results: int = 15,
    whitelist_dict: dict[str, list[str]] | None = None,
) -> ExaSearchResponse:
    """Удобная функция: один вызов с дефолтным whitelist из кода."""
    return ExaSearchClient().search(
        query,
        num_results=num_results,
        whitelist_dict=whitelist_dict,
    )
