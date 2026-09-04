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
    search_type: str = "auto"
    category: str = ""
    used_unrestricted_fallback: bool = False


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


def build_exa_search_kwargs(
    query: str,
    *,
    num_results: int = 15,
    search_type: str = "auto",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    exclude_text: list[str] | None = None,
    category: str | None = None,
    highlight_query: str = DEFAULT_HIGHLIGHT_QUERY,
    highlight_max_characters: int = 2000,
    highlight_num_sentences: int = 5,
    whitelist_dict: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Pure kwargs builder for Exa.search (unit-testable, no network)."""
    inc = include_domains
    if inc is None:
        wl = (
            whitelist_dict if whitelist_dict is not None else APPROVED_SOURCES_WHITELIST
        )
        inc = get_clean_exa_domains(wl)
    inc = [d for d in (inc or []) if (d or "").strip()]
    exc = merge_exa_exclude_domains(exclude_domains)
    if exclude_text is not None:
        excl_text = normalize_exa_exclude_text(exclude_text)
    else:
        excl_text = normalize_exa_exclude_text(EXA_EXCLUDE_TEXT)
    contents = build_exa_contents_dict(
        highlight_query=highlight_query,
        highlight_max_characters=highlight_max_characters,
        highlight_num_sentences=highlight_num_sentences,
    )
    kwargs: dict[str, Any] = {
        "num_results": num_results,
        "type": (search_type or "auto").strip() or "auto",
        "contents": contents,
        "exclude_domains": exc,
    }
    if inc:
        kwargs["include_domains"] = inc
    cat = (category or "").strip()
    if cat:
        kwargs["category"] = cat
    if excl_text:
        kwargs["exclude_text"] = excl_text
    _ = query
    return kwargs


def _run_exa_sdk_search(exa: Any, query: str, search_kwargs: dict[str, Any]) -> Any:
    """Call exa.search; drop `category` once if the SDK/API rejects it."""
    try:
        return exa.search(query, **search_kwargs)
    except TypeError:
        if "category" not in search_kwargs:
            raise
        retry = dict(search_kwargs)
        retry.pop("category", None)
        return exa.search(query, **retry)
    except Exception:
        if "category" not in search_kwargs:
            raise
        retry = dict(search_kwargs)
        retry.pop("category", None)
        return exa.search(query, **retry)


class ExaSearchClient:
    """Обёртка над exa-py: whitelist, dynamic domains, category, hybrid type."""

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
        category: str | None = None,
        highlight_query: str = DEFAULT_HIGHLIGHT_QUERY,
        highlight_max_characters: int = 2000,
        highlight_num_sentences: int = 5,
        whitelist_dict: dict[str, list[str]] | None = None,
        allow_unrestricted_fallback: bool = False,
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

        search_kwargs = build_exa_search_kwargs(
            q,
            num_results=num_results,
            search_type=search_type,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            exclude_text=exclude_text,
            category=category,
            highlight_query=highlight_query,
            highlight_max_characters=highlight_max_characters,
            highlight_num_sentences=highlight_num_sentences,
            whitelist_dict=whitelist_dict,
        )
        exa = Exa(api_key=self._api_key)
        response = _run_exa_sdk_search(exa, q, search_kwargs)
        results = getattr(response, "results", None) or []
        hits = [_parse_hit(r) for r in results if r is not None]
        used_fallback = False
        if (
            not hits
            and allow_unrestricted_fallback
            and search_kwargs.get("include_domains")
        ):
            retry = dict(search_kwargs)
            retry.pop("include_domains", None)
            retry.pop("category", None)
            response = _run_exa_sdk_search(exa, q, retry)
            results = getattr(response, "results", None) or []
            hits = [_parse_hit(r) for r in results if r is not None]
            used_fallback = True
            search_kwargs = retry

        inc_used = list(search_kwargs.get("include_domains") or [])
        return ExaSearchResponse(
            query=q,
            hits=hits,
            include_domains=inc_used,
            exclude_domains=list(search_kwargs.get("exclude_domains") or []),
            search_type=str(search_kwargs.get("type") or search_type or "auto"),
            category=str(search_kwargs.get("category") or ""),
            used_unrestricted_fallback=used_fallback,
        )

    def search_expanded(
        self,
        query: str,
        *,
        num_results: int = 15,
        highlight_query: str = DEFAULT_HIGHLIGHT_QUERY,
    ) -> ExaSearchResponse:
        """Lite domains → HTTP validate → Exa Pass 1; category Pass 2 on miss."""
        from knowledge_engine.services.search.exa_domain_validate import (
            prepare_exa_pass1_domains_blocking,
        )
        from knowledge_engine.services.search.exa_source_expand import (
            absorb_new_exa_hosts,
            exa_pass2_categories,
            expand_search_context_with_flash_lite,
            filter_pass1_official_hosts,
        )
        from knowledge_engine.ui.run_log import trace

        ctx = expand_search_context_with_flash_lite(query)
        live = prepare_exa_pass1_domains_blocking(ctx.primary_domains)
        validated = filter_pass1_official_hosts(live)
        exclude_text: list[str] | None = [] if ctx.include_official_docs else None

        response: ExaSearchResponse | None = None
        if validated:
            trace(f"EXA pass 1 ▶ | include_domains={validated} category=None")
            response = self.search(
                query,
                num_results=num_results,
                search_type=ctx.search_type,
                include_domains=validated,
                category=None,
                exclude_text=exclude_text,
                highlight_query=highlight_query,
                allow_unrestricted_fallback=False,
            )
            trace(f"EXA pass 1 ✓ | hits={len(response.hits)}")

        if response is None or not response.hits:
            for cat in exa_pass2_categories(ctx):
                trace(f"EXA pass 2 ▶ | include_domains=∅ category={cat}")
                response = self.search(
                    query,
                    num_results=num_results,
                    search_type=ctx.search_type,
                    include_domains=[],
                    category=cat,
                    exclude_text=exclude_text,
                    highlight_query=highlight_query,
                    allow_unrestricted_fallback=False,
                )
                trace(f"EXA pass 2 ✓ | category={cat} hits={len(response.hits)}")
                if response.hits:
                    break
            if (response is None or not response.hits) and ctx.use_broader_search:
                trace("EXA pass 2 ▶ | include_domains=∅ category=None")
                response = self.search(
                    query,
                    num_results=num_results,
                    search_type=ctx.search_type,
                    include_domains=[],
                    category=None,
                    exclude_text=exclude_text,
                    highlight_query=highlight_query,
                    allow_unrestricted_fallback=False,
                )

        if response is None:
            response = ExaSearchResponse(
                query=query,
                hits=[],
                include_domains=[],
                exclude_domains=[],
            )
        absorb_new_exa_hosts([h.url for h in response.hits])
        return response


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
