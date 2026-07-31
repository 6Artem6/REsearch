"""Exa API → dict / CurriculumSearchHit; dual-query, domain cap, Lite rerank."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from knowledge_engine.config import (
    EXA_API_KEY,
    EXA_DUAL_QUERY_EN_RATIO,
    EXA_DOMAIN_CAP_PER_HOST,
    EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN,
    EXA_RERANK_LITE_THRESHOLD,
    EXA_SEARCH_ENABLED,
    CURRICULUM_PRACTICAL_EXA_LIMIT,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.search.exa_client import (
    ExaNotConfiguredError,
    ExaSearchClient,
    ExaSearchHit,
    ExaSearchResponse,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

FLASH_EXTRACTS_PER_HIT = 8
_SKIP_OLLAMA_MIN_HIGHLIGHTS = 1
_SKIP_OLLAMA_MIN_WORDS = 100

_EXA_QUERY_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Search Query Architect для Exa (engineering blogs, case studies, EN whitelist).\n"
    "На вход: учебный контекст (часто русский).\n\n"
    "Цель query_en: инженерные разборы и гайды по реализации, НЕ страницы API/SDK.\n"
    "ЗАПРЕЩЕНО: запросы вида «langchain vectorstores API», «vertexai SDK reference», "
    "«cloud provider setup docs» без контекста архитектуры.\n"
    "ТРЕБУЙ в query_en фокус на концепты и архитектуру — используй фреймы (если уместно): "
    "deep dive, architecture, implementation guide, how it works, benchmark, trade-offs.\n"
    "Пример темы «гибридный поиск»:\n"
    "  Плохо: hybrid search vectorstores indexing\n"
    "  Хорошо: hybrid search implementation sparse dense BM25 RRF reciprocal rank fusion "
    "local vector database\n\n"
    "Выдай:\n"
    "- query_en: английский запрос 12–120 слов, термины CS, без «API reference».\n"
    "- query_ru: тот же инженерный фокус на русском (сокращённо).\n"
    "JSON: query_en, query_ru (строки, 8–400 символов)."
)

_EXA_DOC_URL_MARKERS: tuple[str, ...] = (
    "/docs/",
    "/reference/",
    "/api/",
    "v1_operations",
    "v2_operations",
    "/sdk/",
    "/sdk-reference",
    "/api-reference",
    "/developers/docs",
)

_EXA_ARTICLE_URL_MARKERS: tuple[str, ...] = (
    "/blog/",
    "/posts/",
    "/post/",
    "/guides/",
    "/guide/",
    "/engineering/",
    "/learn/",
    "/articles/",
    "/story/",
)


class ExaDualQueryOut(BaseModel):
    query_en: str = Field(default="", max_length=400)
    query_ru: str = Field(default="", max_length=400)


def _published_date_from_raw(raw: dict[str, Any]) -> str:
    for key in ("published_date", "publishedDate", "published"):
        val = raw.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()[:32]
    return ""


def _highlights_to_key_extracts(highlights: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for h in highlights:
        s = (h or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        if len(s) > 2000:
            s = s[:1997] + "…"
        out.append(s)
        if len(out) >= FLASH_EXTRACTS_PER_HIT:
            break
    return out


def _snippet_from_highlights(extracts: list[str]) -> str:
    if not extracts:
        return ""
    return " ".join(extracts)[:1200]


def _extract_word_total(extracts: list[str]) -> int:
    return sum(len((e or "").split()) for e in extracts)


def exa_domain_key(url: str) -> str:
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _exa_url_quality_score(url: str) -> int:
    """Выше = больше «статья/гайд», ниже = API docs."""
    low = (url or "").strip().lower()
    if not low:
        return 0
    score = 0
    for marker in _EXA_ARTICLE_URL_MARKERS:
        if marker in low:
            score += 2
    for marker in _EXA_DOC_URL_MARKERS:
        if marker in low:
            score -= 5
    path = low.split("?", 1)[0]
    if path.endswith("/docs") or path.endswith("/reference"):
        score -= 4
    if "readme.io" in low or "readthedocs" in low:
        score -= 4
    return score


def filter_and_rank_exa_curriculum_hits(
    hits: list[CurriculumSearchHit],
) -> list[CurriculumSearchHit]:
    if not hits:
        return []
    scored = [(h, _exa_url_quality_score(h.url)) for h in hits]
    rejected = [h for h, s in scored if s <= -5]
    kept = [h for h, s in scored if s > -5]
    if rejected:
        trace(
            f"CURRICULUM exa url filter ⊘ | dropped_api_doc_urls={len(rejected)}"
        )
    if not kept:
        kept = [h for h, _ in scored]
    kept.sort(key=lambda h: -_exa_url_quality_score(h.url))
    return kept


_THit = TypeVar("_THit")


def fair_domain_round_robin(
    hits: list[_THit],
    cap: int,
    *,
    max_per_domain: int | None = None,
    get_url: Callable[[_THit], str] | None = None,
) -> list[_THit]:
    """
    Round-robin по доменам: 1-я статья с каждого host, затем 2-я только если cap не заполнен.
    """
    if not hits or cap <= 0:
        return []
    per_cap = max_per_domain if max_per_domain is not None else EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN
    per_cap = max(1, per_cap)
    url_fn = get_url or (lambda h: getattr(h, "url", "") or "")

    buckets: dict[str, list[_THit]] = {}
    domain_order: list[str] = []
    for h in hits:
        dom = exa_domain_key(url_fn(h))
        if not dom:
            dom = "_unknown"
        if dom not in buckets:
            buckets[dom] = []
            domain_order.append(dom)
        buckets[dom].append(h)

    per_domain: dict[str, int] = {d: 0 for d in domain_order}
    pointers: dict[str, int] = {d: 0 for d in domain_order}
    seen_urls: set[str] = set()
    out: list[_THit] = []

    while len(out) < cap:
        took = False
        for dom in domain_order:
            if len(out) >= cap:
                break
            if per_domain[dom] >= per_cap:
                continue
            batch = buckets[dom]
            ptr = pointers[dom]
            while ptr < len(batch):
                h = batch[ptr]
                ptr += 1
                key = (url_fn(h) or "").strip().lower()
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                pointers[dom] = ptr
                per_domain[dom] += 1
                out.append(h)
                took = True
                break
        if not took:
            break

    domains_used = len({exa_domain_key(url_fn(h)) for h in out if url_fn(h)})
    trace(
        f"CURRICULUM exa fair_round_robin ✓ | in={len(hits)} out={len(out)} "
        f"cap={cap} domains={domains_used} max_per_domain={per_cap}"
    )
    return out[:cap]


def apply_exa_domain_cap(
    hits: list[ExaSearchHit],
    *,
    max_per_domain: int | None = None,
    cap: int | None = None,
) -> list[ExaSearchHit]:
    limit = cap if cap is not None else len(hits)
    return fair_domain_round_robin(
        hits,
        limit,
        max_per_domain=max_per_domain,
        get_url=lambda h: h.url,
    )


def merge_dual_exa_hits(
    en_hits: list[ExaSearchHit],
    ru_hits: list[ExaSearchHit],
    *,
    cap: int,
    en_ratio: float | None = None,
) -> list[ExaSearchHit]:
    ratio = en_ratio if en_ratio is not None else EXA_DUAL_QUERY_EN_RATIO
    ratio = max(0.2, min(0.9, ratio))
    en_target = max(1, round(cap * ratio))
    ru_target = max(0, cap - en_target)
    seen: set[str] = set()
    out: list[ExaSearchHit] = []
    en_added = 0
    ru_added = 0

    for h in en_hits:
        if en_added >= en_target or len(out) >= cap:
            break
        key = (h.url or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
        en_added += 1

    for h in ru_hits:
        if ru_added >= ru_target or len(out) >= cap:
            break
        key = (h.url or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
        ru_added += 1

    for h in en_hits:
        if len(out) >= cap:
            break
        key = (h.url or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)

    for h in ru_hits:
        if len(out) >= cap:
            break
        key = (h.url or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)

    trace(
        f"CURRICULUM exa merge ✓ | en={en_added}/{en_target} "
        f"ru={ru_added}/{ru_target} total={len(out)} cap={cap}"
    )
    return out[:cap]


def exa_hit_to_provider_dict(hit: ExaSearchHit) -> dict[str, Any]:
    published = _published_date_from_raw(hit.raw)
    extracts = _highlights_to_key_extracts(hit.highlights)
    skip = bool(
        extracts
        and len(extracts) >= _SKIP_OLLAMA_MIN_HIGHLIGHTS
        and _extract_word_total(extracts) >= _SKIP_OLLAMA_MIN_WORDS
    )
    snippet = _snippet_from_highlights(extracts)
    return {
        "title": hit.title,
        "url": hit.url,
        "snippet": snippet,
        "source": "exa",
        "engine": "exa",
        "published_date": published,
        "key_extracts": extracts,
        "skip_ollama_summary": skip,
    }


def exa_response_to_provider_dicts(response: ExaSearchResponse) -> list[dict[str, Any]]:
    capped = apply_exa_domain_cap(list(response.hits))
    rows: list[dict[str, Any]] = []
    for hit in capped:
        if not (hit.url or "").strip().startswith("http"):
            continue
        rows.append(exa_hit_to_provider_dict(hit))
    return rows


def exa_hit_to_curriculum_hit(hit: ExaSearchHit) -> CurriculumSearchHit:
    row = exa_hit_to_provider_dict(hit)
    return CurriculumSearchHit(
        url=row["url"],
        title=row["title"],
        snippet=row["snippet"],
        published_date=row["published_date"],
        key_extracts=list(row["key_extracts"]),
        source_tier="exa",
        skip_ollama_summary=bool(row["skip_ollama_summary"]),
    )


def exa_response_to_curriculum_hits(response: ExaSearchResponse) -> list[CurriculumSearchHit]:
    capped = apply_exa_domain_cap(list(response.hits))
    return [exa_hit_to_curriculum_hit(h) for h in capped if h.url.startswith("http")]


async def build_exa_dual_queries(
    context: str,
    *,
    anchor: str,
) -> tuple[str, str]:
    """query_ru + query_en (Lite); fallback — heuristic EN."""
    ru = (context or "").strip()[:1200]
    if len(ru) < 8:
        return "", ""
    from knowledge_engine.src.curriculum.lite_search_pipeline import _lite_structured

    trace("CURRICULUM exa queries ▶ | Lite dual EN/RU")
    try:
        out = await _lite_structured(
            _EXA_QUERY_SYSTEM,
            json.dumps(
                {
                    "learning_context": ru,
                    "avoid": "SDK/API reference pages, cloud console setup, v2_operations",
                    "prefer": "architecture, implementation guide, deep dive, benchmark",
                },
                ensure_ascii=False,
            ),
            f"{anchor}:exa_dual_query",
            ExaDualQueryOut,
            "curriculum / exa_dual_query",
        )
        en = (out.query_en or "").strip()[:400]
        ru_q = (out.query_ru or ru).strip()[:400]
        if len(en) < 8:
            raise ValueError("weak query_en")
        trace(f"CURRICULUM exa queries ✓ | en={en[:80]}… ru={ru_q[:60]}…")
        return ru_q, en
    except Exception as exc:
        from knowledge_engine.src.curriculum.search_query_builder import build_search_queries

        trace(f"CURRICULUM exa queries fallback | {exc}")
        built = build_search_queries(ru)
        en = (built.practical_query or built.academic_query or ru)[:400]
        return ru[:400], en


async def _lite_rerank_exa_hits(
    hits: list[CurriculumSearchHit],
    goal: str,
    core_concepts: list[str],
    *,
    anchor: str,
    cap: int,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.curriculum.lite_search_pipeline import (
        batch_lite_eval_curriculum_hits,
    )

    focus = goal.strip()
    if core_concepts:
        focus = f"{focus} | concepts: {', '.join(core_concepts[:8])}"
    trace(
        f"CURRICULUM exa lite rerank ▶ | candidates={len(hits)} cap={cap}"
    )
    approved = await batch_lite_eval_curriculum_hits(
        hits,
        focus[:1200],
        anchor=f"{anchor}:exa_rerank",
        strict=True,
    )
    if approved:
        trace(f"CURRICULUM exa lite rerank ✓ | approved={len(approved)}")
        diversified = fair_domain_round_robin(
            approved,
            cap,
            max_per_domain=EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN,
            get_url=lambda h: h.url,
        )
        return diversified[:cap]
    trace("CURRICULUM exa lite rerank ⊘ | approved=0 — keep domain-capped order")
    return hits[:cap]


async def _exa_search_one(
    client: ExaSearchClient,
    query: str,
    num_results: int,
    *,
    exclude_text: list[str] | None = None,
    highlight_query: str | None = None,
) -> list[ExaSearchHit]:
    if not (query or "").strip():
        return []
    from knowledge_engine.config import EXA_EXCLUDE_TEXT, EXA_PRACTICAL_HIGHLIGHT_QUERY
    from knowledge_engine.services.search.exa_client import normalize_exa_exclude_text

    exc_text = normalize_exa_exclude_text(exclude_text or EXA_EXCLUDE_TEXT)
    hl = (highlight_query or EXA_PRACTICAL_HIGHLIGHT_QUERY).strip()
    response = await asyncio.to_thread(
        client.search,
        query.strip(),
        num_results=max(1, min(num_results, CURRICULUM_PRACTICAL_EXA_LIMIT)),
        exclude_text=exc_text,
        highlight_query=hl,
    )
    return list(response.hits)


def _learning_context_for_node(node: CurriculumNode, course_goal: str) -> str:
    parts = [
        course_goal.strip()[:400],
        node.title.strip(),
        ", ".join(node.core_concepts[:6]),
        (node.brief_summary or "")[:500],
    ]
    text = " | ".join(p for p in parts if p)
    return text[:1200]


async def fetch_exa_curriculum_hits_for_node(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
    cap: int,
) -> list[CurriculumSearchHit]:
    """DEEP-нода: dual EN/RU Exa, domain cap, Lite rerank при избытке кандидатов."""
    if not EXA_SEARCH_ENABLED or not EXA_API_KEY:
        trace("CURRICULUM exa ⊘ | EXA_API_KEY not set or EXA_SEARCH_ENABLED=false")
        return []

    context = _learning_context_for_node(node, course_goal)
    query_ru, query_en = await build_exa_dual_queries(
        context,
        anchor=f"{anchor}:practical:{node.node_id}",
    )
    if not query_ru and not query_en:
        return []

    from knowledge_engine.src.curriculum.practical_url_filters import filter_practical_search_row
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_collectible_article_url,
    )

    fetch_cap = max(cap + 4, cap * 2)
    en_fetch = max(3, round(fetch_cap * EXA_DUAL_QUERY_EN_RATIO))
    ru_fetch = max(2, fetch_cap - en_fetch)

    trace(
        f"CURRICULUM exa ▶ | node={node.node_id} dual en={en_fetch} ru={ru_fetch} "
        f"en_q={query_en[:70]}…"
    )

    try:
        client = ExaSearchClient(api_key=EXA_API_KEY)
        en_raw, ru_raw = await asyncio.gather(
            _exa_search_one(client, query_en, en_fetch),
            _exa_search_one(client, query_ru, ru_fetch),
        )
    except ExaNotConfiguredError as exc:
        trace(f"CURRICULUM exa ✗ | {exc}")
        return []
    except Exception as exc:
        trace(f"CURRICULUM exa ✗ | {exc}")
        return []

    merged_raw = merge_dual_exa_hits(en_raw, ru_raw, cap=fetch_cap)
    capped_raw = fair_domain_round_robin(
        merged_raw,
        fetch_cap,
        max_per_domain=EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN,
        get_url=lambda h: h.url,
    )
    hits = [exa_hit_to_curriculum_hit(h) for h in capped_raw]
    hits = filter_and_rank_exa_curriculum_hits(hits)

    out: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for h in hits:
        if not is_collectible_article_url(h.url):
            continue
        row = {"url": h.url, "title": h.title, "snippet": h.snippet}
        if not filter_practical_search_row(row):
            continue
        key = h.url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)

    threshold = max(1, EXA_RERANK_LITE_THRESHOLD)
    if len(out) > threshold:
        out = await _lite_rerank_exa_hits(
            out,
            context,
            list(node.core_concepts or []),
            anchor=f"{anchor}:practical:{node.node_id}",
            cap=cap,
        )
    else:
        out = out[:cap]

    skip_n = sum(1 for h in out if h.skip_ollama_summary)
    trace(
        f"CURRICULUM exa ✓ | node={node.node_id} hits={len(out)} "
        f"skip_ollama={skip_n}"
    )
    return out[:cap]


async def fetch_exa_curriculum_hits_simple(
    query: str,
    *,
    limit: int,
    anchor: str = "",
) -> list[CurriculumSearchHit]:
    """Один запрос (bulk practical fetch) — domain cap, без dual-query."""
    if not EXA_SEARCH_ENABLED or not EXA_API_KEY:
        return []
    q = (query or "").strip()
    if len(q) < 8:
        return []
    from knowledge_engine.src.curriculum.practical_url_filters import filter_practical_search_row
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_collectible_article_url,
    )

    trace(f"CURRICULUM exa ▶ | {q[:120]}")
    try:
        client = ExaSearchClient(api_key=EXA_API_KEY)
        response = await asyncio.to_thread(
            client.search,
            q,
            num_results=max(1, min(limit, CURRICULUM_PRACTICAL_EXA_LIMIT)),
        )
    except (ExaNotConfiguredError, Exception) as exc:
        trace(f"CURRICULUM exa ✗ | {exc}")
        return []

    hits = exa_response_to_curriculum_hits(response)
    out: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for h in hits:
        if not is_collectible_article_url(h.url):
            continue
        row = {"url": h.url, "title": h.title, "snippet": h.snippet}
        if not filter_practical_search_row(row):
            continue
        key = h.url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    skip_n = sum(1 for h in out if h.skip_ollama_summary)
    trace(f"CURRICULUM exa ✓ | hits={len(out)} skip_ollama={skip_n}")
    return out
