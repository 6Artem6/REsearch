"""Exa API → dict / CurriculumSearchHit; multi-vector query plan, domain cap, Lite rerank."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from knowledge_engine.config import (
    CURRICULUM_PRACTICAL_EXA_LIMIT,
    CURRICULUM_PREFLIGHT_ENABLED,
    CURRICULUM_PREFLIGHT_FETCH_CAP,
    CURRICULUM_PREFLIGHT_FINAL_ARTICLES,
    EXA_API_KEY,
    EXA_DUAL_QUERY_EN_RATIO,
    EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN,
    EXA_FETCH_NUM_RESULTS,
    EXA_MAX_CONCURRENT_SEARCH,
    EXA_RECALL_MAX_PER_DOMAIN,
    EXA_RERANK_LITE_THRESHOLD,
    EXA_SEARCH_ENABLED,
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

_logger = logging.getLogger(__name__)

FLASH_EXTRACTS_PER_HIT = 8
_SKIP_OLLAMA_MIN_HIGHLIGHTS = 1
_SKIP_OLLAMA_MIN_WORDS = 100

_EXA_QUERY_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a Search Query Architect for Exa (official docs + engineering blogs).\n"
    "Input: learning context (often Russian).\n\n"
    "Goal for query_en: mix of (1) official specifications and language/framework "
    "documentation (PEP, CPython, MDN, PyTorch docs) and (2) architecture deep-dives.\n"
    "FORBIDDEN: swagger/openapi consoles, SDK class lists, cloud setup wizards.\n"
    "ALLOWED: canonical spec and vendor documentation hosts for THIS topic, "
    "plus architecture deep-dives.\n"
    "Prefer frames when useful: specification, internals, how it works, architecture, "
    "trade-offs, benchmark.\n\n"
    "Emit:\n"
    "- query_en: English 12–120 words, CS terms.\n"
    "- query_ru: same engineering focus in Russian (shorter).\n"
    "JSON: query_en, query_ru (strings, 8–400 chars)."
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
    "/swagger/",
    "/openapi/",
    "/apidocs/",
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


ExaQueryRole = Literal[
    "en_declarative",
    "en_technical",
    "en_edge_cases",
    "ru_short",
    "ru_expert_article",
    "ru_practical_cases",
]


class ExaQuerySpec(BaseModel):
    role: ExaQueryRole
    query: str = Field(min_length=8, max_length=400)
    highlight_query: str = Field(min_length=8, max_length=500)


class ExaQueryPlanOut(BaseModel):
    en_declarative: str = Field(default="", max_length=400)
    en_technical: str = Field(default="", max_length=400)
    en_edge_cases: str = Field(default="", max_length=400)
    ru_short: str = Field(default="", max_length=400)
    ru_expert_article: str = Field(default="", max_length=400)
    ru_practical_cases: str = Field(default="", max_length=400)
    keywords: list[str] = Field(default_factory=list, max_length=8)
    core_theme: str = Field(default="", max_length=600)


_EXA_QUERY_PLAN_SYSTEM = (
    "You are a Search Query Architect for Exa (official documentation AND "
    "engineering blogs / long-form case studies).\n"
    "Input: learning context (title, concepts, goal — often Russian).\n\n"
    "Produce diverse search vectors covering: language/runtime specs (PEP, CPython, "
    "framework docs), architecture explainers, internals, failure modes, and Russian "
    "engineering longreads.\n\n"
    "Rules for ALL English vectors:\n"
    "- Include official spec/docs queries when the topic is a language, runtime, "
    "or API (e.g. PEP 703, CPython ceval, PyTorch docs) — not only blogs.\n"
    "- Still avoid swagger/openapi consoles and cloud product setup wizards.\n"
    "- Prefer frames: specification, internals, architecture, implementation, "
    "how it works, benchmark, trade-offs, failure modes.\n\n"
    "Rules for Russian vectors (ru_short, ru_expert_article, ru_practical_cases):\n"
    "- Write queries IN RUSSIAN.\n"
    "- Use declarative longread style as on Habr when asking for blogs, AND "
    "mention официальная документация / PEP / спецификация when relevant.\n"
    "- ru_expert_article: expert deep-dive / architecture article.\n"
    "- ru_practical_cases: production cases, optimization war stories, incidents.\n\n"
    "For each vector also output a short English highlight_query (1–2 sentences) "
    "telling Exa which sentences to extract: spec rules, architecture, trade-offs, "
    "benchmarks — not API parameter lists.\n\n"
    "Additionally output two fields for a local pre-flight relevance gate "
    "(NOT sent to Exa):\n"
    "- keywords: 5–8 strict, specific technical entities/terms for this node "
    "(exact names — API/function/algorithm/spec names, not generic words).\n"
    "- core_theme: 1–2 dense English sentences summarizing what a relevant source "
    "MUST cover, written for a cross-encoder relevance query (not a search query).\n\n"
    "JSON fields: en_declarative, en_technical, en_edge_cases, ru_short, "
    "ru_expert_article, ru_practical_cases (8–400 chars each), "
    "keywords (5–8 strings), core_theme (1–600 chars)."
)

_EXA_HIGHLIGHT_BY_ROLE: dict[ExaQueryRole, str] = {
    "en_declarative": (
        "Declarative engineering article: system design narrative, how components interact, "
        "architecture rationale — not API lists."
    ),
    "en_technical": (
        "Deep technical internals: source-level behavior, algorithms, data structures, "
        "implementation details in engineering blogs."
    ),
    "en_edge_cases": (
        "Failure modes, bottlenecks, edge cases, production incidents, performance limits."
    ),
    "ru_short": (
        "Ключевые фрагменты: архитектура, реализация, trade-offs — не списки параметров API."
    ),
    "ru_expert_article": (
        "Разбор архитектуры и внутренней реализации в стиле лонгрида на Хабре."
    ),
    "ru_practical_cases": (
        "Практический опыт, оптимизация в продакшене, узкие места, кейсы и постмортемы."
    ),
}


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
    """Выше = больше «статья/гайд» или официальные docs; ниже = swagger/API consoles."""
    low = (url or "").strip().lower()
    if not low:
        return 0
    from knowledge_engine.services.search.exa_domains import is_official_docs_host

    if is_official_docs_host(low):
        score = 4
        for marker in _EXA_ARTICLE_URL_MARKERS:
            if marker in low:
                score += 1
        if "/swagger/" in low or "/openapi/" in low:
            return -6
        return score
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


def _normalize_exa_score(score: float | None) -> float:
    if score is None:
        return 0.5
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.5
    if s <= 0:
        return 0.0
    if s >= 1:
        return 1.0
    return s


def _normalize_url_quality_score(url_score: int) -> float:
    """Map heuristic url score (~-10..+10) to 0..1."""
    clamped = max(-10, min(10, url_score))
    return (clamped + 10) / 20.0


def _combined_exa_rank_score(hit: CurriculumSearchHit) -> float:
    url_q = _exa_url_quality_score(hit.url)
    exa_n = _normalize_exa_score(hit.exa_relevance_score)
    url_n = _normalize_url_quality_score(url_q)
    return 0.65 * exa_n + 0.35 * url_n


def _log_exa_score_distribution(
    hits: list[CurriculumSearchHit],
    *,
    label: str,
) -> None:
    scores = [h.exa_relevance_score for h in hits if h.exa_relevance_score is not None]
    if not scores:
        _logger.debug("CURRICULUM exa scores | %s | no exa scores", label)
        return
    lo = min(scores)
    hi = max(scores)
    avg = sum(scores) / len(scores)
    _logger.info(
        "CURRICULUM exa scores | %s | n=%d min=%.3f max=%.3f avg=%.3f",
        label,
        len(scores),
        lo,
        hi,
        avg,
    )


def filter_and_rank_exa_curriculum_hits(
    hits: list[CurriculumSearchHit],
) -> list[CurriculumSearchHit]:
    if not hits:
        return []
    trace(f"CURRICULUM exa rank ▶ | raw_hits={len(hits)}")
    _log_exa_score_distribution(hits, label="before_url_filter")

    scored_url = [(h, _exa_url_quality_score(h.url)) for h in hits]
    rejected = [h for h, s in scored_url if s <= -5]
    kept = [h for h, s in scored_url if s > -5]
    trace(
        f"CURRICULUM exa url filter | kept={len(kept)} dropped_api_doc_urls={len(rejected)}"
    )
    if not kept:
        kept = [h for h, _ in scored_url]

    kept.sort(key=lambda h: -_combined_exa_rank_score(h))
    _log_exa_score_distribution(kept, label="after_composite_rank")
    trace(f"CURRICULUM exa rank ✓ | ranked={len(kept)}")
    return kept


def postprocess_exa_hits_for_external_recall(
    hits: list[ExaSearchHit],
    *,
    cap: int,
) -> list[CurriculumSearchHit]:
    """
    Shared Exa post-filter for lecture Stage 2 and similar paths:
    URL heuristics, composite Exa+URL rank, practical filters, canonical URL dedupe.
    """
    from knowledge_engine.src.curriculum.practical_url_filters import (
        filter_practical_search_row,
    )
    from knowledge_engine.utils.link_sanitizer import normalize_lecture_url

    raw_n = len(hits)
    curriculum = [
        exa_hit_to_curriculum_hit(h)
        for h in hits
        if (h.url or "").strip().startswith("http")
    ]
    ranked = filter_and_rank_exa_curriculum_hits(curriculum)

    out: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for h in ranked:
        row = {"url": h.url, "title": h.title, "snippet": h.snippet}
        if not filter_practical_search_row(row):
            continue
        key = normalize_lecture_url(h.url)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= max(1, cap):
            break

    trace(
        f"EXA postprocess ✓ | raw={raw_n} ranked={len(ranked)} "
        f"out={len(out)} cap={cap}"
    )
    return out


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
    per_cap = (
        max_per_domain
        if max_per_domain is not None
        else EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN
    )
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


def fill_round_robin_tail(
    selected: list[_THit],
    all_hits: list[_THit],
    target: int,
    *,
    get_url: Callable[[_THit], str] | None = None,
) -> list[_THit]:
    """Добор до `target` из хвоста, отсечённого `fair_domain_round_robin`'s
    per-domain cap'ом — в исходном порядке релевантности (`all_hits`
    предполагается уже отсортированным по Exa score, как на выходе
    `merge_multi_vector_exa_hits`), не ослабляя anti-monoculture защиту там,
    где кандидатов реально хватает на несколько доменов.

    Раньше при низком разнообразии доменов в конкретном Exa-ответе (например
    все raw-хиты с одного хоста) round-robin честно резал итог до
    `max_per_domain`, а остаток того же ответа просто терялся — добор не
    происходил, хотя материал был в наличии (аудит: `domains=1` → `out=2`
    вместо квоты в 4, при raw_total=18 — сетевой Pass 2 при этом не
    триггерился, т.к. зависел только от `raw_total==0`, а не от итогового
    дефицита после cap'а). Эта функция — Шаг Б двухэтапного добора.
    """
    if len(selected) >= target:
        return selected
    url_fn = get_url or (lambda h: getattr(h, "url", "") or "")
    seen = {(url_fn(h) or "").strip().lower() for h in selected}
    out = list(selected)
    for h in all_hits:
        if len(out) >= target:
            break
        key = (url_fn(h) or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


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


def merge_multi_vector_exa_hits(
    batches: list[list[ExaSearchHit]],
    *,
    cap: int,
) -> list[ExaSearchHit]:
    """Dedup by URL; keep best Exa score; order by score then fair interleave prep."""
    best: dict[str, ExaSearchHit] = {}
    order: list[str] = []
    for batch in batches:
        for h in batch:
            key = (h.url or "").strip().lower()
            if not key or not key.startswith("http"):
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = h
                order.append(key)
                continue
            ps = prev.score if prev.score is not None else -1.0
            ns = h.score if h.score is not None else -1.0
            if ns > ps:
                best[key] = h

    merged = [best[k] for k in order]
    merged.sort(
        key=lambda h: -(h.score if h.score is not None else 0.0),
    )
    trace(
        f"CURRICULUM exa multi_merge ✓ | batches={len(batches)} "
        f"unique={len(merged)} cap={cap}"
    )
    return merged


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
        exa_relevance_score=hit.score,
    )


def exa_response_to_curriculum_hits(
    response: ExaSearchResponse,
) -> list[CurriculumSearchHit]:
    capped = apply_exa_domain_cap(list(response.hits))
    return [exa_hit_to_curriculum_hit(h) for h in capped if h.url.startswith("http")]


@dataclass(frozen=True)
class ExaQueryPlan:
    """Multi-vector Exa specs plus the local pre-flight triage payload."""

    specs: list[ExaQuerySpec]
    keywords: list[str] = field(default_factory=list)
    core_theme: str = ""


async def build_exa_query_plan(
    context: str,
    *,
    anchor: str,
) -> ExaQueryPlan:
    """Multi-vector Exa plan (Lite); ≥1–2 Russian vectors guaranteed when possible."""
    ru = (context or "").strip()[:1200]
    if len(ru) < 8:
        return ExaQueryPlan(specs=[])

    from knowledge_engine.src.curriculum.lite_search_pipeline import _lite_structured

    trace("CURRICULUM exa query_plan ▶ | Lite multi-vector")
    try:
        out = await _lite_structured(
            _EXA_QUERY_PLAN_SYSTEM,
            json.dumps(
                {
                    "learning_context": ru,
                    "avoid": "swagger, openapi consoles, cloud console setup wizards",
                    "prefer": (
                        "official docs and PEPs, CPython/runtime internals, "
                        "architecture, production cases, Habr-style longreads"
                    ),
                },
                ensure_ascii=False,
            ),
            f"{anchor}:exa_query_plan",
            ExaQueryPlanOut,
            "curriculum / exa_query_plan",
        )
        specs: list[ExaQuerySpec] = []
        field_roles: list[tuple[str, ExaQueryRole]] = [
            ("en_declarative", "en_declarative"),
            ("en_technical", "en_technical"),
            ("en_edge_cases", "en_edge_cases"),
            ("ru_short", "ru_short"),
            ("ru_expert_article", "ru_expert_article"),
            ("ru_practical_cases", "ru_practical_cases"),
        ]
        for field_name, role in field_roles:
            q = (getattr(out, field_name) or "").strip()[:400]
            if len(q) < 8:
                continue
            hl = _EXA_HIGHLIGHT_BY_ROLE[role]
            specs.append(ExaQuerySpec(role=role, query=q, highlight_query=hl))

        ru_specs = [s for s in specs if s.role.startswith("ru_")]
        if len(ru_specs) < 1:
            fallback_ru = (out.ru_short or out.ru_expert_article or ru)[:400]
            if len(fallback_ru) >= 8:
                specs.append(
                    ExaQuerySpec(
                        role="ru_expert_article",
                        query=fallback_ru,
                        highlight_query=_EXA_HIGHLIGHT_BY_ROLE["ru_expert_article"],
                    )
                )

        if not specs:
            raise ValueError("empty query plan")

        keywords = [k.strip()[:80] for k in (out.keywords or []) if (k or "").strip()][
            :8
        ]
        core_theme = (out.core_theme or "").strip()[:600]

        roles = ",".join(s.role for s in specs)
        trace(
            f"CURRICULUM exa query_plan ✓ | vectors={len(specs)} roles={roles} "
            f"keywords={len(keywords)} core_theme={'set' if core_theme else 'empty'}"
        )
        return ExaQueryPlan(specs=specs, keywords=keywords, core_theme=core_theme)
    except Exception as exc:
        trace(f"CURRICULUM exa query_plan fallback | {exc}")
        ru_q, en_q = await build_exa_dual_queries(context, anchor=anchor)
        fallback: list[ExaQuerySpec] = []
        if len(en_q) >= 8:
            fallback.append(
                ExaQuerySpec(
                    role="en_declarative",
                    query=en_q,
                    highlight_query=_EXA_HIGHLIGHT_BY_ROLE["en_declarative"],
                )
            )
        if len(ru_q) >= 8:
            fallback.append(
                ExaQuerySpec(
                    role="ru_expert_article",
                    query=ru_q,
                    highlight_query=_EXA_HIGHLIGHT_BY_ROLE["ru_expert_article"],
                )
            )
        return ExaQueryPlan(specs=fallback)


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
                    "avoid": "swagger/openapi consoles, cloud console setup",
                    "prefer": "official specs/docs, architecture, implementation, deep dive",
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
        from knowledge_engine.src.curriculum.search_query_builder import (
            build_search_queries,
        )

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
    max_per_domain: int | None = None,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.curriculum.lite_search_pipeline import (
        batch_lite_eval_curriculum_hits,
    )

    focus = goal.strip()
    if core_concepts:
        focus = f"{focus} | concepts: {', '.join(core_concepts[:8])}"
    trace(f"CURRICULUM exa lite rerank ▶ | candidates={len(hits)} cap={cap}")
    approved = await batch_lite_eval_curriculum_hits(
        hits,
        focus[:1200],
        anchor=f"{anchor}:exa_rerank",
        strict=True,
    )
    if approved:
        trace(f"CURRICULUM exa lite rerank ✓ | approved={len(approved)}")
        per_dom = (
            max_per_domain
            if max_per_domain is not None
            else EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN
        )
        diversified = fair_domain_round_robin(
            approved,
            cap,
            max_per_domain=per_dom,
            get_url=lambda h: h.url,
        )
        if len(diversified) < min(cap, len(approved)):
            diversified = fill_round_robin_tail(
                diversified, approved, min(cap, len(approved)), get_url=lambda h: h.url
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
    max_num_results: int | None = None,
    include_domains: list[str] | None = None,
    search_type: str = "auto",
    category: str | None = None,
    allow_unrestricted_fallback: bool = False,
) -> list[ExaSearchHit]:
    if not (query or "").strip():
        return []
    from knowledge_engine.config import EXA_EXCLUDE_TEXT, EXA_PRACTICAL_HIGHLIGHT_QUERY
    from knowledge_engine.services.search.exa_client import normalize_exa_exclude_text

    cap = (
        max_num_results
        if max_num_results is not None
        else CURRICULUM_PRACTICAL_EXA_LIMIT
    )
    if exclude_text is not None:
        exc_text = normalize_exa_exclude_text(exclude_text)
    else:
        exc_text = normalize_exa_exclude_text(EXA_EXCLUDE_TEXT)
    hl = (highlight_query or EXA_PRACTICAL_HIGHLIGHT_QUERY).strip()
    response = await asyncio.to_thread(
        client.search,
        query.strip(),
        num_results=max(1, min(num_results, cap)),
        search_type=search_type,
        include_domains=include_domains,
        exclude_text=exc_text,
        category=category,
        highlight_query=hl,
        allow_unrestricted_fallback=allow_unrestricted_fallback,
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
    """DEEP-нода: multi-vector Exa, domain cap, Lite rerank при избытке кандидатов."""
    if not EXA_SEARCH_ENABLED or not EXA_API_KEY:
        trace("CURRICULUM exa ⊘ | EXA_API_KEY not set or EXA_SEARCH_ENABLED=false")
        return []

    context = _learning_context_for_node(node, course_goal)
    from knowledge_engine.services.search.exa_domain_validate import (
        prepare_exa_pass1_domains,
    )
    from knowledge_engine.services.search.exa_source_expand import (
        absorb_new_exa_hosts,
        exa_pass2_categories,
        expand_search_context_with_flash_lite,
        filter_pass1_official_hosts,
    )

    expansion = await asyncio.to_thread(expand_search_context_with_flash_lite, context)
    live_domains = await prepare_exa_pass1_domains(expansion.primary_domains)
    validated_domains = filter_pass1_official_hosts(live_domains)
    docs_exclude: list[str] | None = [] if expansion.include_official_docs else None

    qplan = await build_exa_query_plan(
        context,
        anchor=f"{anchor}:practical:{node.node_id}",
    )
    plan = qplan.specs
    if not plan:
        return []

    from knowledge_engine.src.curriculum.practical_url_filters import (
        filter_practical_search_row,
    )
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_collectible_article_url,
    )

    per_vector = max(3, EXA_FETCH_NUM_RESULTS // max(1, len(plan)))
    fetch_cap = (
        CURRICULUM_PREFLIGHT_FETCH_CAP
        if CURRICULUM_PREFLIGHT_ENABLED
        else max(cap + 8, cap * 2, EXA_FETCH_NUM_RESULTS)
    )
    recall_per_domain = max(1, EXA_RECALL_MAX_PER_DOMAIN)
    sem = asyncio.Semaphore(max(1, EXA_MAX_CONCURRENT_SEARCH))

    trace(
        f"CURRICULUM exa ▶ | node={node.node_id} vectors={len(plan)} "
        f"per_vector={per_vector} fetch_cap={fetch_cap} "
        f"max_per_domain={recall_per_domain} concurrency={EXA_MAX_CONCURRENT_SEARCH}"
    )

    async def _search_vectors(
        *,
        include_domains: list[str],
        category: str | None,
    ) -> list[list[ExaSearchHit]]:
        async def _one(spec: ExaQuerySpec) -> list[ExaSearchHit]:
            docs_lane = spec.role in ("en_technical", "en_declarative")
            async with sem:
                return await _exa_search_one(
                    client,
                    spec.query,
                    per_vector,
                    highlight_query=spec.highlight_query,
                    max_num_results=EXA_FETCH_NUM_RESULTS,
                    include_domains=include_domains,
                    search_type=(expansion.search_type if docs_lane else "auto"),
                    category=category,
                    exclude_text=docs_exclude,
                    allow_unrestricted_fallback=False,
                )

        return list(await asyncio.gather(*[_one(s) for s in plan]))

    # Квота, которую реально нужно набрать для ноды — не путать с `fetch_cap`
    # (широкая recall-ёмкость round-robin'а). Двухэтапный добор (Шаг А/Б) и
    # решение о сетевом Pass 2 меряются именно по ней, а не по голому
    # raw_total: раньше Pass 2 триггерился ТОЛЬКО при raw_total==0, из-за чего
    # прогон с raw_total=18, но domains=1 (все хиты с одного хоста —
    # обычная непредсказуемость Exa между запросами) молча отдавал 2
    # источника вместо 4, хотя сетевой добор мог бы найти остальные.
    target_cap = (
        min(cap, CURRICULUM_PREFLIGHT_FINAL_ARTICLES)
        if CURRICULUM_PREFLIGHT_ENABLED
        else cap
    )

    def _merge_and_cap(
        all_batches: list[list[ExaSearchHit]],
    ) -> tuple[list[ExaSearchHit], list[ExaSearchHit]]:
        merged = merge_multi_vector_exa_hits(all_batches, cap=fetch_cap)
        capped = fair_domain_round_robin(
            merged,
            fetch_cap,
            max_per_domain=recall_per_domain,
            get_url=lambda h: h.url,
        )
        if len(capped) < target_cap:
            before = len(capped)
            capped = fill_round_robin_tail(
                capped, merged, target_cap, get_url=lambda h: h.url
            )
            if len(capped) > before:
                trace(
                    f"CURRICULUM exa tail_fill ✓ | {before}→{len(capped)} "
                    f"target_cap={target_cap} (добор из хвоста round-robin'а)"
                )
        return merged, capped

    try:
        client = ExaSearchClient(api_key=EXA_API_KEY)
        batches: list[list[ExaSearchHit]] = []
        raw_total = 0
        merged_raw: list[ExaSearchHit] = []
        capped_raw: list[ExaSearchHit] = []
        if validated_domains:
            trace(
                f"CURRICULUM exa pass 1 ▶ | include_domains={validated_domains} "
                f"category=None"
            )
            batches = await _search_vectors(
                include_domains=validated_domains,
                category=None,
            )
            raw_total = sum(len(b) for b in batches)
            trace(f"CURRICULUM exa pass 1 ✓ | hits={raw_total}")
            if raw_total:
                merged_raw, capped_raw = _merge_and_cap(batches)

        # Сетевой Pass 2 — ТОЛЬКО если после Pass 1 + добора из хвоста всё
        # ещё не хватает до target_cap (не по голому raw_total==0).
        if len(capped_raw) < target_cap:
            for cat in exa_pass2_categories(expansion):
                trace(
                    f"CURRICULUM exa pass 2 ▶ | include_domains=∅ category={cat} "
                    f"(after_pass1_and_tail={len(capped_raw)}/{target_cap})"
                )
                extra_batches = await _search_vectors(
                    include_domains=[],
                    category=cat,
                )
                extra_total = sum(len(b) for b in extra_batches)
                raw_total += extra_total
                trace(f"CURRICULUM exa pass 2 ✓ | category={cat} hits={extra_total}")
                if extra_total:
                    batches = batches + extra_batches
                    merged_raw, capped_raw = _merge_and_cap(batches)
                    if len(capped_raw) >= target_cap:
                        break
            if len(capped_raw) < target_cap and expansion.use_broader_search:
                extra = await _exa_search_one(
                    client,
                    plan[0].query,
                    fetch_cap,
                    highlight_query=plan[0].highlight_query,
                    max_num_results=EXA_FETCH_NUM_RESULTS,
                    include_domains=[],
                    search_type=expansion.search_type,
                    exclude_text=docs_exclude,
                    allow_unrestricted_fallback=False,
                )
                raw_total += len(extra)
                trace(f"CURRICULUM exa pass 2 ✓ | category=None hits={len(extra)}")
                if extra:
                    batches = batches + [extra]
                    merged_raw, capped_raw = _merge_and_cap(batches)
    except ExaNotConfiguredError as exc:
        trace(f"CURRICULUM exa ✗ | {exc}")
        return []
    except Exception as exc:
        trace(f"CURRICULUM exa ✗ | {exc}")
        return []

    trace(
        f"CURRICULUM exa raw hits | total={raw_total} from_vectors={len(batches)} "
        f"capped={len(capped_raw)}/{target_cap}"
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

    trace(
        f"CURRICULUM exa after_collectible+practical | hits={len(out)} "
        f"(from_ranked={len(hits)})"
    )

    if CURRICULUM_PREFLIGHT_ENABLED and qplan.core_theme:
        from knowledge_engine.src.curriculum.pre_flight_triage import (
            run_pre_flight_triage,
        )

        out = await run_pre_flight_triage(
            out,
            core_theme=qplan.core_theme,
            keywords=qplan.keywords,
            final_articles=min(cap, CURRICULUM_PREFLIGHT_FINAL_ARTICLES),
        )
    else:
        threshold = max(1, EXA_RERANK_LITE_THRESHOLD)
        if len(out) > threshold:
            out = await _lite_rerank_exa_hits(
                out,
                context,
                list(node.core_concepts or []),
                anchor=f"{anchor}:practical:{node.node_id}",
                cap=cap,
                max_per_domain=recall_per_domain,
            )
        else:
            before_cap_robin = out
            out = fair_domain_round_robin(
                out,
                cap,
                max_per_domain=recall_per_domain,
                get_url=lambda h: h.url,
            )
            if len(out) < min(cap, len(before_cap_robin)):
                out = fill_round_robin_tail(
                    out,
                    before_cap_robin,
                    min(cap, len(before_cap_robin)),
                    get_url=lambda h: h.url,
                )
            out = out[:cap]

    skip_n = sum(1 for h in out if h.skip_ollama_summary)
    absorb_new_exa_hosts([h.url for h in out])
    trace(
        f"CURRICULUM exa ✓ | node={node.node_id} hits={len(out)} "
        f"skip_ollama={skip_n} intent={expansion.intent} "
        f"primary_domains={len(validated_domains)}"
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
    from knowledge_engine.src.curriculum.practical_url_filters import (
        filter_practical_search_row,
    )
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_collectible_article_url,
    )

    trace(f"CURRICULUM exa ▶ | {q[:120]}")
    try:
        client = ExaSearchClient(api_key=EXA_API_KEY)
        response = await asyncio.to_thread(
            client.search_expanded,
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
