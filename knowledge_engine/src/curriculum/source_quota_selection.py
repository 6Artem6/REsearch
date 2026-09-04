"""Pre-ingest quota buckets: academic vs practical до heavy ingest."""

from __future__ import annotations

from typing import Literal

from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    academic_source_dedupe_key,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.src.curriculum.source_quota_policy import (
    SourceQuota,
    get_source_quota,
)
from knowledge_engine.src.parsers.paper_structure_analyzer import is_academic_pdf_url
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_academic_open_host,
)
from knowledge_engine.ui.run_log import trace

ContentBucket = Literal["academic", "practical"]

_ACADEMIC_TIERS = frozenset(
    {
        "consensus",
        "arxiv",
        "semantic_scholar",
        "searxng_science",
        "openalex",
        "academic",
    }
)


def classify_hit_content_bucket(hit: CurriculumSearchHit) -> ContentBucket:
    tier = (hit.source_tier or "").strip().lower()
    if tier in _ACADEMIC_TIERS or tier.startswith("consensus"):
        return "academic"
    url = (hit.url or "").strip().lower()
    if is_academic_open_host(url) or is_academic_pdf_url(url):
        return "academic"
    if "arxiv.org" in url or "biorxiv.org" in url or "medrxiv.org" in url:
        return "academic"
    return "practical"


def _hit_selection_score(hit: CurriculumSearchHit) -> float:
    if hit.exa_relevance_score is not None:
        return float(hit.exa_relevance_score)
    words = sum(len((e or "").split()) for e in (hit.key_extracts or []))
    snippet = (hit.snippet or "").strip()
    if snippet:
        words += min(len(snippet.split()), 120)
    tier = (hit.source_tier or "").strip().lower()
    tier_boost = 0.15 if tier in _ACADEMIC_TIERS else 0.05 if tier == "exa" else 0.0
    return words / 200.0 + tier_boost


def select_hits_by_quota(
    candidates: list[CurriculumSearchHit],
    quota: SourceQuota,
    *,
    limit: int | None = None,
) -> list[CurriculumSearchHit]:
    """
    TOP-N academic + TOP-M practical с fallback до total_max (без url_validate).

    ``limit`` (опционально) расширяет итоговый предел сверх ``quota.total_max``
    — нужен replenish_valid_hits_until_cap's backfill_margin, чтобы держать
    лишних ранжированных кандидатов в резерве для добора ALIAS/пустых
    extracts ниже по пайплайну (см. targeted_hit_replenishment.py). TOP-N/
    TOP-M выборки по бакетам (academic_max/practical_max) не расширяются —
    растёт только fallback-долив и финальный срез.
    """
    if not candidates:
        return []

    academic: list[CurriculumSearchHit] = []
    practical: list[CurriculumSearchHit] = []
    for h in candidates:
        if classify_hit_content_bucket(h) == "academic":
            academic.append(h)
        else:
            practical.append(h)

    academic.sort(key=_hit_selection_score, reverse=True)
    practical.sort(key=_hit_selection_score, reverse=True)

    picked_a = academic[: max(0, quota.academic_max)]
    picked_p = practical[: max(0, quota.practical_max)]
    selected: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    def _add(lst: list[CurriculumSearchHit]) -> None:
        for h in lst:
            key = academic_source_dedupe_key(h.url)
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(h)

    _add(picked_a)
    _add(picked_p)

    effective_total = (
        max(quota.total_max, limit) if limit is not None else quota.total_max
    )

    need = max(0, effective_total - len(selected))
    if need > 0:
        fallback: list[CurriculumSearchHit] = []
        if len(picked_a) < quota.academic_max:
            fallback.extend(practical[len(picked_p) :])
            fallback.extend(academic[len(picked_a) :])
        elif len(picked_p) < quota.practical_max:
            fallback.extend(academic[len(picked_a) :])
            fallback.extend(practical[len(picked_p) :])
        else:
            fallback.extend(academic[len(picked_a) :])
            fallback.extend(practical[len(picked_p) :])
        for h in fallback:
            if len(selected) >= effective_total:
                break
            key = academic_source_dedupe_key(h.url)
            if key and key not in seen:
                seen.add(key)
                selected.append(h)

    trace(
        f"CURRICULUM quota select | academic_pool={len(academic)} "
        f"practical_pool={len(practical)} picked={len(selected)} "
        f"want_a={quota.academic_max} want_p={quota.practical_max} "
        f"total_max={quota.total_max} limit={effective_total}"
    )
    return selected[:effective_total]


def quota_for_node(node: CurriculumNode) -> SourceQuota:
    return get_source_quota(node.layer, node.node_risk_kind)


def order_candidates_for_node(
    candidates: list[CurriculumSearchHit],
    node: CurriculumNode,
    *,
    limit: int | None = None,
) -> list[CurriculumSearchHit]:
    quota = quota_for_node(node)
    trace(
        f"CURRICULUM quota ▶ | node={node.node_id} layer={node.layer} "
        f"risk={node.node_risk_kind} a={quota.academic_max} p={quota.practical_max} "
        f"total={quota.total_max}"
    )
    return select_hits_by_quota(candidates, quota, limit=limit)
