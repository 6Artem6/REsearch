"""Курация hits: архив, Lite batch-валидация, динамический пул в .source_archive."""

from __future__ import annotations

import asyncio

from knowledge_engine.config import (
    CURRICULUM_SEARCH_TARGET_HITS,
    SOURCE_ARCHIVE_ENABLED,
)
from knowledge_engine.services.search.url_filter import url_priority_score
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
    is_fast_trusted_source,
    resolve_source_provenance,
)
from knowledge_engine.ui.run_log import trace

_TIER_RANK = {
    "consensus": 0,
    "gemini_grounding": 1,
    "gemini_web": 2,
    "lite_suggested_site": 3,
    "whitelist_blog": 4,
    "archive": 5,
    "": 6,
}


def _anchor_from_goal(goal: str) -> str:
    return f"curriculum_sources:{(goal or '').strip()[:500]}"


def collect_archived_practical_hits(
    target_goal: str,
    *,
    exclude_url_keys: set[str] | None = None,
    limit: int = 6,
    strict: bool = False,
) -> list[CurriculumSearchHit]:
    """Повторное использование URL из .source_archive (Lite-approved + static whitelist)."""
    if not SOURCE_ARCHIVE_ENABLED or limit <= 0:
        return []
    exclude = exclude_url_keys or set()
    min_trust = 0.55 if strict else 0.45
    try:
        from knowledge_engine.db.source_links import get_source_link_archive

        archive = get_source_link_archive()
        urls = archive.get_reusable_urls(
            target_goal,
            explored=set(),
            limit=limit * 4,
            min_trust=min_trust,
            high_trust_only=False,
        )
    except Exception as exc:
        trace(f"CURRICULUM archive skip | {exc}")
        return []

    anchor = _anchor_from_goal(target_goal)
    out: list[CurriculumSearchHit] = []
    pending_strict: list[CurriculumSearchHit] = []

    for url in urls:
        if len(out) >= limit:
            break
        key = _normalize_url_key(url)
        if not key or key in exclude:
            continue
        if not is_collectible_article_url(url):
            continue
        cat, _origin = resolve_source_provenance(url)
        hit = CurriculumSearchHit(
            url=url,
            title=url[:400],
            snippet=f"Reused from source archive ({cat}).",
            source_tier="archive",
        )
        if strict:
            if is_fast_trusted_source(url):
                exclude.add(key)
                out.append(hit)
                continue
            pending_strict.append(hit)
            continue
        exclude.add(key)
        out.append(hit)

    if strict and pending_strict and len(out) < limit:
        from knowledge_engine.src.curriculum.lite_search_pipeline import (
            batch_evaluate_sources_sync,
        )

        batch_src = [
            {
                "id": i,
                "url": h.url,
                "title": h.title,
                "snippet": h.snippet or "",
            }
            for i, h in enumerate(pending_strict, start=1)
        ]
        evals = batch_evaluate_sources_sync(
            target_goal,
            batch_src,
            anchor=f"{anchor}:archive",
        )
        approved_ids = {e.id for e in evals if e.status == "APPROVED"}
        for i, h in enumerate(pending_strict, start=1):
            if len(out) >= limit:
                break
            if i not in approved_ids:
                continue
            key = _normalize_url_key(h.url)
            if key and key not in exclude:
                exclude.add(key)
                out.append(h)

    if out:
        trace(f"CURRICULUM archive ✓ | reused={len(out)} strict={strict}")
    return out


def rank_practical_hits(hits: list[CurriculumSearchHit]) -> list[CurriculumSearchHit]:
    """Tier + domain trust score (как discovery_collect)."""

    def sort_key(h: CurriculumSearchHit) -> tuple[int, float]:
        tier = (h.source_tier or "").strip().lower()
        return (_TIER_RANK.get(tier, 9), -url_priority_score(h.url))

    return sorted(hits, key=sort_key)


async def curate_practical_hits_async(
    hits: list[CurriculumSearchHit],
    target_goal: str,
    *,
    anchor: str | None = None,
    max_out: int | None = None,
    lite_batch: bool = True,
) -> list[CurriculumSearchHit]:
    """Ранжирование + опциональный пакетный Lite-gate на title/snippet."""
    if not hits:
        return hits
    cap = max_out if max_out is not None else CURRICULUM_SEARCH_TARGET_HITS
    anchor = anchor or _anchor_from_goal(target_goal)
    ranked = rank_practical_hits(hits)

    collectible: list[CurriculumSearchHit] = []
    for h in ranked:
        if len(collectible) >= cap * 2:
            break
        if is_collectible_article_url(h.url):
            collectible.append(h)

    if lite_batch:
        from knowledge_engine.src.curriculum.lite_search_pipeline import (
            batch_lite_eval_curriculum_hits,
        )

        kept = await batch_lite_eval_curriculum_hits(
            collectible,
            target_goal,
            anchor=f"{anchor}:batch",
        )
    else:
        kept = collectible

    kept = kept[:cap]
    trace(
        f"CURRICULUM hits curate ✓ | in={len(hits)} collectible={len(collectible)} "
        f"out={len(kept)} lite_batch={lite_batch}"
    )
    return kept


def curate_practical_hits(
    hits: list[CurriculumSearchHit],
    target_goal: str,
    *,
    anchor: str | None = None,
    max_out: int | None = None,
    lite_review_cap: int = 16,
    lite_batch: bool = True,
) -> list[CurriculumSearchHit]:
    """Sync обёртка; lite_review_cap оставлен для совместимости вызовов."""
    del lite_review_cap

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            curate_practical_hits_async(
                hits,
                target_goal,
                anchor=anchor,
                max_out=max_out,
                lite_batch=lite_batch,
            )
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            curate_practical_hits_async(
                hits,
                target_goal,
                anchor=anchor,
                max_out=max_out,
                lite_batch=lite_batch,
            ),
        ).result()
