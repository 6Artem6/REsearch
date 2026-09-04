"""
Production replenishment for DEEP node grounding (see targeted_node_search).

After Lite approve, walk candidates in order and fill up to cap using shared
URL policy: practical filters, SQLite domain blocklist, live url_validate.
"""

from __future__ import annotations

from knowledge_engine.config import (
    CURRICULUM_PREFLIGHT_ENABLED,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
)
from knowledge_engine.db.domain_blocklist import (
    add_blocked_domain,
    load_blocked_domain_set,
)
from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    academic_source_dedupe_key,
    canonicalize_academic_url_pure,
    canonicalize_curriculum_hit,
)
from knowledge_engine.src.curriculum.practical_url_filters import (
    practical_url_reject_reason,
)
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.src.curriculum.source_quota_selection import (
    order_candidates_for_node,
    quota_for_node,
)
from knowledge_engine.src.curriculum.url_validate import check_url_live
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_academic_open_host,
    is_collectible_article_url,
)
from knowledge_engine.ui.run_log import trace

_BLOCKLIST_ON_HTTP_REASONS = frozenset(
    {
        "http_403",
        "http_401",
        "http_429",
        "http_503",
    }
)


_ACADEMIC_URL_PRECHECK_TIERS = frozenset(
    {
        "consensus",
        "arxiv",
        "semantic_scholar",
        "searxng_science",
        "openalex",
        "academic",
        "exa",
    }
)


def _hit_skips_practical_url_precheck(hit: CurriculumSearchHit) -> bool:
    tier = (hit.source_tier or "").strip().lower()
    if tier in _ACADEMIC_URL_PRECHECK_TIERS or tier.startswith("consensus"):
        return True
    if tier == "searxng":
        return False
    return is_academic_open_host(hit.url)


def precheck_candidate_url(
    url: str,
    *,
    blocked_domains: set[str],
    skip_practical_filter: bool = False,
) -> str | None:
    """Static gates before HTTP probe; None = proceed to url_validate."""
    u = (url or "").strip()
    pure = canonicalize_academic_url_pure(u)
    if pure:
        u = pure
    if not u.startswith("http"):
        return "not_http"
    if not skip_practical_filter:
        practical = practical_url_reject_reason(u)
        if practical:
            return practical
    if not is_collectible_article_url(u):
        return "not_collectible"
    from knowledge_engine.db.domain_blocklist import extract_domain_from_url

    dom = extract_domain_from_url(u)
    if dom and dom in blocked_domains:
        return f"domain_blocklist:{dom}"
    return None


def precheck_candidate_hit(
    hit: CurriculumSearchHit,
    *,
    blocked_domains: set[str],
) -> str | None:
    return precheck_candidate_url(
        hit.url,
        blocked_domains=blocked_domains,
        skip_practical_filter=_hit_skips_practical_url_precheck(hit),
    )


def _maybe_blocklist_domain(url: str, reason: str) -> None:
    if reason in _BLOCKLIST_ON_HTTP_REASONS:
        dom = add_blocked_domain(url, f"replenish_{reason}")
        if dom:
            trace(f"CURRICULUM replenish blocklist + | domain={dom} reason={reason}")


async def replenish_valid_hits_until_cap(
    candidates: list[CurriculumSearchHit],
    cap: int,
    *,
    timeout: float | None = None,
    node: CurriculumNode | None = None,
    backfill_margin: int = 0,
) -> list[CurriculumSearchHit]:
    """
    Quota buckets (layer × risk) → precheck + url_validate until ``cap`` hits.

    ``backfill_margin`` (see DEEP_INGEST_BACKFILL_MARGIN) widens the stopping
    point to ``cap + backfill_margin`` valid hits, drawn from the same
    already-fetched ``candidates`` pool (no extra network calls) — the extra
    margin hits give downstream Pre-MAP Dedup (_ingest_blog_hits_batch_async)
    a reserve to backfill ALIAS drops from instead of shrinking the final set.
    """
    if cap <= 0 or not candidates:
        return []

    ordered = list(candidates)
    if node is not None:
        quota = quota_for_node(node)
        cap = min(cap, quota.total_max)
        # RU: раньше order_candidates_for_node всегда резал до quota.total_max
        # ДО того, как цикл ниже успевал воспользоваться backfill_margin —
        # margin физически не мог найти запасные кандидаты, даже если сырой
        # candidates-пул был больше total_max. limit= расширяет срез до
        # эффективной цели (cap+margin), не трогая сами TOP-N/TOP-M выборки.
        ordered = order_candidates_for_node(
            candidates, node, limit=cap + max(0, backfill_margin)
        )

    effective_cap = cap + max(0, backfill_margin)
    tmo = timeout if timeout is not None else CURRICULUM_URL_VALIDATE_TIMEOUT_SEC
    blocked_domains = load_blocked_domain_set()
    valid: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    skipped = 0

    for hit in ordered:
        if len(valid) >= effective_cap:
            break
        hit = await canonicalize_curriculum_hit(hit)
        raw_url = (hit.url or "").strip()
        pre = precheck_candidate_hit(hit, blocked_domains=blocked_domains)
        if pre:
            skipped += 1
            trace(f"CURRICULUM replenish ⊘ | {raw_url[:70]} | {pre}")
            continue
        key = academic_source_dedupe_key(hit.url)
        if not key:
            continue
        if key in seen:
            continue

        # Exa hits already passed Pre-Flight Triage Stage 2 (parallel GET +
        # liveness/soft-404 check on the downloaded body) before reaching
        # this pool — a second sequential check_url_live() HEAD+GET round
        # trip here would be pure double HTTP for the same URL.
        if (
            CURRICULUM_PREFLIGHT_ENABLED
            and (hit.source_tier or "").strip().lower() == "exa"
        ):
            seen.add(key)
            valid.append(hit)
            continue

        ok, reason = await check_url_live(raw_url, timeout=tmo)
        if not ok:
            skipped += 1
            trace(f"CURRICULUM replenish ⊘ | {raw_url[:70]} | {reason}")
            _maybe_blocklist_domain(raw_url, reason)
            continue

        seen.add(key)
        valid.append(hit)

    trace(
        f"CURRICULUM replenish ✓ | candidates={len(candidates)} "
        f"valid={len(valid)} cap={cap} backfill_margin={backfill_margin} "
        f"skipped={skipped} blocklist_domains={len(blocked_domains)}"
    )
    return valid
