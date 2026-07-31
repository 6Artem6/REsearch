"""Этап 3: точечный поиск только для DEEP-нод (без fallback approve)."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST,
    CURRICULUM_DEEP_NODE_MAX_HITS,
)
from knowledge_engine.src.curriculum.academic_consensus import (
    consensus_allowed_for_policy,
    is_sota_rd_node,
)
from knowledge_engine.src.curriculum.academic_source_fetch import fetch_academic_sources_async
from knowledge_engine.src.curriculum.lite_search_pipeline import (
    batch_lite_eval_curriculum_hits,
)
from knowledge_engine.src.curriculum.practical_searxng_search import collect_searxng_practical_rows
from knowledge_engine.services.search.exa_transform import fetch_exa_curriculum_hits_for_node
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.source_policy import normalize_source_policy
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
    is_academic_open_host,
    resolve_source_provenance,
)
from knowledge_engine.ui.run_log import trace

_ACADEMIC_SOURCE_TIERS = frozenset(
    {
        "consensus",
        "arxiv",
        "semantic_scholar",
        "searxng_science",
        "openalex",
        "academic",
    }
)


def _is_academic_tier_hit(hit: CurriculumSearchHit) -> bool:
    tier = (hit.source_tier or "").strip().lower()
    if tier in _ACADEMIC_SOURCE_TIERS or tier.startswith("consensus"):
        return True
    return is_academic_open_host(hit.url)


def _node_search_goal(node: CurriculumNode, course_goal: str) -> str:
    parts = [
        course_goal.strip()[:400],
        node.title.strip(),
        ", ".join(node.core_concepts[:6]),
        (node.brief_summary or "")[:500],
    ]
    text = " | ".join(p for p in parts if p)
    return text[:1200]


def _hit_extract_word_count(hit: CurriculumSearchHit) -> int:
    return sum(len((e or "").split()) for e in (hit.key_extracts or []))


def _merge_hits(
    parts: list[list[CurriculumSearchHit]],
    cap: int,
    exclude: set[str],
) -> list[CurriculumSearchHit]:
    out: list[CurriculumSearchHit] = []
    for batch in parts:
        for h in batch:
            key = _normalize_url_key(h.url)
            if not key or key in exclude:
                continue
            exclude.add(key)
            out.append(h)
            if len(out) >= cap:
                return out
    return out


def _merge_hits_interleaved(
    parts: list[list[CurriculumSearchHit]],
    cap: int,
    exclude: set[str],
) -> list[CurriculumSearchHit]:
    """Round-robin по провайдерам — не вытеснять Exa/SS одним блоком Consensus."""
    buckets = [list(p) for p in parts if p]
    if not buckets or cap <= 0:
        return []
    indices = [0] * len(buckets)
    out: list[CurriculumSearchHit] = []
    while len(out) < cap:
        took = False
        for i, batch in enumerate(buckets):
            if len(out) >= cap:
                break
            while indices[i] < len(batch):
                h = batch[indices[i]]
                indices[i] += 1
                key = _normalize_url_key(h.url)
                if not key or key in exclude:
                    continue
                exclude.add(key)
                out.append(h)
                took = True
                break
        if not took:
            break
    return out


def _log_provider_hit_counts(
    label: str,
    node_id: str,
    hits: list[CurriculumSearchHit],
) -> None:
    exa_n = sum(1 for h in hits if (h.source_tier or "").strip().lower() == "exa")
    consensus_n = sum(
        1
        for h in hits
        if (h.source_tier or "").strip().lower() == "consensus"
        or (h.source_tier or "").strip().lower().startswith("consensus")
    )
    arxiv_n = sum(1 for h in hits if (h.source_tier or "").strip().lower() == "arxiv")
    ss_n = sum(
        1 for h in hits if (h.source_tier or "").strip().lower() == "semantic_scholar"
    )
    searx_pr = sum(1 for h in hits if (h.source_tier or "").strip().lower() == "searxng")
    searx_ac = sum(
        1 for h in hits if (h.source_tier or "").strip().lower() == "searxng_science"
    )
    trace(
        f"[SEARCH] {label} | node={node_id} | "
        f"Exa={exa_n} Academic/Consensus={consensus_n} arXiv={arxiv_n} "
        f"SemanticScholar={ss_n} SearXNG={searx_pr + searx_ac} "
        f"(practical={searx_pr} science={searx_ac}) total={len(hits)}"
    )


async def _lite_approve_merged_hits(
    hits: list[CurriculumSearchHit],
    goal: str,
    *,
    anchor: str,
    node_id: str,
) -> list[CurriculumSearchHit]:
    """Academic/consensus: мягкий Lite; practical/exa: строгий."""
    academic: list[CurriculumSearchHit] = []
    practical: list[CurriculumSearchHit] = []
    for h in hits:
        if _is_academic_tier_hit(h):
            academic.append(h)
        else:
            practical.append(h)

    parts: list[list[CurriculumSearchHit]] = []
    cap = CURRICULUM_DEEP_NODE_MAX_HITS * 2

    async def _approve_academic_part() -> list[CurriculumSearchHit]:
        if not academic:
            return []
        approved_ac = await batch_lite_eval_curriculum_hits(
            academic,
            goal,
            anchor=f"{anchor}:batch_academic:{node_id}",
            strict=False,
        )
        trace(
            f"CURRICULUM targeted lite academic | node={node_id} "
            f"in={len(academic)} approved={len(approved_ac)} strict=False"
        )
        if approved_ac:
            return approved_ac
        return []

    async def _approve_practical_part() -> list[CurriculumSearchHit]:
        if not practical:
            return []
        approved_pr = await batch_lite_eval_curriculum_hits(
            practical,
            goal,
            anchor=f"{anchor}:batch_practical:{node_id}",
            strict=True,
        )
        trace(
            f"CURRICULUM targeted lite practical | node={node_id} "
            f"in={len(practical)} approved={len(approved_pr)} strict=True"
        )
        if approved_pr:
            return approved_pr
        return []

    if academic and practical:
        ac_part, pr_part = await asyncio.gather(
            _approve_academic_part(),
            _approve_practical_part(),
        )
        if ac_part:
            parts.append(ac_part)
        if pr_part:
            parts.append(pr_part)
    elif academic:
        ac_part = await _approve_academic_part()
        if ac_part:
            parts.append(ac_part)
    elif practical:
        pr_part = await _approve_practical_part()
        if pr_part:
            parts.append(pr_part)

    return _merge_hits_interleaved(parts, cap, set())


async def _practical_hits_for_node(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
) -> list[CurriculumSearchHit]:
    cap = CURRICULUM_DEEP_NODE_MAX_HITS + 2
    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    exa_hits = await fetch_exa_curriculum_hits_for_node(
        node,
        course_goal,
        anchor=anchor,
        cap=cap,
    )
    # Exa query_en/query_ru: Lite prompt в exa_transform (architecture, не API docs)
    _log_provider_hit_counts("Exa pass", node.node_id, exa_hits)
    for h in exa_hits:
        key = _normalize_url_key(h.url)
        if not key or key in seen:
            continue
        seen.add(key)
        hits.append(h)
        if len(hits) >= cap:
            trace(
                f"CURRICULUM targeted practical ✓ | node={node.node_id} "
                f"exa_only hits={len(hits)}"
            )
            return hits

    need = cap - len(hits)
    if need > 0:
        vec = _node_search_goal(node, course_goal)
        lite_plan = need > 2
        max_q = min(3, need + 1) if not lite_plan else None
        rows = await collect_searxng_practical_rows(
            vec,
            limit=need + 2,
            anchor=f"{anchor}:practical:{node.node_id}",
            lite_query_plan=lite_plan,
            max_queries=max_q,
        )
        for row in rows:
            url = (row.get("url") or "").strip()
            if not is_collectible_article_url(url):
                continue
            key = _normalize_url_key(url)
            if not key or key in seen:
                continue
            seen.add(key)
            snippet = (row.get("snippet") or "")[:1200]
            title = (row.get("title") or url)[:400]
            extracts = [snippet[:800]] if len(snippet) >= 80 else []
            hits.append(
                CurriculumSearchHit(
                    url=url,
                    title=title,
                    snippet=snippet,
                    key_extracts=extracts,
                    source_tier="searxng",
                )
            )
            if len(hits) >= cap:
                break

    exa_n = sum(1 for h in hits if h.source_tier == "exa")
    _log_provider_hit_counts("practical_merged", node.node_id, hits)
    trace(
        f"CURRICULUM targeted practical ✓ | node={node.node_id} "
        f"hits={len(hits)} exa={exa_n} searxng={len(hits) - exa_n}"
    )
    return hits


async def _academic_hits_for_node(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
    source_policy: str,
    on_demand: bool = False,
    registry_entries: list | None = None,
    exclude_url_keys: set[str] | None = None,
) -> list[CurriculumSearchHit]:
    vec = _node_search_goal(node, course_goal)
    allow_consensus = consensus_allowed_for_policy(source_policy)
    hits = await fetch_academic_sources_async(
        vec,
        node=node,
        anchor=f"{anchor}:academic:{node.node_id}",
        min_hits=CURRICULUM_DEEP_NODE_MAX_HITS,
        allow_consensus=allow_consensus,
        on_demand=on_demand,
        registry_entries=registry_entries,
        exclude_url_keys=exclude_url_keys,
    )
    out = hits[:CURRICULUM_DEEP_NODE_MAX_HITS + 2]
    _log_provider_hit_counts("academic_merged", node.node_id, out)
    return out


async def _gather_practical_and_academic(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
    source_policy: str,
    on_demand: bool,
    registry_entries: list | None,
    exclude_url_keys: set[str],
) -> tuple[list[CurriculumSearchHit], list[CurriculumSearchHit]]:
    from knowledge_engine.config import (
        CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC,
        CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN,
    )

    p_task = asyncio.create_task(
        _practical_hits_for_node(node, course_goal, anchor=anchor)
    )
    a_task = asyncio.create_task(
        _academic_hits_for_node(
            node,
            course_goal,
            anchor=anchor,
            source_policy=source_policy,
            on_demand=on_demand,
            registry_entries=registry_entries,
            exclude_url_keys=exclude_url_keys,
        )
    )

    if (
        on_demand
        and CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC > 0
    ):
        practical_hits = await p_task
        academic_hits: list[CurriculumSearchHit] = []
        if len(practical_hits) >= CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN:
            try:
                academic_hits = await asyncio.wait_for(
                    a_task,
                    timeout=CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC,
                )
            except asyncio.TimeoutError:
                trace(
                    f"CURRICULUM on_demand fast-return ▶ | node={node.node_id} "
                    f"practical={len(practical_hits)} "
                    f"academic_wait>{CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC:.0f}s"
                )
                a_task.cancel()
                try:
                    await a_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                academic_hits = []
        else:
            practical_hits, academic_hits = await asyncio.gather(p_task, a_task)
        return practical_hits, academic_hits

    practical_hits, academic_hits = await asyncio.gather(p_task, a_task)
    return practical_hits, academic_hits


async def search_sources_for_deep_node_async(
    node: CurriculumNode,
    course_goal: str,
    *,
    source_policy: str,
    anchor: str,
    exclude_url_keys: set[str],
    on_demand: bool = False,
    registry_entries: list | None = None,
) -> list[CurriculumSearchHit]:
    policy = normalize_source_policy(source_policy, default="hybrid")
    goal = _node_search_goal(node, course_goal)
    trace(
        f"CURRICULUM targeted search ▶ | node={node.node_id} "
        f"policy={policy} on_demand={on_demand} goal={goal[:80]}…"
    )

    cap = CURRICULUM_DEEP_NODE_MAX_HITS
    exclude = set(exclude_url_keys)

    practical_hits: list[CurriculumSearchHit] = []
    academic_hits: list[CurriculumSearchHit] = []

    if policy == "practical_only":
        practical_hits = await _practical_hits_for_node(
            node, course_goal, anchor=anchor
        )
        if node.node_risk_kind == "DEEP" and is_sota_rd_node(node):
            trace(
                f"CURRICULUM targeted policy ▶ | node={node.node_id} "
                "DEEP SOTA: academic+consensus дополнительно (practical_only override)"
            )
            practical_hits, academic_hits = await _gather_practical_and_academic(
                node,
                course_goal,
                anchor=anchor,
                source_policy="hybrid",
                on_demand=on_demand,
                registry_entries=registry_entries,
                exclude_url_keys=exclude,
            )
            merged = _merge_hits_interleaved(
                [academic_hits, practical_hits],
                cap * 2,
                exclude,
            )
        else:
            trace(
                f"CURRICULUM targeted policy ⊘ | node={node.node_id} "
                "academic/consensus blocked (practical_only)"
            )
            merged = _merge_hits([practical_hits], cap * 2, exclude)
    elif policy == "academic_only":
        academic_hits = await _academic_hits_for_node(
            node, course_goal, anchor=anchor, source_policy=policy,
            on_demand=on_demand,
            registry_entries=registry_entries,
            exclude_url_keys=exclude,
        )
        merged = _merge_hits([academic_hits], cap * 2, exclude)
    else:
        trace(
            f"CURRICULUM targeted parallel ▶ | node={node.node_id} "
            "practical + academic gather"
        )
        practical_hits, academic_hits = await _gather_practical_and_academic(
            node,
            course_goal,
            anchor=anchor,
            source_policy=policy,
            on_demand=on_demand,
            registry_entries=registry_entries,
            exclude_url_keys=exclude,
        )
        _log_provider_hit_counts("after_gather_practical", node.node_id, practical_hits)
        _log_provider_hit_counts("after_gather_academic", node.node_id, academic_hits)
        if CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST and practical_hits:
            if is_sota_rd_node(node):
                trace(
                    f"CURRICULUM hybrid sota ▶ | node={node.node_id} "
                    "— academic+consensus union with practical"
                )
            else:
                trace(
                    f"CURRICULUM hybrid practical_first ▶ | node={node.node_id} "
                    "— union academic+practical (no early return)"
                )
        merged = _merge_hits_interleaved(
            [academic_hits, practical_hits],
            cap * 2,
            exclude,
        )

    _log_provider_hit_counts("pre_lite_merged", node.node_id, merged)

    if not merged:
        trace(f"CURRICULUM targeted search ⊘ | node={node.node_id} hits=0")
        return []

    approved = await _lite_approve_merged_hits(
        merged,
        goal,
        anchor=anchor,
        node_id=node.node_id,
    )
    _log_provider_hit_counts("post_lite_approved", node.node_id, approved)
    if not approved:
        trace(
            f"CURRICULUM targeted search ⊘ | node={node.node_id} "
            "lite approved=0 — early exit (no ingest/summarize)"
        )
        return []
    trace(
        f"CURRICULUM targeted search ✓ | node={node.node_id} "
        f"raw={len(merged)} approved={len(approved)}"
    )
    return approved[:cap]


def search_sources_for_deep_node(
    node: CurriculumNode,
    course_goal: str,
    *,
    source_policy: str,
    anchor: str,
    exclude_url_keys: set[str],
) -> list[CurriculumSearchHit]:
    return asyncio.run(
        search_sources_for_deep_node_async(
            node,
            course_goal,
            source_policy=source_policy,
            anchor=anchor,
            exclude_url_keys=exclude_url_keys,
        )
    )


def hit_to_registry_entry(
    hit: CurriculumSearchHit,
    source_id: str,
) -> tuple[CurriculumSearchHit, "CurriculumSourceRegistryEntry"]:
    from knowledge_engine.src.curriculum.schemas import CurriculumSourceRegistryEntry

    matched, cat = resolve_source_provenance(hit.url)
    if cat == "academic_open":
        domain = (urlparse(hit.url).netloc or "academic_open").lower()
    else:
        domain = cat if cat != "open_candidate" else (urlparse(hit.url).netloc or "").lower()
    extracts = list(hit.key_extracts or [])
    why = (hit.snippet or "")[:800]
    if extracts and not why:
        why = extracts[0][:800]
    entry = CurriculumSourceRegistryEntry(
        source_id=source_id[:16],
        title=(hit.title or domain or source_id)[:400],
        whitelist_domain=domain[:200],
        source_type="Article",
        url=hit.url[:2000],
        why_read=why,
        snippet=(hit.snippet or "")[:1200],
        key_extracts=extracts[:12],
        source_tier=(hit.source_tier or "")[:24],
    )
    hit = hit.model_copy(update={"source_id": source_id[:16]})
    return hit, entry
