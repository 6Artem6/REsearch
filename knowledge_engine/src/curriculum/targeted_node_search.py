"""Этап 3: точечный поиск только для DEEP-нод (без fallback approve)."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_ACADEMIC_ARXIV_LIMIT,
    CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
    CURRICULUM_ACADEMIC_SS_LIMIT,
    CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST,
    CURRICULUM_DEEP_NODE_MAX_HITS,
    CURRICULUM_DEEP_NODE_REPLENISH_POOL,
    CURRICULUM_ON_DEMAND_V08_MAX_PAPERS,
    CURRICULUM_ON_DEMAND_V08_POOL_SIZE,
    CURRICULUM_USE_V08_CONSENSUS,
    DEEP_INGEST_BACKFILL_MARGIN,
)
from knowledge_engine.services.search.exa_transform import (
    fetch_exa_curriculum_hits_for_node,
)
from knowledge_engine.src.curriculum.academic_consensus import (
    consensus_allowed_for_policy,
    harvest_consensus_for_node,
    is_sota_rd_node,
)
from knowledge_engine.src.curriculum.academic_searxng_search import (
    collect_searxng_academic_rows,
)
from knowledge_engine.src.curriculum.academic_source_fetch import (
    HITS_QUEUE_SENTINEL,
    fetch_academic_sources_async,
    hit_from_searxng_academic_row,
    process_hits_stream,
    stream_hit_from_paper,
)
from knowledge_engine.src.curriculum.lite_search_pipeline import (
    AcademicSearchPlan,
    batch_lite_eval_curriculum_hits,
    build_academic_search_plan,
)
from knowledge_engine.src.curriculum.practical_searxng_search import (
    collect_searxng_practical_rows,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumNode,
    CurriculumSearchHit,
    CurriculumSourceRegistryEntry,
)
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.source_policy import normalize_source_policy
from knowledge_engine.src.curriculum.source_quota_policy import get_source_quota
from knowledge_engine.src.curriculum.targeted_hit_replenishment import (
    replenish_valid_hits_until_cap,
)
from knowledge_engine.src.retrieval.semantic_scholar import (
    search_arxiv_fallback,
    search_semantic_scholar,
)
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_academic_open_host,
    is_collectible_article_url,
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
    searx_pr = sum(
        1 for h in hits if (h.source_tier or "").strip().lower() == "searxng"
    )
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
    pool_cap: int | None = None,
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
    merge_cap = pool_cap if pool_cap is not None else CURRICULUM_DEEP_NODE_MAX_HITS * 2

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

    return _merge_hits_interleaved(parts, merge_cap, set())


async def _practical_hits_for_node(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
) -> list[CurriculumSearchHit]:
    replenish_pool = max(
        CURRICULUM_DEEP_NODE_REPLENISH_POOL,
        CURRICULUM_DEEP_NODE_MAX_HITS + 2,
    )
    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    exa_hits = await fetch_exa_curriculum_hits_for_node(
        node,
        course_goal,
        anchor=anchor,
        cap=replenish_pool,
    )
    # Exa query_en/query_ru: Lite prompt в exa_transform (architecture, не API docs)
    _log_provider_hit_counts("Exa pass", node.node_id, exa_hits)
    for h in exa_hits:
        key = _normalize_url_key(h.url)
        if not key or key in seen:
            continue
        seen.add(key)
        hits.append(h)
        if len(hits) >= replenish_pool:
            trace(
                f"CURRICULUM targeted practical ✓ | node={node.node_id} "
                f"exa_only hits={len(hits)}"
            )
            return hits

    need = replenish_pool - len(hits)
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
            if len(hits) >= replenish_pool:
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
    out = hits[: CURRICULUM_DEEP_NODE_MAX_HITS + 2]
    _log_provider_hit_counts("academic_merged", node.node_id, out)
    return out


_PAPER_STREAM_SEM = asyncio.Semaphore(3)


async def _stream_pool_has_room(
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
    pool_cap: int,
) -> bool:
    async with seen_lock:
        return len(out_hits) < pool_cap


def _stream_queue_put(
    queue: asyncio.Queue,
    hit: CurriculumSearchHit,
) -> None:
    try:
        queue.put_nowait(hit)
    except Exception as exc:
        trace(f"CURRICULUM stream producer queue | {exc}")


class _StreamProducerGate:
    def __init__(
        self,
        queue: asyncio.Queue,
        producer_count: int,
        consumer_count: int,
    ) -> None:
        self._queue = queue
        self._remaining = producer_count
        self._consumer_count = max(1, consumer_count)

    async def producer_finished(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            for _ in range(self._consumer_count):
                await self._queue.put(HITS_QUEUE_SENTINEL)


class StreamSearchContext:
    def __init__(
        self,
        node: CurriculumNode,
        goal: str,
        anchor: str,
        search_vector: str,
    ) -> None:
        self.node = node
        self.goal = goal
        self.anchor = anchor
        self.search_vector = search_vector
        lite_anchor = anchor or f"curriculum_academic:{search_vector[:400]}"
        self._academic_plan_task = asyncio.create_task(
            build_academic_search_plan(search_vector, anchor=lite_anchor)
        )

    async def academic_plan(self) -> AcademicSearchPlan:
        return await self._academic_plan_task

    async def academic_query(self) -> str:
        plan = await self._academic_plan_task
        q = (plan.academic_query_en or "").strip()
        if q:
            return q
        return self.search_vector[:400]


async def run_exa_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    node = ctx.node
    hits = await fetch_exa_curriculum_hits_for_node(
        node,
        ctx.goal,
        anchor=ctx.anchor,
        cap=pool_cap,
    )
    _log_provider_hit_counts("stream Exa", node.node_id, hits)
    for h in hits:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            break
        _stream_queue_put(hits_queue, h)


async def run_searxng_practical_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    node = ctx.node
    vec = ctx.search_vector
    need = pool_cap
    lite_plan = need > 2
    max_q = min(3, need + 1) if not lite_plan else None
    rows = await collect_searxng_practical_rows(
        vec,
        limit=need + 2,
        anchor=f"{ctx.anchor}:practical:{node.node_id}",
        lite_query_plan=lite_plan,
        max_queries=max_q,
    )
    for row in rows:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            break
        url = (row.get("url") or "").strip()
        if not is_collectible_article_url(url):
            continue
        snippet = (row.get("snippet") or "")[:1200]
        title = (row.get("title") or url)[:400]
        extracts = [snippet[:800]] if len(snippet) >= 80 else []
        _stream_queue_put(
            hits_queue,
            CurriculumSearchHit(
                url=url,
                title=title,
                snippet=snippet,
                key_extracts=extracts,
                source_tier="searxng",
            ),
        )


async def run_consensus_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    on_demand: bool,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    node = ctx.node
    hits = await harvest_consensus_for_node(
        node,
        ctx.search_vector,
        ctx.anchor,
        "stream_pipeline",
        on_demand=on_demand,
        defer_ingest=True,
        force_playwright=True,
    )
    _log_provider_hit_counts("stream consensus", node.node_id, hits)
    for h in hits:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            break
        _stream_queue_put(hits_queue, h)


async def run_ss_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    query = await ctx.academic_query()
    papers = await search_semantic_scholar(
        query,
        limit=CURRICULUM_ACADEMIC_SS_LIMIT,
        ignore_enabled_flag=True,
    )
    trace(f"CURRICULUM stream SS | node={ctx.node.node_id} papers={len(papers)}")

    async def _one(paper) -> None:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            return
        async with _PAPER_STREAM_SEM:
            hit = await stream_hit_from_paper(paper)
            if hit and await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
                _stream_queue_put(hits_queue, hit)

    if papers:
        await asyncio.gather(*[_one(p) for p in papers])


async def run_arxiv_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    plan = await ctx.academic_plan()
    query = (plan.academic_query_en or "").strip() or ctx.search_vector[:400]
    papers = await search_arxiv_fallback(
        query,
        limit=CURRICULUM_ACADEMIC_ARXIV_LIMIT,
        arxiv_params=plan.arxiv_params,
    )
    trace(f"CURRICULUM stream arXiv | node={ctx.node.node_id} papers={len(papers)}")

    async def _one(paper) -> None:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            return
        async with _PAPER_STREAM_SEM:
            hit = await stream_hit_from_paper(paper)
            if hit and await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
                _stream_queue_put(hits_queue, hit)

    if papers:
        await asyncio.gather(*[_one(p) for p in papers])


async def run_searxng_academic_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    query = await ctx.academic_query()
    rows = await collect_searxng_academic_rows(
        query,
        limit=CURRICULUM_ACADEMIC_SEARXNG_LIMIT,
    )
    for row in rows:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            break
        url = (row.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        _stream_queue_put(hits_queue, hit_from_searxng_academic_row(row))


async def run_reuse_producer(
    ctx: StreamSearchContext,
    hits_queue: asyncio.Queue,
    *,
    registry_entries: list | None,
    exclude_url_keys: set[str],
    pool_cap: int,
    out_hits: list[CurriculumSearchHit],
    seen_lock: asyncio.Lock,
) -> None:
    from knowledge_engine.src.curriculum.on_demand_reuse import (
        merge_on_demand_reuse_hits,
    )

    reuse_pool_cap = max(
        CURRICULUM_ON_DEMAND_V08_POOL_SIZE,
        CURRICULUM_ON_DEMAND_V08_MAX_PAPERS + 4,
    )
    reuse_raw = await merge_on_demand_reuse_hits(
        ctx.search_vector,
        list(registry_entries or []),
        cap=reuse_pool_cap,
        exclude_url_keys=exclude_url_keys,
    )
    if not reuse_raw:
        return
    trace(
        f"CURRICULUM stream reuse | node={ctx.node.node_id} "
        f"hits={len(reuse_raw)} (lite in consumer)"
    )
    for h in reuse_raw:
        if not await _stream_pool_has_room(out_hits, seen_lock, pool_cap):
            break
        _stream_queue_put(hits_queue, h)


async def _run_stream_search_pipeline(
    node: CurriculumNode,
    course_goal: str,
    *,
    anchor: str,
    source_policy: str,
    on_demand: bool,
    registry_entries: list | None,
    exclude_url_keys: set[str],
    include_practical: bool,
    include_academic: bool,
) -> list[CurriculumSearchHit]:
    goal = _node_search_goal(node, course_goal)
    search_vector = goal
    cap = CURRICULUM_DEEP_NODE_MAX_HITS
    pool_cap = max(CURRICULUM_DEEP_NODE_REPLENISH_POOL, cap * 2)
    exclude = set(exclude_url_keys)
    allow_consensus = consensus_allowed_for_policy(source_policy)

    hits_queue: asyncio.Queue = asyncio.Queue()
    out_hits: list[CurriculumSearchHit] = []
    seen_urls: set[str] = set(exclude)
    seen_lock = asyncio.Lock()
    ctx = StreamSearchContext(node, goal, anchor, search_vector)

    producer_specs: list[tuple[str, object]] = []
    o, sl, pc = out_hits, seen_lock, pool_cap
    if include_practical:
        producer_specs.append(
            (
                "exa",
                lambda: run_exa_producer(
                    ctx, hits_queue, pool_cap=pc, out_hits=o, seen_lock=sl
                ),
            )
        )
        producer_specs.append(
            (
                "searxng_practical",
                lambda: run_searxng_practical_producer(
                    ctx, hits_queue, pool_cap=pc, out_hits=o, seen_lock=sl
                ),
            )
        )
    if include_academic:
        if on_demand:
            producer_specs.append(
                (
                    "reuse",
                    lambda: run_reuse_producer(
                        ctx,
                        hits_queue,
                        registry_entries=registry_entries,
                        exclude_url_keys=exclude,
                        pool_cap=pc,
                        out_hits=o,
                        seen_lock=sl,
                    ),
                )
            )
        if allow_consensus and CURRICULUM_USE_V08_CONSENSUS:
            producer_specs.append(
                (
                    "consensus",
                    lambda: run_consensus_producer(
                        ctx,
                        hits_queue,
                        on_demand=on_demand,
                        pool_cap=pc,
                        out_hits=o,
                        seen_lock=sl,
                    ),
                )
            )
        producer_specs.append(
            (
                "semantic_scholar",
                lambda: run_ss_producer(
                    ctx, hits_queue, pool_cap=pc, out_hits=o, seen_lock=sl
                ),
            )
        )
        producer_specs.append(
            (
                "searxng_academic",
                lambda: run_searxng_academic_producer(
                    ctx, hits_queue, pool_cap=pc, out_hits=o, seen_lock=sl
                ),
            )
        )
        producer_specs.append(
            (
                "arxiv",
                lambda: run_arxiv_producer(
                    ctx, hits_queue, pool_cap=pc, out_hits=o, seen_lock=sl
                ),
            )
        )

    if not producer_specs:
        return []

    from knowledge_engine.config import KE_INGEST_URL_CONCURRENCY

    consumer_workers = max(1, min(KE_INGEST_URL_CONCURRENCY, pool_cap))
    gate = _StreamProducerGate(
        hits_queue,
        len(producer_specs),
        consumer_workers,
    )
    trace(
        f"CURRICULUM stream pipeline ▶ | node={node.node_id} "
        f"producers={len(producer_specs)} consumers={consumer_workers} "
        f"policy={source_policy} pool_cap={pool_cap} ingest=post-replenish"
    )

    async def _run_producer(name: str, fn) -> None:
        try:
            await fn()
            trace(f"CURRICULUM stream producer ✓ | node={node.node_id} | {name}")
        except Exception as exc:
            trace(
                f"CURRICULUM stream producer ✗ | node={node.node_id} | {name} | {exc}"
            )
        finally:
            await gate.producer_finished()

    consumer_tasks = [
        asyncio.create_task(
            process_hits_stream(
                hits_queue,
                node,
                goal=goal,
                anchor=anchor,
                source_policy=source_policy,
                out_hits=out_hits,
                pool_cap=pool_cap,
                seen_urls=seen_urls,
                seen_lock=seen_lock,
            )
        )
        for _ in range(consumer_workers)
    ]
    producer_tasks = [
        asyncio.create_task(_run_producer(name, fn)) for name, fn in producer_specs
    ]
    await asyncio.gather(*producer_tasks)
    await asyncio.gather(*consumer_tasks)
    _log_provider_hit_counts("stream_merged", node.node_id, out_hits)
    trace(
        f"CURRICULUM stream pipeline ✓ | node={node.node_id} "
        f"pool_hits={len(out_hits)} ingest=post-replenish"
    )
    return out_hits


def _on_demand_post_practical_academic_wait_sec(
    node: CurriculumNode,
    source_policy: str,
) -> float:
    """Extra wait for academic task after practical finishes (on_demand fast-return)."""
    from knowledge_engine.config import (
        CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC,
        CURRICULUM_USE_V08_CONSENSUS,
    )

    base = float(CURRICULUM_ON_DEMAND_ACADEMIC_WAIT_SEC)
    policy = normalize_source_policy(source_policy, default="hybrid")
    if (
        policy != "practical_only"
        and is_sota_rd_node(node)
        and consensus_allowed_for_policy(policy)
        and CURRICULUM_USE_V08_CONSENSUS
    ):
        # Academic arm may still be in Lite reuse + Consensus after practical returns.
        return max(base, 40.0)
    return base


def _must_await_full_academic_gather(
    node: CurriculumNode,
    source_policy: str,
    on_demand: bool,
) -> bool:
    """SOTA hybrid + Consensus: не отменять academic (Playwright harvest) по fast-return."""
    if not on_demand:
        return True
    from knowledge_engine.config import CURRICULUM_USE_V08_CONSENSUS

    policy = normalize_source_policy(source_policy, default="hybrid")
    return (
        policy != "practical_only"
        and is_sota_rd_node(node)
        and consensus_allowed_for_policy(policy)
        and CURRICULUM_USE_V08_CONSENSUS
    )


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
        CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN,
    )
    from knowledge_engine.src.curriculum.source_material_pipeline import (
        summarize_whitelist_blog_hits_async,
    )

    post_practical_wait = _on_demand_post_practical_academic_wait_sec(
        node, source_policy
    )

    practical_ingest_task: asyncio.Task | None = None
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
        and post_practical_wait > 0
        and not _must_await_full_academic_gather(node, source_policy, on_demand)
    ):
        practical_hits = await p_task
        if practical_hits:
            practical_ingest_task = asyncio.create_task(
                summarize_whitelist_blog_hits_async(practical_hits, course_goal)
            )
        academic_hits: list[CurriculumSearchHit] = []
        if len(practical_hits) >= CURRICULUM_ON_DEMAND_MIN_PRACTICAL_FOR_FAST_RETURN:
            if a_task.done():
                academic_hits = a_task.result()
            else:
                try:
                    academic_hits = await asyncio.wait_for(
                        a_task,
                        timeout=post_practical_wait,
                    )
                except asyncio.TimeoutError:
                    trace(
                        f"CURRICULUM on_demand fast-return ▶ | node={node.node_id} "
                        f"practical={len(practical_hits)} "
                        f"academic_wait>{post_practical_wait:.0f}s"
                    )
                    if not a_task.done():
                        a_task.cancel()
                        try:
                            await a_task
                        except asyncio.CancelledError:
                            pass
                except asyncio.CancelledError:
                    academic_hits = []
        else:
            academic_hits = await a_task
        if practical_ingest_task is not None:
            practical_hits = await practical_ingest_task
        return practical_hits, academic_hits

    if on_demand and _must_await_full_academic_gather(node, source_policy, on_demand):
        trace(
            f"CURRICULUM on_demand ▶ | node={node.node_id} "
            "await full academic gather (SOTA+Consensus, no fast-return cancel)"
        )
    practical_hits = await p_task
    if practical_hits:
        trace(
            f"CURRICULUM parallel ingest ▶ | practical hits={len(practical_hits)} "
            "while academic arm still running"
        )
        practical_ingest_task = asyncio.create_task(
            summarize_whitelist_blog_hits_async(practical_hits, course_goal)
        )
    academic_hits = await a_task
    if practical_ingest_task is not None:
        practical_hits = await practical_ingest_task
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

    cap = min(
        CURRICULUM_DEEP_NODE_MAX_HITS,
        get_source_quota(node.layer, node.node_risk_kind).total_max,
    )
    max(CURRICULUM_DEEP_NODE_REPLENISH_POOL, cap * 2)
    exclude = set(exclude_url_keys)

    include_academic = policy != "practical_only"
    include_practical = policy != "academic_only"

    if (
        policy == "practical_only"
        and node.node_risk_kind == "DEEP"
        and is_sota_rd_node(node)
    ):
        trace(
            f"CURRICULUM targeted policy ▶ | node={node.node_id} "
            "DEEP SOTA: academic+consensus дополнительно (practical_only override)"
        )
        include_academic = True
        source_policy = "hybrid"

    if policy == "practical_only" and not include_academic:
        trace(
            f"CURRICULUM targeted policy ⊘ | node={node.node_id} "
            "academic/consensus blocked (practical_only)"
        )

    merged = await _run_stream_search_pipeline(
        node,
        course_goal,
        anchor=anchor,
        source_policy=source_policy,
        on_demand=on_demand,
        registry_entries=registry_entries,
        exclude_url_keys=exclude,
        include_practical=include_practical,
        include_academic=include_academic,
    )

    if (
        CURRICULUM_DEEP_HYBRID_PRACTICAL_FIRST
        and include_practical
        and include_academic
    ):
        if is_sota_rd_node(node):
            trace(
                f"CURRICULUM hybrid sota ▶ | node={node.node_id} "
                "stream: academic+consensus ∥ practical"
            )
        else:
            trace(
                f"CURRICULUM hybrid practical_first ▶ | node={node.node_id} "
                "stream pipeline"
            )

    _log_provider_hit_counts("pre_replenish", node.node_id, merged)

    if not merged:
        trace(f"CURRICULUM targeted search ⊘ | node={node.node_id} hits=0")
        return []

    valid = await replenish_valid_hits_until_cap(
        merged,
        cap,
        node=node,
        backfill_margin=DEEP_INGEST_BACKFILL_MARGIN,
    )
    from knowledge_engine.src.curriculum.source_material_pipeline import (
        ingest_mandatory_academic_hits_async,
        summarize_whitelist_blog_hits_async,
    )

    valid = await ingest_mandatory_academic_hits_async(
        valid,
        label=f"post_replenish:{node.node_id}",
        defer_missing=on_demand,
    )
    valid = await summarize_whitelist_blog_hits_async(
        valid,
        goal,
        backfill_margin=DEEP_INGEST_BACKFILL_MARGIN,
        desired_count=cap,
    )
    trace(
        f"CURRICULUM targeted search ✓ | node={node.node_id} "
        f"stream_hits={len(merged)} valid_after_replenish={len(valid)}"
    )
    return valid


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
) -> tuple[CurriculumSearchHit, CurriculumSourceRegistryEntry]:
    matched, cat = resolve_source_provenance(hit.url)
    if cat == "academic_open":
        domain = (urlparse(hit.url).netloc or "academic_open").lower()
    else:
        domain = (
            cat if cat != "open_candidate" else (urlparse(hit.url).netloc or "").lower()
        )
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
