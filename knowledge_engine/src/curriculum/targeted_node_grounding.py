"""Targeted Node Grounding: Model-First → Risk → (lazy) Search на node/init."""

from __future__ import annotations

import time

from knowledge_engine.config import (
    CURRICULUM_DEEP_NODE_MAX_HITS,
    CURRICULUM_MODEL_FIRST_MIN_NODES,
)
from knowledge_engine.src.curriculum.curriculum_lancedb_persist import (
    persist_approved_curriculum_hits_to_lancedb,
)
from knowledge_engine.src.curriculum.dag_validator import validate_curriculum_dag
from knowledge_engine.src.curriculum.model_first_flash import generate_model_first_graph
from knowledge_engine.src.curriculum.node_risk_classification import (
    classify_and_apply_node_risks,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
    NodeCurriculumBreakdown,
    NodeSourceRef,
)
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.curriculum.source_material_pipeline import (
    enrich_search_hits_with_extracts,
    summarize_whitelist_blog_hits_async,
)
from knowledge_engine.src.curriculum.source_registry import (
    cap_curriculum_sources_registry,
    sync_route_sources_from_registry,
    validate_curriculum_source_links,
)
from knowledge_engine.src.curriculum.targeted_node_search import (
    hit_to_registry_entry,
    search_sources_for_deep_node_async,
)
from knowledge_engine.ui.run_log import trace


def _attach_hits_to_node(
    node: CurriculumNode,
    hits: list,
    source_ids: list[str],
) -> CurriculumNode:
    primary_hit = hits[0]
    primary_sid = source_ids[0][:16]
    extracts = [
        e.strip() for e in (primary_hit.key_extracts or []) if e and str(e).strip()
    ][:12]
    if not extracts and primary_hit.snippet:
        extracts = [primary_hit.snippet[:800]]
    ref = NodeSourceRef(
        source_id=primary_sid,
        url=primary_hit.url[:2000],
        relevant_extracts=extracts,
    )
    breakdown = node.node_curriculum_breakdown
    if breakdown is None and extracts:
        breakdown = NodeCurriculumBreakdown(
            key_concepts=node.core_concepts[:8],
            architectural_focus=(node.brief_summary or "")[:800],
        )
    mapped = [s[:16] for s in source_ids]
    resource_urls = [h.url[:2000] for h in hits]
    return node.model_copy(
        update={
            "mapped_source_ids": mapped,
            "primary_source_id": primary_sid,
            "resource_urls": resource_urls,
            "source_ref": ref,
            "node_curriculum_breakdown": breakdown,
            "grounding_status": "grounded",
            "node_risk_kind": "DEEP",
        }
    )


def _attach_hit_to_node(
    node: CurriculumNode,
    hit,
    source_id: str,
) -> CurriculumNode:
    extracts = [e.strip() for e in (hit.key_extracts or []) if e and str(e).strip()][
        :12
    ]
    if not extracts and hit.snippet:
        extracts = [hit.snippet[:800]]
    ref = NodeSourceRef(
        source_id=source_id[:16],
        url=hit.url[:2000],
        relevant_extracts=extracts,
    )
    breakdown = node.node_curriculum_breakdown
    if breakdown is None and extracts:
        breakdown = NodeCurriculumBreakdown(
            key_concepts=node.core_concepts[:8],
            architectural_focus=(node.brief_summary or "")[:800],
        )
    return node.model_copy(
        update={
            "mapped_source_ids": [source_id[:16]],
            "primary_source_id": source_id[:16],
            "resource_urls": [hit.url[:2000]],
            "source_ref": ref,
            "node_curriculum_breakdown": breakdown,
            "grounding_status": "grounded",
            "node_risk_kind": "DEEP",
        }
    )


def generate_curriculum_targeted_grounding(
    inp: CurriculumGenerateInput,
    anchor: str,
) -> CurriculumGraph:
    t0 = time.monotonic()
    trace(
        "CURRICULUM targeted grounding ▶ | Model-First → Risk → lazy (no search on create)"
    )

    graph = generate_model_first_graph(inp, anchor)
    node_count_initial = len(graph.nodes)

    graph = classify_and_apply_node_risks(
        graph,
        inp.target_goal,
        anchor=f"{anchor}:risk",
    )

    updated_nodes: list[CurriculumNode] = []
    pending = 0
    for node in graph.nodes:
        if node.node_risk_kind != "DEEP":
            trace(
                f"CURRICULUM targeted skip search | node={node.node_id} "
                "risk=BASE (lazy: no search on create)"
            )
            updated_nodes.append(
                node.model_copy(
                    update={
                        "grounding_status": "model_only",
                        "node_risk_kind": "BASE",
                    }
                )
            )
            continue

        pending += 1
        updated_nodes.append(
            node.model_copy(
                update={
                    "grounding_status": "pending_grounding",
                    "node_risk_kind": "DEEP",
                    "mapped_source_ids": [],
                    "primary_source_id": "",
                    "resource_urls": [],
                    "source_ref": None,
                }
            )
        )

    if len(updated_nodes) != node_count_initial:
        raise ValueError(
            "Targeted Grounding: число нод изменилось — структура нарушена"
        )

    graph = graph.model_copy(
        update={
            "nodes": updated_nodes,
            "total_nodes": len(updated_nodes),
            "curriculum_sources_registry": [],
        }
    )
    graph = sync_route_sources_from_registry(graph)

    errors = validate_curriculum_dag(graph)
    if errors:
        raise ValueError(
            "Targeted Grounding: DAG invalid после risk: " + "; ".join(errors[:5])
        )
    if len(graph.nodes) < CURRICULUM_MODEL_FIRST_MIN_NODES:
        raise ValueError(
            f"Targeted Grounding: nodes={len(graph.nodes)} "
            f"< min {CURRICULUM_MODEL_FIRST_MIN_NODES}"
        )

    link_errors = validate_curriculum_source_links(
        graph,
        allow_model_only_nodes=True,
    )
    if link_errors:
        trace(f"CURRICULUM targeted warn | {link_errors[0]}")

    base = sum(1 for n in graph.nodes if n.grounding_status == "model_only")
    elapsed = time.monotonic() - t0
    trace(
        f"CURRICULUM DAG created in {elapsed:.1f}s (lazy grounding enabled) | "
        f"nodes={graph.total_nodes} pending_grounding={pending} model_only={base}"
    )
    return graph


async def lazy_ground_deep_node_on_demand(
    graph: CurriculumGraph,
    node: CurriculumNode,
    *,
    target_goal: str,
    source_policy: str,
    anchor: str,
    reground_academic: bool = False,
) -> tuple[CurriculumGraph, CurriculumNode]:
    """Один DEEP-нода: Exa / Consensus / academic → registry + LanceDB."""
    if node.node_risk_kind != "DEEP":
        return graph, node
    status = (node.grounding_status or "").strip()
    if status == "grounded" and node.source_ref and not reground_academic:
        trace(
            f"NODE_DIVE lazy grounding ⊘ | node_id={node.node_id} "
            "already grounded — skip full search"
        )
        return graph, node
    if status not in ("pending_grounding", "unverified_deep", "model_only", ""):
        return graph, node

    trace(f"NODE_DIVE lazy grounding ▶ | node_id={node.node_id}")

    registry = list(graph.curriculum_sources_registry)
    exclude: set[str] = set()
    for e in registry:
        key = _normalize_url_key(e.url)
        if key:
            exclude.add(key)

    hits = await search_sources_for_deep_node_async(
        node,
        target_goal,
        source_policy=source_policy,
        anchor=anchor,
        exclude_url_keys=exclude,
        on_demand=True,
        registry_entries=registry,
    )
    if hits:
        summarized = await summarize_whitelist_blog_hits_async(hits, target_goal)
        by_url = {h.url.strip().rstrip("/").lower(): h for h in summarized}
        hits = [by_url.get(h.url.strip().rstrip("/").lower(), h) for h in hits]
        hits = enrich_search_hits_with_extracts(hits, target_goal)
        persist_approved_curriculum_hits_to_lancedb(
            hits,
            label=f"lazy_ground:{node.node_id}",
        )

    next_src_idx = len(registry) + 1
    updated_node = node

    if hits:
        batch = hits[:CURRICULUM_DEEP_NODE_MAX_HITS]
        sids: list[str] = []
        registry_hits: list = []
        for i, raw in enumerate(batch):
            sid = f"src_{next_src_idx + i}"
            hit, entry = hit_to_registry_entry(raw, sid)
            registry.append(entry)
            sids.append(sid)
            registry_hits.append(hit)
        updated_node = _attach_hits_to_node(node, registry_hits, sids)
        tiers = ", ".join(sorted({(h.source_tier or "?")[:12] for h in registry_hits}))
        trace(
            f"NODE_DIVE lazy grounding ✓ | node_id={node.node_id} "
            f"status=grounded registry+={len(batch)} mapped={sids} tiers={tiers} "
            f"primary={registry_hits[0].url[:80]}"
        )
    else:
        updated_node = node.model_copy(
            update={
                "grounding_status": "unverified_deep",
                "node_risk_kind": "DEEP",
                "mapped_source_ids": [],
                "primary_source_id": "",
                "resource_urls": [],
                "source_ref": None,
            }
        )
        trace(
            f"NODE_DIVE lazy grounding ⊘ | node_id={node.node_id} "
            "status=unverified_deep (search empty / Lite reject)"
        )

    new_nodes = [updated_node if n.node_id == node.node_id else n for n in graph.nodes]
    graph = graph.model_copy(update={"nodes": new_nodes})
    registry = cap_curriculum_sources_registry(registry, graph=graph)
    graph = graph.model_copy(update={"curriculum_sources_registry": registry})
    graph = sync_route_sources_from_registry(graph)

    ingest_urls = [
        str(h.url or "").strip()
        for h in (hits or [])
        if str(h.url or "").startswith("http")
    ]
    if ingest_urls or hits:
        from knowledge_engine.src.node_deep_dive.diagram_session import (
            refresh_node_session_diagrams_from_articles,
        )

        n = refresh_node_session_diagrams_from_articles(
            graph.curriculum_id,
            updated_node,
            extra_urls=ingest_urls,
            rebuild=True,
        )
        trace(
            f"NODE_DIVE hydrate session diagrams | node={node.node_id} "
            f"content_diagrams={n} extra_urls={len(ingest_urls)}"
        )

    return graph, updated_node
