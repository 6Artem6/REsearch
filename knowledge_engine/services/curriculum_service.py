"""SOLID expansion: Lite vector → grounding sources → Flash patch."""

from __future__ import annotations

from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
    save_curriculum_record,
)
from knowledge_engine.src.curriculum.curriculum_expansion import (
    apply_expansion_patch,
    flash_build_expansion_patch,
    lite_plan_expansion_vector,
    merge_graph_source_registry,
    validate_and_repair_expansion_dag,
)
from knowledge_engine.src.curriculum.schemas import CurriculumGraph
from knowledge_engine.src.curriculum.search_prestep import assign_source_ids
from knowledge_engine.src.curriculum.source_material_pipeline import (
    collect_sources_for_expand,
    enrich_search_hits_with_extracts,
    merge_expansion_source_pool,
    summarize_whitelist_blog_hits,
)
from knowledge_engine.ui.run_log import trace


def expand_curriculum(
    curriculum_id: str,
    user_request: str,
    *,
    generation_mode: str = "fast",
    source_policy: str | None = "practical_only",
) -> CurriculumGraph:
    """
    1. Lite → expansion_vector
    2. Grounding-only по вектору → Summarizer → LanceDB
    3. Flash → new_nodes / new_edges (пул: реестр + LanceDB + новые hits)
    4. Merge graph (статусы нод в session_store не затрагиваются)
    """
    raw = get_curriculum_graph(curriculum_id)
    if not raw:
        raise ValueError(f"curriculum not found: {curriculum_id}")
    graph = CurriculumGraph.model_validate(raw)
    req = (user_request or "").strip()
    if len(req) < 8:
        raise ValueError("user_request слишком короткий")

    from knowledge_engine.src.curriculum.source_policy import (
        normalize_source_policy,
        resolve_source_policy,
    )

    policy = resolve_source_policy(
        source_policy,
        generation_mode,
        default="practical_only",
    )
    policy = normalize_source_policy(policy, default="practical_only")
    anchor = f"curriculum_expand:{curriculum_id}"
    trace(
        f"CURRICULUM expand ▶ | {curriculum_id} | source_policy={policy} | "
        f"{req[:80]}…"
    )

    vector_out = lite_plan_expansion_vector(graph, req, anchor)
    vector_text = vector_out.expansion_vector.strip()

    pre_existing_ids = {n.node_id for n in graph.nodes}

    new_hits = collect_sources_for_expand(vector_text, source_policy=policy)
    fresh_grounding_count = len(new_hits)
    if fresh_grounding_count == 0:
        trace(
            "CURRICULUM expand ⚠ | new_grounding=0 — Flash без свежих выжимок "
            "(429/пустой grounding)"
        )
    if new_hits:
        new_hits = summarize_whitelist_blog_hits(new_hits, vector_text)
        new_hits = enrich_search_hits_with_extracts(new_hits, vector_text)
        new_hits = assign_source_ids(new_hits)

    pool_hits = merge_expansion_source_pool(graph, new_hits)
    if fresh_grounding_count > 0:
        pool_hits = enrich_search_hits_with_extracts(pool_hits, req)

    patch = flash_build_expansion_patch(
        graph,
        vector_text,
        pool_hits,
        req,
        anchor,
        fresh_grounding_hits=fresh_grounding_count,
    )
    new_node_ids = {n.node_id for n in patch.new_nodes}
    expanded = apply_expansion_patch(graph, patch)
    expanded = validate_and_repair_expansion_dag(
        expanded,
        pre_existing_ids,
        new_node_ids,
        patch.new_edges,
        new_nodes_ordered=[n.node_id for n in patch.new_nodes],
    )
    from knowledge_engine.src.curriculum.dag_validator import (
        repair_curriculum_dag_cycles,
        validate_curriculum_dag,
    )

    expanded, broke = repair_curriculum_dag_cycles(
        expanded,
        prefer_remove_node_ids=new_node_ids,
    )
    if broke:
        trace(f"CURRICULUM expand DAG cycles ▶ removed {broke} cycle edge(s)")

    dag_errors = validate_curriculum_dag(expanded)
    if dag_errors:
        trace(
            f"CURRICULUM expand DAG warn | {len(dag_errors)} issues: "
            f"{'; '.join(dag_errors[:3])}"
        )
    expanded = merge_graph_source_registry(expanded, new_hits)

    meta = get_curriculum_meta(curriculum_id) or {}
    save_curriculum_record(
        expanded,
        target_goal=meta.get("target_goal") or graph.title,
        generation_mode=meta.get("generation_mode") or generation_mode,
        depth_level=meta.get("depth_level") or "Standard",
        user_level=meta.get("user_level") or "Intermediate/Advanced",
    )
    trace(
        f"CURRICULUM expand ✓ | +nodes={len(patch.new_nodes)} "
        f"+edges={len(patch.new_edges)} total={expanded.total_nodes}"
    )
    return expanded
