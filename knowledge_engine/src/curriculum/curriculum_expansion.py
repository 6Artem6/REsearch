"""Lite (вектор) + Flash (граф patch) для expand_curriculum."""

from __future__ import annotations

from knowledge_engine.config import (
    GEMINI_FLASH_MODEL,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.curriculum import ExpansionVectorContract
from knowledge_engine.services.gemini_stateless import (
    gemini_reasoner_model_chain,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.curriculum.dag_validator import (
    CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumExpansionEdge,
    CurriculumExpansionPatch,
    CurriculumGraph,
    CurriculumNode,
    CurriculumSearchHit,
)
from knowledge_engine.src.curriculum.search_first_flash import (
    _FlashExpansionPatch,
    coerce_expansion_patch_from_flash,
)
from knowledge_engine.src.curriculum.search_prestep import (
    _normalize_url_key,
    search_hits_as_prompt_json,
)
from knowledge_engine.ui.run_log import trace

_LITE_EXPANSION_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are the Lite strategist of a learning path (direction vector).\n"
    "Input: the current graph (brief) and the user's expansion request.\n\n"
    "Task: produce ONLY the expansion_vector field — a coherent text description of "
    "the direction to deepen (2-6 sentences).\n"
    "FORBIDDEN: a list of nodes, links, URLs, graph JSON structure, DAG.\n"
)
"""
RU (пояснение): Lite-этап расширения курса — только текстовый вектор
направления (2-6 предложений), без нод/DAG/JSON.
"""

_FLASH_EXPANSION_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are the Flash architect of a learning DAG (path expansion).\n\n"
    "Input:\n"
    "- expansion_vector from Lite (direction);\n"
    "- current nodes (id, title, prerequisites, layer);\n"
    "- source excerpts (if provided).\n\n"
    "Task:\n"
    "1. Unfold the vector into 2-3 atomic new_nodes (not one crammed node covering several technologies).\n"
    "2. For each node: node_curriculum_breakdown, source_ref with relevant_extracts "
    "(only if sources were provided and are relevant).\n"
    "3. new_edges: prerequisite from_node_id → to_node_id (from must be completed before to).\n"
    "4. node_id of new nodes — unique snake_case.\n"
    "5. Do not rewrite existing nodes — only new_nodes and new_edges.\n\n"
    "TOPOLOGY (critical):\n"
    "1. SINGLE ENTRY POINT (anchor): the new branch attaches to ONE node of the old graph. "
    "Only the first new_node in the chain may have prerequisites on existing node_id "
    "(at most 1-2 parents from the old graph).\n"
    "2. NO SPIDERWEBS: do not link one new_node to several unrelated old branches "
    '(e.g. "Streaming" + IoT + RDMA + serialization in a single node).\n'
    "3. All other new_nodes: prerequisites ONLY on other new_nodes — a single chain "
    "(new_1 → new_2 → new_3), no branching and no references to the old graph.\n"
    "4. new_edges must reflect that same chain; do not add cross-edges between new_nodes.\n"
    "5. Do not add new_edges that attach EXISTING old nodes to other old nodes.\n"
    "6. DECOMPOSITION: sequential steps, not one final SOTA node with every prereq of the graph.\n\n"
    "**User anchor topics:** if the path goal (target_goal / graph description) "
    "or user_expansion_request lists specific topics in parentheses, comma-separated, or as a list "
    '(e.g. "Storage architecture (WAL, Ring Buffer, P99)"):\n'
    "   - You MUST weave each of these topics into the graph as separate nodes or key concepts.\n"
    "   - Build logical dependencies (prerequisites) around them: place basic topics earlier, "
    "advanced ones — in deeper layers of the graph.\n"
    "   - Preserve the substance and terminology of the suggested topics, adapting their names "
    "naturally to the course's engineering style.\n"
    f"{CURRICULUM_DAG_REPAIR_PRESERVE_ANCHOR_TOPICS}\n"
)
"""
RU (пояснение): Flash-этап расширения — 2-3 атомарные new_nodes от Lite-
вектора; топология: один anchor-родитель из старого графа, дальше цепочка
без ветвления и без паутины к другим старым нодам.
"""

_NO_FRESH_GROUNDING_USER_BLOCK = (
    "### sources_context\n"
    "Свежих внешних статей не найдено (Gemini Search grounding пуст или rate limit).\n"
    "Сформируй чистое академическое расщепление expansion_vector на 2–3 атомарные new_nodes.\n"
    "НЕ притягивай старые нерелевантные выжимки из базы; source_ref/relevant_extracts — "
    "минимальные или пустые.\n"
)


def _graph_summary(graph: CurriculumGraph) -> str:
    lines = [
        f"curriculum_id={graph.curriculum_id}",
        f"title={graph.title}",
        f"nodes={graph.total_nodes}",
    ]
    for n in graph.nodes[:40]:
        prereq = ",".join(n.prerequisites[:6])
        lines.append(f"- {n.node_id} | {n.title} | layer={n.layer} | prereq=[{prereq}]")
    return "\n".join(lines)


def lite_plan_expansion_vector(
    graph: CurriculumGraph,
    user_request: str,
    anchor: str,
) -> ExpansionVectorContract:
    payload = (
        f"### user_expansion_request\n{user_request.strip()}\n\n"
        f"### current_graph\n{_graph_summary(graph)}\n"
    )
    return run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        _LITE_EXPANSION_SYSTEM,
        payload,
        anchor,
        ExpansionVectorContract,
        "curriculum / expansion_lite",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=[GEMINI_LITE_MODEL],
    )


def flash_build_expansion_patch(
    graph: CurriculumGraph,
    expansion_vector: str,
    hits: list[CurriculumSearchHit],
    user_request: str,
    anchor: str,
    *,
    fresh_grounding_hits: int = 0,
) -> CurriculumExpansionPatch:
    if fresh_grounding_hits <= 0:
        sources_json = "[]"
        sources_section = _NO_FRESH_GROUNDING_USER_BLOCK
        trace("CURRICULUM expansion_flash ▶ | sources=0 (no fresh grounding)")
    else:
        sources_json = search_hits_as_prompt_json(hits)
        sources_section = f"### lancedb_and_harvest_extracts\n{sources_json}\n"
        trace(f"CURRICULUM expansion_flash ▶ | sources={len(hits)}")
    payload = (
        f"### user_expansion_request\n{user_request.strip()}\n\n"
        f"### expansion_vector\n{expansion_vector.strip()}\n\n"
        f"### current_graph\n{_graph_summary(graph)}\n\n"
        f"{sources_section}"
    )
    raw_patch = run_gemini_structured_with_chain(
        GEMINI_FLASH_MODEL,
        _FLASH_EXPANSION_SYSTEM,
        payload,
        anchor,
        _FlashExpansionPatch,
        "curriculum / expansion_flash",
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(GEMINI_FLASH_MODEL),
    )
    registry = list(graph.curriculum_sources_registry or [])
    patch = coerce_expansion_patch_from_flash(raw_patch, hits, registry)
    if not patch.new_nodes:
        raise ValueError(
            "expansion_flash: модель не вернула валидные new_nodes "
            f"(raw_nodes={len(raw_patch.new_nodes or [])})"
        )
    return patch


def apply_expansion_patch(
    graph: CurriculumGraph,
    patch: CurriculumExpansionPatch,
) -> CurriculumGraph:
    """Merge new_nodes/edges; существующие node_id и прогресс не трогаем."""
    existing_ids = {n.node_id for n in graph.nodes}
    nodes = list(graph.nodes)
    for n in patch.new_nodes:
        if n.node_id in existing_ids:
            continue
        existing_ids.add(n.node_id)
        nodes.append(n)

    edge_map: dict[str, set[str]] = {}
    for n in nodes:
        edge_map.setdefault(n.node_id, set()).update(n.prerequisites)

    for e in patch.new_edges:
        fr = e.from_node_id.strip()
        to = e.to_node_id.strip()
        if not fr or not to or to not in existing_ids:
            continue
        if fr not in existing_ids:
            continue
        edge_map.setdefault(to, set()).add(fr)

    merged_nodes = []
    for n in nodes:
        extra = sorted(edge_map.get(n.node_id, set()))
        prereq = list(dict.fromkeys([*n.prerequisites, *extra]))
        merged_nodes.append(n.model_copy(update={"prerequisites": prereq}))

    return graph.model_copy(
        update={
            "nodes": merged_nodes,
            "total_nodes": len(merged_nodes),
        }
    )


_LAYER_RANK = {"foundation": 0, "advanced": 1, "sota": 2}


def _ancestors_of(node_id: str, by_id: dict[str, CurriculumNode]) -> set[str]:
    seen: set[str] = set()
    stack = list(by_id[node_id].prerequisites)
    while stack:
        u = stack.pop()
        if u in seen or u not in by_id:
            continue
        seen.add(u)
        stack.extend(by_id[u].prerequisites)
    return seen


def _pick_anchor_old_prereqs(
    candidates: list[str],
    by_id: dict[str, CurriculumNode],
    new_node: CurriculumNode,
    max_keep: int = 2,
) -> list[str]:
    if not candidates:
        return []
    rank = _LAYER_RANK.get(str(new_node.layer), 1)
    scored = sorted(
        candidates,
        key=lambda pid: (
            abs(_LAYER_RANK.get(str(by_id[pid].layer), 1) - rank),
            pid,
        ),
    )
    return scored[:max_keep]


def _enforce_new_node_chain(
    by_id: dict[str, CurriculumNode],
    new_ids: set[str],
    anchor_id: str | None,
    ordered_new: list[str],
) -> int:
    """Вторичные new_nodes: prereq только на предыдущую в ветке (цепочка)."""
    removed = 0
    order = [nid for nid in ordered_new if nid in new_ids]
    if not order:
        order = sorted(new_ids)
    if anchor_id and anchor_id in order:
        order = [anchor_id] + [n for n in order if n != anchor_id]
    for i, nid in enumerate(order):
        if nid not in by_id:
            continue
        node = by_id[nid]
        if nid == anchor_id:
            continue
        if i == 0:
            continue
        prev = order[i - 1]
        old_links = [p for p in node.prerequisites if p not in new_ids]
        new_links = [p for p in node.prerequisites if p in new_ids]
        want = [prev] if prev in new_ids else new_links[:1]
        if old_links or new_links != want:
            removed += len(node.prerequisites) - len(want)
            by_id[nid] = node.model_copy(update={"prerequisites": want})
    return removed


def validate_and_repair_expansion_dag(
    graph: CurriculumGraph,
    pre_existing_ids: set[str],
    new_node_ids: set[str],
    patch_edges: list[CurriculumExpansionEdge],
    *,
    new_nodes_ordered: list[str] | None = None,
) -> CurriculumGraph:
    """
    После merge: убрать спагетти-привязки, transitive redundant prereq, лимит anchor.
    """
    removed = 0
    by_id: dict[str, CurriculumNode] = {n.node_id: n for n in graph.nodes}
    new_ids = {nid for nid in new_node_ids if nid in by_id}

    patch_old_old: set[tuple[str, str]] = set()
    for e in patch_edges or []:
        fr, to = e.from_node_id.strip(), e.to_node_id.strip()
        if fr in pre_existing_ids and to in pre_existing_ids:
            patch_old_old.add((fr, to))

    # Убрать old→old ребра, добавленные expand
    for nid, node in list(by_id.items()):
        if nid not in pre_existing_ids:
            continue
        kept = [
            p
            for p in node.prerequisites
            if p not in pre_existing_ids or (p, nid) not in patch_old_old
        ]
        if len(kept) != len(node.prerequisites):
            removed += len(node.prerequisites) - len(kept)
            by_id[nid] = node.model_copy(update={"prerequisites": kept})

    # Anchor: одна new_node с prereq на старый граф
    anchor_id: str | None = None
    for nid in sorted(new_ids):
        old_links = [p for p in by_id[nid].prerequisites if p in pre_existing_ids]
        if old_links:
            anchor_id = nid
            break
    if anchor_id is None and new_ids:
        anchor_id = sorted(new_ids)[0]

    for nid in new_ids:
        node = by_id[nid]
        old_links = [p for p in node.prerequisites if p in pre_existing_ids]
        if not old_links:
            continue
        if nid != anchor_id:
            kept = [p for p in node.prerequisites if p not in pre_existing_ids]
            removed += len(old_links)
            by_id[nid] = node.model_copy(update={"prerequisites": kept})
        elif len(old_links) > 2:
            trimmed = _pick_anchor_old_prereqs(old_links, by_id, node, max_keep=2)
            removed += len(old_links) - len(trimmed)
            other = [p for p in node.prerequisites if p not in pre_existing_ids]
            by_id[nid] = node.model_copy(update={"prerequisites": trimmed + other})

    # Transitive reduction на prerequisites каждой ноды
    for nid, node in list(by_id.items()):
        prereqs = list(node.prerequisites)
        if len(prereqs) < 2:
            continue
        kept: list[str] = []
        for p in prereqs:
            drop = False
            for q in prereqs:
                if q != p and p in _ancestors_of(q, by_id):
                    drop = True
                    break
            if not drop:
                kept.append(p)
        if len(kept) != len(prereqs):
            removed += len(prereqs) - len(kept)
            by_id[nid] = node.model_copy(update={"prerequisites": kept})

    removed += _enforce_new_node_chain(
        by_id,
        new_ids,
        anchor_id,
        new_nodes_ordered or [],
    )

    if removed:
        trace(
            f"CURRICULUM expand DAG repaired ▶ removed {removed} redundant cross-edges"
        )

    nodes = [by_id[n.node_id] for n in graph.nodes if n.node_id in by_id]
    return graph.model_copy(update={"nodes": nodes, "total_nodes": len(nodes)})


def merge_graph_source_registry(
    graph: CurriculumGraph,
    new_hits: list[CurriculumSearchHit],
) -> CurriculumGraph:
    """Добавить новые источники (expand grounding) в curriculum_sources_registry."""
    from knowledge_engine.src.curriculum.search_first_flash import _registry_from_hits
    from knowledge_engine.src.curriculum.source_registry import (
        sync_route_sources_from_registry,
    )

    existing_keys = {
        _normalize_url_key(e.url) for e in graph.curriculum_sources_registry
    }
    extra_hits = [h for h in new_hits if _normalize_url_key(h.url) not in existing_keys]
    if not extra_hits:
        return graph

    new_entries = _registry_from_hits(extra_hits)
    used_ids = {e.source_id for e in graph.curriculum_sources_registry}
    next_i = 1
    adjusted: list = []
    for entry in new_entries:
        while f"src_{next_i}" in used_ids:
            next_i += 1
        sid = f"src_{next_i}"
        used_ids.add(sid)
        next_i += 1
        adjusted.append(entry.model_copy(update={"source_id": sid}))

    registry = list(graph.curriculum_sources_registry) + adjusted
    out = graph.model_copy(update={"curriculum_sources_registry": registry[:30]})
    return sync_route_sources_from_registry(out)
