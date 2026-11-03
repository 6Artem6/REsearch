"""Node Grounding Gate в worker: discover (Gate 1) / digest (Gate 2) / finalize.

Все три этапа используют ML-веса — BGE-M3 (дедуп кандидатов, грубый отбор
абзацев), Cross-Encoder (реранжировка), Map-Reduce ingest. В процессе API
веса запрещены (``ml_runtime.assert_ml_weights_allowed``), там каждый шаг
молча уходил в fail-open. Поэтому роуты ``/node/grounding-*`` только
проверяют курс/ноду и ставят ``WorkJobKind.NODE_GROUNDING``, а работа идёт
здесь, в worker (``work_handlers._dispatch_work_job``).

Payload задачи: ``{"op": "discover"|"digest"|"finalize", "curriculum_id",
"node_id", "approved_urls"}``. Результат задачи — тот же JSON, что раньше
возвращали роуты синхронно.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from knowledge_engine.src.core.observability import set_span_attributes
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.shared.ml_runtime.ml_memory_guard import (
    rag_request_finished,
    rag_request_started,
)
from knowledge_engine.src.shared.node_grounding import (
    node_gate_state_store as gate_state_store,
)
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NodeGateState,
    NodeSearchProfile,
)

if TYPE_CHECKING:
    from knowledge_engine.src.domains.curriculum.schemas import (
        CurriculumGraph,
        CurriculumNode,
    )

NODE_GROUNDING_OPS = ("discover", "digest", "finalize")


class NodeGroundingNotFoundError(LookupError):
    """Курс или нода не найдены (роут отвечает 404)."""


def load_graph_and_node(
    curriculum_id: str, node_id: str
) -> tuple["CurriculumGraph", "CurriculumNode"]:
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph
    from knowledge_engine.src.shared.skill_tree_store import get_curriculum_graph

    raw = get_curriculum_graph(curriculum_id)
    if not raw:
        raise NodeGroundingNotFoundError(f"curriculum {curriculum_id} not found")
    graph = CurriculumGraph.model_validate(raw)
    node = next((n for n in graph.nodes if n.node_id == node_id), None)
    if node is None:
        raise NodeGroundingNotFoundError(f"node {node_id} not found")
    return graph, node


def node_needs_gate(node: "CurriculumNode") -> bool:
    """Гейт только для DEEP-ноды, ещё не заземлённой реальными источниками."""
    status = (node.grounding_status or "").strip()
    return node.node_risk_kind == "DEEP" and not (
        status == "grounded" and node.source_ref
    )


def empty_discovery(curriculum_id: str, node_id: str) -> dict[str, Any]:
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate1DiscoveryResponse,
        NodeSearchProfile,
    )

    return NodeGate1DiscoveryResponse(
        curriculum_id=curriculum_id,
        node_id=node_id,
        profile=NodeSearchProfile(),
        candidates=[],
    ).model_dump()


async def run_discover(curriculum_id: str, node_id: str) -> dict[str, Any]:
    """Этап 0+1: поисковый профиль ноды + сбор/дедуп/паспорта для Gate 1."""
    from knowledge_engine.src.domains.curriculum.academic_consensus import (
        consensus_allowed_for_policy,
        is_sota_rd_node,
    )
    from knowledge_engine.src.domains.curriculum.services import (
        node_candidate_collection_service as collection,
    )
    from knowledge_engine.src.domains.curriculum.source_policy import (
        resolve_source_policy,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate1DiscoveryResponse,
    )
    from knowledge_engine.src.shared.node_grounding.node_search_profile_service import (
        build_node_academic_plan,
        generate_node_search_profile,
    )
    from knowledge_engine.src.shared.skill_tree_store import get_curriculum_meta

    _graph, node = load_graph_and_node(curriculum_id, node_id)
    if not node_needs_gate(node):
        return empty_discovery(curriculum_id, node.node_id)

    meta = get_curriculum_meta(curriculum_id) or {}
    policy = resolve_source_policy(
        meta.get("source_policy"),
        str(meta.get("generation_mode") or "fast"),
        default="hybrid",
    )
    # SOTA-override — так же, как targeted_node_search.py: решение «нужна ли
    # академия» совпадает с боевым lazy-grounding.
    sota_override = policy == "practical_only" and is_sota_rd_node(node)
    effective_policy = "hybrid" if sota_override else policy
    include_academic = consensus_allowed_for_policy(effective_policy)
    trace(
        f"NODE_GATE discover ▶ | {curriculum_id}/{node.node_id} policy={policy} "
        f"sota_override={sota_override} include_academic={include_academic}"
    )

    profile = await generate_node_search_profile(
        node_title=node.title,
        node_summary=node.brief_summary,
        core_concepts=node.core_concepts,
        include_academic=include_academic,
        curriculum_id=curriculum_id,
        node_id=node.node_id,
    )
    academic_plan = None
    if include_academic:
        # Academic Query Architect (docs/ACADEMIC_AND_CONSENSUS.md): query для
        # показа на Gate 1 + arxiv_params для сбора.
        academic_plan = await build_node_academic_plan(
            node.title,
            node.brief_summary,
            anchor=f"node_gate_academic_plan:{curriculum_id}:{node.node_id}",
        )
        if (academic_plan.academic_query_en or "").strip():
            profile = profile.model_copy(
                update={"academic_queries": [academic_plan.academic_query_en]}
            )

    candidates = await collection.collect_node_candidates(
        profile,
        curriculum_id=curriculum_id,
        node_id=node.node_id,
        node=node,
        academic_plan=academic_plan,
    )
    if candidates:
        # Перезагрузка страницы до утверждения — тот же Gate 1 без поиска.
        gate_state_store.save_gate_state(
            NodeGateState(
                curriculum_id=curriculum_id,
                node_id=node.node_id,
                status="awaiting_gate_1",
                profile=profile,
                candidates=candidates,
            )
        )
    return NodeGate1DiscoveryResponse(
        curriculum_id=curriculum_id,
        node_id=node.node_id,
        profile=profile,
        candidates=candidates,
    ).model_dump()


async def run_digest(
    curriculum_id: str, node_id: str, approved_urls: list[str]
) -> dict[str, Any]:
    """Этап 2: полный текст → BGE-M3 + Cross-Encoder top-K → Gemma дайджест
    для Gate 2. Ничего не персистит."""
    from knowledge_engine.src.domains.curriculum.services import (
        node_digest_service,
    )

    _graph, node = load_graph_and_node(curriculum_id, node_id)
    digests = await node_digest_service.generate_node_digests(
        approved_urls,
        node_title=node.title,
        node_summary=node.brief_summary,
        curriculum_id=curriculum_id,
        node_id=node.node_id,
        anchor=f"node_gate_digest:{curriculum_id}:{node.node_id}",
    )
    if digests.digests:
        prior = gate_state_store.get_gate_state(curriculum_id, node.node_id)
        gate_state_store.save_gate_state(
            NodeGateState(
                curriculum_id=curriculum_id,
                node_id=node.node_id,
                status="awaiting_gate_2",
                profile=prior.profile if prior else NodeSearchProfile(),
                candidates=prior.candidates if prior else [],
                digests=digests.digests,
                approved_urls=list(approved_urls),
            )
        )
    return digests.model_dump()


async def run_finalize(
    curriculum_id: str, node_id: str, approved_urls: list[str]
) -> dict[str, Any]:
    """Этап 3: Gate 2 согласие → Map-Reduce ingest → нода ``grounded``,
    курс сохраняется. После этого ``/node/init`` поиск не повторяет."""
    from knowledge_engine.src.domains.curriculum.source_policy import (
        resolve_source_policy,
    )
    from knowledge_engine.src.shared import skill_tree_store
    from knowledge_engine.src.shared.node_grounding import (
        node_grounding_finalize_service,
    )

    graph, node = load_graph_and_node(curriculum_id, node_id)
    meta = skill_tree_store.get_curriculum_meta(curriculum_id) or {}
    target_goal = str(meta.get("target_goal") or graph.description or "").strip()
    generation_mode = str(meta.get("generation_mode") or "fast")
    graph, updated_node = await node_grounding_finalize_service.finalize_node_grounding(
        graph, node, approved_urls, target_goal=target_goal
    )
    gate_state_store.clear_gate_state(curriculum_id, node.node_id)
    skill_tree_store.save_curriculum_record(
        graph,
        target_goal=target_goal,
        generation_mode=generation_mode,
        depth_level=str(meta.get("depth_level") or "Standard"),
        user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
        source_policy=resolve_source_policy(
            meta.get("source_policy"), generation_mode, default="hybrid"
        ),
    )
    return {
        "curriculum_id": curriculum_id,
        "node_id": updated_node.node_id,
        "grounding_status": updated_node.grounding_status,
        "mapped_source_ids": list(updated_node.mapped_source_ids or []),
        "resource_urls": list(updated_node.resource_urls or []),
    }


async def _run_op(payload: dict[str, Any]) -> dict[str, Any]:
    op = str(payload.get("op") or "").strip().lower()
    cid = str(payload.get("curriculum_id") or "").strip()
    nid = str(payload.get("node_id") or "").strip()
    urls = [str(u) for u in (payload.get("approved_urls") or [])]
    if op == "discover":
        return await run_discover(cid, nid)
    if op == "digest":
        return await run_digest(cid, nid, urls)
    if op == "finalize":
        return await run_finalize(cid, nid, urls)
    raise ValueError(f"Unknown node_grounding op: {op!r}")


def run_node_grounding_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Точка входа worker (синхронный поток задачи)."""
    op = str(payload.get("op") or "")
    cid = str(payload.get("curriculum_id") or "")
    nid = str(payload.get("node_id") or "")
    set_span_attributes(
        **{
            "ke.node_gate.op": op,
            "ke.tutor.curriculum_id": cid,
            "ke.tutor.node_id": nid,
        }
    )
    trace(f"WORKER node_grounding ▶ {op} | {cid}/{nid}")
    t0 = time.perf_counter()
    # Весь этап — один ML-запрос для ML_MEMORY_GUARD: пока он активен, guard
    # не выгружает BGE-M3/Cross-Encoder между статьями (иначе цикл digest
    # выгружал и тут же грузил модель заново). После — обычный cooldown
    # RAG_MPS_REQUEST_COOLDOWN_SEC: модели нужны следующему этапу гейта.
    rag_request_started()
    try:
        result = asyncio.run(_run_op(payload))
    finally:
        rag_request_finished()
    trace(
        f"WORKER node_grounding ✓ {op} | {cid}/{nid} | "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return result


__all__ = [
    "NODE_GROUNDING_OPS",
    "NodeGroundingNotFoundError",
    "empty_discovery",
    "load_graph_and_node",
    "node_needs_gate",
    "run_digest",
    "run_discover",
    "run_finalize",
    "run_node_grounding_job",
]
