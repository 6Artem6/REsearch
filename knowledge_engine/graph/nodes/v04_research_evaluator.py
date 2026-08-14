"""v0.4: Gemini evaluator — gaps и достаточность L2."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from knowledge_engine.config import MAX_RESEARCH_DEPTH
from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState
from knowledge_engine.schemas.llm_contracts.v04_gemini import ResearchEvaluationContract
from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    run_stateless_gemini,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start, trace

_console = Console()


def _collect_l2_summary(state: EngineGraphState, store: VectorStore) -> str:
    lines: list[str] = []
    for nid in state.get("knowledge_node_ids") or []:
        node = store.get_knowledge_node(nid)
        if node and node.level == "L2_EVIDENCE":
            src = f" ({node.source_url})" if node.source_url else ""
            lines.append(f"-{src} {node.content[:400]}")
    return "\n".join(lines[:24]) or "(ещё нет L2-фактов)"


def research_evaluator_node_v04(state: EngineGraphState) -> dict[str, Any]:
    node_start("research_evaluator_node_v04 (Gemini)")
    pipeline_phase("Evaluator (Gemini)")
    depth = int(state.get("depth") or 0)
    query = state.get("original_query") or state.get("user_problem") or ""
    constraints = state.get("constraints") or state.get("context_constraints") or ""
    l0 = state.get("l0_summary") or ""

    store = VectorStore()
    l2_block = _collect_l2_summary(state, store)
    anchor = global_anchor_from_state(query, constraints, l0)

    set_status(f"[evaluator] Gemini достаточность L2 (depth={depth})…")
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Сравни L0/L1 карту с L2-фактами в LanceDB. "
        "Оцени достаточность для Trade-off матрицы (failure modes, tail latency, RAM, LanceDB). "
        "JSON ResearchEvaluation."
    )
    user = (
        f"L0:\n{l0}\n\nL2:\n{l2_block}\n\n"
        f"URL explored: {len(state.get('explored_urls') or [])}"
    )

    evaluation = run_stateless_gemini(
        system,
        user,
        anchor,
        response_schema=ResearchEvaluationContract,
        label="v04 evaluator / ResearchEvaluation",
        rpm_pause=True,
    )

    updates: dict[str, Any] = {
        "research_sufficient": evaluation.is_sufficient,
        "last_research_gaps": evaluation.missing_gaps,
    }

    if not evaluation.is_sufficient and depth < MAX_RESEARCH_DEPTH:
        new_q = [q.strip() for q in evaluation.new_search_queries if q.strip()]
        if new_q:
            updates["search_queries"] = new_q[:4]
        updates["depth"] = depth + 1
        gap_preview = ", ".join(evaluation.missing_gaps[:4]) or "не указаны"
        trace(f"RE-ACT gaps: {gap_preview}")
        set_status(f"[Re-Act] допоиск: {gap_preview[:100]}")
        _console.print(
            f"\n[yellow]Re-Act {depth + 1}/{MAX_RESEARCH_DEPTH}[/yellow] {gap_preview}\n"
        )
    elif evaluation.is_sufficient:
        trace("RE-ACT ✓ sufficient")

    node_end("research_evaluator_node_v04", f"sufficient={evaluation.is_sufficient}")
    return updates
