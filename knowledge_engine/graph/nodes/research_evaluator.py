"""v0.3.1: Re-Act оценка — достаточно ли L2 для matrix (локальная модель)."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from knowledge_engine.config import MAX_RESEARCH_DEPTH
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, ResearchEvaluation
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.hybrid_llm import run_react_evaluation
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
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


def research_evaluator_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("research_evaluator_node (Re-Act local)")
    depth = int(state.get("depth") or 0)
    query = state.get("original_query") or state.get("user_problem") or ""
    constraints = state.get("constraints") or state.get("context_constraints") or ""
    l0 = state.get("l0_summary") or ""

    store = VectorStore()
    l2_block = _collect_l2_summary(state, store)
    anchor = global_anchor_from_state(query, constraints, l0)

    set_status(f"[Re-Act] локальная оценка L2 (depth={depth})…")
    system = (
        f"{RUSSIAN_OUTPUT_RULE} "
        "Оцени, достаточно ли собранных L2-фактов для Trade-off матрицы по задаче. "
        "Учитывай: failure modes, tail latency, RAM Mac M1, LanceDB invalidation. "
        "Если нет — missing_gaps и 1–3 new_search_queries (точечные, не общие). "
        "Ответ строго JSON ResearchEvaluation."
    )
    user = (
        f"L0 summary:\n{l0}\n\n"
        f"Собранные L2:\n{l2_block}\n\n"
        f"Уже исследовано URL: {len(state.get('explored_urls') or [])}"
    )

    evaluation = run_react_evaluation(
        system,
        user,
        anchor,
        ResearchEvaluation,
        "research_evaluator / ResearchEvaluation",
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
        trace(
            f"RE-ACT → discovery | depth {depth}→{depth + 1} | gaps: {gap_preview} | "
            f"queries={new_q[:3]}"
        )
        set_status(f"[Re-Act] шаг {depth + 1}: допоиск — {gap_preview[:120]}")
        _console.print(
            f"\n[yellow]Re-Act шаг {depth + 1}/{MAX_RESEARCH_DEPTH}[/yellow] "
            f"— закрываем пробелы: {gap_preview}\n"
            f"[dim]Запросы: {new_q or evaluation.new_search_queries}[/dim]\n"
        )
    elif evaluation.is_sufficient:
        trace("RE-ACT ✓ sufficient → matrix")
        set_status("[Re-Act] данных достаточно → матрица")
    else:
        trace("RE-ACT max depth → matrix anyway")
        set_status("[Re-Act] лимит глубины → матрица")

    node_end(
        "research_evaluator_node (Re-Act local)",
        f"sufficient={evaluation.is_sufficient}, depth={updates.get('depth', depth)}",
    )
    return updates
