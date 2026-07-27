"""v0.4: Gemini unraveling."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    run_stateless_gemini,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start


def unraveling_node_v04(state: EngineGraphState) -> dict[str, Any]:
    node_start("unraveling_node_v04 (Gemini)")
    pipeline_phase("Unraveling (Gemini)")
    parsed = EngineState.model_validate(state)
    if parsed.report is None or parsed.selected_option_id is None:
        node_end("unraveling_node_v04", "skip")
        return {}

    report = parsed.report
    option = next(
        (o for o in report.options if o.id == parsed.selected_option_id), None
    )
    if option is None:
        raise RuntimeError(f"Вариант id={parsed.selected_option_id} не найден")

    store = VectorStore()
    hits = store.hybrid_search_nodes(parsed.user_problem, limit=4)
    ctx_parts = [store.get_hierarchical_context(n.id) for n in hits]
    hierarchy = "\n\n---\n\n".join(p for p in ctx_parts if p)
    draft = state.get("pre_synthesis_draft") or ""

    anchor = global_anchor_from_state(
        parsed.user_problem,
        parsed.context_constraints,
        parsed.l0_summary or state.get("l0_summary") or "",
    )

    set_status(f"[unraveling] Gemini вариант {option.id}…")
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Engineering unraveling: implementation, failure modes, RAM/latency Mac M-series, LanceDB. Markdown."
    )
    user = (
        f"Вариант: {option.pattern_name} — {option.fundamental_idea}\n"
        f"Pros: {option.pros}\nCons: {option.cons_and_risks}\n"
        f"Cost: {option.operational_cost}\n\n"
        f"Pre-synthesis:\n{draft[:8000]}\n\nКонтекст:\n{hierarchy[:8000]}"
    )

    details = run_stateless_gemini(
        system,
        user,
        anchor,
        label="v04 unraveling / markdown",
    )
    text = details if isinstance(details, str) else str(details)
    node_end("unraveling_node_v04", f"chars={len(text)}")
    return {"unraveled_details": text.strip()}
