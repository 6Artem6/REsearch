"""Unified Unraveling node — structured UnravelingNodeResponse, Host markdown."""

from __future__ import annotations

from typing import Any

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.schemas.unraveling_schemas import UnravelingNodeResponse
from knowledge_engine.services.gemini_stateless import global_anchor_from_state
from knowledge_engine.services.hybrid_llm import run_structured_hybrid
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start

_UNRAVELING_SYSTEM = (
    "You are a staff engineer unraveling ONE selected trade-off option.\n"
    "Return strictly valid JSON matching UnravelingNodeResponse. No free-form Markdown "
    "outside JSON fields.\n"
    "Required keys: summary, ram_and_latency_impact, failure_modes, "
    "technical_breakdown_markdown.\n"
    "failure_modes: 1–12 objects with scenario, impact, mitigation.\n"
    "technical_breakdown_markdown MUST be at least 300 words: algorithms, data "
    "structures, code or config listings, production failure modes.\n"
    "ram_and_latency_impact: concrete RAM/latency analysis for Apple Silicon / "
    "Mac M-series and local LanceDB where relevant.\n"
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "User-facing string fields MUST be in natural Russian.\n"
)


def render_unraveling_markdown(response_obj: UnravelingNodeResponse) -> str:
    """Host-owned UI markdown — the model never emits this wrapping."""
    blocks = [
        f"## Резюме\n\n{response_obj.summary.strip()}",
        (
            "## RAM и latency (Apple Silicon)\n\n"
            f"{response_obj.ram_and_latency_impact.strip()}"
        ),
    ]
    fm_parts: list[str] = []
    for i, fm in enumerate(response_obj.failure_modes, start=1):
        fm_parts.append(
            f"### {i}. {fm.scenario.strip()}\n\n"
            f"**Влияние:** {fm.impact.strip()}\n\n"
            f"**Митигация:** {fm.mitigation.strip()}"
        )
    blocks.append("## Failure modes\n\n" + "\n\n".join(fm_parts))
    blocks.append(
        "## Технический разбор\n\n" + response_obj.technical_breakdown_markdown.strip()
    )
    return "\n\n".join(blocks)


def unraveling_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("unraveling_node")
    pipeline_phase("Unraveling")
    parsed = EngineState.model_validate(state)
    if parsed.report is None or parsed.selected_option_id is None:
        node_end("unraveling_node", "skip")
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
    draft = (state.get("pre_synthesis_draft") or "").strip()

    anchor = global_anchor_from_state(
        parsed.user_problem,
        parsed.context_constraints,
        parsed.l0_summary or state.get("l0_summary") or "",
    )

    set_status(f"[unraveling] structured JSON вариант {option.id}…")
    user = (
        f"Selected option id={option.id}:\n"
        f"{option.pattern_name} — {option.fundamental_idea}\n"
        f"Category: {option.category}\n"
        f"Pros: {option.pros}\n"
        f"Cons: {option.cons_and_risks}\n"
        f"Operational cost: {option.operational_cost}\n"
    )
    if draft:
        user += f"\nPre-synthesis draft:\n{draft[:8000]}\n"
    if hierarchy:
        user += f"\nHierarchical knowledge context:\n{hierarchy[:12000]}\n"

    response_obj = run_structured_hybrid(
        _UNRAVELING_SYSTEM,
        user,
        anchor,
        UnravelingNodeResponse,
        "unraveling / UnravelingNodeResponse",
        prefer_gemini=True,
    )
    details = render_unraveling_markdown(response_obj)
    node_end("unraveling_node", f"chars={len(details)}")
    return {"unraveled_details": details}
