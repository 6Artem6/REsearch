"""Этап 2: Lite классификация нод BASE vs DEEP (риск галлюцинаций)."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.curriculum.lite_search_pipeline import _lite_structured
from knowledge_engine.src.curriculum.schemas import CurriculumGraph, CurriculumNode
from knowledge_engine.ui.run_log import trace

_RISK_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a curriculum graph inspector. For each node, determine its knowledge type:\n\n"
    "**BASE** (General practice): standard concepts, syntax, basic patterns, "
    "topic introduction. No web search is needed — the model's own knowledge is enough.\n\n"
    "**DEEP** (Nuances and academia): architectural risks, edge cases, fault-tolerance "
    "guarantees, formal/distributed algorithms, SOTA, verification, "
    "complex trade-offs. These REQUIRE external RAG.\n\n"
    "Rules:\n"
    "- foundation nodes with basic syntax → usually BASE.\n"
    "- advanced/sota nodes with risks, guarantees, formal models → DEEP.\n"
    "- Do not mark everything DEEP: ~30-50% of nodes is typical for DEEP.\n\n"
    "JSON: assignments — an array of { node_id, risk_kind (BASE|DEEP), reason (Russian) } "
    "for every node_id in the input."
)
"""
RU (пояснение): классификация нод BASE (хватит знаний модели) vs DEEP
(нужен внешний RAG) — этап 2 генерации курса, перед добором источников.
"""


class NodeRiskAssignment(BaseModel):
    node_id: str = ""
    risk_kind: Literal["BASE", "DEEP"] = "BASE"
    reason: str = ""


class NodeRiskBatchResult(BaseModel):
    assignments: list[NodeRiskAssignment] = Field(default_factory=list)


def _anchor_risk(goal: str) -> str:
    return f"curriculum_node_risk:{(goal or '').strip()[:400]}"


def _nodes_payload(nodes: list[CurriculumNode]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for n in nodes:
        out.append(
            {
                "node_id": n.node_id,
                "layer": n.layer,
                "title": n.title[:300],
                "category": n.category[:200],
                "brief_summary": n.brief_summary[:800],
                "core_concepts": ", ".join(n.core_concepts[:6]),
            }
        )
    return out


async def classify_node_risks_async(
    graph: CurriculumGraph,
    target_goal: str,
    *,
    anchor: str | None = None,
) -> dict[str, NodeRiskAssignment]:
    nodes = graph.nodes
    if not nodes:
        return {}

    trace(f"CURRICULUM node_risk ▶ | Lite inspect nodes={len(nodes)}")
    user_obj = {
        "course_goal": (target_goal or "").strip()[:1200],
        "nodes": _nodes_payload(nodes),
    }
    try:
        out = await _lite_structured(
            _RISK_SYSTEM,
            json.dumps(user_obj, ensure_ascii=False),
            anchor or _anchor_risk(target_goal),
            NodeRiskBatchResult,
            "curriculum / node_risk_classification",
        )
    except Exception as exc:
        trace(f"CURRICULUM node_risk fallback | heuristic DEEP=sota/advanced | {exc}")
        by_id: dict[str, NodeRiskAssignment] = {}
        for n in nodes:
            kind: Literal["BASE", "DEEP"] = (
                "DEEP" if n.layer in ("advanced", "sota") else "BASE"
            )
            by_id[n.node_id] = NodeRiskAssignment(
                node_id=n.node_id,
                risk_kind=kind,
                reason="heuristic по layer",
            )
        return by_id

    by_id: dict[str, NodeRiskAssignment] = {}
    for a in out.assignments or []:
        nid = (a.node_id or "").strip()
        if not nid:
            continue
        rk = (a.risk_kind or "BASE").strip().upper()
        if rk not in ("BASE", "DEEP"):
            rk = "BASE"
        by_id[nid] = NodeRiskAssignment(
            node_id=nid,
            risk_kind=rk,
            reason=(a.reason or "")[:400],
        )

    for n in nodes:
        if n.node_id not in by_id:
            by_id[n.node_id] = NodeRiskAssignment(
                node_id=n.node_id,
                risk_kind="BASE",
                reason="не классифицировано Lite",
            )

    deep_n = sum(1 for a in by_id.values() if a.risk_kind == "DEEP")
    trace(f"CURRICULUM node_risk ✓ | BASE={len(by_id) - deep_n} DEEP={deep_n}")
    return by_id


def apply_risk_classifications(
    graph: CurriculumGraph,
    assignments: dict[str, NodeRiskAssignment],
) -> CurriculumGraph:
    updated: list[CurriculumNode] = []
    for n in graph.nodes:
        a = assignments.get(n.node_id)
        if not a:
            updated.append(n)
            continue
        risk = a.risk_kind
        status = n.grounding_status
        if risk == "BASE":
            status = "model_only"
        elif status not in ("grounded",):
            status = "pending_grounding"
        updated.append(
            n.model_copy(
                update={
                    "node_risk_kind": risk,
                    "grounding_status": status,
                }
            )
        )
    return graph.model_copy(update={"nodes": updated, "total_nodes": len(updated)})


def classify_and_apply_node_risks(
    graph: CurriculumGraph,
    target_goal: str,
    *,
    anchor: str | None = None,
) -> CurriculumGraph:
    import asyncio

    assignments = asyncio.run(
        classify_node_risks_async(graph, target_goal, anchor=anchor)
    )
    return apply_risk_classifications(graph, assignments)
