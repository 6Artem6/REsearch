"""Штурвал: init DEEP-ноды не запускает автопилотный lazy grounding.

Источники такой ноды приходят только через Node Grounding Gate. Если UI
гейт пропустил (0 кандидатов, сбой discover, старый бандл), init даёт
лекцию без поиска, а не полный пайплайн. Режим — из запроса, иначе
``control_axis`` курса в skill_tree_store."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGraph,
    CurriculumNode,
)
from knowledge_engine.src.domains.grounding import engine
from knowledge_engine.src.domains.grounding.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.shared import skill_tree_store


def _graph() -> CurriculumGraph:
    def node(node_id: str, **kw) -> CurriculumNode:
        return CurriculumNode(
            node_id=node_id,
            title=f"Тема {node_id}",
            layer="advanced",
            category="Категория",
            brief_summary="Достаточно длинное краткое описание ноды для теста.",
            core_concepts=["концепция"],
            **kw,
        )

    return CurriculumGraph(
        curriculum_id="cur_guard",
        title="Guard",
        description="Курс для теста защиты Штурвала.",
        total_nodes=3,
        nodes=[
            node("deep", node_risk_kind="DEEP", grounding_status="pending_grounding"),
            node("base_one"),
            node("base_two"),
        ],
    )


def _req(control_axis: str | None) -> NodeDeepDiveRequest:
    return NodeDeepDiveRequest(
        curriculum_id="cur_guard",
        node_data=NodeDataInput(
            node_id="deep", title="Тема deep", layer="advanced", core_concepts=["c"]
        ),
        user_action="init",
        control_axis=control_axis,
    )


@pytest.fixture
def store(monkeypatch):
    meta = {"target_goal": "g", "source_policy": "practical_only"}
    monkeypatch.setattr(
        skill_tree_store, "get_curriculum_graph", lambda cid: _graph().model_dump()
    )
    monkeypatch.setattr(skill_tree_store, "get_curriculum_meta", lambda cid: meta)
    monkeypatch.setattr(skill_tree_store, "save_curriculum_record", lambda *a, **k: {})
    return meta


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("request_axis", "stored_axis"),
    [("steering", None), (None, "steering")],
)
async def test_steering_skips_autopilot_search(store, request_axis, stored_axis):
    if stored_axis:
        store["control_axis"] = stored_axis
    search = AsyncMock()
    with patch(
        "knowledge_engine.src.domains.curriculum.targeted_node_grounding."
        "lazy_ground_deep_node_on_demand",
        search,
    ):
        out = await engine._apply_lazy_grounding_for_init(_req(request_axis))
    search.assert_not_called()
    assert out.node_data.node_id == "deep"


@pytest.mark.anyio
async def test_autopilot_still_grounds_pending_deep_node(store):
    """Явный «Автопилот» из переключателя UI перекрывает режим курса."""
    store["control_axis"] = "steering"
    graph = _graph()
    deep = graph.nodes[0]
    search = AsyncMock(return_value=(graph, deep))
    with patch(
        "knowledge_engine.src.domains.curriculum.targeted_node_grounding."
        "lazy_ground_deep_node_on_demand",
        search,
    ):
        await engine._apply_lazy_grounding_for_init(_req("autopilot"))
    search.assert_awaited_once()


def test_save_keeps_control_axis_when_not_passed(tmp_path, monkeypatch):
    """Lazy grounding / finalize гейта сохраняют курс без control_axis —
    режим, заданный при создании, не должен сбрасываться в autopilot."""
    monkeypatch.setattr(skill_tree_store, "_STORE_PATH", tmp_path / "store.json")
    graph = _graph()
    skill_tree_store.save_curriculum_record(
        graph, target_goal="g", control_axis="steering"
    )
    skill_tree_store.save_curriculum_record(graph, target_goal="g")
    assert skill_tree_store.get_curriculum_control_axis("cur_guard") == "steering"

    doc = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
    assert doc["curricula"][0]["control_axis"] == "steering"


def test_legacy_record_without_axis_is_autopilot(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_tree_store, "_STORE_PATH", tmp_path / "store.json")
    skill_tree_store.save_curriculum_record(_graph(), target_goal="g")
    assert skill_tree_store.get_curriculum_control_axis("cur_guard") == "autopilot"
