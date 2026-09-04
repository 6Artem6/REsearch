"""finalize_graph_chat_response: source_registry И mapped_source_ids
ТЕКУЩЕГО хода должны строиться по node_for_lecture (пере-обогащён в
run_dense_lecture_turn ПОСЛЕ persist_verified_external_sources_to_node), а
не по req.node_data (снимок, захваченный запросом ДО persist) — иначе:
- session.sourceRegistry на фронте (NodeDrawer.js читает его для панели
  источников лекции) остаётся пустым до перезагрузки страницы;
- selectedNode/curriculum.nodes.mapped_source_ids (NodeDrawer «Адресация
  ноды») тоже остаются пустыми, т.к. RoadmapDashboard.applyNodeResponse
  раньше не получал mapped_source_ids в ответе вообще и не мог их
  применить к selectedNode/curriculum без полного рефетча графа.
См. [[node_for_lecture]] фикс в run_dense_lecture_turn (та же гонка в
рамках хода, что и с текстом лекции, но для payload'а ответа)."""

from __future__ import annotations

import pytest

import knowledge_engine.src.node_deep_dive.engine as engine_mod
from knowledge_engine.src.node_deep_dive.engine import finalize_graph_chat_response
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import (
    NodeContentBlock,
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.node_deep_dive.session_store import _SessionRecord


def _node(mapped_source_ids: list[str]) -> NodeDataInput:
    return NodeDataInput(
        node_id="b_tree_indexes",
        title="B-Tree индексы",
        layer="foundation",
        category="indexes",
        brief_summary="Summary",
        core_concepts=["btree"],
        mapped_source_ids=mapped_source_ids,
    )


@pytest.mark.anyio
async def test_finalize_uses_node_for_lecture_registry_not_stale_request_node(
    monkeypatch,
):
    stale_node = _node([])  # как пришёл в req.node_data — ДО persist в этом ходе
    fresh_node = _node(["src_1", "src_2", "src_3"])  # пере-обогащён после persist

    req = NodeDeepDiveRequest(
        curriculum_id="indexes_and_data_structures",
        node_data=stale_node,
        user_action="chat",
        user_message="",
    )
    memory = SessionMemory()

    monkeypatch.setattr(
        engine_mod,
        "get_session",
        lambda _cid, _nid: _SessionRecord(history=[], memory=memory),
    )
    monkeypatch.setattr(engine_mod, "save_session", lambda *a, **kw: "session_key")
    monkeypatch.setattr(
        engine_mod, "get_all_sessions_for_curriculum", lambda _cid: {}
    )
    monkeypatch.setattr(
        engine_mod,
        "merge_mastery_from_session_memory",
        lambda *a, **kw: None,
    )

    captured: dict = {}

    def _fake_build_registry(curriculum_id, mapped_source_ids):
        captured["mapped_source_ids"] = list(mapped_source_ids)
        return []

    monkeypatch.setattr(
        engine_mod, "build_session_source_registry", _fake_build_registry
    )

    state = {
        "request": req,
        "memory": memory,
        "content": NodeContentBlock(),
        "tutor_message": "Ответ с материалом.",
        "node_for_lecture": fresh_node,
    }

    resp = await finalize_graph_chat_response(state)

    assert captured["mapped_source_ids"] == ["src_1", "src_2", "src_3"]
    assert resp.mapped_source_ids == ["src_1", "src_2", "src_3"]


@pytest.mark.anyio
async def test_finalize_falls_back_to_request_node_without_dense_lecture(
    monkeypatch,
):
    """Не-lecture ходы (chat без dense_lecture) не кладут node_for_lecture в
    state — должен использоваться req.node_data как раньше."""
    node = _node(["src_9"])
    req = NodeDeepDiveRequest(
        curriculum_id="indexes_and_data_structures",
        node_data=node,
        user_action="chat",
        user_message="",
    )
    memory = SessionMemory()

    monkeypatch.setattr(
        engine_mod,
        "get_session",
        lambda _cid, _nid: _SessionRecord(history=[], memory=memory),
    )
    monkeypatch.setattr(engine_mod, "save_session", lambda *a, **kw: "session_key")
    monkeypatch.setattr(
        engine_mod, "get_all_sessions_for_curriculum", lambda _cid: {}
    )
    monkeypatch.setattr(
        engine_mod, "merge_mastery_from_session_memory", lambda *a, **kw: None
    )

    captured: dict = {}

    def _fake_build_registry(curriculum_id, mapped_source_ids):
        captured["mapped_source_ids"] = list(mapped_source_ids)
        return []

    monkeypatch.setattr(
        engine_mod, "build_session_source_registry", _fake_build_registry
    )

    state = {
        "request": req,
        "memory": memory,
        "content": NodeContentBlock(),
        "tutor_message": "Ответ.",
    }

    resp = await finalize_graph_chat_response(state)

    assert captured["mapped_source_ids"] == ["src_9"]
    assert resp.mapped_source_ids == ["src_9"]
