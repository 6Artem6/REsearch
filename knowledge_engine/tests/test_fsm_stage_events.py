"""FSM stage-progress events (schemas/fsm.py + graph/stage_events.py) — см.
prompt.txt: "Интеграция FSM-статусов LangGraph c SSE-стримингом"."""

from __future__ import annotations

import pytest

from knowledge_engine.schemas.fsm import FSMStatus, StageProgressEvent, TutorStage
from knowledge_engine.src.node_deep_dive.graph.stage_events import (
    emit_stage,
    stage_scope,
)


class _FakeNode:
    node_id = "n1"


class _FakeReq:
    curriculum_id = "c1"
    node_data = _FakeNode()


def test_to_sse_dict_has_type_discriminator():
    event = StageProgressEvent(
        session_id="c1/n1",
        stage=TutorStage.LLM_GENERATE,
        status=FSMStatus.RUNNING,
        message="Генерируем ответ…",
        elapsed_sec=1.23,
    )
    d = event.to_sse_dict()
    assert d["type"] == "stage"
    assert d["stage"] == "LLM_GENERATE"
    assert d["status"] == "RUNNING"
    assert d["session_id"] == "c1/n1"
    assert "timestamp" in d


def test_emit_stage_noop_without_callback_in_config():
    # Не должно падать — просто ничего не делает, если стрима нет вообще
    # (обычный non-streaming init/chat запрос).
    emit_stage({"request": _FakeReq()}, None, TutorStage.INIT, FSMStatus.RUNNING, "x")
    emit_stage(
        {"request": _FakeReq()},
        {"configurable": {}},
        TutorStage.INIT,
        FSMStatus.RUNNING,
        "x",
    )


def test_emit_stage_calls_callback_with_correct_fields():
    received: list[StageProgressEvent] = []
    config = {
        "configurable": {"stage_callback": received.append, "turn_started_at": 0.0}
    }
    emit_stage(
        {"request": _FakeReq()},
        config,
        TutorStage.VECTOR_SEARCH,
        FSMStatus.COMPLETED,
        "Контекст найден",
        extra="value",
    )
    assert len(received) == 1
    event = received[0]
    assert event.session_id == "c1/n1"
    assert event.stage == TutorStage.VECTOR_SEARCH
    assert event.status == FSMStatus.COMPLETED
    assert event.payload == {"extra": "value"}
    assert event.elapsed_sec >= 0


def test_emit_stage_callback_exception_does_not_propagate():
    def _boom(_event):
        raise RuntimeError("frontend disconnected")

    config = {"configurable": {"stage_callback": _boom}}
    # Не должно поднять исключение наружу — best-effort эмиссия.
    emit_stage({"request": _FakeReq()}, config, TutorStage.INIT, FSMStatus.RUNNING, "x")


def test_stage_scope_emits_running_then_completed_on_success():
    received: list[StageProgressEvent] = []
    config = {"configurable": {"stage_callback": received.append}}
    with stage_scope(
        {"request": _FakeReq()},
        config,
        TutorStage.FINALIZE,
        running_message="Завершаем…",
    ):
        pass
    assert [e.status for e in received] == [FSMStatus.RUNNING, FSMStatus.COMPLETED]


def test_stage_scope_emits_running_then_failed_and_reraises():
    received: list[StageProgressEvent] = []
    config = {"configurable": {"stage_callback": received.append}}
    with pytest.raises(ValueError, match="boom"):
        with stage_scope(
            {"request": _FakeReq()},
            config,
            TutorStage.LLM_GENERATE,
            running_message="Генерируем…",
        ):
            raise ValueError("boom")
    assert [e.status for e in received] == [FSMStatus.RUNNING, FSMStatus.FAILED]
    assert "ValueError" in (received[1].payload or {}).get("error", "")


def test_stage_scope_survives_early_return_inside_with_block():
    """Узлы графа (напр. sub_concept_eval_node) имеют несколько early return
    внутри with-блока — контекст-менеджер должен корректно эмитить COMPLETED
    и в этом случае, не только на "естественном" конце функции."""
    received: list[StageProgressEvent] = []
    config = {"configurable": {"stage_callback": received.append}}

    def _node():
        with stage_scope(
            {"request": _FakeReq()},
            config,
            TutorStage.INTENT_ANALYSIS,
            running_message="x",
        ):
            if True:
                return "early"
            return "late"  # pragma: no cover

    result = _node()
    assert result == "early"
    assert [e.status for e in received] == [FSMStatus.RUNNING, FSMStatus.COMPLETED]
