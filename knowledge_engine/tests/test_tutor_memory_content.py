"""Tutor memory window content vs UI tutor_message."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.src.node_deep_dive.tutor_memory_content import (
    tutor_content_for_active_window,
)


def test_tutor_content_for_active_window_uses_semantic_fields():
    out = DeepDiveLLMOutput(
        feedback_on_answer="В ответе не хватило кворума.",
        technical_explanation="При failover split-brain снижается через sticky routing.",
        follow_up_question="Какой QPS ты держишь при failover?",
    )
    body = tutor_content_for_active_window(out)
    assert "кворума" in body
    assert "split-brain" in body
    assert "?" not in body
    assert "Какой QPS" not in body


def test_tutor_content_fallback_sanitizes_questions():
    out = DeepDiveLLMOutput(
        technical_explanation="",
        follow_up_question="Что думаешь о latency?",
    )
    body = tutor_content_for_active_window(
        out,
        fallback_compose_text="Разбор. Что думаешь о latency?",
    )
    assert "Разбор" in body
    assert "?" not in body


def test_compose_tutor_message_joins_fields():
    out = DeepDiveLLMOutput(
        feedback_on_answer="A",
        technical_explanation="B",
        follow_up_question="C?",
    )
    assert out.compose_tutor_message() == "A\n\nB\n\nC?"


def test_sync_history_uses_window_msg_id_with_full_tutor_text():
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        dialog_message,
        sync_session_history_turns,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

    mem = SessionMemory()
    mem.active_window = [
        dialog_message("user", "ответ", 1),
        dialog_message("tutor", "только разбор без вопроса", 2),
    ]
    mem.dialog_seq = 2
    hist = sync_session_history_turns(
        [],
        mem,
        tutor_message="полный UI текст с вопросом?",
    )
    tutor_rows = [h for h in hist if h.get("role") == "tutor"]
    assert len(tutor_rows) == 1
    assert tutor_rows[0]["content"] == "полный UI текст с вопросом?"
    assert tutor_rows[0]["msg_id"] == "2"


def test_reconcile_keeps_full_tutor_history_over_window_without_follow_up():
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        dialog_message,
        reconcile_dialog_history,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
    from knowledge_engine.src.node_deep_dive.session_store import (
        repair_history_with_memory,
    )

    full = "Разбор трассировки.\n\n" "Как ты организуешь персистентное хранение шагов?"
    short = "Разбор трассировки."
    hist = [
        dialog_message("user", "ответ", 1),
        dialog_message("tutor", full, 2),
    ]
    mem = SessionMemory()
    mem.active_window = [
        dialog_message("user", "ответ", 1),
        dialog_message("tutor", short, 2),
    ]
    mem.dialog_seq = 2
    merged, _ = reconcile_dialog_history(hist, mem.active_window)
    tutor = [m for m in merged if m.get("role") == "tutor"][0]
    assert "персистентное" in tutor["content"]
    repaired = repair_history_with_memory(hist, mem)
    tutor2 = [m for m in repaired if m.get("role") == "tutor"][0]
    assert "персистентное" in tutor2["content"]


def test_repair_history_keeps_repeated_identical_user_text():
    """Repeated identical answers are distinct turns — keep both msg_ids."""
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        dialog_message,
        parse_msg_id,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
    from knowledge_engine.src.node_deep_dive.session_store import (
        repair_history_with_memory,
    )

    text = "Финальный ответ по WAL и recovery."
    hist = [
        dialog_message("user", "первый", 1),
        dialog_message("tutor", "вопрос?", 2),
        dialog_message("user", text, 3),
        dialog_message("tutor", "итог ноды", 4),
        dialog_message("user", text, 5),
        dialog_message("tutor", "ещё раз", 6),
    ]
    mem = SessionMemory()
    mem.active_window = hist[-4:]
    mem.dialog_seq = 6
    repaired = repair_history_with_memory(hist, mem)
    assert [parse_msg_id(m) for m in repaired] == [1, 2, 3, 4, 5, 6]
    users = [m for m in repaired if m.get("role") == "user"]
    assert len(users) == 3


def test_repair_history_keeps_both_lecture_requests():
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        dialog_message,
        parse_msg_id,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
    from knowledge_engine.src.node_deep_dive.session_store import (
        repair_history_with_memory,
    )

    lecture = "[mode:lecture] Дай плотный материал по теме."
    hist = [
        dialog_message("user", lecture, 1),
        dialog_message("tutor", "первая лекция", 2),
        dialog_message("user", "ответ по теме", 3),
        dialog_message("tutor", "следующий вопрос?", 4),
        dialog_message("user", lecture, 5),
        dialog_message("tutor", "вторая лекция без зачёта", 6),
    ]
    mem = SessionMemory()
    mem.active_window = hist[-4:]
    mem.dialog_seq = 6
    repaired = repair_history_with_memory(hist, mem)
    assert [parse_msg_id(m) for m in repaired] == [1, 2, 3, 4, 5, 6]
    lecture_rows = [
        m
        for m in repaired
        if m.get("role") == "user" and "плотный материал" in (m.get("content") or "")
    ]
    assert len(lecture_rows) == 2
    assert repaired[-2]["msg_id"] == "5"
    assert repaired[-1]["role"] == "tutor"


def test_recover_tutor_display_from_chat_sessions():
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
    from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
        recover_tutor_display_from_chat_sessions,
    )

    mem = SessionMemory()
    mem.chat_sessions = {
        "node_deep_dive/tutor": {
            "api_turns": [
                {
                    "role": "model",
                    "content": (
                        '{"feedback_on_answer":"A","technical_explanation":"B",'
                        '"follow_up_question":"C?"}'
                    ),
                }
            ]
        }
    }
    display, fu = recover_tutor_display_from_chat_sessions(mem)
    assert "A" in display and "B" in display and "C?" in display
    assert fu == "C?"
