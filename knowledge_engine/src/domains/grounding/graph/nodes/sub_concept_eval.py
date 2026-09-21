"""Sub-concept evaluator node (single-writer for sub_concepts coverage status)."""

from __future__ import annotations

import logging
from typing import Any

from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.grounding.concept_map import (
    process_sub_concept_user_answer,
    stored_pending_evaluation_id,
)
from knowledge_engine.src.domains.grounding.fsm import TutorStage
from knowledge_engine.src.domains.grounding.graph.stage_events import stage_scope
from knowledge_engine.src.domains.grounding.graph.state import TutorGraphState

logger = logging.getLogger(__name__)


def _with_memory(state: TutorGraphState, memory) -> TutorGraphState:
    return {
        **state,
        "memory": memory,
        "is_layer_just_completed": bool(
            getattr(memory, "is_layer_just_completed", False)
        ),
    }


def sub_concept_eval_node(
    state: TutorGraphState,
    config: dict[str, Any] | None = None,
) -> TutorGraphState:
    """FSM stage wrapper — см. graph/stage_events.py. "Оценка ответа" эмитится
    только вокруг реального вызова process_sub_concept_user_answer (см.
    _sub_concept_eval_node_impl), не вокруг всего узла: если оценивать
    нечего (пустое сообщение / нет pending / lecture request / quick-reply),
    узел молча пропускает шаг, и первым видимым статусом хода становится
    LLM_GENERATE ("Генерация нового сообщения…") на tutor_generate_node."""
    return _sub_concept_eval_node_impl(state, config)


def _sub_concept_eval_node_impl(
    state: TutorGraphState, config: dict[str, Any] | None = None
) -> TutorGraphState:
    """Gap eval for ``pending_evaluation_concept_id`` only; skip if no pending.

    Topic Q&A (``req.interaction_axis == "topic_qna"``) has NO axis-specific
    branch here on purpose (removed — see "self_check_node" task
    and docs/STEERING_AND_TOPIC_QNA_ROADMAP.md): the generic "no pending"
    skip below already covers plain Q&A, because TopicQnaLectureResponse /
    TopicQnaExplainContract structurally cannot ask a question, so ordinary
    answers never set ``pending_evaluation_concept_id`` in the first place
    (see ``commit_turn.py`` — it only binds pending when
    ``llm_out.follow_up_question`` is non-empty). Pending is set ONLY by the
    explicit ``[mode:self_check]`` turn (``TopicQnaTutorContract``/
    ``DeepDiveExplainContract`` is skipped there too, but for that specific
    turn ``engine.py`` swaps back to a question-bearing schema — see
    ``_invoke_tutor``'s ``factory_mode != "self_check"`` check). So when
    pending IS set, it is normally a genuine Self-Check answer awaiting
    grading, and must be evaluated normally — unconditionally skipping it
    (the old behavior) was exactly the bug: the learner had to switch to
    "Лекция" to ever get credit for a subtopic.

    Two exceptions where pending is stale rather than an answer attempt —
    both clear it via ``clear_pending_evaluation_state`` instead of grading
    the incoming message: (1) the session's ``interaction_axis`` differs
    from the one that set the pending target (``pending_evaluation_
    interaction_axis``) — e.g. the learner asked the Self-Check question
    while in ``lecture_self_check`` then switched to Topic Q&A; (2) the
    incoming message is itself a lecture/dense-material request — asking
    for a lecture is never an attempt to answer the pending question,
    even without an axis switch."""
    req = state["request"]
    memory = state["memory"]
    memory.evaluator_skipped = False
    user_message = (req.user_message or "").strip()
    if not user_message:
        logger.info("sub_concept_eval_node skip | empty user_message")
        from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
            mark_evaluator_skipped,
        )

        mark_evaluator_skipped(memory, "empty user_message")
        return _with_memory(state, memory)

    pending = stored_pending_evaluation_id(memory)
    if not pending:
        logger.warning(
            "sub_concept_eval_node skip | no pending target "
            "(asked=%r pending_field=%r) — tutor may run without credit",
            getattr(memory, "asked_question_sub_concept_id", ""),
            getattr(memory, "pending_evaluation_concept_id", ""),
        )
        from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
            mark_evaluator_skipped,
        )

        mark_evaluator_skipped(memory, "no pending (silent credit loss risk)")
        return _with_memory(state, memory)

    pending_axis = (memory.pending_evaluation_interaction_axis or "").strip().lower()
    current_axis = (req.interaction_axis or "").strip().lower()
    if pending_axis and current_axis and pending_axis != current_axis:
        # RU: самопроверка была задана в другом interaction_axis (напр.
        # lecture_self_check), а пользователь пришёл этим сообщением уже
        # после переключения (напр. на topic_qna) — pending протух, это
        # не попытка ответить на тот вопрос. Не сбрасывать здесь означало
        # бы оценить произвольное новое сообщение как ответ на самопроверку
        # (см. отчёт по Topic Q&A оценке/вопросу).
        logger.info(
            "sub_concept_eval_node skip | pending stale after axis switch "
            "(pending_axis=%r current_axis=%r)",
            pending_axis,
            current_axis,
        )
        from knowledge_engine.src.domains.grounding.concept_map_state import (
            clear_pending_evaluation_state,
        )
        from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
            mark_evaluator_skipped,
        )

        clear_pending_evaluation_state(memory)
        mark_evaluator_skipped(
            memory, f"interaction_axis switched ({pending_axis} → {current_axis})"
        )
        return _with_memory(state, memory)

    if current_axis == "topic_qna":
        from knowledge_engine.src.config.settings import (
            TOPIC_QNA_SELF_CHECK_MAX_ATTEMPTS,
        )
        from knowledge_engine.src.domains.grounding.concept_map import find_sub_concept

        row = find_sub_concept(memory, pending)
        if row is not None and int(row.failed_attempts or 0) >= (
            TOPIC_QNA_SELF_CHECK_MAX_ATTEMPTS
        ):
            # RU: Topic Q&A — свободная консультация, а не обязательный gate
            # мастерства. После N подряд незачтённых попыток по одной и той
            # же подтеме (пользователь раз за разом не отвечает на
            # поставленный вопрос — переспрашивает/уточняет/повторяет его)
            # дальнейшее переспрашивание того же вопроса даёт эффект
            # бесконечного цикла оценки вместо ответа (см. отчёт по
            # Topic Q&A). Сдаёмся по ЭТОЙ подтеме: pending сбрасывается,
            # статус подтемы (partial/gap) НЕ трогаем — зачёта не даём,
            # просто прекращаем зацикленный допрос. lecture_self_check
            # не затронут — там анти-геймерская логика (off-topic/refusal
            # тоже оценивается) остаётся как есть.
            logger.info(
                "sub_concept_eval_node skip | topic_qna gave up on concept=%s "
                "after failed_attempts=%s >= %s",
                pending,
                row.failed_attempts,
                TOPIC_QNA_SELF_CHECK_MAX_ATTEMPTS,
            )
            from knowledge_engine.src.domains.grounding.concept_map_state import (
                clear_pending_evaluation_state,
            )
            from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
                mark_evaluator_skipped,
            )

            clear_pending_evaluation_state(memory)
            mark_evaluator_skipped(
                memory,
                f"topic_qna gave up on concept={pending} after "
                f"{row.failed_attempts} failed attempts (not credited)",
            )
            return _with_memory(state, memory)

    from knowledge_engine.src.domains.grounding.lecture_scope import (
        is_lecture_request_message,
    )

    if is_lecture_request_message(user_message):
        logger.info("sub_concept_eval_node skip | lecture request")
        from knowledge_engine.src.domains.grounding.concept_map_state import (
            clear_pending_evaluation_state,
        )
        from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
            mark_evaluator_skipped,
        )

        clear_pending_evaluation_state(memory)
        mark_evaluator_skipped(memory, "lecture request (not a user answer)")
        return _with_memory(state, memory)

    from knowledge_engine.src.domains.grounding.concept_map import (
        is_quick_reply_control_message,
    )

    if is_quick_reply_control_message(user_message, memory):
        logger.info(
            "sub_concept_eval_node skip | quick-reply control chip "
            "(not a scored answer)"
        )
        from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
            mark_evaluator_skipped,
        )

        mark_evaluator_skipped(
            memory, "quick-reply control (Gloss / Дожать / next — evaluator off)"
        )
        return _with_memory(state, memory)

    try:
        from knowledge_engine.src.domains.grounding.node_session_reset import (
            node_deep_dive_anchor,
        )

        anchor = state.get("anchor") or node_deep_dive_anchor(
            req.curriculum_id, req.node_data.node_id
        )
        with stage_scope(
            state,
            config,
            TutorStage.INTENT_ANALYSIS,
            running_message="Оценка ответа…",
        ):
            process_sub_concept_user_answer(
                user_message,
                memory,
                req.node_data,
                anchor,
            )
    except Exception as exc:
        logger.exception("sub_concept_eval_node FAILED pending=%s", pending)
        trace(f"EVALUATOR_ERROR | pipeline | {type(exc).__name__}: {exc}")
        trace(f"NODE_DIVE sub_concept evaluation FAILED | {type(exc).__name__}: {exc}")
        # Last-resort: do not leave unchecked without a mark.
        try:
            from knowledge_engine.src.domains.grounding.concept_map import (
                find_sub_concept,
            )
            from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
                apply_degraded_threshold,
            )

            row = find_sub_concept(memory, pending)
            if row is not None and row.status == "unchecked":
                layer = str(getattr(req.node_data, "layer", "") or "foundation")
                directive = apply_degraded_threshold(
                    row,
                    layer=layer,
                    reason=f"node_exc:{type(exc).__name__}",
                    memory=memory,
                )
                memory.last_eval_directive = directive
                from knowledge_engine.src.domains.grounding.concept_map_state import (
                    build_evaluator_feedback,
                )

                memory.last_evaluator_feedback = build_evaluator_feedback(row)
        except Exception as inner:
            logger.exception("degraded threshold also failed: %s", inner)
    return _with_memory(state, memory)
