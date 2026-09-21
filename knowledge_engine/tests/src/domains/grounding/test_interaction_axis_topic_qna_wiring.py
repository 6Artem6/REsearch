"""Interaction Axis (Topic Q&A) подключён к живому пути тьютора (см.

docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Interaction Axis подключён").
Раньше ``select_interaction_axis_system_prompt`` существовал, но не
вызывался НИОТКУДА из ``_invoke_tutor``/``coverage_router_node`` — выбор
"Topic Q&A" в UI был чисто визуальным (``interactionAxis`` в session state
никогда не попадал в тело запроса). Эти тесты проверяют саму диспетчеризацию
(``select_interaction_axis_system_prompt``) и её реальное подключение в
``generate_dense_material`` (services/node_content_generator.py) — система
теперь получает ``TOPIC_QNA_SYSTEM_PROMPT`` поверх обычного dense-system
именно когда ``interaction_axis="topic_qna"``, и НЕ получает его для
дефолтного ``"lecture_self_check"``."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from knowledge_engine.src.domains.grounding.prompt_factory import (
    select_interaction_axis_system_prompt,
)
from knowledge_engine.src.domains.grounding.schemas import NodeDeepDiveRequest


def test_select_interaction_axis_system_prompt_topic_qna() -> None:
    from knowledge_engine.src.domains.steering.prompts.topic_qna_prompt import (
        TOPIC_QNA_SYSTEM_PROMPT,
    )

    assert select_interaction_axis_system_prompt("topic_qna") == TOPIC_QNA_SYSTEM_PROMPT


def test_topic_qna_uses_a_schema_with_no_follow_up_field_at_all() -> None:
    """Regression, escalated twice: a text override telling the model to

    leave the closing-question field empty was ignored (it referenced the
    wrong field name at first — the lecture contract used to call this
    field `checkpoint_prompt`, later unified to `follow_up_question` across
    both dialogue and lecture contracts — then the right one). Softening
    the Field(description=...) was ALSO ignored live (confirmed on a
    brand-new node, sql_cte — the field came back filled despite a clean
    system prompt and a fresh, uncontaminated chat session). Both attempts
    relied on the model choosing to obey a conditional instruction for an
    invariant that was already known at request time. The actual fix:
    TopicQnaLectureResponse has NO follow_up_question field in its JSON
    Schema — Gemini structured output cannot write to a slot that isn't
    there, independent of prompt wording. StructuredLectureResponse
    (lecture_self_check) is untouched and still has the field,
    mandatory-by-convention as before."""
    from knowledge_engine.src.domains.grounding.tutor import (
        StructuredLectureResponse,
        TopicQnaLectureResponse,
    )
    from knowledge_engine.src.domains.steering.prompts.topic_qna_prompt import (
        TOPIC_QNA_SYSTEM_PROMPT,
    )

    assert "follow_up_question" not in TopicQnaLectureResponse.model_fields
    assert "follow_up_question" in StructuredLectureResponse.model_fields
    shared = set(TopicQnaLectureResponse.model_fields) - {"follow_up_question"}
    assert shared == set(StructuredLectureResponse.model_fields) - {"follow_up_question"}
    assert "follow_up_question" not in TOPIC_QNA_SYSTEM_PROMPT


def test_generate_dense_material_picks_schema_by_interaction_axis(monkeypatch) -> None:
    """generate_dense_material must select response_schema by axis, not

    just compose a matching system prompt — otherwise Gemini is still
    given the field-bearing StructuredLectureResponse schema and the
    schema split above buys nothing."""
    # Explicit, not relied-on-default: this must hold regardless of the real
    # .env's ENABLE_TUTOR_EXECUTION_PLAN (dev machines may have it on).
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", False
    )
    from knowledge_engine.src.domains.grounding import node_content_generator as svc
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
    from knowledge_engine.src.domains.grounding.tutor import (
        StructuredLectureResponse,
        TopicQnaLectureResponse,
    )

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["c"]
    )
    memory = SessionMemory()

    def _run_once(interaction_axis: str) -> type:
        captured: list[type] = []

        def _capture_and_stop(_model, _system, _payload, _anchor, schema, *_a, **_kw):
            captured.append(schema)
            raise _StopEarly()

        with (
            patch.object(
                svc, "run_gemini_structured_with_chain", side_effect=_capture_and_stop
            ),
            patch.object(svc, "run_local_structured", side_effect=_StopEarly),
        ):
            with pytest.raises(_StopEarly):
                svc.generate_dense_material(
                    node, memory, "", "anchor-1", interaction_axis=interaction_axis
                )
        assert captured
        return captured[0]

    assert _run_once("topic_qna") is TopicQnaLectureResponse
    assert _run_once("lecture_self_check") is StructuredLectureResponse


def test_topic_qna_chat_turn_also_uses_a_schema_with_no_follow_up_field() -> None:
    """The dense_material ("Дай плотный материал по теме") schema split

    fixed only ONE of the two paths a Topic Q&A session actually takes —
    confirmed live on the sql_cte node: free-form chat
    turns (the user just asking questions, not requesting a lecture) kept
    ending in a technical question anyway. Root cause: engine.py's
    _invoke_tutor routes evaluator_skipped turns to DeepDiveExplainContract,
    whose follow_up_question is "Optional... if set" with NO axis
    condition at all — the same class of bug, in a second, previously
    unexamined schema. Fix: TopicQnaExplainContract (no follow_up_question
    /question_sub_concept_id field) is used whenever
    interaction_axis="topic_qna", mirroring TopicQnaLectureResponse."""
    from knowledge_engine.src.domains.grounding.tutor import (
        DeepDiveExplainContract,
        TopicQnaExplainContract,
    )

    assert "follow_up_question" not in TopicQnaExplainContract.model_fields
    assert "question_sub_concept_id" not in TopicQnaExplainContract.model_fields
    assert "follow_up_question" in DeepDiveExplainContract.model_fields
    shared = set(TopicQnaExplainContract.model_fields)
    assert shared == set(DeepDiveExplainContract.model_fields) - {
        "follow_up_question",
        "question_sub_concept_id",
    }


def test_select_interaction_axis_system_prompt_lecture_self_check_is_none() -> None:
    assert select_interaction_axis_system_prompt("lecture_self_check") is None


def test_select_interaction_axis_system_prompt_unknown_is_none() -> None:
    assert select_interaction_axis_system_prompt("") is None
    assert select_interaction_axis_system_prompt("bogus") is None


def test_node_deep_dive_request_defaults_to_lecture_self_check() -> None:
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data={
            "node_id": "n1",
            "title": "Test",
            "layer": "foundation",
            "core_concepts": ["c"],
        },
        user_action="chat",
    )
    assert req.interaction_axis == "lecture_self_check"


class _StopEarly(Exception):
    """Прерывает generate_dense_material сразу после сборки system prompt —

    не даёт дойти до реального сетевого вызова LLM. Само значение system
    захватывается в mutable-список ДО raise (см. _capture_system_and_stop) —
    generate_dense_material ловит любое исключение из primary-вызова и
    откатывается на run_local_structured (иная сигнатура, system другим
    позиционным аргументом), поэтому не полагаемся на текст исключения."""


def test_generate_dense_material_appends_topic_qna_prompt() -> None:
    """system, переданный в LLM-вызов, содержит TOPIC_QNA_SYSTEM_PROMPT

    когда interaction_axis="topic_qna" — и НЕ содержит для дефолта."""
    from knowledge_engine.src.domains.grounding import node_content_generator as svc
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
    from knowledge_engine.src.domains.steering.prompts.topic_qna_prompt import (
        TOPIC_QNA_SYSTEM_PROMPT,
    )

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["c"]
    )
    memory = SessionMemory()

    def _run_once(interaction_axis: str) -> str:
        captured: list[str] = []

        def _capture_and_stop(_model, system, *_a, **_kw):
            captured.append(system)
            raise _StopEarly()

        with (
            patch.object(
                svc, "run_gemini_structured_with_chain", side_effect=_capture_and_stop
            ),
            # Фоллбэк на локальный LLM тоже должен молчать (иначе тест
            # реально ждёт local model десятки секунд) — сигнатура другая
            # (system третьим позиционным), но нам важно только НЕ дать ей
            # выполниться по-настоящему.
            patch.object(svc, "run_local_structured", side_effect=_StopEarly),
        ):
            with pytest.raises(_StopEarly):
                svc.generate_dense_material(
                    node, memory, "", "anchor-1", interaction_axis=interaction_axis
                )
        assert captured, "run_gemini_structured_with_chain не был вызван"
        return captured[0]

    system_topic = _run_once("topic_qna")
    system_lecture = _run_once("lecture_self_check")

    assert TOPIC_QNA_SYSTEM_PROMPT.strip() in system_topic
    assert TOPIC_QNA_SYSTEM_PROMPT.strip() not in system_lecture


def test_build_dense_system_strips_all_mandatory_checkpoint_directives_for_topic_qna() -> (
    None
):
    """Regression: TOPIC_QNA_SYSTEM_PROMPT is appended as a single override

    on top of build_dense_system's output (see generate_dense_material) —
    but the base dense-system prompt independently baked "mandatory
    follow_up_question" directives into FIVE separate fragments
    (LECTURE_MODE_STRUCTURE_RULES / STRUCTURED_LECTURE_FIELD_RULES rule 8 /
    NO_CLOSING_QUESTIONNAIRES / DENSE_FUNDAMENTALS_BLOCK /
    LECTURE_GAP_STEERING_RULES / DENSE_LECTURE_INTERACTION_MODE — found one
    at a time via live testing). A single appended
    "override" sentence does not reliably beat several independently-worded
    MANDATORY/MUST directives baked into the base prompt itself — the fix is
    to never generate the conflicting fragment for topic_qna in the first
    place. This test locks in that build_dense_system(interaction_axis=
    "topic_qna") itself is clean, and that the default variant is
    byte-identical to before (no accidental behavior change for the
    unaffected lecture_self_check axis)."""
    from knowledge_engine.src.domains.grounding.tutor_prompt_builder import (
        build_dense_system,
    )

    topic_qna_prompt = build_dense_system(interaction_axis="topic_qna")
    default_prompt = build_dense_system(interaction_axis="lecture_self_check")

    mandatory_checkpoint_phrases = [
        "MANDATORY CLOSING QUESTION",
        "PART 2 is mandatory",
        "Every scored criterion MUST be named here",
        "NEVER omit",
        "the ONLY JSON field for the one technical question",
        "exactly ONE technical question EXCLUSIVELY",
        "CHECKPOINT ALIGNMENT: The question in follow_up_question MUST directly",
        "MUST steer lecture depth and follow_up_question",
    ]
    for phrase in mandatory_checkpoint_phrases:
        assert (
            phrase not in topic_qna_prompt
        ), f"leaked mandatory-checkpoint directive into topic_qna prompt: {phrase!r}"
        assert (
            phrase in default_prompt
        ), f"default lecture_self_check prompt lost expected directive: {phrase!r}"


def test_sub_concept_eval_node_evaluates_a_pending_self_check_answer_in_topic_qna() -> (
    None
):
    """Regression: the earlier unconditional topic_qna skip in

    sub_concept_eval.py was itself the bug "self_check_node"
    task reported — "чтобы зачесть подтему, пользователю приходится
    вручную переключать селектор на Лекцию". With TopicQnaLectureResponse /
    TopicQnaExplainContract structurally unable to ask a question, ordinary
    Q&A never sets pending_evaluation_concept_id any more (see
    commit_turn.py — it only binds pending when llm_out.follow_up_question
    is non-empty), so whenever pending IS set in a topic_qna session, it
    is genuinely a Self-Check answer awaiting grading, and must be
    evaluated exactly like lecture_self_check — never skipped just because
    of the axis."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["sc1"]
    )
    memory = SessionMemory()
    memory.sub_concepts = [SubConceptRecord(id="sc1", label="Sub Concept 1")]
    memory.pending_evaluation_concept_id = "sc1"
    memory.asked_question_sub_concept_id = "sc1"
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="Материализация кэширует промежуточный набор строк.",
        interaction_axis="topic_qna",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is False


def test_sub_concept_eval_node_still_skips_topic_qna_with_no_pending() -> None:
    """Control: plain Q&A in topic_qna (no Self-Check ever asked) has

    nothing to grade — same generic "no pending" skip as any other axis,
    proving the fix above is not a blanket "always evaluate topic_qna"
    regression."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["c"]
    )
    memory = SessionMemory()
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="What does this actually mean in practice?",
        interaction_axis="topic_qna",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is True


def test_sub_concept_eval_node_resets_stale_pending_after_axis_switch() -> None:
    """Regression: Self-Check was asked in lecture_self_check (pending +

    pending_evaluation_interaction_axis="lecture_self_check" bound), the
    learner switched the session to Topic Q&A without answering it, and
    asked a brand-new question instead. Grading that unrelated message as
    the Self-Check answer produced exactly the reported symptom — an
    evaluation plaque + a fresh follow_up_question in what the learner
    experiences as plain Q&A. The axis mismatch must clear pending and
    skip evaluation instead."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["sc1"]
    )
    memory = SessionMemory()
    memory.sub_concepts = [SubConceptRecord(id="sc1", label="Sub Concept 1")]
    memory.pending_evaluation_concept_id = "sc1"
    memory.asked_question_sub_concept_id = "sc1"
    memory.pending_evaluation_interaction_axis = "lecture_self_check"
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="А как вообще работает партиционирование в этой БД?",
        interaction_axis="topic_qna",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is True
    assert out["memory"].pending_evaluation_concept_id == ""
    assert out["memory"].pending_evaluation_interaction_axis == ""


def test_sub_concept_eval_node_resets_pending_on_lecture_request() -> None:
    """Regression: a lecture/dense-material request right after Self-Check

    must drop the pending target, not just skip grading for this one turn —
    otherwise the NEXT ordinary message (still seeing pending set) gets
    graded as the abandoned Self-Check answer."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["sc1"]
    )
    memory = SessionMemory()
    memory.sub_concepts = [SubConceptRecord(id="sc1", label="Sub Concept 1")]
    memory.pending_evaluation_concept_id = "sc1"
    memory.asked_question_sub_concept_id = "sc1"
    memory.pending_evaluation_interaction_axis = "lecture_self_check"
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="[mode:lecture] Дай плотный материал по теме.",
        interaction_axis="lecture_self_check",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is True
    assert out["memory"].pending_evaluation_concept_id == ""
    assert out["memory"].pending_evaluation_interaction_axis == ""


def test_sub_concept_eval_node_gives_up_in_topic_qna_after_max_failed_attempts() -> (
    None
):
    """Regression: a live session (clickhouse_mergetree_f32db532a801/

    clickhouse) got stuck re-probing the same sub_concept in Topic Q&A —
    failed_attempts reached 5, each turn the learner asked a clarifying
    counter-question instead of answering, and each one was graded as a
    wrong attempt and re-asked (GAP_EVAL_SYSTEM deliberately scores
    off-topic/refusal as wrong, by design, to stop gaming the mandatory
    lecture_self_check path). Topic Q&A is a free consultation, not a
    mastery gate, so after TOPIC_QNA_SELF_CHECK_MAX_ATTEMPTS consecutive
    fails it must give up on that sub_concept (clear pending, stop
    re-probing) WITHOUT crediting it — status stays whatever it already
    was (never forced to verified)."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["sc1"]
    )
    memory = SessionMemory()
    memory.sub_concepts = [
        SubConceptRecord(
            id="sc1", label="Sub Concept 1", status="partial", failed_attempts=3
        )
    ]
    memory.pending_evaluation_concept_id = "sc1"
    memory.asked_question_sub_concept_id = "sc1"
    memory.pending_evaluation_interaction_axis = "topic_qna"
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="А как вообще работает партиционирование в этой БД?",
        interaction_axis="topic_qna",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is True
    assert out["memory"].pending_evaluation_concept_id == ""
    assert out["memory"].sub_concepts[0].status == "partial"
    assert out["memory"].sub_concepts[0].failed_attempts == 3


def test_sub_concept_eval_node_still_evaluates_topic_qna_below_attempt_ceiling() -> (
    None
):
    """Control: below the attempt ceiling, Topic Q&A Self-Check retries

    still evaluate normally — the circuit breaker only trips once the
    threshold is actually reached, it is not a blanket topic_qna skip."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import (
        SessionMemory,
        SubConceptRecord,
    )
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["sc1"]
    )
    memory = SessionMemory()
    memory.sub_concepts = [
        SubConceptRecord(
            id="sc1", label="Sub Concept 1", status="partial", failed_attempts=2
        )
    ]
    memory.pending_evaluation_concept_id = "sc1"
    memory.asked_question_sub_concept_id = "sc1"
    memory.pending_evaluation_interaction_axis = "topic_qna"
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="Материализация кэширует промежуточный набор строк.",
        interaction_axis="topic_qna",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is False


def test_dialogue_turn_also_uses_schemas_with_no_follow_up_field_in_topic_qna() -> None:
    """Confirmed live on the sql_cte node: free-form chat

    turns (not "Дай плотный материал по теме") kept ending in a technical
    question too. Root cause: engine.py._invoke_tutor routes
    evaluator_skipped turns to DeepDiveExplainContract (Optional follow_up
    with no axis condition) and non-skipped (audited/graded) turns to
    DeepDiveTutorContract (same issue) — a second and third instance of
    the checkpoint_prompt bug class, in schemas dense_material never
    touches. Fix: TopicQnaExplainContract / TopicQnaTutorContract, same
    split pattern as TopicQnaLectureResponse, selected whenever
    interaction_axis="topic_qna" (except the one turn that must ask a
    question: [mode:self_check] itself — see _invoke_tutor)."""
    from knowledge_engine.src.domains.grounding.tutor import (
        DeepDiveExplainContract,
        DeepDiveTutorContract,
        TopicQnaExplainContract,
        TopicQnaTutorContract,
    )

    assert "follow_up_question" not in TopicQnaExplainContract.model_fields
    assert "follow_up_question" not in TopicQnaTutorContract.model_fields
    assert "follow_up_question" in DeepDiveExplainContract.model_fields
    assert "follow_up_question" in DeepDiveTutorContract.model_fields
    # audit/feedback_on_answer — shared, unaffected by the split.
    assert "audit" in TopicQnaTutorContract.model_fields
    assert "ready_for_transition" in TopicQnaTutorContract.model_fields


def test_resolve_tutor_response_schema_keeps_asking_when_self_check_not_credited(
    monkeypatch,
) -> None:
    """Regression: the first version of the DeepDiveTutorContract split used

    TopicQnaTutorContract (no follow_up_question) for EVERY evaluated
    topic_qna turn, including a Self-Check answer that was NOT credited —
    confirmed live: instead of a follow-up question giving the learner
    another attempt, the model was left improvising an "уточните" text
    inside technical_explanation, which explicitly forbids «?». Only a
    TERMINAL eval directive (subtopic actually passed) should drop the
    question field; "PROBE_NEXT_LAYER:*" (same layer still open) must keep
    DeepDiveTutorContract so the Self-Check loop can continue."""
    # Explicit, not relied-on-default: this must hold regardless of the real
    # .env's ENABLE_TUTOR_EXECUTION_PLAN (dev machines may have it on).
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", False
    )
    from knowledge_engine.src.domains.grounding.engine import (
        _resolve_tutor_response_schema,
    )
    from knowledge_engine.src.domains.grounding.tutor import (
        DeepDiveTutorContract,
        TopicQnaTutorContract,
    )

    not_credited = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="PROBE_NEXT_LAYER:HOW",
    )
    assert not_credited is DeepDiveTutorContract

    credited = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="PASSED_CLEAN",
    )
    assert credited is TopicQnaTutorContract

    deep_mastery = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="DEEP_MASTERY_EARNED",
    )
    assert deep_mastery is TopicQnaTutorContract

    # lecture_self_check is entirely unaffected by the directive check.
    lecture_axis = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="lecture_self_check",
        last_eval_directive="PASSED_CLEAN",
    )
    assert lecture_axis is DeepDiveTutorContract


def test_resolve_tutor_response_schema_self_check_trigger_and_skip_reasons() -> None:
    """The [mode:self_check] turn itself (asking the check question) and

    other evaluator-skip reasons (empty message, explicit lecture request,
    no pending) must still get a question-bearing schema — only plain
    topic_qna Q&A gets the field-less TopicQnaExplainContract."""
    from knowledge_engine.src.domains.grounding.engine import (
        _resolve_tutor_response_schema,
    )
    from knowledge_engine.src.domains.grounding.tutor import (
        DeepDiveExplainContract,
        TopicQnaExplainContract,
    )

    self_check_turn = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=True,
        factory_mode="self_check",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="",
    )
    assert self_check_turn is DeepDiveExplainContract

    plain_qna = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=True,
        factory_mode="",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="",
    )
    assert plain_qna is TopicQnaExplainContract

    lecture_skip = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=True,
        factory_mode="",
        drill_schema=None,
        interaction_axis="lecture_self_check",
        last_eval_directive="",
    )
    assert lecture_skip is DeepDiveExplainContract


def test_resolve_tutor_response_schema_keeps_question_while_self_check_pending() -> (
    None
):
    """Regression: pressing "Переформулируй вопрос" (intent "clarify")

    while a Self-Check question was awaiting an answer was correctly
    recognized (confirmed via worker trace: chip=clarify), but the reply
    had no question at all — confirmed live. Root cause:
    "clarify" is classified by is_quick_reply_control_message as a UI
    control chip, so sub_concept_eval_node skips it (not a scored answer)
    WITHOUT clearing pending_evaluation_concept_id — the self-check is
    still open. _invoke_tutor only kept the question-bearing schema for
    the literal [mode:self_check] trigger turn, so any OTHER control chip
    hit while a Self-Check is pending (clarify, or any future one) fell
    through to the field-less TopicQnaExplainContract with nowhere to put
    a rephrased question. has_pending_self_check covers every such chip
    generically, not just "clarify" by name — consistent with the
    "no hardcoded per-intent text matching" constraint."""
    from knowledge_engine.src.domains.grounding.engine import (
        _resolve_tutor_response_schema,
    )
    from knowledge_engine.src.domains.grounding.tutor import DeepDiveExplainContract

    clarify_while_pending = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=True,
        factory_mode="",
        drill_schema=None,
        interaction_axis="topic_qna",
        last_eval_directive="",
        has_pending_self_check=True,
    )
    assert clarify_while_pending is DeepDiveExplainContract


def test_resolve_for_model_starts_new_session_on_interaction_axis_switch() -> None:
    """Regression: even with a fully-clean topic_qna system prompt, the

    model kept ending Topic Q&A responses with a checkpoint question
    (confirmed live against the sql_cte_group_by node) —
    root cause was that ChatSessionManager.resolve_for_model only compared
    model_name, so switching interaction_axis for the same chat label
    (e.g. "node_deep_dive/dense_material") reused the OLD StoredChatSession
    and replayed its accumulated api_turns — raw JSON responses from
    earlier lecture_self_check turns, each ending in a filled
    follow_up_question — as conversation history into the new Gemini chat.
    The model then imitated its own established pattern regardless of the
    current system_instruction. The fix: an interaction_axis mismatch must
    start a new session (Summary handoff, same as a model-name mismatch),
    exactly like a model switch does."""
    from knowledge_engine.src.domains.grounding.chat_session_manager import (
        ChatSessionManager,
    )

    mgr = ChatSessionManager("test-scope")
    label = "node_deep_dive/dense_material"
    model = "gemini-3.5-flash-lite"

    s1 = mgr.resolve_for_model(label, model, "", interaction_axis="lecture_self_check")
    mgr.record_turn(
        label,
        "Дай плотный материал по теме.",
        '{"follow_up_question": "Как дефицит work_mem провоцирует spill-to-disk?"}',
    )
    s2 = mgr.get(label)
    assert s2 is not None and s2.session_id == s1.session_id
    assert len(s2.api_turns) == 2  # прежний паттерн реально накопился в истории

    # Тот же label, ось не менялась — история переиспользуется как раньше.
    s3 = mgr.resolve_for_model(
        label, model, "handoff", interaction_axis="lecture_self_check"
    )
    assert s3.session_id == s2.session_id
    assert len(s3.api_turns) == 2

    # Переключение оси на topic_qna для ТОГО ЖЕ label — новая сессия, старые
    # api_turns (с follow_up_question-вопросами) не реплеятся как history.
    s4 = mgr.resolve_for_model(label, model, "handoff", interaction_axis="topic_qna")
    assert s4.session_id != s2.session_id
    assert s4.interaction_axis == "topic_qna"
    assert not any(
        "follow_up_question" in (t.get("content") or "") for t in s4.api_turns
    )

    # Обратное переключение снова стартует новую сессию (не восстанавливает s2).
    s5 = mgr.resolve_for_model(
        label, model, "handoff", interaction_axis="lecture_self_check"
    )
    assert s5.session_id not in (s2.session_id, s4.session_id)


def test_generate_dense_material_threads_interaction_axis_into_chat_session() -> None:
    """system prompt being clean is not enough — the actual Gemini chat call

    must also receive interaction_axis, or ChatSessionManager can't tell a
    Topic Q&A turn apart from a lecture_self_check one for the same node
    (see test_resolve_for_model_starts_new_session_on_interaction_axis_switch)."""
    from knowledge_engine.src.domains.grounding import node_content_generator as svc
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["c"]
    )
    memory = SessionMemory()

    captured_axis: list[str] = []

    def _capture_and_stop(*_a: object, **kw: object) -> None:
        captured_axis.append(str(kw.get("interaction_axis")))
        raise _StopEarly()

    with (
        patch.object(
            svc, "run_gemini_structured_with_chain", side_effect=_capture_and_stop
        ),
        patch.object(svc, "run_local_structured", side_effect=_StopEarly),
    ):
        with pytest.raises(_StopEarly):
            svc.generate_dense_material(
                node, memory, "", "anchor-1", interaction_axis="topic_qna"
            )
        with pytest.raises(_StopEarly):
            svc.generate_dense_material(
                node, memory, "", "anchor-1", interaction_axis="lecture_self_check"
            )

    assert captured_axis == ["topic_qna", "lecture_self_check"]


def test_sub_concept_eval_node_evaluates_normally_for_lecture_self_check() -> None:
    """Control: the default interaction_axis does not trip the new skip

    branch — reaches the existing "no pending" skip path instead, proving
    the topic_qna check is additive, not a blanket bypass."""
    from knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval import (
        _sub_concept_eval_node_impl,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
    from knowledge_engine.src.domains.grounding.schemas import NodeDataInput

    node = NodeDataInput(
        node_id="n1", title="Test", layer="foundation", core_concepts=["c"]
    )
    memory = SessionMemory()
    req = NodeDeepDiveRequest(
        curriculum_id="cur-1",
        node_data=node,
        user_action="chat",
        user_message="Some answer",
        interaction_axis="lecture_self_check",
    )
    state = {"request": req, "memory": memory}

    out = _sub_concept_eval_node_impl(state)

    assert out["memory"].evaluator_skipped is True  # no pending target set up
