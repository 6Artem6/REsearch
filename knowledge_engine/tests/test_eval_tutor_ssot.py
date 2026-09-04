"""Host SSOT: EvalDirective binds Tutor audit; skip; transparency plaque."""

from __future__ import annotations

from knowledge_engine.schemas.drill_schemas import (
    AnswerAccuracyGrade,
    TechnicalConceptAudit,
    coerce_audit_to_host_directive,
    directive_required_feedback_kind,
)
from knowledge_engine.schemas.llm_contracts.tutor import DeepDiveExplainContract
from knowledge_engine.src.node_deep_dive.concept_map_state import (
    compose_host_transparency_plaque,
    format_concept_map_for_tutor,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    apply_threshold_to_sub_concept,
    mark_evaluator_skipped,
)
from knowledge_engine.src.node_deep_dive.tutor_critique_prompt import (
    ANTI_SYCOPHANCY_INVARIANTS,
    EVALUATOR_SKIPPED_TUTOR_RULES,
)
from knowledge_engine.src.node_deep_dive.dialogue_prompt_en import (
    DIALOGUE_JSON_CONTRACT_EN,
    FEEDBACK_TRANSPARENCY_REQUIREMENT_EN,
)


def _exact_audit() -> TechnicalConceptAudit:
    return TechnicalConceptAudit(
        feedback_kind="EXACT",
        accuracy_grade=AnswerAccuracyGrade.EXACT_AND_CORRECT,
        user_claims_analysis=["Тезисы по владению указателем точны."],
        detected_errors_or_misconceptions=[],
        confirmation="Разбор точный: refcnt и ob_type согласованы.",
        correction_breakdown="",
    )


def _correction_audit() -> TechnicalConceptAudit:
    return TechnicalConceptAudit(
        feedback_kind="NEEDS_CORRECTION",
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        user_claims_analysis=["Путаница mutex и LOCK XADD."],
        detected_errors_or_misconceptions=["Mutex вместо атомарной инструкции."],
        confirmation="",
        correction_breakdown=(
            "LOCK XADD не паркует поток в ОС и не использует mutex ядра."
        ),
        praise_points=["LOCK XADD — атомарная инструкция процессора, не mutex ядра."],
    )


def test_directive_maps_pass_and_probe() -> None:
    assert directive_required_feedback_kind("PASSED_CLEAN") == "EXACT"
    assert directive_required_feedback_kind("PASSED_WITH_GLOSS") == "EXACT"
    assert directive_required_feedback_kind("DEEP_MASTERY_EARNED") == "EXACT"
    assert directive_required_feedback_kind("PROBE_NEXT_LAYER:HOW") == "NEEDS_CORRECTION"
    assert (
        directive_required_feedback_kind("STAR_TASK_NEEDS_REFINEMENT")
        == "NEEDS_CORRECTION"
    )
    assert directive_required_feedback_kind("") is None


def test_coerce_short_evidence_does_not_pad() -> None:
    coerced = coerce_audit_to_host_directive(
        None,
        "PASSED_CLEAN",
        evidence="WHY ок.",
    )
    assert coerced is not None
    assert coerced.confirmation == "WHY ок."
    assert coerced.confirmation.count("WHY ок.") == 1
    coerced = coerce_audit_to_host_directive(
        _correction_audit(),
        "PASSED_CLEAN",
        evidence="WHY и HOW закрыты в этом ответе.",
    )
    assert coerced is not None
    assert coerced.feedback_kind == "EXACT"
    assert coerced.accuracy_grade is AnswerAccuracyGrade.EXACT_AND_CORRECT
    assert not coerced.detected_errors_or_misconceptions
    assert coerced.correction_breakdown == ""
    assert "WHY и HOW" in coerced.confirmation
    assert "Нужно закрыть открытый критерий" not in coerced.confirmation


def test_coerce_probe_overrides_tutor_exact() -> None:
    coerced = coerce_audit_to_host_directive(
        _exact_audit(),
        "PROBE_NEXT_LAYER:HOW",
        focus_hint="Нужно раскрыть HOW: архитектура и инварианты.",
    )
    assert coerced is not None
    assert coerced.feedback_kind == "NEEDS_CORRECTION"
    assert coerced.detected_errors_or_misconceptions
    assert coerced.confirmation == ""
    assert "HOW" in coerced.correction_breakdown or "архитектура" in (
        coerced.correction_breakdown or ""
    ).lower()
    assert "Нужно закрыть открытый критерий" not in coerced.correction_breakdown


def test_host_transparency_plaque_on_probe_only() -> None:
    mem = SessionMemory(
        last_eval_directive="PROBE_NEXT_LAYER:HOW",
        asked_question_sub_concept_id="pyobject",
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject",
                status="partial",
                why_passed=True,
                evidence="Мотивация владения указателем закрыта.",
                focus_hint="Нужно раскрыть HOW: header и ob_type.",
            )
        ],
    )
    plaque = compose_host_transparency_plaque(mem)
    assert "**Слои:**" not in plaque
    assert "WHY ✅" not in plaque
    assert "📋" in plaque
    assert "🎯" in plaque
    assert "ob_type" in plaque
    mem.last_eval_directive = "PASSED_CLEAN"
    assert compose_host_transparency_plaque(mem) == ""
    mem.last_eval_directive = "PROBE_NEXT_LAYER:HOW"
    mem.evaluator_skipped = True
    assert compose_host_transparency_plaque(mem) == ""


def test_stream_host_transparency_plaque_emits_before_body() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        stream_host_transparency_plaque,
    )

    mem = SessionMemory(
        last_eval_directive="PROBE_NEXT_LAYER:HOW",
        asked_question_sub_concept_id="pyobject",
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject",
                status="partial",
                why_passed=True,
                evidence="Мотивация владения указателем закрыта.",
                focus_hint="Нужно раскрыть HOW: header и ob_type.",
            )
        ],
    )
    chunks: list[str] = []
    plaque = stream_host_transparency_plaque(chunks.append, mem)
    assert plaque.startswith("---")
    assert "**Слои:**" not in plaque
    assert "Что уже зачтено" in plaque
    assert "".join(chunks) == plaque + "\n\n"
    assert stream_host_transparency_plaque(None, mem) == plaque
    mem.evaluator_skipped = True
    assert stream_host_transparency_plaque(chunks.append, mem) == ""
    assert "".join(chunks) == plaque + "\n\n"


def test_host_transparency_plaque_layer_status_line() -> None:
    mem = SessionMemory(
        last_eval_directive="PROBE_NEXT_LAYER:HOW",
        asked_question_sub_concept_id="pyobject",
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject",
                status="partial",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
                evidence="WHY закрыт.",
                focus_hint="Нужно раскрыть HOW.",
            )
        ],
    )
    plaque = compose_host_transparency_plaque(mem)
    assert "**Слои:**" not in plaque
    assert "Что уже зачтено" in plaque
    assert "Чего не хватило" in plaque

    mem.last_eval_directive = "STAR_TASK_NEEDS_REFINEMENT"
    plaque_star = compose_host_transparency_plaque(mem)
    assert "**Слои:**" not in plaque_star
    assert "Что уже зачтено" in plaque_star

    mem.last_eval_directive = "PASSED_CLEAN"
    assert compose_host_transparency_plaque(mem) == ""
    mem.evaluator_skipped = True
    assert compose_host_transparency_plaque(mem) == ""


def test_evaluator_skip_explain_contract_has_no_audit() -> None:
    parsed = DeepDiveExplainContract(
        technical_explanation="Разберём header PyObject без оценки прошлого ответа.",
        follow_up_question="Как связан ob_type с диспетчеризацией слотов?",
    )
    assert parsed.feedback_on_answer == ""
    assert "audit" not in DeepDiveExplainContract.model_fields


def test_mark_evaluator_skipped_flag() -> None:
    mem = SessionMemory()
    mark_evaluator_skipped(mem, "lecture request")
    assert mem.evaluator_skipped is True


def test_core_map_does_not_inject_stale_overlay_critique() -> None:
    mem = SessionMemory(
        last_evaluator_critique={
            "target_layer": "DEEP",
            "passes_threshold": False,
            "analyzed_ideas": [],
        },
        last_eval_directive="PROBE_NEXT_LAYER:HOW",
        last_evaluator_feedback="core feedback",
        sub_concepts=[
            SubConceptRecord(id="agg", label="agg", status="partial", why_passed=True)
        ],
    )
    block = format_concept_map_for_tutor(mem, node_layer="advanced")
    assert "[EVALUATOR_CRITIQUE_JSON]" not in block
    assert "core feedback" in block


def test_soft_verified_does_not_invent_layer_flags() -> None:
    row = SubConceptRecord(id="kv", label="KV", status="unchecked")
    apply_threshold_to_sub_concept(
        row, layer="foundation", why=False, how=False, mechanic=False
    )
    assert row.why_passed is False
    assert row.how_passed is False
    assert row.status == "partial"


def test_prompts_forbid_oneof_and_model_plaques() -> None:
    blob = (
        ANTI_SYCOPHANCY_INVARIANTS
        + DIALOGUE_JSON_CONTRACT_EN
        + FEEDBACK_TRANSPARENCY_REQUIREMENT_EN
        + EVALUATOR_SKIPPED_TUTOR_RULES
    )
    assert "not a JSON-Schema oneOf" in ANTI_SYCOPHANCY_INVARIANTS.lower() or (
        "NOT a JSON-Schema oneOf" in ANTI_SYCOPHANCY_INVARIANTS
    )
    assert "discriminated union on" not in blob.lower()
    assert "Host prepends" in FEEDBACK_TRANSPARENCY_REQUIREMENT_EN or (
        "Host prepends" in ANTI_SYCOPHANCY_INVARIANTS
    )
    assert "DeepDiveExplainContract" in EVALUATOR_SKIPPED_TUTOR_RULES
