"""Anti-sycophancy: schema rejects grade / error-list / feedback-branch mismatches."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge_engine.schemas.drill_schemas import (
    ActiveDrillStepResponse,
    AnswerAccuracyGrade,
    TechnicalConceptAudit,
)
from knowledge_engine.schemas.llm_contracts.tutor import (
        DeepDiveDeepAnalysisContract,
        DeepDiveExplainContract,
        DeepDiveTutorContract,
)


def _dense_theory() -> str:
    return ("архитектура потока данных изоляция состояния аллокация ") * 60


def _correction_audit() -> TechnicalConceptAudit:
    return TechnicalConceptAudit(
        feedback_kind="NEEDS_CORRECTION",
        user_claims_analysis=[
            "Утверждение про накладные расходы на согласованность — верно.",
            "Утверждение про «кэш процесса» и блокировку потока ОС — неверно.",
        ],
        detected_errors_or_misconceptions=[
            "Смешение атомарной инструкции LOCK XADD с OS mutex / блокировкой потока."
        ],
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        correction_breakdown=(
            "LOCK XADD не блокирует поток на уровне ОС и не использует "
            "«кэш процесса»: замедление идёт через MESI bouncing."
        ),
        confirmation="",
        praise_points=[
            "Накладные расходы на согласованность кэш-линий — верный тезис."
        ],
    )


def test_audit_validator_fails_on_inconsistent_grade() -> None:
    with pytest.raises(ValidationError):
        TechnicalConceptAudit(
            feedback_kind="EXACT",
            accuracy_grade=AnswerAccuracyGrade.EXACT_AND_CORRECT,
            user_claims_analysis=["Тезис пользователя содержит ошибку в MESI."],
            detected_errors_or_misconceptions=[
                "Путаница L1/L2 ядер с межпроцессным кэшем."
            ],
            confirmation="Краткое подтверждение точных тезисов.",
        )
    with pytest.raises(ValidationError):
        TechnicalConceptAudit.model_validate(
            {
                "feedback_kind": "NEEDS_CORRECTION",
                "user_claims_analysis": ["Есть ошибка."],
                "detected_errors_or_misconceptions": ["Mutex вместо LOCK XADD."],
                "accuracy_grade": "EXACT_AND_CORRECT",
                "correction_breakdown": (
                    "Mutex не эквивалентен LOCK XADD: атомарная инструкция "
                    "не паркует поток в ОС."
                ),
            }
        )
    with pytest.raises(ValidationError):
        ActiveDrillStepResponse.model_validate(
            {
                "audit": {
                    "feedback_kind": "EXACT",
                    "accuracy_grade": "EXACT_AND_CORRECT",
                    "user_claims_analysis": ["Есть ошибка."],
                    "detected_errors_or_misconceptions": ["Mutex вместо LOCK XADD."],
                    "confirmation": "Краткое подтверждение точных тезисов.",
                },
                "status_header": (
                    "[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]"
                ),
                "theory_body": _dense_theory(),
                "next_question": "Как инвалидируется кэш-линия при LOCK XADD?",
            }
        )


def test_audit_rejects_correction_branch_without_errors() -> None:
    with pytest.raises(ValidationError):
        TechnicalConceptAudit(
            feedback_kind="NEEDS_CORRECTION",
            user_claims_analysis=["Тезисы по MESI точны."],
            detected_errors_or_misconceptions=[],
            accuracy_grade=AnswerAccuracyGrade.PARTIAL,
            correction_breakdown=(
                "Сухой разбор без списка ошибок не должен проходить валидацию."
            ),
        )


def test_valid_audit_response_passes() -> None:
    parsed = ActiveDrillStepResponse(
        audit=_correction_audit(),
        status_header="[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]",
        theory_body=_dense_theory(),
        next_question="Почему bouncing бьёт по L1/L2 разных ядер?",
    )
    assert parsed.audit.accuracy_grade is AnswerAccuracyGrade.PARTIAL
    assert parsed.audit.detected_errors_or_misconceptions
    assert parsed.audit.praise_points
    assert "Накладные расходы" in parsed.feedback_on_previous
    assert "LOCK XADD" in parsed.feedback_on_previous

    exact = TechnicalConceptAudit(
        feedback_kind="EXACT",
        user_claims_analysis=["Тезисы по MESI и cache-line bouncing точны."],
        detected_errors_or_misconceptions=[],
        accuracy_grade=AnswerAccuracyGrade.EXACT_AND_CORRECT,
        confirmation="Разбор точный: инвалидация идёт по MESI между ядрами.",
    )
    tutor = DeepDiveTutorContract(
        audit=exact,
        technical_explanation="Дальше разберём false sharing на соседней линии.",
        follow_up_question="Как выровнять атомарный счётчик, чтобы избежать false sharing?",
    )
    assert tutor.audit.accuracy_grade is AnswerAccuracyGrade.EXACT_AND_CORRECT
    assert tutor.feedback_on_answer.startswith("Разбор точный")


def test_partial_audit_requires_praise_points() -> None:
    with pytest.raises(ValidationError):
        TechnicalConceptAudit(
            feedback_kind="NEEDS_CORRECTION",
            user_claims_analysis=["Часть тезисов верна."],
            detected_errors_or_misconceptions=["Пропущен инвариант владения."],
            accuracy_grade=AnswerAccuracyGrade.PARTIAL,
            correction_breakdown="Нужен инвариант владения указателем.",
            confirmation="",
            praise_points=[],
        )


def test_exact_audit_rejects_praise_points() -> None:
    with pytest.raises(ValidationError):
        TechnicalConceptAudit(
            feedback_kind="EXACT",
            user_claims_analysis=["Тезисы точны."],
            detected_errors_or_misconceptions=[],
            accuracy_grade=AnswerAccuracyGrade.EXACT_AND_CORRECT,
            confirmation="Разбор точный: refcnt согласован.",
            praise_points=["refcnt согласован."],
        )


def test_audit_json_schema_has_no_oneof_or_discriminator() -> None:
    """Gemini Schema rejects JSON-Schema oneOf / discriminator on nested fields."""
    from google.genai import _transformers

    class _Client:
        vertexai = False

    client = _Client()
    for cls in (
        TechnicalConceptAudit,
        ActiveDrillStepResponse,
        DeepDiveTutorContract,
        DeepDiveDeepAnalysisContract,
        DeepDiveExplainContract,
    ):
        schema = cls.model_json_schema()
        audit = schema.get("properties", {}).get("audit", schema)
        if "$ref" in audit:
            ref_name = str(audit["$ref"]).rsplit("/", 1)[-1]
            audit = schema.get("$defs", {}).get(ref_name, audit)
        raw = json.dumps(audit)
        assert '"oneOf"' not in raw, cls.__name__
        assert '"discriminator"' not in raw, cls.__name__
        converted = _transformers.t_schema(client, cls)
        assert converted is not None
