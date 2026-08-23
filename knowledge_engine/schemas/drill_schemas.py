"""Structured-output contracts for Layer Drill Session tutor turns."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


def _word_count(text: str) -> int:
    return len((text or "").split())


THEORY_BODY_HARD_MIN_WORDS = 150
THEORY_BODY_SOFT_TARGET_WORDS = 300


class AnswerAccuracyGrade(str, Enum):
    """Critic verdict on the learner's previous answer."""

    EXACT_AND_CORRECT = "EXACT_AND_CORRECT"
    PARTIAL = "PARTIAL"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"
    MISUNDERSTANDING = "MISUNDERSTANDING"


class TechnicalConceptAudit(BaseModel):
    """
    Critic pass before learner-facing feedback.

    Single Gemini-compatible object (no JSON-Schema oneOf). Branch rules are
    enforced by ``validate_grade_matches_errors``, not by a discriminated union.
    """

    feedback_kind: Literal["EXACT", "NEEDS_CORRECTION"] = Field(
        ...,
        description=(
            "EXACT when detected_errors_or_misconceptions is empty; "
            "NEEDS_CORRECTION when that list is non-empty."
        ),
    )
    # RU: ветка фидбека: EXACT или NEEDS_CORRECTION.
    accuracy_grade: AnswerAccuracyGrade = Field(
        ...,
        description=(
            "Accuracy verdict. EXACT_AND_CORRECT only with feedback_kind=EXACT. "
            "PARTIAL / NEEDS_CORRECTION / MISUNDERSTANDING only with "
            "feedback_kind=NEEDS_CORRECTION."
        ),
    )
    # RU: вердикт точности; EXACT_AND_CORRECT запрещён при непустом списке ошибок.
    user_claims_analysis: list[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description=(
            "Step-by-step breakdown of the learner's claims: mark which theses "
            "are correct and which are not."
        ),
    )
    # RU: пошаговый разбор тезисов пользователя (верные и неверные).
    detected_errors_or_misconceptions: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Errors, mixed-up terms, or hallucinations. Empty list on EXACT. "
            "Non-empty on NEEDS_CORRECTION."
        ),
    )
    # RU: список ошибок; пустой на ветке EXACT, непустой на NEEDS_CORRECTION.
    confirmation: str = Field(
        default="",
        max_length=4000,
        description=(
            "EXACT branch only: brief technical confirmation. "
            "MUST be an empty string when feedback_kind is NEEDS_CORRECTION."
        ),
    )
    # RU: подтверждение точных тезисов; на ветке коррекции обязана быть пустой строкой.
    praise_points: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Correct technical theses from THIS answer (facts only, no praise). "
            "REQUIRED non-empty when accuracy_grade is PARTIAL. "
            "MUST be empty on EXACT (use confirmation). "
            "May be empty on MISUNDERSTANDING."
        ),
    )
    # RU: верные тезисы (не похвала); обязательны на PARTIAL; на EXACT пусто.
    correction_breakdown: str = Field(
        default="",
        max_length=4000,
        description=(
            "NEEDS_CORRECTION branch only: dry technical correction of the "
            "missing or wrong fragment. "
            "MUST be an empty string when feedback_kind is EXACT."
        ),
    )
    # RU: сухой разбор ошибок / пробела; на ветке EXACT обязана быть пустой строкой.

    @model_validator(mode="after")
    def grade_must_match_detected_errors(self) -> TechnicalConceptAudit:
        validate_grade_matches_errors(self)
        return self


def _nonempty_errors(audit: TechnicalConceptAudit) -> list[str]:
    return [
        e.strip()
        for e in (audit.detected_errors_or_misconceptions or [])
        if (e or "").strip()
    ]


def audit_feedback_text(audit: TechnicalConceptAudit | None) -> str:
    """Host-facing learner text from the active audit branch."""
    if audit is None:
        return ""
    if audit.feedback_kind == "NEEDS_CORRECTION":
        points = [
            p.strip() for p in (audit.praise_points or []) if (p or "").strip()
        ]
        correction = (audit.correction_breakdown or "").strip()
        if points and correction:
            return "\n".join(points) + "\n\n" + correction
        if points:
            return "\n".join(points)
        return correction
    return (audit.confirmation or "").strip()


HOST_PASS_DIRECTIVES = frozenset(
    {
        "PASSED_WITH_GLOSS",
        "PASSED_CLEAN",
        "DEEP_MASTERY_EARNED",
    }
)
HOST_FAIL_DIRECTIVES = frozenset({"STAR_TASK_NEEDS_REFINEMENT"})


def directive_required_feedback_kind(directive: str) -> str | None:
    """Map Host EvalDirective → audit.feedback_kind, or None if unbound."""
    raw = (directive or "").strip()
    if not raw:
        return None
    if raw in HOST_PASS_DIRECTIVES:
        return "EXACT"
    if raw in HOST_FAIL_DIRECTIVES or raw.startswith("PROBE_NEXT_LAYER:"):
        return "NEEDS_CORRECTION"
    return None


def _first_nonempty(*parts: str) -> str:
    """First stripped non-empty string, truncated. No length padding."""
    for part in parts:
        text = (part or "").strip()
        if text:
            return text[:4000]
    return ""


def coerce_audit_to_host_directive(
    audit: TechnicalConceptAudit | None,
    directive: str,
    *,
    focus_hint: str = "",
    evidence: str = "",
    accuracy_grade: AnswerAccuracyGrade | str | None = None,
) -> TechnicalConceptAudit | None:
    """Force Tutor audit.feedback_kind to match Host EvalDirective.

    Returns a new audit when coercion is required; ``None`` stays ``None``
    when the directive is unbound. Prose comes from evidence / focus_hint /
    the existing audit — never from looping filler.
    """
    required = directive_required_feedback_kind(directive)
    if required is None:
        return audit
    hint = (focus_hint or "").strip()
    ev = (evidence or "").strip()
    claims = list((audit.user_claims_analysis or []) if audit is not None else [])
    if not claims:
        claims = [_first_nonempty(ev, hint, directive) or "Host coverage verdict"]
    if required == "EXACT":
        if (
            audit is not None
            and audit.feedback_kind == "EXACT"
            and audit.accuracy_grade == AnswerAccuracyGrade.EXACT_AND_CORRECT
        ):
            return audit
        confirmation = ""
        if audit is not None and audit.feedback_kind == "EXACT":
            confirmation = (audit.confirmation or "").strip()
        confirmation = _first_nonempty(confirmation, ev, directive)
        return TechnicalConceptAudit(
            feedback_kind="EXACT",
            accuracy_grade=AnswerAccuracyGrade.EXACT_AND_CORRECT,
            user_claims_analysis=claims[:16],
            detected_errors_or_misconceptions=[],
            confirmation=confirmation,
            praise_points=[],
            correction_breakdown="",
        )
    host_grade = _coerce_probe_accuracy_grade(audit, accuracy_grade)
    praise = _praise_points_for_probe(audit, ev)
    if (
        audit is not None
        and audit.feedback_kind == "NEEDS_CORRECTION"
        and _nonempty_errors(audit)
        and (audit.correction_breakdown or "").strip()
        and (
            host_grade != AnswerAccuracyGrade.PARTIAL
            or (audit.praise_points or [])
            or praise
        )
    ):
        if host_grade == AnswerAccuracyGrade.PARTIAL and not (
            audit.praise_points or []
        ):
            return audit.model_copy(update={"praise_points": praise[:8]})
        return audit
    errors = list((audit.detected_errors_or_misconceptions or []) if audit else [])
    errors = [e.strip() for e in errors if (e or "").strip()]
    if not errors:
        errors = [_first_nonempty(hint, directive) or "PROBE"]
    correction = ""
    if audit is not None and audit.feedback_kind == "NEEDS_CORRECTION":
        correction = (audit.correction_breakdown or "").strip()
    correction = _first_nonempty(correction, hint, errors[0], directive)
    if host_grade == AnswerAccuracyGrade.PARTIAL and not praise:
        praise = [_first_nonempty(ev, claims[0]) or claims[0]][:8]
    if host_grade == AnswerAccuracyGrade.MISUNDERSTANDING:
        out_praise: list[str] = []
    else:
        out_praise = praise[:8]
    return TechnicalConceptAudit(
        feedback_kind="NEEDS_CORRECTION",
        accuracy_grade=host_grade,
        user_claims_analysis=claims[:16],
        detected_errors_or_misconceptions=errors[:16],
        confirmation="",
        praise_points=out_praise,
        correction_breakdown=correction,
    )


def _coerce_probe_accuracy_grade(
    audit: TechnicalConceptAudit | None,
    raw: AnswerAccuracyGrade | str | None,
) -> AnswerAccuracyGrade:
    """Map Host/Evaluator grade onto the NEEDS_CORRECTION branch."""
    for candidate in (raw, getattr(audit, "accuracy_grade", None)):
        parsed = _parse_accuracy_grade(candidate)
        if parsed is None or parsed == AnswerAccuracyGrade.EXACT_AND_CORRECT:
            continue
        return parsed
    return AnswerAccuracyGrade.PARTIAL


def _parse_accuracy_grade(raw: object) -> AnswerAccuracyGrade | None:
    if raw is None:
        return None
    if isinstance(raw, AnswerAccuracyGrade):
        return raw
    text = str(getattr(raw, "value", raw) or "").strip()
    try:
        return AnswerAccuracyGrade(text)
    except ValueError:
        return None


def _praise_points_for_probe(
    audit: TechnicalConceptAudit | None,
    evidence: str,
) -> list[str]:
    points = [
        p.strip()
        for p in ((audit.praise_points or []) if audit is not None else [])
        if (p or "").strip()
    ]
    if points:
        return points[:8]
    ev = (evidence or "").strip()
    if ev:
        return [ev[:500]]
    return []


def align_structured_tutor_with_host(raw: object, memory: object | None) -> object:
    """Post-LLM: coerce ``audit.feedback_kind`` to Host ``EvalDirective``.

    Evaluator-skip turns use ``DeepDiveExplainContract`` (no audit). If a
    scored schema still arrives, Host still rewrites audit to the directive.
    """
    if raw is None or not hasattr(raw, "model_copy"):
        return raw
    if bool(getattr(memory, "evaluator_skipped", False)):
        return raw
    if not hasattr(raw, "audit"):
        return raw
    directive = str(getattr(memory, "last_eval_directive", "") or "").strip()
    hint, evidence, row_grade = "", "", ""
    if memory is not None:
        from knowledge_engine.src.node_deep_dive.concept_map_state import (
            find_sub_concept,
            resolve_evaluation_target_id,
        )

        cid = ""
        try:
            cid = resolve_evaluation_target_id(memory) or ""
        except Exception:
            cid = str(getattr(memory, "asked_question_sub_concept_id", "") or "")
        row = find_sub_concept(memory, cid) if cid else None
        if row is None:
            for sc in getattr(memory, "sub_concepts", None) or []:
                if (getattr(sc, "focus_hint", "") or "").strip():
                    row = sc
                    break
        if row is not None:
            hint = (row.focus_hint or "").strip()
            evidence = (row.evidence or "").strip()
            row_grade = (row.last_accuracy_grade or "").strip()
    new_audit = coerce_audit_to_host_directive(
        getattr(raw, "audit", None),
        directive,
        focus_hint=hint,
        evidence=evidence,
        accuracy_grade=row_grade,
    )
    if new_audit is getattr(raw, "audit", None):
        return raw
    return raw.model_copy(update={"audit": new_audit})


def validate_grade_matches_errors(audit: TechnicalConceptAudit) -> None:
    """Reject logically incompatible grade / error-list / feedback_kind triples."""
    errors = _nonempty_errors(audit)
    confirmation = (audit.confirmation or "").strip()
    correction = (audit.correction_breakdown or "").strip()
    if errors:
        if audit.feedback_kind != "NEEDS_CORRECTION":
            raise ValueError(
                "Inconsistency: detected_errors_or_misconceptions is not empty, "
                "but feedback_kind is EXACT. Use NEEDS_CORRECTION with "
                "correction_breakdown."
            )
        if audit.accuracy_grade == AnswerAccuracyGrade.EXACT_AND_CORRECT:
            raise ValueError(
                "Inconsistency: detected_errors_or_misconceptions is not empty, "
                "but accuracy_grade is EXACT_AND_CORRECT. "
                "Grade must be PARTIAL, NEEDS_CORRECTION, or MISUNDERSTANDING."
            )
        if not correction:
            raise ValueError(
                "NEEDS_CORRECTION requires non-empty correction_breakdown."
            )
        if confirmation:
            raise ValueError(
                "Inconsistency: feedback_kind is NEEDS_CORRECTION, but "
                "confirmation is not empty. Leave confirmation empty and put "
                "the review in correction_breakdown."
            )
        praise = [
            p.strip() for p in (audit.praise_points or []) if (p or "").strip()
        ]
        if audit.accuracy_grade == AnswerAccuracyGrade.PARTIAL and not praise:
            raise ValueError(
                "PARTIAL requires non-empty praise_points "
                "(correct theses, not cheerleading)."
            )
        if (
            audit.accuracy_grade
            in (
                AnswerAccuracyGrade.NEEDS_CORRECTION,
                AnswerAccuracyGrade.MISUNDERSTANDING,
            )
            and praise
            and not correction
        ):
            raise ValueError(
                "NEEDS_CORRECTION / MISUNDERSTANDING still require "
                "correction_breakdown; praise_points cannot replace it."
            )
        return
    praise = [
        p.strip() for p in (audit.praise_points or []) if (p or "").strip()
    ]
    if praise:
        raise ValueError(
            "Inconsistency: feedback_kind is EXACT, but praise_points is not "
            "empty. Leave praise_points empty and put the review in confirmation."
        )
    if audit.feedback_kind != "EXACT":
        raise ValueError(
            "Inconsistency: detected_errors_or_misconceptions is empty, "
            "but feedback_kind is NEEDS_CORRECTION. Use EXACT with confirmation."
        )
    if audit.accuracy_grade != AnswerAccuracyGrade.EXACT_AND_CORRECT:
        raise ValueError(
            "Inconsistency: detected_errors_or_misconceptions is empty, "
            "but accuracy_grade is not EXACT_AND_CORRECT."
        )
    if not confirmation:
        raise ValueError("EXACT requires non-empty confirmation.")
    if correction:
        raise ValueError(
            "Inconsistency: feedback_kind is EXACT, but correction_breakdown "
            "is not empty. Leave correction_breakdown empty."
        )


class ActiveDrillStepResponse(BaseModel):
    """Tutor payload while the drill still has a queued sub-topic to teach."""

    audit: TechnicalConceptAudit = Field(
        ...,
        description=(
            "Strict technical audit of the learner's previous answer, filled "
            "BEFORE any other learner-facing text. "
            "feedback_kind EXACT → confirmation (correction_breakdown empty); "
            "NEEDS_CORRECTION → correction_breakdown (confirmation empty)."
        ),
    )
    status_header: str = Field(
        ...,
        min_length=8,
        max_length=400,
        description=(
            "Progress status line, e.g. "
            "'[Слой MECH: Проверено 1/4 подтем. Переходим к подтеме №2: «…»]'"
        ),
    )
    theory_body: str = Field(
        ...,
        min_length=300,
        max_length=24_000,
        description=(
            "Dense theoretical treatment of the current sub-topic "
            f"(target {THEORY_BODY_SOFT_TARGET_WORDS}+ words; "
            f"hard minimum {THEORY_BODY_HARD_MIN_WORDS}). "
            "Include C structures, schemes, code listings, memory allocation, "
            "or architectural trade-offs as required by the active layer."
        ),
    )
    # RU: теория текущего шага; жёсткий пол 150 слов, цель 300 (ниже 300 — WARN).
    next_question: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description=(
            "Exactly one direct checkpoint question about theory_body. "
            "Every criterion the Evaluator may require MUST appear here and "
            "in theory_body first. FORBIDDEN: a surface question whose hidden "
            "rubric is an unasked deeper layer."
        ),
    )

    @computed_field
    @property
    def feedback_on_previous(self) -> str:
        return audit_feedback_text(self.audit)

    @field_validator("theory_body")
    @classmethod
    def theory_must_be_dense(cls, value: str) -> str:
        words = _word_count(value)
        if words < THEORY_BODY_HARD_MIN_WORDS:
            raise ValueError(
                f"theory_body must contain at least {THEORY_BODY_HARD_MIN_WORDS} "
                f"words (got {words})"
            )
        if words < THEORY_BODY_SOFT_TARGET_WORDS:
            from knowledge_engine.ui.run_log import trace

            trace(
                "WARN drill theory_body short of soft target | "
                f"words={words} target={THEORY_BODY_SOFT_TARGET_WORDS} "
                f"hard_min={THEORY_BODY_HARD_MIN_WORDS}"
            )
        return value

    @field_validator("next_question")
    @classmethod
    def question_must_ask(cls, value: str) -> str:
        text = (value or "").strip()
        if "?" not in text and "？" not in text:
            raise ValueError("next_question must contain a question mark")
        return text

    @model_validator(mode="after")
    def validate_audit_branch_consistency(self) -> ActiveDrillStepResponse:
        validate_grade_matches_errors(self.audit)
        return self


class LayerCompletionTutorOutput(BaseModel):
    """Facilitation payload when Evaluator closed the current layer this turn.

    There is no ``next_question`` field — the model cannot emit a checkpoint quiz.
    Host attaches ``ready_for_transition`` and Quick Reply chips after validation.
    """

    praise: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description=(
            "Congratulate the learner on closing this layer. "
            "No technical or evaluative checkpoint."
        ),
    )
    # RU: похвала за закрытие слоя; не проверочный вопрос.
    layer_summary: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description=(
            "Short recap of what this layer established. "
            "No theory_body lecture and no checkpoint question."
        ),
    )
    # RU: резюме закрытого слоя.
    transition_framing: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description=(
            "Invite the learner to choose HOW/MECH/Advanced/Deep mode "
            "or the next topic. Not a technical/evaluative question. "
            "Do not invent chip labels — Host owns Quick Replies."
        ),
    )
    # RU: фрейминг выбора следующего шага; чипы ставит Host.


# Active drill checkpoint contract (required next_question).
StandardDrillTutorOutput = ActiveDrillStepResponse

DrillStepResponse = Union[ActiveDrillStepResponse, LayerCompletionTutorOutput]

# Compat aliases — same Gemini-safe model, not JSON-Schema union variants.
ExactAnswerAudit = TechnicalConceptAudit
NeedsCorrectionAudit = TechnicalConceptAudit
