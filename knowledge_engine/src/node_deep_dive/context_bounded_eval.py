"""SSOT замкнутого контекста: слой оценки = слой, явно запрошенный вопросом.

Мета-доменный инвариант (Scope Ceiling): Evaluator и Question Factory
делят одни правила для любой предметной области.
"""

from __future__ import annotations

CONTEXT_BOUNDED_EVAL_RULES = (
    "=== STRICT SCOPE & ABSTRACTION CEILING (UNIVERSAL INVARIANT) ===\n"
    "The evaluation scope is STRICTLY BOUNDED by the explicitly requested abstraction layer "
    "and explicit constraints of the asked question (last_tutor_question + current intro).\n\n"
    "EVALUATION RULES:\n"
    "1. BOUNDED ABSTRACTION CEILING:\n"
    "   Evaluate the answer ONLY at the abstraction layer requested by the question.\n"
    "   - FORBIDDEN: Lowering accuracy_grade, marking partial/gap, or generating focus_hint "
    "     based on missing facts, micro-details, or mechanisms from deeper or adjacent layers "
    "     that were NOT explicitly demanded by the question text.\n"
    "   - FORBIDDEN: Penalty for omitting unasked jargon, internal implementations, or background context.\n\n"
    "2. ACCURACY GRADING CONTRACT:\n"
    "   - EXACT_AND_CORRECT: Set when the answer is factually correct and complete FOR THE ASKED "
    "     QUESTION'S LAYER. Volunteered deeper knowledge is bonus credit, NEVER a required threshold.\n"
    "   - PARTIAL: Set ONLY when an aspect EXPLICITLY requested by the question is missing or incomplete.\n\n"
    "3. ERROR DEFINITION:\n"
    "   - detected_errors_or_misconceptions MUST contain ONLY explicit, factually false statements "
    "     made in the user's text.\n"
    "   - Silence or omission regarding unasked topics is NOT an error.\n\n"
    "4. FOCUS_HINT RESTRICTION:\n"
    "   - focus_hint MUST describe ONLY the single missing element from the explicitly asked scope.\n"
    "   - NEVER suggest or demand deeper/unasked abstraction layers in focus_hint.\n"
)
"""
RU (пояснение): оценка строго на запрошенном слое абстракции; опущение незапрошенного — не ошибка.
"""

CONTEXT_BOUNDED_QUESTION_RULES = (
    "=== CONTEXT-BOUNDED QUESTION FACTORY (UNIVERSAL INVARIANT) ===\n"
    "A question MUST explicitly define its scope and state every required evaluative criterion "
    "in its text.\n\n"
    "GENERATION RULES:\n"
    "1. EXPLICIT SCOPE LOCK:\n"
    "   Any concept, mechanism, or detail that the Evaluator will be expected to verify MUST be "
    "   explicitly named, introduced, or scope-locked in the question text.\n"
    "2. NO HIDDEN RUBRICS:\n"
    "   Do NOT ask a high-level motivation or intuitive question if the hidden evaluation "
    "   criteria expects deep structural, mathematical, or low-level mechanics.\n"
    "3. LAYER CONGRUENCE:\n"
    "   The prompt layer MUST match the evaluation layer:\n"
    "   - WHY / Motivation probes -> Evaluate ONLY purpose, cause, and consequences.\n"
    "   - HOW / Structural probes -> Evaluate ONLY components, roles, and invariants.\n"
    "   - MECHANIC / Detail probes -> Evaluate execution steps, code, math, or formal logic ONLY after naming them.\n"
)
"""
RU (пояснение): вопрос обязан назвать и зафиксировать слой всех критериев, которые потом оценит Evaluator.
"""
