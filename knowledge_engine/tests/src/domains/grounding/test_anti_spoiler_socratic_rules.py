"""Anti-Spoiler / Socratic Scaffolding rules: text explains mechanics, the
closing question asks for a consequence/trade-off the user derives
themselves. Regression for the tutor quality audit — see prompt.log task
about spoiler effect / orphaned concepts / meta-thesis frames."""

from __future__ import annotations

from knowledge_engine.src.domains.grounding.context_bounded_eval import (
    ANTI_SPOILER_SOCRATIC_RULES,
)
from knowledge_engine.src.domains.grounding.dialogue_prompt_en import (
    dialogue_module_parts,
)
from knowledge_engine.src.domains.grounding.prompt_types import InteractionPromptMode
from knowledge_engine.src.domains.grounding.tutor import (
    DeepDiveTutorContract,
    StructuredLectureResponse,
    TopicQnaTutorContract,
    _DenseLectureFieldsBase,
    _TutorFieldsBase,
)
from knowledge_engine.src.domains.grounding.tutor_prompt_builder import (
    build_critical_rules_recency_tail,
)


def test_anti_spoiler_rule_present_in_dialogue_module_parts():
    assert any(
        "ANTI-SPOILER" in part for part in dialogue_module_parts()
    )


def test_anti_spoiler_rule_present_in_all_three_recency_tails():
    for mode in (
        InteractionPromptMode.DIALOGUE_FEEDBACK,
        InteractionPromptMode.LECTURE_CHAT,
        InteractionPromptMode.LECTURE_DENSE,
    ):
        tail = build_critical_rules_recency_tail(mode=mode)
        assert "ANTI-SPOILER" in tail, mode
        assert "SOCRATIC" in tail, mode


def test_rule_text_bans_stating_the_answer_and_allows_micro_questions():
    assert "never state that consequence" in ANTI_SPOILER_SOCRATIC_RULES
    assert "rhetorical micro-questions" in ANTI_SPOILER_SOCRATIC_RULES
    assert "NO ORPHANED CONCEPTS" in ANTI_SPOILER_SOCRATIC_RULES


def test_technical_explanation_description_carries_anti_spoiler_and_allows_micro_questions():
    desc = _TutorFieldsBase.model_fields["technical_explanation"].description or ""
    assert "never states the consequence" in desc
    assert "rhetorical micro-questions" in desc
    # Must not ban all '?' outright anymore — that would forbid the allowed
    # Socratic micro-questions the system prompt now asks for.
    assert "no «?»" not in desc


def test_follow_up_question_description_requires_consequence_not_restated_fact():
    desc = DeepDiveTutorContract.model_fields["follow_up_question"].description or ""
    assert "CONSEQUENCE/TRADE-OFF/COST" in desc
    assert "never only in this question" in desc


def test_lecture_body_and_follow_up_question_descriptions_carry_anti_spoiler():
    body_desc = _DenseLectureFieldsBase.model_fields["lecture_body"].description or ""
    checkpoint_desc = (
        StructuredLectureResponse.model_fields["follow_up_question"].description or ""
    )
    assert "never states the consequence" in body_desc
    assert "rhetorical micro-questions" in body_desc
    assert "CONSEQUENCE/TRADE-OFF/COST" in checkpoint_desc
    assert "never only here" in checkpoint_desc


def test_message_bullet_summary_forbids_recap_frames_in_both_contracts():
    for cls in (_DenseLectureFieldsBase, _TutorFieldsBase):
        desc = cls.model_fields["message_bullet_summary"].description or ""
        assert "discussed X" in desc or "covered Y" in desc
        assert "Content claims only" in desc


def test_message_bullet_summary_names_the_leak_source_to_ignore():
    """Regression: real observed leak — confirmation's 'подтема разобрана
    верно' resurfaced reworded as a thesis ('разбор завершён'). Naming the
    exact field to ignore (audit / summary+bridge_to_next) is stronger than
    a generic 'no meta-statements' ban — see prompt.log trace
    fe16e76b969a/turn_02_tutor_exchange.md."""
    dialogue_desc = _TutorFieldsBase.model_fields["message_bullet_summary"].description or ""
    assert "IGNORE the `audit` object entirely" in dialogue_desc
    assert "confirmation" in dialogue_desc and "correction_breakdown" in dialogue_desc

    lecture_desc = _DenseLectureFieldsBase.model_fields["message_bullet_summary"].description or ""
    assert "IGNORE `summary` and `bridge_to_next` entirely" in lecture_desc


def test_topic_qna_tutor_contract_inherits_updated_technical_explanation():
    desc = TopicQnaTutorContract.model_fields["technical_explanation"].description or ""
    assert "never states the consequence" in desc
