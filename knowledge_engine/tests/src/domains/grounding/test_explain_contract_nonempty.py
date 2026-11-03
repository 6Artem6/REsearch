"""Ход без Evaluator (explain-контракты) обязан содержать текст для ученика.

Живой случай: Evaluator пропущен (axis switch topic_qna → lecture_self_check),
тьютор ответил по DeepDiveExplainContract только служебными полями
(summary / quick_replies / question_sub_concept_id) — контракт без
обязательных полей это пропустил, tutor_generate подставил заглушку
«Продолжим по теме — один конкретный вопрос или уточнение.»"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge_engine.src.domains.grounding.tutor import (
    EMPTY_EXPLAIN_TURN_ERROR,
    DeepDiveExplainContract,
    TopicQnaExplainContract,
)

_BOOKKEEPING_ONLY = {
    "node_status": "in_progress",
    "summary": "Разбор шардирования векторных индексов.",
    "question_sub_concept_id": "векторные_базы_данных",
    "quick_replies": ["Хочу Gloss", "Идем дальше"],
}


@pytest.mark.parametrize("schema", [DeepDiveExplainContract, TopicQnaExplainContract])
def test_bookkeeping_only_reply_is_rejected_with_retry_marker(schema) -> None:
    with pytest.raises(ValidationError) as exc_info:
        schema.model_validate(_BOOKKEEPING_ONLY)
    # engine.py распознаёт retryable-ошибку по этому маркеру в str(exc).lower()
    assert EMPTY_EXPLAIN_TURN_ERROR.lower() in str(exc_info.value).lower()


def test_explain_with_text_or_question_is_valid() -> None:
    DeepDiveExplainContract.model_validate(
        {**_BOOKKEEPING_ONLY, "technical_explanation": "Шардирование делит индекс."}
    )
    DeepDiveExplainContract.model_validate(
        {**_BOOKKEEPING_ONLY, "follow_up_question": "Что ограничивает шард?"}
    )
    TopicQnaExplainContract.model_validate(
        {**_BOOKKEEPING_ONLY, "technical_explanation": "Шардирование делит индекс."}
    )


def test_whitespace_only_text_is_still_empty() -> None:
    with pytest.raises(ValidationError):
        DeepDiveExplainContract.model_validate(
            {
                **_BOOKKEEPING_ONLY,
                "technical_explanation": "   ",
                "follow_up_question": "\n",
            }
        )
