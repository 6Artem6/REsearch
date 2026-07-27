"""SLM (1.5B) Clarification Check — Frugal routing перед discovery."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from knowledge_engine.config import CLARIFY_VIA_GEMINI, ROUTER_MODEL, SKIP_GEMINI
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import (
    ClarificationAssessment,
    EngineGraphState,
    EngineState,
)
from knowledge_engine.services.context_manager import rolling_summarize_dialogue
from knowledge_engine.services.gemini_research_session import ask_gemini_research
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start

_PLACEHOLDER_Q_RE = re.compile(r"\bX\b|что такое X", re.I)


def _constraints_already_rich(parsed: EngineState) -> bool:
    c = (parsed.context_constraints or "").lower()
    p = parsed.user_problem.lower()
    has_stack = "lancedb" in c or "python" in c
    has_topic = "rag" in c or "кэш" in p or "cache" in p or "invalidation" in p
    return has_stack and has_topic


def _gemini_clarify(question: str, parsed: EngineState) -> str:
    payload = (
        "Ответь на русском, 5–8 буллетов, без вступления.\n\n"
        f"Контекст задачи: {parsed.user_problem}\n"
        f"Стек: {parsed.context_constraints or '(нет)'}\n\n"
        f"Вопрос: {question}"
    )
    set_status("[intent_and_clarify] уточнение → Gemini (короткий запрос)…")
    return ask_gemini_research(payload)


def intent_and_clarify_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("intent_and_clarify_node (1.5B routing)")
    parsed = EngineState.model_validate(state)

    if _constraints_already_rich(parsed):
        set_status("[intent_and_clarify] constraints достаточны — без уточнений")
        node_end("intent_and_clarify_node (1.5B routing)", "skip rich constraints")
        return {"pending_clarification": False}

    set_status("[intent_and_clarify] 1.5B: достаточно ли вводных?…")

    structured = structured_chat(ROUTER_MODEL, ClarificationAssessment, temperature=0.1)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Оцень данные для архитектурного анализа RAG/кэширования. "
            "Если в constraints есть стек (Python/LanceDB/Mac) и тема ясна — "
            "needs_clarification=false. "
            "clarification_question: конкретный вопрос с реальными терминами из задачи "
            "(LanceDB, invalidation, tail latency). Никогда не пиши букву «X» или шаблоны. "
            "Пользователя interrupt только для бизнес-контекста (SLA, объём продукта)."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача:\n{parsed.user_problem}\n\n"
            f"Ограничения:\n{parsed.context_constraints or '(пусто)'}"
        )
    )
    assessment = invoke_logged(
        structured, [system, human], "intent_and_clarify / ClarificationAssessment"
    )
    if assessment is None:
        assessment = ClarificationAssessment(needs_clarification=False)

    updates: dict[str, Any] = {"pending_clarification": False}
    history = list(parsed.external_ai_dialogue_history)

    q = (assessment.clarification_question or "").strip()
    if assessment.needs_clarification and q and _PLACEHOLDER_Q_RE.search(q):
        set_status("[intent_and_clarify] шаблонный вопрос 1.5B — пропуск уточнения")
        assessment.needs_clarification = False

    if assessment.needs_clarification and assessment.clarification_question:
        question = assessment.clarification_question
        use_gemini = CLARIFY_VIA_GEMINI and not SKIP_GEMINI

        if use_gemini:
            answer_text = _gemini_clarify(question, parsed).strip()
            history.append({"role": "assistant", "content": question})
            history.append({"role": "assistant", "content": f"[Gemini] {answer_text}"})
        else:
            set_status("[intent_and_clarify] interrupt → уточнение пользователю")
            user_answer = interrupt(
                {
                    "kind": "clarification",
                    "question": question,
                }
            )
            answer_text = str(user_answer).strip()
            history.append({"role": "assistant", "content": question})
            history.append({"role": "user", "content": answer_text})

        merged_constraints = parsed.context_constraints
        block = (
            f"Уточнение ({'Gemini' if use_gemini else 'пользователь'}): {answer_text}"
        )
        merged_constraints = (
            f"{merged_constraints}\n{block}" if merged_constraints else block
        )
        rolling = rolling_summarize_dialogue(
            EngineState.model_validate(
                {
                    **parsed.model_dump(),
                    "external_ai_dialogue_history": history,
                }
            ),
            new_user_line=answer_text,
        )
        updates.update(
            {
                "context_constraints": merged_constraints,
                "external_ai_dialogue_history": history,
                "dialogue_rolling_summary": rolling,
            }
        )

    node_end("intent_and_clarify_node (1.5B routing)", "ok")
    return updates
