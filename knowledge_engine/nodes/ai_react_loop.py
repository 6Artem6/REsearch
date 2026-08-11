"""Re-Act диалог 1.5B ⇄ внешний ИИ (Playwright)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAX_AI_DIALOGUE_TURNS, REQUIRE_GEMINI, ROUTER_MODEL
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import (
    AIDialogueEvaluation,
    AIDialoguePrompt,
    EngineGraphState,
    EngineState,
)
from knowledge_engine.services.ai_dialogue.gemini_session import (
    BrowserGeminiDialogueSession,
)
from knowledge_engine.src.processors.question_formation_rules import (
    QUESTION_FORMATION_RULES,
)
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def _build_initial_prompt(state: EngineState) -> AIDialoguePrompt:
    structured = structured_chat(ROUTER_MODEL, AIDialoguePrompt, temperature=0.15)
    abs_text = "\n".join(f"- {a.title} ({a.cs_concept})" for a in state.abstractions)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Внешний ИИ (Gemini) — основная research-работа. В system_prompt для Gemini: "
            f"{GEMINI_RUSSIAN_ROLE} "
            "Требуй: SOTA (ArXiv), infra, prod-практики, URL, failure modes, trade-offs, "
            "метрики RAM/latency на Apple Silicon. user_message — конкретная задача."
        )
    )
    human = HumanMessage(
        content=(
            f"Задача: {state.user_problem}\n"
            f"Ограничения: {state.context_constraints}\n"
            f"CS-абстракции:\n{abs_text}\n"
        )
    )
    result = invoke_logged(
        structured, [system, human], "ai_react_loop / AIDialoguePrompt"
    )
    if result is None:
        raise RuntimeError("ai_react_loop: не удалось сформировать промпт")
    return result


def _evaluate_response(state: EngineState, answer: str) -> AIDialogueEvaluation:
    structured = structured_chat(ROUTER_MODEL, AIDialogueEvaluation, temperature=0.1)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Оцени ответ внешнего ИИ: достаточно ли ссылок и фактов для Trade-off матрицы. "
            "Если нет — один уточняющий вопрос на русском (follow_up_question).\n\n"
            f"{QUESTION_FORMATION_RULES}"
        )
    )
    human = HumanMessage(
        content=f"Задача: {state.user_problem}\n\nОтвет внешнего ИИ:\n{answer[:6000]}"
    )
    result = invoke_logged(
        structured, [system, human], "ai_react_loop / AIDialogueEvaluation"
    )
    return result


def ai_react_loop_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("ai_react_loop_node (Gemini Playwright)")
    parsed = EngineState.model_validate(state)
    set_status("[ai_react_loop] BrowserGeminiDialogueSession (.browser_state/)…")
    session = BrowserGeminiDialogueSession()
    collected_urls = list(parsed.collected_urls)
    found_facts = list(parsed.found_facts)

    try:
        prompt_pack = _build_initial_prompt(parsed)
        first_message = f"{prompt_pack.system_prompt}\n\n{prompt_pack.user_message}"
        answer = session.send(first_message)

        for turn in range(MAX_AI_DIALOGUE_TURNS):
            evaluation = _evaluate_response(parsed, answer)
            for url in session.extract_reference_urls(answer):
                if url not in collected_urls:
                    collected_urls.append(url)
            for url in evaluation.extracted_urls:
                if url not in collected_urls:
                    collected_urls.append(url)
            for fact in evaluation.extracted_facts:
                if fact not in found_facts:
                    found_facts.append(fact)

            if evaluation.has_sufficient_links:
                set_status("[Dialogue] роутер: достаточно ссылок и фактов")
                break
            if not evaluation.follow_up_question:
                break
            set_status(
                f"[Dialogue] уточнение #{turn + 1}: {evaluation.follow_up_question[:80]}…"
            )
            answer = session.send(evaluation.follow_up_question)

    finally:
        session.close()

    history = [t for t in session.as_chat_dicts()]
    assistant_chars = sum(
        len(t.get("content", "")) for t in history if t.get("role") == "assistant"
    )
    if REQUIRE_GEMINI and assistant_chars < 200:
        node_end("ai_react_loop_node (Gemini Playwright)", "FAIL: пустой диалог")
        raise RuntimeError(
            "Gemini: нет содержательного ответа. Если чат не открыт — один раз "
            "python -m knowledge_engine.main browser-login (гость или Google)."
        )

    sufficient = (
        len(collected_urls) >= 2 or len(found_facts) >= 4 or assistant_chars >= 1500
    )
    node_end(
        "ai_react_loop_node (Gemini Playwright)",
        f"urls={len(collected_urls)}, facts={len(found_facts)}",
    )
    return {
        "external_ai_dialogue_history": history,
        "collected_urls": collected_urls,
        "found_facts": found_facts,
        "is_facts_sufficient": sufficient,
        "search_iterations": parsed.search_iterations + 1,
    }
