"""Qwen 1.5B: оценка выжимки Gemini vs user_profile.md + LanceDB."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import ROUTER_MODEL
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import (
    DocumentSummary,
    EngineGraphState,
    EngineState,
    GeminiSourceExtraction,
    ProfileValidationResult,
)
from knowledge_engine.services.context_manager import load_personal_orchestrator_focus
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.prompts.engineering_context import GLOBAL_ENGINEERING_CRITERIA
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start


def profile_validator_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("profile_validator_node (1.5B)")
    parsed = EngineState.model_validate(state)
    if not parsed.last_extraction:
        node_end("profile_validator_node", "skip (no extraction)")
        return {"last_validator_signal": "REJECTED: нет выжимки источника"}

    extraction = GeminiSourceExtraction.model_validate(parsed.last_extraction)
    profile = load_personal_orchestrator_focus()

    structured = structured_chat(ROUTER_MODEL, ProfileValidationResult, temperature=0.1)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Ты Personal Profile Validator. Оцени выжимку источника по инженерным критериям "
            "и личному фокусу разработчика:\n"
            f"{GLOBAL_ENGINEERING_CRITERIA}\n"
            "- Практическая ценность для заявленного фокуса?\n"
            "- Реальные failure modes (OOM, tail latency, stale reads)?\n"
            "- Нет ли маркетинговой абстракции?\n"
            "Если ценно: is_valuable=true, actions включает save_to_lancedb. "
            "validator_signal — коротко для Gemini (VALIDATED/REJECTED + аспекты), "
            "НЕ возвращай полный текст статьи."
        )
    )
    human = HumanMessage(
        content=(
            f"Профиль:\n{profile}\n\n"
            f"Задача: {parsed.user_problem}\n\n"
            f"Выжимка Gemini:\n"
            f"URL: {extraction.source_url}\n"
            f"Findings: {extraction.key_engineering_findings}\n"
            f"Failure modes: {extraction.extracted_failure_modes}\n"
            f"Next: {extraction.proposed_next_steps}"
        )
    )
    validation = invoke_logged(
        structured, [system, human], "profile_validator / ProfileValidationResult"
    )
    if validation is None:
        validation = ProfileValidationResult(
            is_valuable=False,
            reason="Не удалось оценить",
            actions=["continue_deep_dive"],
            validator_signal="REJECTED: ошибка валидатора",
        )

    updates: dict[str, Any] = {
        "last_validation": validation.model_dump(),
        "last_validator_signal": validation.validator_signal,
        "research_source_index": parsed.research_source_index + 1,
    }

    facts = list(parsed.found_facts)
    sig = validation.validator_signal
    if sig not in facts:
        facts.append(sig[:500])
    updates["found_facts"] = facts

    summaries = list(parsed.found_summaries)
    validated_count = parsed.validated_source_count

    if validation.is_valuable and "save_to_lancedb" in validation.actions:
        set_status("[profile_validator] ценный источник → LanceDB…")
        fm = [
            s.strip()
            for s in extraction.extracted_failure_modes.split("\n")
            if s.strip()
        ]
        takeaways = [
            s.strip()
            for s in extraction.key_engineering_findings.split("\n")
            if s.strip()
        ]
        summary = DocumentSummary(
            title=extraction.source_url[:80],
            url=extraction.source_url,
            key_takeaways=takeaways[:8] or [extraction.key_engineering_findings[:400]],
            failure_modes=fm[:8],
            cs_concepts=[],
        )
        store = VectorStore()
        store.save_summary(summary)
        if not any(s.url == summary.url for s in summaries):
            summaries.append(summary)
        validated_count += 1
        updates["found_summaries"] = [s.model_dump() for s in summaries]
        updates["validated_source_count"] = validated_count
        updates["is_facts_sufficient"] = validated_count >= 1

    node_end(
        "profile_validator_node (1.5B)",
        f"valuable={validation.is_valuable}, validated={validated_count}",
    )
    return updates
