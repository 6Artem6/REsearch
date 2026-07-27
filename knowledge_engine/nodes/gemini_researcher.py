"""Gemini Deep Researcher: find → extract → (validator loop) → final matrix."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import (
    MAX_RESEARCH_SOURCES,
    ROUTER_MODEL,
    SKIP_GEMINI,
)
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_ROUTER_RULE
from knowledge_engine.nodes.decomposition import decomposition_node
from knowledge_engine.nodes.matrix import matrix_node
from knowledge_engine.schemas import (
    EngineGraphState,
    EngineState,
    GeminiSourceExtraction,
    GeminiSourceList,
)
from knowledge_engine.services.analysis_report_structure import (
    structure_analysis_report,
)
from knowledge_engine.services.gemini_research_session import (
    ask_gemini_research,
    close_gemini_research_session,
    research_dialogue_history,
)
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start

_URL_RE = re.compile(r"https?://[^\s\]<\"')]+")


def _parse_source_list(raw: str) -> GeminiSourceList:
    structured = structured_chat(ROUTER_MODEL, GeminiSourceList, temperature=0.05)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Из ответа Gemini извлеки список URL. Только реальные http(s) ссылки из текста."
        )
    )
    human = HumanMessage(content=raw[:8000])
    result = invoke_logged(
        structured, [system, human], "gemini_research / SourceList parse"
    )
    if result is None or not result.urls:
        urls = _URL_RE.findall(raw)
        return GeminiSourceList(
            urls=urls[:MAX_RESEARCH_SOURCES], search_notes="extracted regex"
        )
    return result


def _parse_extraction(raw: str, url: str) -> GeminiSourceExtraction:
    structured = structured_chat(ROUTER_MODEL, GeminiSourceExtraction, temperature=0.05)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Структурируй разбор источника из ответа Gemini. Поля на русском."
        )
    )
    human = HumanMessage(content=f"URL: {url}\n\nОтвет Gemini:\n{raw[:10000]}")
    result = invoke_logged(
        structured, [system, human], "gemini_research / SourceExtraction parse"
    )
    if result is None:
        return GeminiSourceExtraction(
            source_url=url,
            key_engineering_findings=raw[:2000],
            extracted_failure_modes="",
            proposed_next_steps="",
        )
    if not result.source_url:
        result.source_url = url
    return result


def _build_find_sources_prompt(state: EngineState) -> str:
    signals = state.last_validator_signal or "(нет сигналов валидатора)"
    return (
        f"{GEMINI_RUSSIAN_ROLE}\n\n"
        "[ШАГ A — ПОИСК ИСТОЧНИКОВ]\n"
        f"Задача: {state.user_problem}\n"
        f"Ограничения: {state.context_constraints}\n\n"
        f"Сигналы валидатора 1.5B (не полные тексты):\n{signals}\n\n"
        "Дай список до "
        f"{MAX_RESEARCH_SOURCES} конкретных URL: eng-блоги, postmortems, RFC, papers, Habr. "
        "Без маркетинга. Формат: нумерованный список URL + 1 строка почему источник."
    )


def _build_extract_prompt(state: EngineState, url: str) -> str:
    return (
        f"{GEMINI_RUSSIAN_ROLE}\n\n"
        "[ШАГ B — ГЛУБОКИЙ РАЗБОР ОДНОГО ИСТОЧНА]\n"
        f"Задача: {state.user_problem}\n"
        f"URL: {url}\n\n"
        "Открой/проанализируй источник. Ответ структурируй:\n"
        "1) key_engineering_findings — инженерные нюансы\n"
        "2) extracted_failure_modes — OOM, stale, tail latency и т.д.\n"
        "3) proposed_next_steps — что проверить дальше\n"
        "Коротко, тезисно, на русском. Без воды."
    )


def _build_final_synthesis_prompt(state: EngineState) -> str:
    validated_lines = []
    for s in state.found_summaries[-10:]:
        validated_lines.append(
            f"- {s.title} ({s.url}): {'; '.join(s.key_takeaways[:3])}"
        )
    signals = state.last_validator_signal or ""
    block = "\n".join(validated_lines) or "(используй сигналы валидатора)"
    return (
        f"{GEMINI_RUSSIAN_ROLE}\n\n"
        "[ШАГ C — ФИНАЛЬНЫЙ СИНТЕЗ]\n"
        f"Задача: {state.user_problem}\n"
        f"Ограничения: {state.context_constraints}\n\n"
        "[VALIDATED SOURCES — только выжимки, не полные статьи]\n"
        f"{block}\n\n"
        f"Последний сигнал валидатора: {signals}\n\n"
        "Построй Trade-off матрицу (Классика / SOTA / Минимализм), failure modes, "
        "RAM/latency Apple Silicon. Всё на русском."
    )


def gemini_find_sources_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("gemini_find_sources_node (Step A)")
    parsed = EngineState.model_validate(state)
    if SKIP_GEMINI:
        node_end("gemini_find_sources_node", "SKIP_GEMINI")
        return {}

    set_status("[Gemini Research] Step A: список целевых URL…")
    raw = ask_gemini_research(_build_find_sources_prompt(parsed))
    source_list = _parse_source_list(raw)
    urls = [u.rstrip(".,);]") for u in source_list.urls if u.startswith("http")]
    urls = urls[:MAX_RESEARCH_SOURCES]

    node_end("gemini_find_sources_node", f"urls={len(urls)}")
    return {
        "research_source_urls": urls,
        "research_source_index": 0,
        "research_find_rounds": parsed.research_find_rounds + 1,
        "collected_urls": list(dict.fromkeys(parsed.collected_urls + urls)),
    }


def gemini_extract_source_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("gemini_extract_source_node (Step B)")
    parsed = EngineState.model_validate(state)
    urls = parsed.research_source_urls
    idx = parsed.research_source_index
    if idx >= len(urls):
        node_end("gemini_extract_source_node", "no url")
        return {}

    url = urls[idx]
    set_status(f"[Gemini Research] Step B: разбор {url[:60]}…")
    raw = ask_gemini_research(_build_extract_prompt(parsed, url))
    extraction = _parse_extraction(raw, url)

    node_end("gemini_extract_source_node", extraction.source_url[:40])
    return {
        "last_extraction": extraction.model_dump(),
        "gemini_raw_response": raw,
    }


def gemini_final_matrix_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("gemini_final_matrix_node (Step C)")
    parsed = EngineState.model_validate(state)

    try:
        if SKIP_GEMINI:
            node_end("gemini_final_matrix_node", "fallback matrix")
            return matrix_node(state)

        if not parsed.abstractions:
            decomp = decomposition_node(state)
            parsed = EngineState.model_validate({**parsed.model_dump(), **decomp})

        set_status("[Gemini Research] Step C: финальная матрица…")
        raw = ask_gemini_research(_build_final_synthesis_prompt(parsed))

        report = structure_analysis_report(
            parsed, raw, log_label="gemini_research / AnalysisReport structure"
        )

        node_end("gemini_final_matrix_node", f"report ok, {len(raw)} sym")
        return {
            "report": report.model_dump(),
            "abstractions": [a.model_dump() for a in report.abstractions],
            "gemini_raw_response": raw,
            "external_ai_dialogue_history": research_dialogue_history(),
        }
    finally:
        close_gemini_research_session()
