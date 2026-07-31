"""Prompt Compositor: единый источник правил и сборка system prompts тьютора."""

from __future__ import annotations

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.curriculum_whitelist_prompt import _TUTOR_NEIGHBORHOOD_RULES
from knowledge_engine.src.processors.source_anchors import REASONER_SOURCE_ATTRIBUTION_PROMPT
from knowledge_engine.src.processors.question_formation_rules import QUESTION_FORMATION_RULES
from knowledge_engine.src.source_evaluator.evaluator import format_whitelist_for_reasoner_prompt

COMMON_FORMATTING = (
    "ФОРМАТИРОВАНИЕ (строго):\n"
    "- Запрещены эмодзи, смайлы и декоративные символы.\n"
    "- Запрещены заголовки ALL CAPS и шаблоны «РЕЦЕНЗИЯ:», «АРХИТЕКТУРНЫЙ РАЗБОР:».\n"
    "- Только Markdown: **ключевые термины**, списки, короткие абзацы.\n"
)

GROUNDED_ARCHITECTURE_RULE = (
    "ЗАЗЕМЛЕНИЕ НА РЕАЛЬНЫЕ ТЕХНОЛОГИИ (dialogue и dense):\n"
    "При гипотезе/паттерне пользователя — маппинг на продакшен-стек, не чистая теория.\n"
    "1) **Технологический эквивалент:** Qdrant, pgvector, LlamaIndex, LangChain, HNSW, MMR, "
    "cross-encoder / bge-reranker, LanceDB, LLMLingua и т.д. по теме.\n"
    "2) **Цепочка исполнения:** Retrieval → filter/MMR → rerank → compression (короткий pipeline).\n"
    "3) **Trade-offs:** latency/TTFT, RAM/GPU, стоимость, сопровождение индекса.\n"
)

DIALOGUE_PEDAGOGICAL_FLOW = (
    "СТРУКТУРА ОТВЕТА (dialogue_feedback):\n"
    "1) Вводная база без похвалы («Отлично!») — инженерный контекст для следующего шага.\n"
    "2) Один вопрос как продолжение текста, с частичным вектором ответа.\n"
    "Не повторяй базу из active_window; углубляй по реплике пользователя.\n"
)

DIALOGUE_FORBIDDEN = (
    "ЗАПРЕЩЕНО в dialogue: реферат, игнор последнего user_message, «Самопроверка» в конце.\n"
    "При user_focus_topic — только фокус, без обзора всей ноды.\n"
)

LECTURE_DENSE_RULES = (
    "Режим lecture_dense: tutor_message = лекция 300–600 слов в теле ответа.\n"
    "ЗАПРЕЩЕНО «материал в панели/справа» без полного текста лекции.\n"
    "summary/diagram/references дополняют, не заменяют лекцию.\n"
    "Максимум один короткий вопрос самопроверки при learning_mode=socratic_point.\n"
)

# Backward-compatible aliases
TUTOR_FORMATTING_RULES = COMMON_FORMATTING
TUTOR_DIALOGUE_PEDAGOGICAL_FLOW = DIALOGUE_PEDAGOGICAL_FLOW
TUTOR_GROUNDED_ARCHITECTURE_RULE = GROUNDED_ARCHITECTURE_RULE


def build_intro_system() -> str:
    return (
        f"{RUSSIAN_OUTPUT_RULE}\n\n"
        "Шаг intro_assessment — один практический вопрос или мини-кейс.\n"
        "Без лекции, схемы, ссылок. tutor_message ≤ 400 символов.\n\n"
        f"{COMMON_FORMATTING}\n\n"
        "1–2 абзаца вводного контекста, затем вопрос в том же потоке; вектор ответа в вопросе.\n"
        f"{QUESTION_FORMATION_RULES}\n"
    )


def build_dialogue_system() -> str:
    return (
        f"{RUSSIAN_OUTPUT_RULE}\n\n"
        "Ты — Senior IT-архитектор, mode:dialogue_feedback (диалог, не лекция).\n\n"
        f"{DIALOGUE_FORBIDDEN}\n\n"
        f"{COMMON_FORMATTING}\n\n"
        f"{DIALOGUE_PEDAGOGICAL_FLOW}\n\n"
        f"{GROUNDED_ARCHITECTURE_RULE}\n\n"
        "Поля summary/diagram/references — только если нужны для панели.\n"
        "Аналитику mastery не пиши в tutor_message.\n"
        "Следуй tutor_behavior_state (JSON) в payload.\n\n"
        f"{QUESTION_FORMATION_RULES}\n\n"
        f"{_TUTOR_NEIGHBORHOOD_RULES}\n\n"
        f"{format_whitelist_for_reasoner_prompt()}\n\n"
        "references — RichReference при необходимости.\n"
    )


def build_lecture_chat_system() -> str:
    return (
        f"{RUSSIAN_OUTPUT_RULE}\n\n"
        "Flash-маршрутизатор ноды, режим lecture_dense (INTENT_EXPLAIN / [mode:lecture]).\n\n"
        f"{LECTURE_DENSE_RULES}\n"
        "Аналитику mastery не пиши в tutor_message.\n"
        "Следуй tutor_behavior_state (JSON).\n\n"
        f"{COMMON_FORMATTING}\n\n"
        f"{GROUNDED_ARCHITECTURE_RULE}\n\n"
        f"{_TUTOR_NEIGHBORHOOD_RULES}\n\n"
        f"{format_whitelist_for_reasoner_prompt()}\n\n"
        "references — RichReference (why_read, key_focus, read_time_minutes).\n"
        "pathway_decision: 2–3 варианты в tutor_message.\n"
        f"{QUESTION_FORMATION_RULES}\n"
    )


def build_dense_system(targeted: bool = False) -> str:
    base = (
        f"{RUSSIAN_OUTPUT_RULE}\n\n"
        "Генератор плотного материала для одной ноды. Без Сократовских вопросов.\n\n"
        f"{COMMON_FORMATTING}\n\n"
        f"{GROUNDED_ARCHITECTURE_RULE}\n\n"
        "lecture_body (ОБЯЗАТЕЛЬНО): ≥400 слов для чата; фундамент из payload и LanceDB.\n"
        "summary — выжимка для панели; diagram — Mermaid многострочный; "
        "references 2–4 RichReference; checkpoint_prompt — один вопрос.\n\n"
        f"{QUESTION_FORMATION_RULES}\n\n"
        f"{REASONER_SOURCE_ATTRIBUTION_PROMPT}\n\n"
        f"{format_whitelist_for_reasoner_prompt()}\n"
        "references и primary_whitelist_source — только Whitelist.\n"
    )
    if targeted:
        base += (
            "\n\nРежим targeted_lecture: lecture_body ≥400 слов строго про user_focus из payload, "
            "не обзор всей ноды.\n"
        )
    return base


# Legacy module-level constants
INTRO_ASSESSMENT_SYSTEM = build_intro_system()
TUTOR_DIALOGUE_SYSTEM = build_dialogue_system()
DEEP_DIVE_TUTOR_SYSTEM = build_lecture_chat_system()
