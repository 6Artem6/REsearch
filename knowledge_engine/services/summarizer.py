"""Персонализированное сжатие статей в DocumentSummary."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import (
    MAIN_MODEL,
    SUMMARIZER_MAX_INPUT_CHARS,
    SUMMARIZER_MAX_PROFILE_CHARS,
    USER_PROFILE_PATH,
)
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.ui.logger import set_status


def _load_user_profile() -> str:
    path = Path(USER_PROFILE_PATH)
    if not path.is_file():
        return "(user_profile.md не найден)"
    return path.read_text(encoding="utf-8")[:SUMMARIZER_MAX_PROFILE_CHARS]


def summarize_article(
    title: str,
    url: str,
    raw_text: str,
    diagram_descriptions: list[str] | None = None,
) -> DocumentSummary:
    """
    Сжать текст с учётом user_profile.md → структура DocumentSummary.
    """
    profile = _load_user_profile()
    diagrams = diagram_descriptions or []
    set_status(f"[Summarizer] сжатие: {title[:60]}…")

    structured = structured_chat(MAIN_MODEL, DocumentSummary, temperature=0.25)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_OUTPUT_RULE} "
            "Ты research-аналитик. Сжимай материал строго под интересы разработчика "
            "из профиля. Выход — JSON DocumentSummary. diagram_descriptions дополни "
            "переданными описаниями схем."
        )
    )
    human = HumanMessage(
        content=(
            f"Профиль разработчика:\n{profile}\n\n"
            f"URL: {url}\nTitle: {title}\n\n"
            f"Описания диаграмм (если есть):\n{diagrams}\n\n"
            f"Исходный текст (фрагмент):\n{raw_text[:SUMMARIZER_MAX_INPUT_CHARS]}"
        )
    )
    result = invoke_logged(
        structured, [system, human], f"summarizer / DocumentSummary ({title[:40]})"
    )
    if result is None:
        raise RuntimeError("Summarizer: structured output returned None")
    # URL/title из источника — надёжнее, чем из модели
    result.url = url
    if not result.title:
        result.title = title
    if diagrams:
        merged = list(result.diagram_descriptions)
        for d in diagrams:
            if d not in merged:
                merged.append(d)
        result.diagram_descriptions = merged
    return result


def summarize_gemini_bundle(
    user_problem: str,
    dialogue_history: list[dict[str, str]],
    api_snippets: list[str],
) -> DocumentSummary:
    """Один вызов 7B: сжать ответы Gemini + сниппеты API для LanceDB."""
    profile = _load_user_profile()
    parts: list[str] = []
    for turn in dialogue_history:
        if turn.get("role") == "assistant":
            parts.append(turn.get("content", "")[:4000])
    bundle = "\n---\n".join(parts)
    if api_snippets:
        bundle += "\n\nAPI snippets:\n" + "\n".join(api_snippets[:12])

    set_status("[Summarizer] один сжатый пакет: Gemini + API (не N×URL)…")
    structured = structured_chat(MAIN_MODEL, DocumentSummary, temperature=0.25)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_OUTPUT_RULE} "
            "Сжать исследовательский пакет (ответы Gemini + сниппеты поиска) в один "
            "DocumentSummary для Trade-off матрицы. JSON DocumentSummary."
        )
    )
    human = HumanMessage(
        content=(
            f"Профиль:\n{profile}\n\nЗадача: {user_problem}\n\n"
            f"Пакет:\n{bundle[:SUMMARIZER_MAX_INPUT_CHARS]}"
        )
    )
    result = invoke_logged(
        structured, [system, human], "summarizer / Gemini bundle DocumentSummary"
    )
    if result is None:
        raise RuntimeError("Summarizer: Gemini bundle returned None")
    result.url = "gemini-research-bundle"
    if not result.title:
        result.title = "Gemini + search API research bundle"
    return result
