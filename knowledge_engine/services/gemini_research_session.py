"""Одна Playwright-сессия Gemini на прогон Deep Researcher."""

from __future__ import annotations

from typing import Optional

from knowledge_engine.services.ai_dialogue.gemini_session import (
    BrowserGeminiDialogueSession,
)

_session: Optional[BrowserGeminiDialogueSession] = None


def get_gemini_research_session() -> BrowserGeminiDialogueSession:
    global _session
    if _session is None:
        _session = BrowserGeminiDialogueSession()
    return _session


def close_gemini_research_session() -> None:
    global _session
    if _session is not None:
        _session.close()
        _session = None


def ask_gemini_research(payload: str) -> str:
    return get_gemini_research_session().ask_gemini(payload)


def research_dialogue_history() -> list[dict[str, str]]:
    if _session is None:
        return []
    return _session.as_chat_dicts()
