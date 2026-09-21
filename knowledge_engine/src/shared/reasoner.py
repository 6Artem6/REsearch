"""Reasoner final response — Gemini contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinalResponseContract(BaseModel):
    user_final_answer: str = Field(
        ...,
        description="Finished in-depth answer for the user (Markdown)",
        # RU: готовый глубокий ответ для пользователя.
    )
    fact_nuggets: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Short fact nuggets for LightRAG",
        # RU: короткие факты для LightRAG.
    )
