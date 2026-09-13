"""Reasoner final response — Gemini contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinalResponseContract(BaseModel):
    user_final_answer: str = Field(
        ...,
        description="Готовый глубокий ответ для пользователя (Markdown)",
    )
    fact_nuggets: list[str] = Field(
        default_factory=list,
        max_length=24,
        description="Короткие fact nuggets для LightRAG",
    )
