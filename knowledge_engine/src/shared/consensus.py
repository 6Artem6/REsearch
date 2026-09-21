"""Consensus / validator — Gemini Lite contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AcademicQueryContract(BaseModel):
    academic_query_en: str = Field(
        ...,
        description="Clean English CS query for Consensus.app",
    )
    notes: str = Field(
        default="",
        description="What was stripped/translated from the user question",
        # RU: что убрано/переведено из вопроса пользователя.
    )


class ConsensusDocContract(BaseModel):
    title: str = Field(default="")
    url: str = Field(default="")
    snippet: str = Field(default="")
    source_anchor: str = Field(default="", description="Sx id if known")


class ValidationResultContract(BaseModel):
    status: Literal["OK", "REJECT", "RETRY"] = Field(
        ...,
        description="OK | REJECT | RETRY",
    )
    docs: list[ConsensusDocContract] = Field(default_factory=list)
    refinement_prompt: str | None = Field(
        default=None,
        description="English follow-up for Consensus retry",
    )
    reason: str = Field(default="", description="Explanation of the status")
    # RU: объяснение статуса.


class ProfileApplicabilityContract(BaseModel):
    apply_personal_profile: bool = Field(
        ...,
        description="Whether the Reasoner needs the personal profile",
        # RU: нужен ли personal profile в Reasoner.
    )
    context_applicability: str = Field(
        default="",
        description="general_academic | engineering_practice | project_specific",
    )
    reason: str = Field(default="")


class RefinementSanitizeContract(BaseModel):
    academic_query_en: str = Field(
        ...,
        description="Cleaned English academic query (one line)",
        # RU: очищенный английский academic query (одна строка).
    )
