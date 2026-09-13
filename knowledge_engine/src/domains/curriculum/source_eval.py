"""Source evaluator Lite — Gemini contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceEvaluatorLiteContract(BaseModel):
    status: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="APPROVED | REJECTED",
    )
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Причина на русском")
    suggested_action: Literal["RETRY_WITH_NEW_SOURCE", "REMOVE_LINK", "KEEP"] = Field(
        default="KEEP",
    )
