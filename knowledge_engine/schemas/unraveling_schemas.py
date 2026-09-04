"""Structured-output contracts for trade-off Unraveling."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def _word_count(text: str) -> int:
    return len((text or "").split())


class TradeoffFailureMode(BaseModel):
    """One failure / bottleneck case for the selected trade-off option."""

    scenario: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Scenario in which the failure or bottleneck appears",
    )
    impact: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Impact on RAM, CPU, latency, or data integrity",
    )
    mitigation: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Engineering mitigation or workaround pattern",
    )


class UnravelingNodeResponse(BaseModel):
    """Validated Unraveling payload — Host assembles UI markdown."""

    summary: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description="Short engineering summary of the conclusion",
    )
    ram_and_latency_impact: str = Field(
        ...,
        min_length=20,
        max_length=8000,
        description=(
            "Detailed memory-load and latency analysis "
            "(Mac M-series / Apple Silicon)"
        ),
    )
    failure_modes: list[TradeoffFailureMode] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Critical failure points and edge cases",
    )
    technical_breakdown_markdown: str = Field(
        ...,
        min_length=300,
        max_length=24_000,
        description=(
            "Deep technical treatment with algorithms and code/config (300+ words)"
        ),
    )

    @field_validator("technical_breakdown_markdown")
    @classmethod
    def breakdown_must_be_dense(cls, value: str) -> str:
        words = _word_count(value)
        if words < 300:
            raise ValueError(
                f"technical_breakdown_markdown must contain at least 300 words "
                f"(got {words})"
            )
        return value
