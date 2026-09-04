"""Pydantic contracts for research-contour helpers (harvest, clarify, REPL)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HarvestedLinkItem(BaseModel):
    """One practical article / document recommended for a curriculum goal."""

    title: str = Field(
        ...,
        min_length=4,
        max_length=400,
        description="Article or document title",
    )
    url: str = Field(
        ...,
        min_length=12,
        max_length=2000,
        description="Direct resource URL",
    )
    relevance_reason: str = Field(
        ...,
        min_length=12,
        max_length=1200,
        description="Why this source is relevant to the topic",
    )

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, value: str) -> str:
        text = (value or "").strip()
        if not text.startswith("http://") and not text.startswith("https://"):
            raise ValueError("url must be an http(s) URL")
        return text.split("#")[0].rstrip("/")


class HarvestedLinksResponse(BaseModel):
    """Structured harvest payload — Host maps items to CurriculumSearchHit."""

    items: list[HarvestedLinkItem] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Authoritative engineering articles (not homepages)",
    )


class ClarificationConstraintsResponse(BaseModel):
    """Technical constraints produced instead of free-text clarification bullets."""

    constraints: list[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Clear technical constraints and research givens",
    )

    @field_validator("constraints")
    @classmethod
    def constraints_must_be_nonempty(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in (value or []) if (c or "").strip()]
        if len(cleaned) < 3 or len(cleaned) > 8:
            raise ValueError("constraints must contain 3–8 non-empty strings")
        return cleaned


def render_clarification_constraints(
    payload: ClarificationConstraintsResponse,
) -> str:
    """Host-owned bullet list for merging into context_constraints."""
    return "\n".join(f"- {c}" for c in payload.constraints)


class ReplFollowUpResponse(BaseModel):
    """v0.7 REPL follow-up — Host uses `answer` as the user-facing string."""

    answer: str = Field(
        ...,
        min_length=20,
        max_length=24_000,
        description="Technical answer grounded in RESEARCH CONTEXT",
    )
    sources_cover_question: bool = Field(
        default=True,
        description="False when the loaded sources do not cover the question",
    )
