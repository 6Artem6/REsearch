"""Pydantic schemas for Gemini paper-structure analysis (Map-Reduce pre-filter)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ParagraphPriority(str, Enum):
    """Map-Reduce routing tier for a PDF paragraph."""

    CORE = "CORE"
    CONTEXT = "CONTEXT"
    DROP = "DROP"


class SemanticLevel(str, Enum):
    """Abstraction grain of the paragraph (host weight, not LLM math)."""

    SPEC_EXACT = "SPEC_EXACT"
    CONCEPTUAL_MODEL = "CONCEPTUAL_MODEL"
    METAPHOR_ONLY = "METAPHOR_ONLY"


class TechnicalCorrectness(str, Enum):
    """Soundness of the paragraph's technical claims."""

    VERIFIED = "VERIFIED"
    SIMPLIFIED = "SIMPLIFIED"
    CONTRADICTION = "CONTRADICTION"


class InformationDensity(str, Enum):
    """How much technical content the paragraph carries."""

    HIGH = "HIGH"
    NEUTRAL = "NEUTRAL"
    WATER_OR_OPINION = "WATER_OR_OPINION"


class ExtractMode(str, Enum):
    """How many sentences of a kept paragraph to send to Map-Reduce."""

    FULL = "full"
    HEAD_1 = "head_1"
    HEAD_2 = "head_2"


# Host weights (Gemini emits names; never send these floats in the LLM schema).
SEMANTIC_LEVEL_WEIGHT: dict[SemanticLevel, float] = {
    SemanticLevel.SPEC_EXACT: 1.0,
    SemanticLevel.CONCEPTUAL_MODEL: 0.6,
    SemanticLevel.METAPHOR_ONLY: 0.2,
}
TECHNICAL_CORRECTNESS_WEIGHT: dict[TechnicalCorrectness, float] = {
    TechnicalCorrectness.VERIFIED: 1.0,
    TechnicalCorrectness.SIMPLIFIED: 0.7,
    TechnicalCorrectness.CONTRADICTION: 0.0,
}
INFORMATION_DENSITY_WEIGHT: dict[InformationDensity, float] = {
    InformationDensity.HIGH: 1.0,
    InformationDensity.NEUTRAL: 0.5,
    InformationDensity.WATER_OR_OPINION: 0.1,
}


class ParagraphStructureVerdict(BaseModel):
    """Pass 1 (structure / need): utility, relevance, Map-Reduce priority. No truth fields."""

    paragraph_id: int = Field(
        description="Unique sequential paragraph id across the full document (1-based).",
    )
    page_number: int = Field(
        description="Page number where this paragraph appears (1-based, PDF order).",
    )
    section_title: str = Field(
        description="Current section or subsection heading governing this paragraph.",
    )
    priority: ParagraphPriority = Field(
        description=(
            "CORE: core math, model architecture, key formulas, algorithms, pseudocode, "
            "and headline experimental results. "
            "CONTEXT: introductory context, domain overview, related work, general discussion. "
            "DROP: footnotes, layout noise, headers/footers, figure captions, licenses, acknowledgments."
        ),
    )
    topic_relevance: int = Field(
        ge=0,
        le=10,
        description="Relevance to the target topic (0 irrelevant, 10 directly answers the task).",
    )
    reason: str = Field(
        description="Brief justification for the assigned priority and relevance score.",
    )


class ParagraphCredibility(BaseModel):
    """Pass 2 LLM row: intrinsic vector traits of one remaining paragraph."""

    paragraph_id: int = Field(
        description="Paragraph id from remaining_paragraphs. Do not invent ids.",
    )
    semantic_level: SemanticLevel = Field(
        description=(
            "SPEC_EXACT: precise runtime mechanics, flags, data structures, formal specs. "
            "CONCEPTUAL_MODEL: accurate high-level system interactions without low-level detail. "
            "METAPHOR_ONLY: everyday analogy or metaphor without technical depth."
        ),
    )
    # RU: зерно абстракции параграфа (спека / модель / метафора).
    technical_correctness: TechnicalCorrectness = Field(
        description=(
            "VERIFIED: technically sound. "
            "SIMPLIFIED: pedagogical shortcut that does not invert the mechanism. "
            "CONTRADICTION: hard error or mixed abstraction (e.g. user-space eval vs hardware IRQ)."
        ),
    )
    # RU: техническая корректность; CONTRADICTION обнуляет P_i на хосте.
    information_density: InformationDensity = Field(
        description=(
            "HIGH: facts, algorithms, state transitions. "
            "NEUTRAL: informational context. "
            "WATER_OR_OPINION: fluff, intros, unverified commentary."
        ),
    )
    # RU: плотность полезного технического содержания.
    extract_mode: ExtractMode = Field(
        default=ExtractMode.FULL,
        description=(
            "Volume of sentences to keep. Independent of priority/importance. "
            "full: technical load is spread across the paragraph. "
            "head_1: load-bearing fact is in the 1st sentence; rest is repetition/water. "
            "head_2: load-bearing fact + 1 essential condition in first 2 sentences. "
            "FORBIDDEN: mapping CORE->full or CONTEXT->head_1 automatically."
        ),
    )
    # RU: сколько предложений взять из абзаца; не путать с CORE/CONTEXT/DROP.
    reason: str = Field(
        description="Brief English justification, required especially for CONTRADICTION.",
    )
    # RU: краткая причина оценки (обязательна при CONTRADICTION).


class PaperCredibilityAnalysis(BaseModel):
    """Pass-2 Flash Lite response: one row per remaining paragraph id."""

    paragraphs: list[ParagraphCredibility] = Field(
        default_factory=list,
        description="Credibility rows for every remaining paragraph_id.",
    )


class ParagraphAnalysis(ParagraphStructureVerdict):
    """Host merge of pass 1 (need) + pass 2 (vector traits). Pass 1 LLM never sees pass-2 fields."""

    semantic_level: SemanticLevel | None = Field(
        default=None,
        description="Pass-2 abstraction grain. Absent until credibility audit runs.",
    )
    technical_correctness: TechnicalCorrectness | None = Field(
        default=None,
        description="Pass-2 soundness. Absent until credibility audit runs.",
    )
    information_density: InformationDensity | None = Field(
        default=None,
        description="Pass-2 density. Absent until credibility audit runs.",
    )
    extract_mode: ExtractMode | None = Field(
        default=None,
        description=(
            "Pass-2 sentence volume. Absent until credibility audit runs; "
            "host treats None as full."
        ),
    )
    accuracy_reason: str = Field(
        default="",
        description="Pass-2 justification. Empty until credibility audit runs.",
    )
    # RU: оси прохода 2; None = аудит не выполнялся.

    def as_credibility(self) -> ParagraphCredibility | None:
        if (
            self.semantic_level is None
            or self.technical_correctness is None
            or self.information_density is None
        ):
            return None
        return ParagraphCredibility(
            paragraph_id=self.paragraph_id,
            semantic_level=self.semantic_level,
            technical_correctness=self.technical_correctness,
            information_density=self.information_density,
            extract_mode=self.extract_mode or ExtractMode.FULL,
            reason=self.accuracy_reason or "scored",
        )


class PaperStructureAnalysis(BaseModel):
    """Pass 1 LLM contract: structure only (ParagraphStructureVerdict rows)."""

    references_start_page: Optional[int] = Field(
        default=None,
        description="First page of the References/Bibliography section, if detected.",
    )
    drop_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Page numbers to remove entirely (references, raw-data appendices, author bios)."
        ),
    )
    paragraphs: list[ParagraphStructureVerdict] = Field(
        default_factory=list,
        description="Per-paragraph utility, relevance, and Map-Reduce priority labels.",
    )


class InputPaperParagraph(BaseModel):
    paragraph_id: int = Field(description="Sequential paragraph id in the document.")
    section_title: str = Field(description="Section heading active for this paragraph.")
    text: str = Field(description="Paragraph body text extracted from the PDF.")


class InputPaperPage(BaseModel):
    page_number: int = Field(description="PDF page number (1-based).")
    paragraphs: list[InputPaperParagraph] = Field(
        default_factory=list,
        description="Text paragraphs detected on this page.",
    )


class InputPaperJson(BaseModel):
    total_pages: int = Field(description="Total number of pages in the PDF.")
    pages: list[InputPaperPage] = Field(
        default_factory=list,
        description="Per-page paragraph lists in reading order.",
    )
