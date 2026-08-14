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


class ParagraphAnalysis(BaseModel):
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


class PaperStructureAnalysis(BaseModel):
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
    paragraphs: list[ParagraphAnalysis] = Field(
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
