"""VLM batch — Gemini Lite multimodal contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DiagramKind = Literal["architecture", "benchmark_chart", "none"]


class VlmDiagramItemContract(BaseModel):
    index: int = Field(ge=0, le=20, description="Индекс в батче 0..N-1")
    is_diagram: bool = Field(
        default=False,
        description="true если архитектура или benchmark chart",
    )
    diagram_kind: DiagramKind = Field(
        default="none",
        description="architecture | benchmark_chart | none",
    )
    title: str = Field(default="", description="Подпись схемы")
    mermaid: str = Field(default="", description="Mermaid flowchart или xychart-beta")
    summary: str = Field(default="", description="2–3 предложения на русском")

    @field_validator("title", "mermaid", "summary", mode="before")
    @classmethod
    def _str_fields(cls, v: object) -> str:
        return str(v or "").strip()

    @field_validator("diagram_kind", mode="before")
    @classmethod
    def _diagram_kind(cls, v: object) -> str:
        raw = str(v or "").strip().lower()
        if raw in ("architecture", "benchmark_chart", "none"):
            return raw
        if raw in ("chart", "benchmark", "graph", "xychart"):
            return "benchmark_chart"
        if raw in ("flowchart", "pipeline", "sequence", "uml", "er"):
            return "architecture"
        return "none"


class VlmBatchResponseContract(BaseModel):
    items: list[VlmDiagramItemContract] = Field(
        default_factory=list,
        description="Один item на каждое изображение в батче",
    )
