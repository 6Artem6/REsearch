"""Pre-Ingest Discovery Manifest — универсальные PDF-кандидаты без доменных хардкодов."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PDFCandidate(BaseModel):
    url: str
    kind: Literal["direct_pdf", "html_reader"] = Field(
        ...,
        description=(
            "direct_pdf — прямая ссылка на файл; "
            "html_reader — HTML-страница просмотрщика (epdf/ReadCube/Viewer)."
        ),
    )
    source_type: Literal[
        "meta_tag",
        "dom_anchor",
        "unpaywall",
        "scihub",
        "llm_validated",
    ]
    priority: int = Field(..., ge=1, le=5)


class ArticleResourceManifest(BaseModel):
    source_id: str = ""
    canonical_url: str
    doi: Optional[str] = None
    pdf_candidates: List[PDFCandidate] = Field(default_factory=list)
    selected_pdf_url: Optional[str] = None
    has_diagrams_extracted: bool = False
    html_snapshot: Optional[str] = None
    fetched_pdf_bytes: Optional[bytes] = None

    model_config = {"arbitrary_types_allowed": True}


class PDFLinkValidationResponse(BaseModel):
    best_pdf_url: Optional[str] = Field(
        None,
        description="URL файла или страницы-просмотрщика",
    )
    kind: Literal["direct_pdf", "html_reader"] = Field(
        "direct_pdf",
        description="Тип найденного ресурса",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Причина выбора ссылки")
