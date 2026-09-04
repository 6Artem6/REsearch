"""Flash Lite contracts for Exa query expansion and domain authority."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ExaSearchIntent = Literal["language_api", "architecture", "mixed"]
ExaSearchType = Literal["auto", "keyword", "neural"]
ExaApiCategory = Literal["company", "research paper", "news", "github", "pdf"]
DomainAuthorityClass = Literal[
    "OFFICIAL_DOCS",
    "VENDOR_BLOG",
    "ACADEMIC_OR_PAPER",
    "COMMUNITY_BLOG",
    "SPAM_AGGREGATOR",
]
# RU: таксономия авторитетности; Pass 1 Exa берёт только OFFICIAL_DOCS.

AUTHORITY_KEEP_CLASSES: frozenset[str] = frozenset(
    {"OFFICIAL_DOCS", "VENDOR_BLOG", "ACADEMIC_OR_PAPER"}
)
# RU: KEEP на хосте — docs/spec/source, vendor blogs, papers.
AUTHORITY_REJECT_CLASSES: frozenset[str] = frozenset(
    {"COMMUNITY_BLOG", "SPAM_AGGREGATOR"}
)
# RU: REJECT — личные/агрегаторные блоги и SEO-фермы.
PASS1_INCLUDE_CLASSES: frozenset[str] = frozenset({"OFFICIAL_DOCS"})
# RU: в include_domains Pass 1 только официальные spec/docs/source.


class ExaSearchContextExpansion(BaseModel):
    """Dynamic Exa filter: domains, category, and search type for one query."""

    intent: ExaSearchIntent = Field(
        default="mixed",
        description=(
            "language_api: canonical specs and official docs/source for THIS topic. "
            "architecture: vendor blogs (Pass 2 / broad search). mixed: both."
        ),
    )
    # RU: интент — спецификации vs архитектура vs оба.
    primary_domains: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Pass 1 host hypotheses: CANONICAL_SPEC, OFFICIAL_DOCS, SOURCE_TREE only. "
            "No aggregator, Q&A, or SEO-academy hosts."
        ),
    )
    # RU: гипотезы канонических spec/docs-хостов для темы (Pass 1, после HTTP).
    allowed_categories: list[ExaApiCategory] = Field(
        default_factory=list,
        max_length=4,
        description="Exa `category` values; empty = do not constrain category.",
    )
    # RU: встроенные категории Exa API; пустой список — без category.
    search_type: ExaSearchType = Field(
        default="auto",
        description="Exa `type`: keyword/auto for specs; neural for narrative blogs.",
    )
    # RU: keyword/auto лучше для PEP/docs; neural — для лонгридов.
    use_broader_search: bool = Field(
        default=True,
        description="If primary_domains yield zero hits, retry without include_domains.",
    )
    # RU: fallback на глобальный Exa с exclude_domains.
    include_official_docs: bool = Field(
        default=True,
        description="Keep official documentation URLs; skip anti-docs excludeText.",
    )
    # RU: не резать PEP/CPython/framework docs пост-фильтрами.
    topic_vector_query: str = Field(
        default="",
        max_length=400,
        description=(
            "English high-level topic gist for BGE-M3 domain_registry search. "
            "Same abstraction as domain general_summary. "
            "FORBIDDEN: narrow subtopic laundry lists."
        ),
    )
    # RU: каноническая формулировка темы для локального векторного lookup.

    @field_validator("primary_domains", mode="before")
    @classmethod
    def _coerce_domains(cls, v: object) -> object:
        return v if v is not None else []

    @field_validator("allowed_categories", mode="before")
    @classmethod
    def _coerce_cats(cls, v: object) -> object:
        return v if v is not None else []


class DomainAuthorityVerdict(BaseModel):
    """Flash Lite verdict for a previously unseen Exa hostname."""

    domain: str = Field(..., description="Hostname without scheme.")
    # RU: проверяемый хост.
    classification: DomainAuthorityClass = Field(
        ...,
        description=(
            "OFFICIAL_DOCS KEEP (Pass 1). VENDOR_BLOG KEEP (not Pass 1). "
            "ACADEMIC_OR_PAPER KEEP (not Pass 1). "
            "COMMUNITY_BLOG REJECT. SPAM_AGGREGATOR REJECT."
        ),
    )
    # RU: класс авторитетности домена.
    status: Literal["KEEP", "REJECT"] = Field(
        ...,
        description=(
            "Host binary gate aligned to classification: KEEP for "
            "OFFICIAL_DOCS / VENDOR_BLOG / ACADEMIC_OR_PAPER; "
            "REJECT for COMMUNITY_BLOG / SPAM_AGGREGATOR."
        ),
    )
    # RU: статус выводится из classification, не из свободного текста модели.
    reason: str = Field(default="", description="Short English justification.")

    @model_validator(mode="after")
    def _align_status_to_classification(self) -> "DomainAuthorityVerdict":
        expected = (
            "REJECT" if self.classification in AUTHORITY_REJECT_CLASSES else "KEEP"
        )
        if self.status != expected:
            self.status = expected
        return self


class DomainAuthorityItem(BaseModel):
    """One hostname in a Flash Lite batch authority response."""

    domain: str = Field(..., description="Hostname without scheme or path.")
    # RU: классифицируемый хост.
    classification: DomainAuthorityClass = Field(
        ...,
        description=(
            "OFFICIAL_DOCS KEEP (Pass 1). VENDOR_BLOG KEEP (not Pass 1). "
            "ACADEMIC_OR_PAPER KEEP (not Pass 1). "
            "COMMUNITY_BLOG REJECT. SPAM_AGGREGATOR REJECT."
        ),
    )
    # RU: класс авторитетности; status на хосте выводится из enum.
    general_summary: str = Field(
        default="",
        max_length=400,
        description=(
            "Canonical high-level gist of the resource (language/kernel/spec "
            "family). FORBIDDEN: enumerating narrow lecture subtopics."
        ),
    )
    # RU: верхнеуровневое описание для BGE-M3; не список узких тем.
    reason: str = Field(default="", description="Short English justification.")


class BatchDomainAuthorityResponse(BaseModel):
    """Flash Lite batch: one item per requested hostname."""

    items: list[DomainAuthorityItem] = Field(
        default_factory=list,
        max_length=16,
        description="One DomainAuthorityItem per input hostname.",
    )
    # RU: пачка вердиктов; один вызов Lite на все неизвестные хосты.

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items(cls, v: object) -> object:
        return v if v is not None else []


__all__ = [
    "AUTHORITY_KEEP_CLASSES",
    "AUTHORITY_REJECT_CLASSES",
    "BatchDomainAuthorityResponse",
    "DomainAuthorityClass",
    "DomainAuthorityItem",
    "DomainAuthorityVerdict",
    "ExaApiCategory",
    "ExaSearchContextExpansion",
    "ExaSearchIntent",
    "ExaSearchType",
    "PASS1_INCLUDE_CLASSES",
]
