"""Knowledge Engine v0.7 — Pydantic contracts and LangGraph state."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

SourceType = Literal[
    "arxiv_html5",
    "github_dom",
    "trafilatura",
    "dom_mask",
    "academic_pdf",
]


class ValidatedQuerySpec(BaseModel):
    cs_formal_query: str = Field(description="Формализованный CS-запрос")
    target_keywords: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    preserved_terms: List[str] = Field(
        default_factory=list,
        description="Аббревиатуры/термины из запроса без искажения (MCP, GUI, …)",
    )


class PersonalContext(BaseModel):
    summary: str = ""
    target_architecture: List[str] = Field(default_factory=list)
    latency_requirements: str = ""
    resource_constraints: str = ""
    target_stack: List[str] = Field(default_factory=list)
    project_focus: str = ""


class ScrapedDocument(BaseModel):
    doc_id: str
    source_url: str = ""
    source_type: SourceType
    raw_markdown: str
    cosine_dedup_passed: bool = False
    title: str = ""
    is_pdf: bool = False


class StructuredChunk(BaseModel):
    chunk_id: str
    doc_id: str = ""
    text: str = ""
    concepts: List[str] = Field(default_factory=list)
    code_snippets: List[str] = Field(default_factory=list)
    p99_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_anchor: str = Field(
        default="",
        description="Source id S1, S2, … tied to SOURCE REGISTRY",
    )


class KnowledgeEngineState(TypedDict, total=False):
    thread_id: str
    user_query: str
    user_profile_md: str
    personal_context: Optional[Dict[str, Any]]
    scholarly_papers: Optional[List[Dict[str, Any]]]
    query_spec: Optional[ValidatedQuerySpec]
    documents: List[ScrapedDocument]
    structured_chunks: List[StructuredChunk]
    concept_graph: Optional[Dict[str, Any]]
    profile_gap_map: Optional[Dict[str, Any]]
    tradeoff_matrix: Optional[List[Dict[str, Any]]]
    failure_modes: Optional[List[Dict[str, Any]]]
    current_step: str
    density_delta: float
    staged_resource_lock_active: bool
    search_depth: NotRequired[int]
    total_scraped_chunks: NotRequired[int]
    unique_chunks_ingested: NotRequired[int]
    # v0.8 Consensus agent
    user_final_answer: NotRequired[str]
    fact_nuggets: NotRequired[List[str]]
    consensus_raw_history: NotRequired[List[str]]
    consensus_docs: NotRequired[List[Dict[str, Any]]]
    validation_status: NotRequired[str]
    validation_reason: NotRequired[str]
    personal_context_profile: NotRequired[str]
    selective_profile_context: NotRequired[str]
    apply_personal_profile: NotRequired[bool]
    context_applicability: NotRequired[str]
    profile_applicability_reason: NotRequired[str]
    consensus_academic_query: NotRequired[str]
    consensus_preserved_terms: NotRequired[List[str]]
    pipeline_version: NotRequired[str]
    source_registry: NotRequired[List[Dict[str, Any]]]
    retrieval_mode: NotRequired[str]
    answer_block_sources: NotRequired[List[Dict[str, Any]]]


def empty_v08_state(
    thread_id: str,
    user_profile_md: str = "",
    user_query: str = "",
) -> KnowledgeEngineState:
    base = empty_v07_state(thread_id, user_profile_md, user_query=user_query)
    base.update(
        {
            "user_final_answer": "",
            "fact_nuggets": [],
            "consensus_raw_history": [],
            "consensus_docs": [],
            "validation_status": "",
            "validation_reason": "",
            "personal_context_profile": "",
            "selective_profile_context": "",
            "apply_personal_profile": True,
            "context_applicability": "",
            "profile_applicability_reason": "",
            "consensus_academic_query": "",
            "consensus_preserved_terms": [],
            "pipeline_version": "",
            "source_registry": [],
            "retrieval_mode": "fast",
        }
    )
    return base


def empty_v07_state(
    thread_id: str,
    user_profile_md: str = "",
    user_query: str = "",
) -> KnowledgeEngineState:
    return {
        "thread_id": thread_id,
        "user_query": user_query,
        "user_profile_md": user_profile_md,
        "personal_context": None,
        "scholarly_papers": None,
        "query_spec": None,
        "documents": [],
        "structured_chunks": [],
        "concept_graph": None,
        "profile_gap_map": None,
        "tradeoff_matrix": None,
        "failure_modes": None,
        "current_step": "init",
        "density_delta": 0.0,
        "staged_resource_lock_active": False,
        "search_depth": 0,
        "total_scraped_chunks": 0,
        "unique_chunks_ingested": 0,
    }
