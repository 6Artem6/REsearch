"""Gemini Flash Lite — санитизация запросов для Consensus и валидация ответов."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from knowledge_engine.src.retrieval.semantic_scholar import (
    ScholarPaper,
    format_papers_block,
)

ACADEMIC_SANITIZER_SYSTEM = """You prepare queries for Consensus.app (peer-reviewed paper search).

Inputs: user_question, preserved_terms, optional WEB SNIPPETS from SearXNG.

Output academic_query_en: ONE or TWO sentences in English for literature search.

TERMINOLOGY (critical):
1. Include every preserved_terms token verbatim (RPG, LLM, lore, RAG, …).
2. If WEB SNIPPETS are present, prefer exact technical phrases that appear in snippets
   (paper titles, definitions) — do NOT guess trendy synonyms.
3. Do NOT replace the user's topic with a different research question.
4. Avoid vague paraphrases (e.g. "interactive fiction") when user said RPG lore / narrative lore;
   use: narrative lore, long-context, world knowledge, story consistency — as fits the question.
5. "локальная LLM" → local LLM or on-device LLM (keep LLM); not only "resource-constrained".

REMOVE: personal hardware brands, Docker/cloud product names, developer projects.

notes: brief — what was translated vs removed (Russian→EN ok to mention)."""

REFINEMENT_SANITIZER_SYSTEM = """Convert the refinement follow-up into academic English for Consensus.app.
Stay on the same topic as the original user question — do not pivot to unrelated indexing/RAG
unless the refinement explicitly asks. Remove personal infrastructure and project names.
English only. One or two sentences."""

VALIDATOR_SYSTEM = """You validate a Consensus.app literature response against the original user question
and optional developer_profile_context (selective constraints from Light RAG).

Relevance Gate: developer_profile_context may be intentionally EMPTY when the user asked a general /
academic / comparative CS question (apply_personal_profile=false). Do NOT treat missing profile as a gap
and do NOT RETRY for "Constraint Mismatch" when profile is empty — general questions need industry facts only.

STATUS RETRY when:
1. Architectural Gap — theory only, no engineering (data structures, complexity, indexing).
2. Constraint Mismatch — ONLY if developer_profile_context is non-empty AND papers clearly ignore those
   stated personal constraints (not applicable when profile is empty).
3. Low Information Diversity — single approach without alternatives or downsides.

STATUS REJECT — empty or off-topic vs user_query.
STATUS OK — sufficient engineering depth relative to user_query (general or personal).

docs: {title, url, snippet, source_anchor?} from consensus_extracted_papers and response.
Preserve bibliographic identity — never drop URLs/titles from reasoning.
refinement_prompt: ONE academic English follow-up for Consensus (no personal/stack/hardware context).
reason: short status explanation; cite [Sx] when referring to specific papers if SOURCE REGISTRY is provided."""


class AcademicQueryPayload(BaseModel):
    academic_query_en: str = Field(description="Clean English CS query for Consensus")
    notes: str = Field(
        default="", description="What was removed from the user question"
    )


class ConsensusDoc(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_anchor: str = Field(default="", description="Sx id if known")


class ValidationResult(BaseModel):
    status: Literal["OK", "REJECT", "RETRY"]
    docs: list[ConsensusDoc] = Field(default_factory=list)
    refinement_prompt: Optional[str] = None
    reason: str = ""


def sanitize_query_for_consensus(
    user_query: str,
    global_anchor: str,
    grounding_context: str = "",
    preserved_terms: list[str] | None = None,
) -> AcademicQueryPayload:
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured
    from knowledge_engine.src.processors.consensus_query_prep import (
        build_consensus_sanitize_payload,
    )

    preserved = preserved_terms or []
    user_payload = build_consensus_sanitize_payload(
        user_query,
        grounding_context,
        preserved,
    )
    return run_gemini_lite_structured(
        ACADEMIC_SANITIZER_SYSTEM,
        user_payload,
        global_anchor,
        AcademicQueryPayload,
        "consensus_query_sanitize",
    )


def sanitize_message_for_consensus(raw_message: str, global_anchor: str) -> str:
    """Очистка RETRY/refinement перед отправкой в Consensus (English academic only)."""
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    class _OneLine(BaseModel):
        academic_query_en: str

    out = run_gemini_lite_structured(
        REFINEMENT_SANITIZER_SYSTEM,
        raw_message.strip(),
        global_anchor,
        _OneLine,
        "consensus_refinement_sanitize",
    )
    return (out.academic_query_en or "").strip()


def validate_consensus_response(
    raw_consensus_text: str,
    user_query: str,
    developer_profile_context: str,
    global_anchor: str,
    *,
    attempt: int,
    max_retries: int,
    extracted_papers: list[ScholarPaper] | None = None,
) -> ValidationResult:
    papers_block = format_papers_block(extracted_papers or [])
    profile_block = (
        developer_profile_context.strip()
        or "(empty — no selective profile from Light RAG)"
    )
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured
    from knowledge_engine.src.processors.source_anchors import (
        format_registry_for_prompt,
    )

    registry_block = ""
    if extracted_papers:
        paper_dicts = [p.model_dump() for p in extracted_papers]
        from knowledge_engine.src.processors.source_anchors import build_source_registry

        reg = build_source_registry(paper_dicts)
        if reg:
            registry_block = (
                "\n\n### SOURCE REGISTRY (for citations in reason)\n"
                + format_registry_for_prompt(reg)
            )

    user_payload = (
        f"attempt={attempt} max_retries={max_retries}\n\n"
        f"### user_query\n{user_query}\n\n"
        f"### developer_profile_context\n{profile_block[:6000]}\n\n"
        f"### consensus_extracted_papers\n{papers_block}\n"
        f"{registry_block}\n\n"
        f"### consensus_raw_response\n{raw_consensus_text[:12000]}"
    )
    return run_gemini_lite_structured(
        VALIDATOR_SYSTEM,
        user_payload,
        global_anchor,
        ValidationResult,
        "consensus_validator",
    )
