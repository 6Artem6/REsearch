"""Подготовка Consensus query: preserved terms + SearXNG grounding (v0.7 fast_grounding)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from knowledge_engine.src.guardrails.term_guard import extract_user_acronyms

_EXTRA_TERM_RULES: tuple[tuple[str, str], ...] = (
    (r"\brpg\b", "RPG"),
    (r"\blore\b|лор", "lore"),
    (r"\bllm\b", "LLM"),
    (r"\brag\b", "RAG"),
)


def extract_preserved_terms_for_consensus(user_query: str) -> list[str]:
    """Термины из запроса — не подменять при переводе в academic EN."""
    preserved = list(extract_user_acronyms(user_query))
    seen = {p.upper() for p in preserved}
    for pattern, term in _EXTRA_TERM_RULES:
        if re.search(pattern, user_query or "", re.I):
            key = term.upper()
            if key not in seen:
                seen.add(key)
                preserved.append(term)
    return preserved[:12]


def build_consensus_sanitize_payload(
    user_query: str,
    grounding_context: str,
    preserved_terms: list[str],
) -> str:
    parts = [f"user_question:\n{(user_query or '').strip()}"]
    if preserved_terms:
        parts.append(
            "preserved_terms (include verbatim in academic_query_en — do not replace RPG/LLM/lore "
            "with unrelated phrases):\n" + ", ".join(preserved_terms)
        )
    grounding = (grounding_context or "").strip()
    if grounding:
        parts.append(
            "### WEB SNIPPETS (SearXNG raw hits — pick ESTABLISHED paper phrases from here; "
            "do not invent synonyms):\n" + grounding[:2800]
        )
    else:
        parts.append(
            "### WEB SNIPPETS: (none — translate faithfully; no generic paraphrase like "
            "'interactive fiction' unless user asked)"
        )
    return "\n\n".join(parts)


RELEVANCE_GATE_SYSTEM = """You are a Relevance Gate before literature search and final synthesis.

Classify whether the user's question requires their personal developer profile in answers
(Jarvis, Knowledge Engine, Apple Silicon, M-series Mac, LanceDB, local LLM stack, etc.).

apply_personal_profile = FALSE when:
- The question is fundamental, academic, comparative, or about general industry / CS practice
  (e.g. "Compare monolith and microservices", "How does HNSW work", architecture patterns,
  complexity trade-offs, consensus in the field) WITHOUT asking what the user should deploy locally.

apply_personal_profile = TRUE when:
- The user explicitly asks about THEIR project, stack, hardware, deployment, or choices
  (e.g. "What should I pick for Jarvis?", "How to run this on my Mac?", "for my local setup").

context_applicability: one of general_academic | engineering_practice | project_specific | hybrid

reason: one short sentence (Russian or English)."""


class ProfileApplicabilityPayload(BaseModel):
    apply_personal_profile: bool = Field(
        description="Whether selective Light RAG profile should reach Reasoner and L2 profile steps"
    )
    context_applicability: str = Field(
        description="general_academic | engineering_practice | project_specific | hybrid"
    )
    reason: str = ""


def _heuristic_apply_personal_profile(user_query: str) -> bool:
    q = (user_query or "").lower()
    markers = (
        "jarvis",
        "джарвис",
        "мой проект",
        "my project",
        "на моем",
        "на моём",
        "for my",
        "мой mac",
        "my mac",
        "apple silicon",
        "m-series",
        "локально",
        "локальный",
        "local setup",
        "knowledge engine",
    )
    return any(m in q for m in markers)


def assess_profile_applicability(
    user_query: str, global_anchor: str
) -> ProfileApplicabilityPayload:
    """Gemini Lite: should personal profile affect Reasoner / constraint checks?"""
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    q = (user_query or "").strip()
    if not q:
        return ProfileApplicabilityPayload(
            apply_personal_profile=False,
            context_applicability="general_academic",
            reason="empty query",
        )
    try:
        out = run_gemini_lite_structured(
            RELEVANCE_GATE_SYSTEM,
            f"### user_query\n{q}",
            global_anchor,
            ProfileApplicabilityPayload,
            "profile_relevance_gate",
        )
        return out
    except Exception:
        apply = _heuristic_apply_personal_profile(q)
        return ProfileApplicabilityPayload(
            apply_personal_profile=apply,
            context_applicability=(
                "project_specific" if apply else "engineering_practice"
            ),
            reason="heuristic fallback (Lite gate unavailable)",
        )
