"""Two-Stage Fact Relevance & Anchor Filtering.

Validated against the real production pipeline in
``knowledge_engine/tests/benchmarks/test_fact_relevance_pipeline.py`` (kept
as the slow, real-Gemma/Gemini A/B harness — gated behind
``RUN_FACT_RELEVANCE_BENCHMARK=1``, skipped by default). Winning strategy
(Synthetic Anchor, Variant A) is wired into production, both add-only and
OFF by default:

1. Static Anchor Filtering (Map/Index phase) — score every extracted
   ``KnowledgeAtom`` against the article's core anchor with the existing
   Cross-Encoder reranker (``rag_gateway/cross_encoder.py``), cut noise
   before REDUCE/storage. Wired into
   ``blog_spatial_summarizer.py::_apply_static_anchor_filter``, gated by
   ``BLOG_SPATIAL_ANCHOR_FILTER_ENABLED`` (default off).
2. Dynamic Topic Relevance (Retrieval phase) — combine the stored static
   score with a per-query dynamic score at retrieval time. Wired into
   ``domains/grounding/dialog_atoms_rag.py``, gated by
   ``DIALOG_ATOMS_HYBRID_RERANK_ENABLED`` (default off — this retrieval path
   currently makes no Cross-Encoder call at all; flipping this on adds one
   to every tutor turn).
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

from pydantic import BaseModel, Field

from knowledge_engine.src.adapters.llm_providers.gemini_stateless import (
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.config.settings import GEMINI_LITE_MODEL
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.src.shared.extraction import KnowledgeAtom

_FAST_ANCHOR_SYSTEM = (
    "You compress ONE technical article into its single core-claim anchor "
    "for downstream relevance filtering. Output 2-3 sentences: the problem "
    "the article addresses and its central mechanism or conclusion. "
    "No preamble, no meta-commentary, no markdown. "
    "CRITICAL: write the anchor in the SAME language as the article body — "
    "do NOT translate to English. The anchor is compared against extracted "
    "facts by a multilingual Cross-Encoder; a cross-lingual anchor/fact pair "
    "scores far lower than a same-language pair regardless of true semantic "
    "relevance, which would silently discard every fact."
)


class ArticleCoreAnchorContract(BaseModel):
    """Structured Output for the ultra-light Article Core Anchor call (Variant B)."""

    anchor: str = Field(
        ...,
        min_length=8,
        max_length=600,
        description="2-3 sentence core-claim summary of the article, no preamble",
    )


def get_synthetic_anchor(article_title: str, lead_text: str) -> str:
    """Variant A: zero-LLM anchor — Title + Lead only."""
    title = (article_title or "").strip()
    lead = (lead_text or "").strip()
    if not title and not lead:
        return ""
    return f"{title}\n\n{lead}".strip()


async def get_fast_llm_anchor(article_text: str) -> str:
    """Variant B: ultra-light Gemini Flash-Lite call — 2-3 sentence core anchor."""
    text = (article_text or "").strip()
    if not text:
        return ""
    result = await asyncio.to_thread(
        run_gemini_structured_with_chain,
        GEMINI_LITE_MODEL,
        _FAST_ANCHOR_SYSTEM,
        f"ARTICLE (first ~6000 chars):\n{text[:6000]}",
        "",
        ArticleCoreAnchorContract,
        "experimental_fast_anchor",
    )
    return (result.anchor or "").strip()


_ANSWER_FROM_FACTS_SYSTEM = (
    "You answer a user's question using ONLY the numbered facts provided. "
    "Do not invent information beyond them; if the facts do not cover the "
    "question, say so explicitly instead of guessing. Be concise (3-6 "
    "sentences). Write the answer in Russian."
)


class AnswerFromFactsContract(BaseModel):
    """Structured Output for the retrieval-quality demo — an answer built
    strictly from a given ranked fact list (Outcome 1 vs Outcome 2)."""

    answer: str = Field(
        ...,
        min_length=20,
        max_length=1200,
        description="Concise Russian answer built only from the given facts",
    )


async def generate_answer_from_facts(query: str, atoms: List[KnowledgeAtom]) -> str:
    """Downstream retrieval-quality probe: what answer would the tutor give
    if these atoms (already ranked/selected) were its retrieved context?"""
    if not atoms:
        return ""
    facts_block = "\n".join(
        f"{i + 1}. [{a.scope.value}] {a.statement}" for i, a in enumerate(atoms)
    )
    result = await asyncio.to_thread(
        run_gemini_structured_with_chain,
        GEMINI_LITE_MODEL,
        _ANSWER_FROM_FACTS_SYSTEM,
        f"QUESTION: {query}\n\nFACTS:\n{facts_block}",
        "",
        AnswerFromFactsContract,
        "experimental_answer_from_facts",
    )
    return (result.answer or "").strip()


class AtomAnchorFilterService:
    """Static Anchor Filtering (Cut) + Dynamic hybrid reranking (Retrieval)."""

    async def filter_atoms_by_anchor(
        self,
        atoms: List[KnowledgeAtom],
        anchor: str,
        threshold: float = 0.35,
    ) -> List[KnowledgeAtom]:
        """Score every atom against ``anchor`` (batched, off event loop) and
        drop atoms scoring below ``threshold``. Stamps ``core_relevance_score``
        on every atom passed in, including ones later dropped by the caller."""
        if not atoms:
            return []
        anchor = (anchor or "").strip()
        if not anchor:
            trace("EXP_ANCHOR_FILTER ⚠ empty anchor — atoms pass through unscored")
            return list(atoms)

        statements = [a.statement for a in atoms]
        # CRITICAL: score_relevance_pairs is a sync Cross-Encoder call —
        # never call it directly inside an async def (see cross_encoder.py /
        # locks.py::run_under_uma_lock precedent for why this matters).
        scores = await asyncio.to_thread(score_relevance_pairs, anchor, statements)

        kept: List[KnowledgeAtom] = []
        for atom, score in zip(atoms, scores):
            atom.core_relevance_score = score
            if score >= threshold:
                kept.append(atom)
        trace(
            f"EXP_ANCHOR_FILTER ✓ | in={len(atoms)} out={len(kept)} "
            f"threshold={threshold}"
        )
        return kept

    @staticmethod
    def calculate_hybrid_fact_score(
        query_score: float,
        core_relevance_score: float,
        alpha: float = 0.7,
    ) -> float:
        """final_score = alpha*query_score + (1-alpha)*core_relevance_score."""
        return (alpha * query_score) + ((1.0 - alpha) * core_relevance_score)

    async def rerank_atoms_for_query(
        self,
        query: str,
        candidate_atoms: List[KnowledgeAtom],
        alpha: float = 0.7,
    ) -> List[Tuple[KnowledgeAtom, float]]:
        """Dynamic query score (Cross-Encoder) combined with the atom's stored
        static ``core_relevance_score``; falls back to pure query score for
        atoms never scored by ``filter_atoms_by_anchor`` (None, not 0.0)."""
        if not candidate_atoms:
            return []
        statements = [a.statement for a in candidate_atoms]
        query_scores = await asyncio.to_thread(score_relevance_pairs, query, statements)
        scored: List[Tuple[KnowledgeAtom, float]] = []
        for atom, q_score in zip(candidate_atoms, query_scores):
            core = atom.core_relevance_score
            final_score = (
                self.calculate_hybrid_fact_score(q_score, core, alpha)
                if core is not None
                else q_score
            )
            scored.append((atom, final_score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
