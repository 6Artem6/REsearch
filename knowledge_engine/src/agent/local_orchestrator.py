"""Локальный агент: оркестрация Consensus + Gemini без генерации финального текста локально."""

from __future__ import annotations

import asyncio
from typing import Any, List

from knowledge_engine.config import CONSENSUS_MAX_RETRIES
from knowledge_engine.services.v07_run_progress import publish_web_run_progress
from knowledge_engine.src.analytics.chunker import extract_structured_chunks
from knowledge_engine.src.analytics.profiler import (
    build_concept_graph,
    build_profile_gap_map,
    build_tradeoff_matrix,
)
from knowledge_engine.src.dedup import ChunkDedupStore, ingest_document_chunks
from knowledge_engine.src.guardrails.fast_grounding import get_term_grounding_context
from knowledge_engine.src.memory.light_rag import LightRAG
from knowledge_engine.src.processors.consensus_query_prep import (
    assess_profile_applicability,
    consensus_sanitize_anchor,
    extract_preserved_terms_for_consensus,
)
from knowledge_engine.src.processors.source_anchors import (
    build_source_registry,
    format_papers_block_with_anchors,
    resolve_source_anchor_for_url,
    url_to_source_id_map,
)
from knowledge_engine.src.processors.validator import (
    ValidationResult,
    sanitize_message_for_consensus,
    sanitize_query_for_consensus,
    validate_consensus_response,
)
from knowledge_engine.src.retrieval.consensus_papers import (
    consensus_docs_to_papers,
    enrich_papers_metadata,
    merge_scholar_papers,
)
from knowledge_engine.src.retrieval.consensus_session import (
    ConsensusLoginRequiredError,
    acquire_consensus_session,
    release_consensus_session,
)
from knowledge_engine.src.retrieval.paper_documents import fetch_all_paper_documents
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.src.state import (
    KnowledgeEngineState,
    StructuredChunk,
    empty_v08_state,
)
from knowledge_engine.ui.run_log import node_end, node_start, trace


def _emit_progress(
    web_run_id: str | None,
    step: str,
    state: KnowledgeEngineState | dict[str, Any],
    keys: list[str],
) -> None:
    publish_web_run_progress(web_run_id, step, state, keys=keys)


def _global_anchor(user_query: str, profile_context: str) -> str:
    parts = [f"Задача: {user_query.strip()}"]
    if profile_context.strip():
        parts.append(f"Selective profile:\n{profile_context[:1500]}")
    return "\n".join(parts)


def _docs_from_validation(v: ValidationResult) -> List[dict[str, Any]]:
    return [d.model_dump() for d in v.docs]


def _scholar_papers_to_docs(papers: List[ScholarPaper]) -> List[dict[str, Any]]:
    return [
        {
            "title": p.title,
            "url": p.source_url,
            "snippet": (p.abstract or p.tldr or "")[:1200],
            "source_anchor": getattr(p, "source_anchor", "") or "",
        }
        for p in papers
    ]


def _docs_from_paper_dicts(paper_dicts: List[dict[str, Any]]) -> List[dict[str, Any]]:
    return [
        {
            "title": (p.get("title") or "paper"),
            "url": (p.get("source_url") or p.get("url") or ""),
            "snippet": (p.get("abstract") or p.get("tldr") or p.get("snippet") or "")[
                :1200
            ],
            "source_anchor": p.get("source_anchor") or "",
        }
        for p in paper_dicts
    ]


async def run_consensus_pipeline(
    user_query: str,
    user_profile_md: str,
    thread_id: str,
    web_run_id: str | None = None,
) -> KnowledgeEngineState:
    """
    Consensus: только академический English query.
    Gemini: user_query + selective Light RAG profile context.
    """
    node_start("consensus_pipeline")
    step = "init"
    state = empty_v08_state(thread_id, user_profile_md, user_query=user_query)
    state["pipeline_version"] = "0.8"
    state["retrieval_mode"] = "consensus"
    _emit_progress(
        web_run_id,
        "init",
        state,
        ["user_query", "pipeline_version", "thread_id"],
    )
    session = await acquire_consensus_session()
    try:
        rag = LightRAG()
        await rag.sync_profile_from_markdown(user_profile_md)
        profile_context = await rag.get_relevant_profile_context(user_query)
        gate_anchor = consensus_sanitize_anchor(user_query)
        applicability = await asyncio.to_thread(
            assess_profile_applicability,
            user_query,
            gate_anchor,
        )
        apply_profile = applicability.apply_personal_profile
        profile_effective = profile_context if apply_profile else ""
        profile_md_for_l2 = user_profile_md if apply_profile else ""
        state["apply_personal_profile"] = apply_profile
        state["context_applicability"] = applicability.context_applicability
        state["profile_applicability_reason"] = applicability.reason
        state["selective_profile_context"] = profile_effective
        trace(
            f"Relevance Gate | apply_personal_profile={apply_profile} | "
            f"{applicability.context_applicability} | {applicability.reason[:120]}"
        )
        _emit_progress(
            web_run_id,
            "profile_context",
            state,
            [
                "user_query",
                "selective_profile_context",
                "apply_personal_profile",
                "context_applicability",
                "profile_applicability_reason",
                "pipeline_version",
            ],
        )
        anchor = _global_anchor(user_query, profile_effective)

        trace("Gemini Lite ▶ sanitize query for Consensus (academic EN only)")
        sanitize_anchor = consensus_sanitize_anchor(user_query)
        preserved_terms = extract_preserved_terms_for_consensus(user_query)
        grounding = await get_term_grounding_context(user_query)
        if preserved_terms:
            trace(f"Consensus ⊶ preserved terms | {', '.join(preserved_terms)}")
        academic = await asyncio.to_thread(
            sanitize_query_for_consensus,
            user_query,
            sanitize_anchor,
            grounding,
            preserved_terms,
        )
        consensus_query = (academic.academic_query_en or "").strip()
        if not consensus_query:
            consensus_query = user_query.strip()
        if (academic.notes or "").strip():
            trace(f"Consensus ⊶ sanitize notes | {academic.notes[:240]}")
        trace(f"Consensus ▶ academic query | {consensus_query[:200]}")
        state["consensus_academic_query"] = consensus_query
        if preserved_terms:
            state["consensus_preserved_terms"] = preserved_terms
        _emit_progress(
            web_run_id,
            "consensus_query",
            state,
            [
                "user_query",
                "consensus_academic_query",
                "consensus_preserved_terms",
                "selective_profile_context",
                "pipeline_version",
            ],
        )

        await session.start()
        await session.begin_new_run()
        turn = await session.send_message(consensus_query)
        raw = turn.raw_text
        raw_history: List[str] = [raw]
        accumulated_papers: List[ScholarPaper] = list(turn.papers)

        best_docs: List[dict[str, Any]] = []
        last_validation: ValidationResult | None = None
        partial_note = ""

        for attempt in range(CONSENSUS_MAX_RETRIES + 1):
            trace(f"Gemini Lite ▶ validate consensus | attempt={attempt}")
            validation = await asyncio.to_thread(
                validate_consensus_response,
                raw,
                user_query,
                profile_effective,
                anchor,
                attempt=attempt,
                max_retries=CONSENSUS_MAX_RETRIES,
                extracted_papers=accumulated_papers,
            )
            last_validation = validation
            trace(
                f"Gemini Lite ✓ status={validation.status} | {validation.reason[:120]}"
            )

            validator_papers = consensus_docs_to_papers(
                _docs_from_validation(validation)
            )
            accumulated_papers = merge_scholar_papers(
                accumulated_papers, validator_papers
            )
            if validation.docs:
                best_docs = _docs_from_validation(validation)
            elif accumulated_papers:
                best_docs = _scholar_papers_to_docs(accumulated_papers)

            if validation.status == "REJECT":
                step = "rejected"
                state.update(
                    {
                        "validation_status": "REJECT",
                        "user_final_answer": (
                            "Поиск в Consensus не дал релевантных материалов. "
                            f"Причина: {validation.reason}"
                        ),
                        "current_step": "rejected",
                        "consensus_raw_history": raw_history,
                        "scholarly_papers": [
                            p.model_dump() for p in accumulated_papers
                        ],
                        "consensus_docs": best_docs,
                        "selective_profile_context": profile_effective,
                    }
                )
                _emit_progress(
                    web_run_id,
                    "rejected",
                    state,
                    [
                        "user_query",
                        "user_final_answer",
                        "validation_status",
                        "validation_reason",
                        "consensus_academic_query",
                        "scholarly_papers",
                        "consensus_docs",
                        "consensus_raw_history",
                    ],
                )
                return state

            if validation.status == "OK":
                state["validation_status"] = "OK"
                break

            if validation.status == "RETRY" and attempt < CONSENSUS_MAX_RETRIES:
                refinement_raw = (validation.refinement_prompt or "").strip()
                if not refinement_raw:
                    refinement_raw = "Compare alternative indexing approaches and their complexity trade-offs."
                trace("Gemini Lite ▶ sanitize refinement for Consensus")
                consensus_refinement = await asyncio.to_thread(
                    sanitize_message_for_consensus,
                    refinement_raw,
                    consensus_sanitize_anchor(user_query),
                )
                trace("Consensus ▶ RETRY refinement (academic EN)")
                turn = await session.send_message(consensus_refinement)
                raw = turn.raw_text
                raw_history.append(raw)
                accumulated_papers = merge_scholar_papers(
                    accumulated_papers, turn.papers
                )
            else:
                partial_note = (
                    f"После {CONSENSUS_MAX_RETRIES} уточнений данные неполные. "
                    f"{validation.reason}"
                )
                state["validation_status"] = "PARTIAL"
                break

        state["validation_reason"] = last_validation.reason if last_validation else ""
        state["consensus_raw_history"] = raw_history
        state["scholarly_papers"] = [p.model_dump() for p in accumulated_papers]
        state["consensus_docs"] = best_docs
        state["validation_status"] = state.get("validation_status") or (
            last_validation.status if last_validation else ""
        )
        _emit_progress(
            web_run_id,
            "validation",
            state,
            [
                "user_query",
                "consensus_academic_query",
                "validation_status",
                "validation_reason",
                "consensus_raw_history",
                "scholarly_papers",
                "consensus_docs",
                "selective_profile_context",
            ],
        )

        trace(f"Consensus ⊶ enrich metadata | papers={len(accumulated_papers)}")
        enriched_papers = await enrich_papers_metadata(accumulated_papers)
        paper_dicts = [p.model_dump() for p in enriched_papers]
        source_registry = build_source_registry(paper_dicts)
        state["source_registry"] = source_registry
        url_map = url_to_source_id_map(source_registry)
        if enriched_papers:
            best_docs = _docs_from_paper_dicts(paper_dicts)
        _emit_progress(
            web_run_id,
            "papers_enriched",
            state,
            [
                "user_query",
                "consensus_academic_query",
                "validation_status",
                "scholarly_papers",
                "consensus_docs",
                "source_registry",
            ],
        )

        trace(f"scholar › fetch {len(enriched_papers)} consensus paper bodies")
        documents = await fetch_all_paper_documents(enriched_papers)
        state["scholarly_papers"] = paper_dicts
        state["consensus_docs"] = best_docs

        store = ChunkDedupStore()
        structured: List[StructuredChunk] = []
        for doc in documents:
            trace("Ollama+LanceDB › dedup ingest consensus chunks")
            accepted, _scraped = await ingest_document_chunks(
                store, doc.doc_id, doc.raw_markdown
            )
            for ch in accepted:
                structured.append(ch)

        for i, doc in enumerate(documents, start=1):
            trace(f"Gemini Lite › chunking consensus doc {i}/{len(documents)}")
            doc_sid = resolve_source_anchor_for_url(
                doc.source_url, url_map, paper_dicts
            )
            chunks = await asyncio.to_thread(
                extract_structured_chunks,
                doc,
                anchor,
                doc_sid,
            )
            structured.extend(chunks)

        state["documents"] = [d.model_dump() for d in documents]
        state["structured_chunks"] = [c.model_dump() for c in structured]
        _emit_progress(
            web_run_id,
            "sources_fetched",
            state,
            [
                "user_query",
                "documents",
                "scholarly_papers",
                "consensus_docs",
                "validation_status",
            ],
        )

        trace("Gemini Flash › L2a ConceptGraph (consensus papers)")
        concept_graph = await asyncio.to_thread(
            build_concept_graph,
            structured,
            anchor,
            documents,
            None,
            paper_dicts,
            user_query,
            source_registry,
        )
        state["concept_graph"] = concept_graph
        _emit_progress(
            web_run_id,
            "l2a",
            state,
            [
                "user_query",
                "concept_graph",
                "documents",
                "scholarly_papers",
                "validation_status",
            ],
        )

        trace("Gemini Flash › L2b ProfileGapMap")
        gap_map = await asyncio.to_thread(
            build_profile_gap_map,
            concept_graph,
            profile_md_for_l2,
            anchor,
            documents,
            None,
            paper_dicts,
            user_query,
            source_registry,
        )
        state["profile_gap_map"] = gap_map
        _emit_progress(
            web_run_id,
            "l2b",
            state,
            [
                "user_query",
                "concept_graph",
                "profile_gap_map",
                "documents",
                "scholarly_papers",
            ],
        )

        trace("Gemini Flash › L2c TradeoffMatrix")
        tradeoff_matrix = await asyncio.to_thread(
            build_tradeoff_matrix,
            concept_graph,
            gap_map,
            profile_md_for_l2,
            anchor,
            None,
            paper_dicts,
            user_query,
            source_registry,
        )
        state["tradeoff_matrix"] = tradeoff_matrix
        _emit_progress(
            web_run_id,
            "l2c",
            state,
            [
                "user_query",
                "concept_graph",
                "profile_gap_map",
                "tradeoff_matrix",
                "documents",
                "scholarly_papers",
            ],
        )

        trace("Gemini Reasoner ▶ final answer")
        from knowledge_engine.src.processors.reasoner import (
            run_reasoner as invoke_reasoner,
        )

        _emit_progress(
            web_run_id,
            "reasoner",
            state,
            [
                "user_query",
                "concept_graph",
                "profile_gap_map",
                "tradeoff_matrix",
                "documents",
                "scholarly_papers",
                "validation_status",
            ],
        )

        reasoner_docs = best_docs
        if enriched_papers:
            reasoner_docs = _docs_from_paper_dicts(paper_dicts)
        papers_block = format_papers_block_with_anchors(enriched_papers)
        final = await asyncio.to_thread(
            invoke_reasoner,
            reasoner_docs,
            user_query,
            profile_effective,
            anchor,
            raw_consensus_text=raw_history[-1],
            partial_data_note=partial_note,
            papers_block=papers_block,
            source_registry=source_registry,
            apply_personal_profile=apply_profile,
            retrieval_mode="consensus",
        )

        await rag.ingest_facts(final.fact_nuggets)

        state.update(
            {
                "user_final_answer": final.user_final_answer,
                "fact_nuggets": list(final.fact_nuggets),
                "consensus_raw_history": raw_history,
                "consensus_docs": best_docs,
                "scholarly_papers": paper_dicts,
                "documents": [d.model_dump() for d in documents],
                "structured_chunks": [c.model_dump() for c in structured],
                "concept_graph": concept_graph,
                "profile_gap_map": gap_map,
                "tradeoff_matrix": tradeoff_matrix,
                "validation_reason": (
                    last_validation.reason if last_validation else ""
                ),
                "current_step": "completed",
                "selective_profile_context": profile_effective,
            }
        )
        from knowledge_engine.src.processors.answer_corpus import (
            finalize_run_answer_corpus,
        )

        finalize_run_answer_corpus(state)
        _emit_progress(
            web_run_id,
            "completed",
            state,
            [
                "user_query",
                "user_final_answer",
                "fact_nuggets",
                "concept_graph",
                "profile_gap_map",
                "tradeoff_matrix",
                "documents",
                "scholarly_papers",
                "validation_status",
                "source_registry",
                "structured_chunks",
                "answer_block_sources",
            ],
        )
        trace(
            f"PIPELINE ✓ consensus | papers={len(enriched_papers)} "
            f"docs={len(documents)} chunks={len(structured)}"
        )
        step = "completed"
        return state
    except ConsensusLoginRequiredError as exc:
        step = "login_required"
        state.update(
            {
                "validation_status": "REJECT",
                "user_final_answer": (
                    "Consensus просит войти. Остановите API и выполните один раз: "
                    "`python -m knowledge_engine.main consensus-login` "
                    "(Google/email). Профиль: knowledge_engine/.browser_state. "
                    "browser-login — только Gemini, не Consensus. "
                    f"Детали: {exc}"
                ),
                "current_step": step,
            }
        )
        _emit_progress(
            web_run_id,
            "login_required",
            state,
            ["user_query", "user_final_answer", "validation_status", "current_step"],
        )
        return state
    finally:
        await release_consensus_session(session)
        node_end("consensus_pipeline", step)


async def run_fast_pipeline(
    user_query: str,
    user_profile_md: str,
    thread_id: str,
    web_run_id: str | None = None,
) -> KnowledgeEngineState:
    """Fast mode: Light RAG + Gemini Reasoner, без Consensus / L2."""
    node_start("fast_pipeline")
    step = "init"
    state = empty_v08_state(thread_id, user_profile_md, user_query=user_query)
    state["pipeline_version"] = "0.8"
    state["retrieval_mode"] = "fast"
    state["validation_status"] = "FAST_MODE"
    _emit_progress(
        web_run_id,
        "init",
        state,
        [
            "user_query",
            "pipeline_version",
            "thread_id",
            "retrieval_mode",
            "validation_status",
        ],
    )
    try:
        rag = LightRAG()
        await rag.sync_profile_from_markdown(user_profile_md)
        profile_context = await rag.get_relevant_profile_context(user_query)
        facts_context = await rag.get_relevant_facts_context(user_query, limit=8)
        gate_anchor = consensus_sanitize_anchor(user_query)
        applicability = await asyncio.to_thread(
            assess_profile_applicability,
            user_query,
            gate_anchor,
        )
        apply_profile = applicability.apply_personal_profile
        profile_effective = profile_context if apply_profile else ""
        state["apply_personal_profile"] = apply_profile
        state["context_applicability"] = applicability.context_applicability
        state["profile_applicability_reason"] = applicability.reason
        state["selective_profile_context"] = profile_effective
        trace(
            f"Fast mode | Light RAG facts={'yes' if facts_context else 'no'} | "
            f"apply_profile={apply_profile}"
        )
        _emit_progress(
            web_run_id,
            "profile_context",
            state,
            [
                "user_query",
                "selective_profile_context",
                "apply_personal_profile",
                "context_applicability",
                "profile_applicability_reason",
                "pipeline_version",
                "retrieval_mode",
                "validation_status",
            ],
        )
        anchor = _global_anchor(user_query, profile_effective)
        partial_note = (
            "Режим fast: внешний поиск Consensus не выполнялся. "
            "Ответ опирается на фундаментальные CS-источники и релевантные факты Light RAG."
        )
        if not facts_context:
            partial_note += (
                " В Light RAG не найдено фактов с высокой релевантностью к запросу."
            )

        from knowledge_engine.src.processors.reasoner import (
            run_reasoner as invoke_reasoner,
        )

        _emit_progress(
            web_run_id,
            "reasoner",
            state,
            [
                "user_query",
                "selective_profile_context",
                "validation_status",
                "retrieval_mode",
            ],
        )
        step = "reasoner"
        final = await asyncio.to_thread(
            invoke_reasoner,
            [],
            user_query,
            profile_effective,
            anchor,
            raw_consensus_text="",
            partial_data_note=partial_note,
            papers_block="",
            source_registry=[],
            apply_personal_profile=apply_profile,
            retrieval_mode="fast",
            light_rag_context=facts_context,
        )
        await rag.ingest_facts(final.fact_nuggets)
        state.update(
            {
                "user_final_answer": final.user_final_answer,
                "fact_nuggets": list(final.fact_nuggets),
                "consensus_raw_history": [],
                "consensus_docs": [],
                "scholarly_papers": [],
                "documents": [],
                "structured_chunks": [],
                "concept_graph": None,
                "profile_gap_map": None,
                "tradeoff_matrix": None,
                "validation_reason": "Consensus пропущен (fast mode)",
                "current_step": "completed",
                "selective_profile_context": profile_effective,
                "source_registry": [],
            }
        )
        from knowledge_engine.src.processors.answer_corpus import (
            finalize_run_answer_corpus,
        )

        finalize_run_answer_corpus(state)
        state["source_registry"] = state.get("source_registry") or []
        _emit_progress(
            web_run_id,
            "completed",
            state,
            [
                "user_query",
                "user_final_answer",
                "fact_nuggets",
                "validation_status",
                "retrieval_mode",
                "selective_profile_context",
                "source_registry",
                "structured_chunks",
                "documents",
                "answer_block_sources",
            ],
        )
        trace("PIPELINE ✓ fast mode | reasoner only")
        return state
    finally:
        node_end("fast_pipeline", step)


async def run_knowledge_engine_v08(
    user_query: str,
    user_profile_md: str,
    thread_id: str,
    web_run_id: str | None = None,
    retrieval_mode: str = "fast",
) -> KnowledgeEngineState:
    mode = (retrieval_mode or "fast").strip().lower()
    if mode == "consensus":
        trace(
            f"GRAPH ▶ v0.8 consensus | thread_id={thread_id} | "
            "retrieval=Consensus+LightRAG"
        )
        result = await run_consensus_pipeline(
            user_query,
            user_profile_md,
            thread_id,
            web_run_id=web_run_id,
        )
    else:
        trace(
            f"GRAPH ▶ v0.8 fast | thread_id={thread_id} | "
            "retrieval=LightRAG+Reasoner"
        )
        result = await run_fast_pipeline(
            user_query,
            user_profile_md,
            thread_id,
            web_run_id=web_run_id,
        )
    trace(f"GRAPH ✓ v0.8 завершён | step={result.get('current_step')} | mode={mode}")
    return result
