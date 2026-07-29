"""Curriculum: Consensus Playwright + Lite validate + Summarizer → LanceDB + hits."""

from __future__ import annotations

import uuid

from knowledge_engine.config import (
    CONSENSUS_MAX_RETRIES,
    CURRICULUM_V08_MAX_PAPERS,
    PACKAGE_ROOT,
)
from knowledge_engine.services.summarizer import summarize_article
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.analytics.chunker import extract_structured_chunks
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.processors.consensus_query_prep import (
    assess_profile_applicability,
    extract_preserved_terms_for_consensus,
)
from knowledge_engine.src.processors.source_anchors import (
    build_source_registry,
    resolve_source_anchor_for_url,
    url_to_source_id_map,
)
from knowledge_engine.src.processors.validator import (
    sanitize_message_for_consensus,
    sanitize_query_for_consensus,
    validate_consensus_response,
)
from knowledge_engine.src.retrieval.consensus_capture import (
    is_generic_consensus_url,
    normalize_paper_urls,
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
from knowledge_engine.src.retrieval.paper_documents import fetch_paper_document
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper, paper_to_document_text
from knowledge_engine.src.guardrails.fast_grounding import get_term_grounding_context
from knowledge_engine.src.state import ScrapedDocument
from knowledge_engine.ui.run_log import trace


def _read_user_profile_md() -> str:
    path = PACKAGE_ROOT / "user_profile.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _consensus_sanitize_anchor(user_query: str) -> str:
    return f"Задача (только для ориентира, не расширять): {user_query.strip()}"


def _word_count(text: str) -> int:
    return len((text or "").split())


def _clip_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def _external_paper_url(url: str) -> bool:
    u = (url or "").strip()
    if not u.startswith("http"):
        return False
    if is_generic_consensus_url(u):
        return False
    if "consensus.app" in u.lower():
        return False
    return True


def _deep_extract_blocks(
    key_takeaways: list[str],
    failure_modes: list[str],
    chunk_texts: list[str],
    min_words: int = 150,
    max_words: int = 300,
) -> list[str]:
    primary = "\n\n".join(
        [t.strip() for t in key_takeaways if (t or "").strip()]
        + [t.strip() for t in failure_modes if (t or "").strip()]
    )
    for ct in chunk_texts:
        if ct.strip():
            primary += "\n\n" + ct.strip()
    primary = primary.strip()
    if not primary:
        return []

    blocks: list[str] = []
    words = primary.split()
    i = 0
    while i < len(words) and len(blocks) < 4:
        chunk_words = words[i : i + max_words]
        if len(chunk_words) < min_words and blocks:
            break
        if len(chunk_words) < 40:
            break
        blocks.append(" ".join(chunk_words))
        i += max_words
    if not blocks and primary:
        blocks.append(_clip_words(primary, max_words))
    return blocks


def _paper_rank_key(p: ScholarPaper) -> tuple[int, int]:
    url = (p.source_url or "").strip()
    alen = len((p.abstract or p.tldr or "") or "")
    return (1 if _external_paper_url(url) else 0, alen)


def _select_curriculum_papers(
    accumulated: list[ScholarPaper],
    best_docs: list[dict],
    cap: int,
) -> list[ScholarPaper]:
    merged = merge_scholar_papers(
        normalize_paper_urls(accumulated),
        consensus_docs_to_papers(best_docs),
    )
    ranked = sorted(merged, key=_paper_rank_key, reverse=True)
    out: list[ScholarPaper] = []
    seen: set[str] = set()
    for p in ranked:
        if len(out) >= cap:
            break
        url = (p.source_url or "").strip()
        title = (p.title or "").strip()
        abstract = (p.abstract or p.tldr or "").strip()
        dedupe = url.lower() if url else title.lower()
        if not dedupe or dedupe in seen:
            continue
        if _external_paper_url(url) or len(abstract) >= 60:
            seen.add(dedupe)
            out.append(p)
    return out


def _scraped_from_paper(paper: ScholarPaper) -> ScrapedDocument | None:
    text = paper_to_document_text(paper)
    if len(text.strip()) < 60:
        return None
    url = (paper.source_url or "").strip()
    if not _external_paper_url(url):
        return None
    return ScrapedDocument(
        doc_id=f"curriculum_{uuid.uuid4().hex[:12]}",
        source_url=url[:2000],
        source_type="trafilatura",
        raw_markdown=text[:14000],
        title=(paper.title or "paper")[:400],
        is_pdf=False,
    )


def _hits_from_summary_and_chunks(
    title: str,
    url: str,
    summary,
    chunks: list,
    snippet_fallback: str = "",
) -> CurriculumSearchHit | None:
    u = (url or "").strip()
    if not u.startswith("http"):
        return None
    chunk_texts = [
        (c.text or "").strip()
        for c in chunks
        if getattr(c, "text", None) and len((c.text or "").strip()) > 40
    ]
    extracts = _deep_extract_blocks(
        list(summary.key_takeaways or []),
        list(summary.failure_modes or []),
        chunk_texts,
    )
    if not extracts and snippet_fallback:
        extracts = _deep_extract_blocks([], [], [snippet_fallback], min_words=80, max_words=300)
    if not extracts:
        return None
    return CurriculumSearchHit(
        url=u[:2000],
        title=(summary.title or title or u)[:400],
        snippet=(snippet_fallback or extracts[0])[:1200],
        key_extracts=extracts[:8],
        source_tier="consensus",
    )


def _process_validator_doc(
    raw: dict,
    anchor: str,
    store: VectorStore,
    seen_urls: set[str],
) -> CurriculumSearchHit | None:
    url = (raw.get("url") or "").strip()
    if not _external_paper_url(url) or url in seen_urls:
        return None
    title = (raw.get("title") or url)[:400]
    snippet = (raw.get("snippet") or "")[:4000]
    body = snippet if len(snippet) >= 80 else f"{title}\n\n{snippet}"
    try:
        summary = summarize_article(title, url, body[:14000])
        store.save_summary(summary)
    except Exception as exc:
        trace(f"CURRICULUM v08 validator doc summarizer skip | {exc}")
        extracts = _deep_extract_blocks([], [], [snippet], min_words=80, max_words=300)
        if not extracts:
            return None
        seen_urls.add(url)
        return CurriculumSearchHit(
            url=url[:2000],
            title=title,
            snippet=snippet[:1200],
            key_extracts=extracts[:8],
            source_tier="consensus",
        )
    hit = _hits_from_summary_and_chunks(title, url, summary, [], snippet_fallback=snippet)
    if hit:
        seen_urls.add(url)
    return hit


async def _process_paper_to_hit(
    paper: ScholarPaper,
    anchor: str,
    store: VectorStore,
    url_map: dict,
    paper_dicts: list[dict],
    seen_urls: set[str],
) -> CurriculumSearchHit | None:
    url = (paper.source_url or "").strip()
    if not _external_paper_url(url) or url in seen_urls:
        return None
    title = (paper.title or url)[:400]
    snippet = (paper.abstract or paper.tldr or "")[:1200]
    doc_sid = resolve_source_anchor_for_url(url, url_map, paper_dicts)
    doc = await fetch_paper_document(paper)
    raw_for_summary = (doc.raw_markdown if doc else "") or paper_to_document_text(paper)
    try:
        summary = summarize_article(title, url, raw_for_summary[:14000])
        store.save_summary(summary)
    except Exception as exc:
        trace(f"CURRICULUM v08 summarizer skip | {url[:60]} | {exc}")
        extracts = _deep_extract_blocks([], [], [snippet], min_words=80, max_words=300)
        if not extracts:
            return None
        seen_urls.add(url)
        return CurriculumSearchHit(
            url=url[:2000],
            title=title,
            snippet=snippet[:1200],
            key_extracts=extracts[:8],
            source_tier="consensus",
        )
    chunks: list = []
    if doc and len((doc.raw_markdown or "").strip()) >= 80:
        chunks = extract_structured_chunks(doc, anchor, doc_sid)
    else:
        synth = _scraped_from_paper(paper)
        if synth:
            chunks = extract_structured_chunks(synth, anchor, doc_sid)
    hit = _hits_from_summary_and_chunks(title, url, summary, chunks, snippet_fallback=snippet)
    if hit:
        seen_urls.add(url)
    return hit


async def harvest_curriculum_sources_v08(
    target_goal: str,
    anchor: str,
) -> list[CurriculumSearchHit]:
    goal = (target_goal or "").strip()
    if len(goal) < 8:
        return []

    trace("CURRICULUM v08 harvest ▶ Consensus Playwright + Lite + Summarizer")
    user_profile_md = _read_user_profile_md()
    sanitize_anchor = _consensus_sanitize_anchor(goal)

    applicability = assess_profile_applicability(goal, sanitize_anchor)
    profile_effective = ""
    if applicability.apply_personal_profile:
        from knowledge_engine.src.memory.light_rag import LightRAG

        rag = LightRAG()
        await rag.sync_profile_from_markdown(user_profile_md)
        profile_effective = await rag.get_relevant_profile_context(goal)

    session = await acquire_consensus_session()
    hits: list[CurriculumSearchHit] = []
    store = VectorStore()

    try:
        preserved = extract_preserved_terms_for_consensus(goal)
        grounding = await get_term_grounding_context(goal)
        academic = sanitize_query_for_consensus(
            goal,
            sanitize_anchor,
            grounding,
            preserved,
        )
        consensus_query = (academic.academic_query_en or goal).strip()
        trace(f"CURRICULUM v08 Consensus query | {consensus_query[:200]}")

        await session.start()
        await session.begin_new_run()
        turn = await session.send_message(consensus_query)
        raw = turn.raw_text
        accumulated: list[ScholarPaper] = list(turn.papers)
        best_docs: list[dict] = []

        for attempt in range(CONSENSUS_MAX_RETRIES + 1):
            validation = validate_consensus_response(
                raw,
                goal,
                profile_effective,
                anchor,
                attempt=attempt,
                max_retries=CONSENSUS_MAX_RETRIES,
                extracted_papers=accumulated,
            )
            trace(
                f"CURRICULUM v08 Lite validate | status={validation.status} "
                f"attempt={attempt} docs={len(validation.docs)}"
            )
            validator_papers = consensus_docs_to_papers(
                [d.model_dump() for d in validation.docs]
            )
            accumulated = merge_scholar_papers(accumulated, validator_papers)
            if validation.docs:
                best_docs = [d.model_dump() for d in validation.docs]
            if validation.status == "OK":
                break
            if validation.status == "REJECT":
                trace("CURRICULUM v08 Consensus REJECT | no consensus hits")
                return []
            if validation.status == "RETRY" and attempt < CONSENSUS_MAX_RETRIES:
                refinement = (validation.refinement_prompt or "").strip()
                if not refinement:
                    refinement = (
                        "Compare alternative approaches and complexity trade-offs."
                    )
                consensus_refinement = sanitize_message_for_consensus(
                    refinement,
                    sanitize_anchor,
                )
                turn = await session.send_message(consensus_refinement)
                raw = turn.raw_text
                accumulated = merge_scholar_papers(accumulated, turn.papers)
            else:
                break

        selected = _select_curriculum_papers(
            accumulated,
            best_docs,
            CURRICULUM_V08_MAX_PAPERS,
        )
        trace(
            f"CURRICULUM v08 paper select | accumulated={len(accumulated)} "
            f"selected={len(selected)} validator_docs={len(best_docs)}"
        )

        enriched = await enrich_papers_metadata(selected)
        paper_dicts = [p.model_dump() for p in enriched]
        url_map = url_to_source_id_map(build_source_registry(paper_dicts))

        seen_urls: set[str] = set()
        for raw_doc in best_docs:
            hit = _process_validator_doc(raw_doc, anchor, store, seen_urls)
            if hit:
                hits.append(hit)

        for paper in enriched:
            hit = await _process_paper_to_hit(
                paper, anchor, store, url_map, paper_dicts, seen_urls
            )
            if hit:
                hits.append(hit)

        deep = sum(1 for h in hits if _word_count(" ".join(h.key_extracts)) >= 120)
        trace(
            f"CURRICULUM v08 harvest ✓ | hits={len(hits)} deep={deep} LanceDB indexed"
        )
        return hits[:CURRICULUM_V08_MAX_PAPERS]

    except ConsensusLoginRequiredError as exc:
        trace(f"CURRICULUM v08 Consensus login required | {exc}")
        return []
    except Exception as exc:
        trace(f"CURRICULUM v08 harvest ✗ | {exc}")
        return []
    finally:
        await release_consensus_session(session)


def should_use_v08_consensus(
    generation_mode: str = "fast",
    depth_level: str = "",
) -> bool:
    """Consensus Playwright только при generation_mode=consensus (UI)."""
    m = (generation_mode or "fast").strip().lower()
    if m in ("consensus", "deep"):
        return True
    if m == "fast":
        return False
    from knowledge_engine.config import CURRICULUM_USE_V08_CONSENSUS

    return CURRICULUM_USE_V08_CONSENSUS
