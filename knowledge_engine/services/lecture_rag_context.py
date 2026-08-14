"""Локальный LanceDB контекст для генерации лекций (Node Deep-Dive)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from knowledge_engine.config import (
    LECTURE_CHUNK_CA_ENABLED,
    LECTURE_CHUNK_CA_TOP_K,
    LECTURE_RAG_CANDIDATE_LIMIT,
    LECTURE_RAG_COLLECT_TIMEOUT_SEC,
    LECTURE_RAG_KNODE_CANDIDATE_LIMIT,
    LECTURE_RAG_LIGHT_TIMEOUT_SEC,
    LECTURE_RAG_MMR_TOP_K,
    LECTURE_RAG_RERANK_TIMEOUT_SEC,
    LECTURE_RAG_TOP_K,
    LIGHT_RAG_MIN_COSINE_SIM,
)
from knowledge_engine.db.source_links import get_source_link_archive
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.article_diagram_context import (
    build_lecture_pinned_diagrams_block,
    format_article_mermaids_for_source,
)
from knowledge_engine.services.blocking_pools import (
    pool_light,
    pool_rag_ce,
    pool_rag_io,
    run_blocking,
    run_blocking_timed,
)
from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
    format_primary_whitelist_foundation,
)
from knowledge_engine.services.lecture_context_rerank import (
    LectureContextCandidate,
    cross_attention_select_lecture_candidates_sync,
    diversify_lecture_candidates_sync,
    fallback_dedupe_candidates,
)
from knowledge_engine.services.lecture_pipeline import (
    LectureRagStats,
    build_lecture_rag_stats,
)
from knowledge_engine.services.lecture_rag_source_scope import LectureRagSourceScope
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.memory.light_rag import LightRAG
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock, NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import get_session
from knowledge_engine.ui.run_log import trace

LECTURE_RAG_FALLBACK = (
    "Локальный конспект не найден. Сгенерируй лекцию на основе фундаментальных "
    "Best Practices архитектуры."
)


@dataclass(frozen=True)
class LectureRagContextResult:
    context: str
    stats: LectureRagStats
    citation_registry_block: str = ""
    inspector_chunks: tuple[dict[str, object], ...] = ()


def _empty_rag_result(context: str = LECTURE_RAG_FALLBACK) -> LectureRagContextResult:
    return LectureRagContextResult(
        context=context,
        stats=build_lecture_rag_stats([], [], []),
        citation_registry_block="",
    )


def empty_lecture_rag_context_result(
    context: str = LECTURE_RAG_FALLBACK,
) -> LectureRagContextResult:
    """Публичный fallback после engine-level timeout."""
    return _empty_rag_result(context)


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _build_search_query(
    node: NodeDataInput,
    user_query: str,
    curriculum_goal: str = "",
) -> str:
    concepts = ", ".join(node.core_concepts[:8])
    parts = [
        curriculum_goal,
        node.title,
        node.brief_summary,
        node.category,
        concepts,
        f"node_id:{node.node_id}",
        (user_query or "").strip(),
    ]
    return "\n".join(p for p in parts if (p or "").strip())[:4000]


def _format_document_summary(
    ds: DocumentSummary, index: int, tag: str = "Конспект"
) -> str:
    from knowledge_engine.schemas.extraction import format_takeaways_for_tutor

    lines = [
        f"### {tag} {index}: {ds.title}",
        f"URL: {ds.url}",
    ]
    if ds.cs_concepts:
        lines.append("Концепты: " + ", ".join(ds.cs_concepts[:16]))
    if ds.key_takeaways:
        # Knowledge Triangulation: 3 явных блока вместо единого полотна
        lines.append(
            format_takeaways_for_tutor(ds.key_takeaways[:24], max_per_bucket=14)
        )
    if ds.failure_modes:
        lines.append(
            "Failure modes:\n" + "\n".join(f"- {t}" for t in ds.failure_modes[:8])
        )
    if ds.diagram_descriptions:
        lines.append(
            "Схемы (описания):\n"
            + "\n".join(f"- {t}" for t in ds.diagram_descriptions[:6])
        )
    mermaid_block = format_article_mermaids_for_source(url=ds.url)
    if mermaid_block:
        lines.append(mermaid_block)
    return "\n".join(lines)


def _format_knowledge_node(
    index: int, content: str, source_url: str | None, level: str
) -> str:
    head = f"### Knowledge graph {index} [{level}]"
    if source_url:
        head += f" ({source_url})"
    return f"{head}\n{content[:6000]}"


def _format_registry_stub(
    index: int,
    url: str,
    title: str,
    snippet: str,
) -> str:
    lines = [f"### Ссылка маршрута/registry {index}: {title or url}", f"URL: {url}"]
    if snippet:
        lines.append(f"Контекст: {snippet[:1200]}")
    return "\n".join(lines)


def _plain_from_document_summary(ds: DocumentSummary) -> str:
    parts = [
        ds.title,
        " ".join(ds.cs_concepts or []),
        " ".join(ds.key_takeaways or []),
        " ".join(ds.failure_modes or []),
        " ".join(ds.diagram_descriptions or []),
    ]
    return "\n".join(p for p in parts if (p or "").strip())[:6000]


def _merge_plain_texts(texts: list[str]) -> str:
    seen_lines: set[str] = set()
    blocks: list[str] = []
    for raw in texts:
        t = (raw or "").strip()
        if not t:
            continue
        chunk_lines: list[str] = []
        for ln in [x.strip() for x in t.splitlines() if x.strip()]:
            key = ln.lower()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            chunk_lines.append(ln)
        if chunk_lines:
            blocks.append("\n".join(chunk_lines))
    return "\n\n".join(blocks).strip()


def _lance_vector(vec: list[float] | None) -> np.ndarray | None:
    if not vec:
        return None
    return np.asarray(vec, dtype=np.float64)


def _finalize_lecture_citation_candidates(
    selected: list[LectureContextCandidate],
) -> list[LectureContextCandidate]:
    """Единые [R1]…[Rn] в formatted + source_index для реестра и склейки.

    Кандидаты сортируются по trust_score (desc), затем по retrieval_score —
    авторитетные docs попадают в промпт первыми (меньшие R-индексы).
    """
    if not selected:
        return []
    ordered = sorted(
        selected,
        key=lambda c: (
            float(c.trust_score if c.trust_score is not None else 1.0),
            float(c.retrieval_score or 0.0),
        ),
        reverse=True,
    )
    out: list[LectureContextCandidate] = []
    for i, c in enumerate(ordered, 1):
        idx = i
        formatted = (c.formatted or "").strip()
        plain = (c.plain or formatted or "").strip()
        title = (c.source_title or c.label or f"Source {idx}").strip()[:200]
        trust = float(c.trust_score if c.trust_score is not None else 1.0)
        if formatted.startswith(f"[R{idx}]"):
            body = formatted
        else:
            chunk_hint = ""
            if c.chunk_index > 0 and c.chunks_in_doc > 0:
                chunk_hint = f", Chunk {c.chunk_index}/{c.chunks_in_doc}"
            body = (
                f"[R{idx}] (Source: {title}{chunk_hint}, "
                f"trust={trust:.2f})\n{plain[:6000]}"
            )
        out.append(
            LectureContextCandidate(
                label=c.label,
                formatted=body,
                plain=plain[:6000] if plain else body,
                url_key=c.url_key,
                source_id=c.source_id,
                source_title=title,
                source_index=idx,
                chunk_index=c.chunk_index,
                chunks_in_doc=c.chunks_in_doc,
                retrieval_score=c.retrieval_score,
                trust_score=trust,
                vector_similarity=float(c.vector_similarity or 0.0),
                doc_id=c.doc_id,
                chunk_vector=c.chunk_vector,
                doc_meta_vector=c.doc_meta_vector,
            )
        )
    return out


def build_rag_chunk_inspector_payload(
    selected: list[LectureContextCandidate],
) -> list[dict[str, object]]:
    """Сериализуемый payload для UI RAG Inspector."""
    rows: list[dict[str, object]] = []
    for c in selected:
        if int(c.source_index or 0) <= 0:
            continue
        url = (c.url_key or "").strip()
        rows.append(
            {
                "rag_id": f"R{c.source_index}",
                "title": (c.source_title or "").strip()[:200],
                "url": url if url.startswith("http") else "",
                "chunk_index": int(c.chunk_index or 0),
                "chunks_in_doc": int(c.chunks_in_doc or 0),
                "cosine_score": round(float(c.retrieval_score or 0.0), 4),
                "trust_score": round(
                    float(c.trust_score if c.trust_score is not None else 1.0), 3
                ),
                "vector_similarity": round(float(c.vector_similarity or 0.0), 4),
                "chunk_text": (c.plain or c.formatted or "")[:2000],
                "doc_id": (c.doc_id or c.source_id or "").strip()[:64],
            }
        )
    return rows[:16]


def lookup_lecture_rag_inspector_chunks(
    inspector_rows: list[dict[str, object]],
    rag_ids: list[str],
) -> list[dict[str, object]]:
    """Сопоставить [R6]… из выделения с payload lecture_rag_inspector в памяти."""
    if not inspector_rows or not rag_ids:
        return []
    by_id: dict[str, dict[str, object]] = {}
    for raw in inspector_rows:
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("rag_id") or "").strip().upper()
        if rid:
            by_id[rid] = raw
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_id in rag_ids:
        rid = str(raw_id or "").strip().upper()
        if not rid:
            continue
        if not rid.startswith("R"):
            rid = f"R{rid}"
        if rid in seen:
            continue
        row = by_id.get(rid)
        if row is not None:
            out.append(row)
            seen.add(rid)
    return out


def format_highlight_rag_chunks_block(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    lines = [
        "--- ТОЧНЫЕ ИСХОДНЫЕ ЧАНКИ ЛЕКЦИИ ДЛЯ ВЫДЕЛЕННОГО ФРАГМЕНТА ---",
    ]
    for row in rows:
        rid = str(row.get("rag_id") or "").strip()
        text = str(row.get("chunk_text") or "").strip()
        if not rid:
            continue
        if len(text) > 6000:
            text = text[:6000] + "…"
        lines.append(f"- [{rid}]: {text or '(пусто)'}")
    return "\n".join(lines)


def build_rag_chunk_citation_registry(
    selected: list[LectureContextCandidate],
) -> str:
    """Реестр R1…Rn для сносок в lecture_body (префиксы в НАЧАЛО МАТЕРИАЛА)."""
    ordered = sorted(
        [c for c in selected if int(c.source_index or 0) > 0],
        key=lambda c: int(c.source_index),
    )
    if not ordered:
        return ""
    lines = [
        "### RAG CHUNK SOURCE INDEX (сноски [R1]…[Rn] для блока НАЧАЛО МАТЕРИАЛА)",
        "Каждая строка материала начинается с [Rx] — в lecture_body используй тот же x для факта из этой строки.",
        "Мульти-цитирование: [R1][R3] подряд, если факт подтверждён несколькими RAG-фрагментами.",
        "Не путать с [S1]… из SOURCE REGISTRY курса (whitelist) — это отдельное пространство имён.",
        "Правило Reduce: факт из строки [RN] → маркер [RN] в конце предложения/абзаца.",
    ]
    seen: set[int] = set()
    for c in ordered:
        idx = int(c.source_index)
        if idx in seen:
            continue
        seen.add(idx)
        sid = f"R{idx}"
        title = (c.source_title or c.label or sid).strip()[:200]
        url = (c.url_key or "").strip()
        trust = float(c.trust_score if c.trust_score is not None else 1.0)
        if url.startswith("http"):
            lines.append(f"- [{sid}] trust={trust:.2f} | {title} | {url}")
        else:
            sid_src = (c.source_id or c.label or "local").strip()[:120]
            lines.append(f"- [{sid}] trust={trust:.2f} | {title} | local:{sid_src}")
    lines.append(
        "used_sources: для каждого [Rx] из RAG-тела добавь запись asset_id=Rx "
        "(title/url — из строки реестра выше). [S*] — только для SOURCE REGISTRY курса."
    )
    return "\n".join(lines)


def _stitch_candidates_by_url(
    pinned: list[str],
    selected: list[LectureContextCandidate],
) -> str:
    if selected and any(c.source_index > 0 for c in selected):
        ordered = sorted(selected, key=lambda c: c.source_index)
        stitched_blocks = [
            (c.formatted or "").strip() for c in ordered if (c.formatted or "").strip()
        ]
    else:
        by_url: dict[str, list[LectureContextCandidate]] = {}
        for c in selected:
            key = (c.url_key or "").strip().lower() or f"label:{c.label}"
            by_url.setdefault(key, []).append(c)

        stitched_blocks: list[str] = []
        for key, group in by_url.items():
            plains = [(c.plain or c.formatted or "").strip() for c in group]
            merged = _merge_plain_texts(plains)
            if not merged:
                merged = _merge_plain_texts(
                    [(c.formatted or "").strip() for c in group]
                )
            head = group[0]
            if head.url_key or key.startswith("http"):
                title = (head.label or head.url_key or "source").strip()
                block = f"### {title}\n{merged}"
            else:
                block = merged or (head.formatted or "").strip()
            if block.strip():
                stitched_blocks.append(block.strip())

    parts = [p.strip() for p in pinned if (p or "").strip()] + stitched_blocks
    out = "\n\n---\n\n".join(parts)
    from knowledge_engine.config import (
        LECTURE_RAG_CONTEXT_MAX_CHARS,
        LECTURE_RAG_PROMPT_MAX_CHARS,
    )

    cap = max(1500, min(LECTURE_RAG_CONTEXT_MAX_CHARS, LECTURE_RAG_PROMPT_MAX_CHARS))
    if len(out) > cap:
        trace(f"LECTURE_RAG context cap | {len(out)} → {cap} chars")
        out = out[: cap - 20].rstrip() + "\n… [truncated]"
    return out


_PINNED_RAG_LABEL = "pinned_rag"
_MAPPED_CHUNKS_PER_DOC = 2


def _chunk_dedupe_key(c: LectureContextCandidate) -> str:
    if c.doc_id and c.chunk_index:
        return f"{c.doc_id}:{c.chunk_index}"
    cid = (c.source_id or "").strip()
    if cid:
        return f"{cid}:{c.chunk_index}"
    return (c.plain or c.formatted or "")[:120]


def _merge_mandatory_rag_after_rerank(
    mandatory: list[LectureContextCandidate],
    selected: list[LectureContextCandidate],
) -> list[LectureContextCandidate]:
    if not mandatory:
        return selected
    seen: set[str] = set()
    out: list[LectureContextCandidate] = []
    for c in mandatory:
        key = _chunk_dedupe_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    for c in selected:
        key = _chunk_dedupe_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _score_rag_chunk_rows_for_query(
    query: str,
    rows: list[dict[str, object]],
) -> list[tuple[float, dict[str, object]]]:
    from knowledge_engine.config import ACADEMIC_RERANK_ENABLED
    from knowledge_engine.db.rag_chunks_schema import COL_CHUNK_VECTOR, COL_TRUST_SCORE
    from knowledge_engine.src.retrieval.academic_rerank import (
        RerankSignals,
        hybrid_academic_score,
    )
    from knowledge_engine.src.services.openalex_evaluator import (
        coerce_trust_score,
        final_retrieval_score,
        passes_trust_hard_cutoff,
    )

    store = VectorStore()
    qv = np.asarray(
        store._embeddings.embed_query((query or "")[:8000]),
        dtype=np.float64,
    )
    qn = float(np.linalg.norm(qv))
    if qn > 0:
        qv = qv / qn
    scored: list[tuple[float, dict[str, object]]] = []
    dropped = 0
    for row in rows:
        cv_raw = row.get(COL_CHUNK_VECTOR)
        if cv_raw is None:
            scored.append((0.0, row))
            continue
        cv = np.asarray(list(cv_raw), dtype=np.float64)
        cn = float(np.linalg.norm(cv))
        cos = float(np.dot(qv, cv / cn)) if cn > 0 else 0.0
        trust = coerce_trust_score(row.get(COL_TRUST_SCORE), default=1.0)
        if not passes_trust_hard_cutoff(cos, trust):
            dropped += 1
            continue
        enriched = dict(row)
        enriched["_cosine_raw"] = cos
        enriched["_trust_score"] = trust
        if ACADEMIC_RERANK_ENABLED:
            rank = hybrid_academic_score(
                RerankSignals(
                    relevance_sim=cos,
                    trust_score=trust,
                    citation_count=int(row.get("citation_count") or 0),
                    year=_year_from_row_meta(row),
                )
            )
        else:
            rank = final_retrieval_score(cos, trust)
        scored.append((rank, enriched))
    if dropped:
        trace(f"LECTURE_RAG hard_cutoff ⊘ | mapped_dropped={dropped}")
    return scored


def _year_from_row_meta(row: dict[str, object]) -> int | None:
    for key in ("year", "published_year", "publication_year"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            year = int(raw)
        except (TypeError, ValueError):
            continue
        if 1990 <= year <= 2100:
            return year
    return None


def _append_mandatory_mapped_rag_chunks(
    query: str,
    candidates: list[LectureContextCandidate],
    seen_chunk_ids: set[str],
    *,
    mapped_doc_ids: list[str],
    node_id: str = "",
    max_per_doc: int = _MAPPED_CHUNKS_PER_DOC,
) -> int:
    """Гарантированные чанки привязанных статей (label pinned_rag)."""
    from knowledge_engine.db.rag_chunks_schema import (
        COL_CHUNK_ID,
        COL_CHUNK_INDEX,
        COL_CHUNK_TEXT,
        COL_CHUNK_VECTOR,
        COL_CHUNKS_IN_DOC,
        COL_DOC_META_VECTOR,
        COL_TITLE,
        COL_URL,
    )

    if not mapped_doc_ids:
        trace(f"LECTURE_RAG_MAPPED_FETCH | node_id={node_id} mapped_doc_ids=0 chunks=0")
        return 0

    store = VectorStore()
    fetched = 0
    for doc_id in mapped_doc_ids:
        rows = store.fetch_rag_chunks_by_doc_id(doc_id)
        if not rows:
            continue
        scored = _score_rag_chunk_rows_for_query(query, rows)
        scored.sort(key=lambda x: -x[0])
        for cos, row in scored[: max(1, max_per_doc)]:
            cid = str(row.get(COL_CHUNK_ID) or "")
            if not cid or cid in seen_chunk_ids:
                continue
            text = (row.get(COL_CHUNK_TEXT) or "").strip()
            if len(text) < 24:
                continue
            seen_chunk_ids.add(cid)
            url = (row.get(COL_URL) or "").strip()
            key = _normalize_url(url) if url.startswith("http") else f"doc:{doc_id}"
            candidates.append(
                LectureContextCandidate(
                    label=_PINNED_RAG_LABEL,
                    formatted="",
                    plain=text[:6000],
                    url_key=key,
                    source_id=doc_id,
                    source_title=str(row.get(COL_TITLE) or url or doc_id)[:200],
                    doc_id=doc_id,
                    chunk_index=int(row.get(COL_CHUNK_INDEX) or 0),
                    chunks_in_doc=int(row.get(COL_CHUNKS_IN_DOC) or 0),
                    retrieval_score=float(cos),
                    trust_score=float(row.get("_trust_score") or 1.0),
                    vector_similarity=float(row.get("_cosine_raw") or cos),
                    doc_meta_vector=_lance_vector(
                        list(row.get(COL_DOC_META_VECTOR) or [])
                    ),
                    chunk_vector=_lance_vector(list(row.get(COL_CHUNK_VECTOR) or [])),
                )
            )
            fetched += 1

    trace(
        f"LECTURE_RAG_MAPPED_FETCH | node_id={node_id} "
        f"mapped_docs={len(mapped_doc_ids)} chunks={fetched}"
    )
    return fetched


def _append_fine_rag_chunk_candidates(
    query: str,
    candidates: list[LectureContextCandidate],
    seen_chunk_ids: set[str],
    *,
    scope: LectureRagSourceScope | None,
    node_id: str = "",
) -> None:
    from knowledge_engine.config import (
        DOC_GATE_THRESHOLD,
        LECTURE_RAG_PREFILTER_MIN_PRIMARY_CHUNKS,
        LECTURE_RAG_SCOPE_SECONDARY_PENALTY,
        LECTURE_RAG_SECONDARY_SCORE_FLOOR,
        RAG_CHUNK_SEARCH_LIMIT,
    )
    from knowledge_engine.db.rag_chunks_schema import (
        COL_CHUNK_ID,
        COL_CHUNK_INDEX,
        COL_CHUNK_TEXT,
        COL_CHUNK_VECTOR,
        COL_CHUNKS_IN_DOC,
        COL_DOC_ID,
        COL_DOC_META_VECTOR,
        COL_TITLE,
        COL_URL,
    )

    store = VectorStore()
    primary_ids: list[str] = []
    secondary_ids: list[str] = []
    if scope is not None:
        primary_ids = list(scope.primary_doc_ids)
        secondary_ids = list(scope.library_doc_ids)

    if scope is not None and not primary_ids and not secondary_ids:
        trace(
            f"LECTURE_RAG_PREFILTER ⊘ | node_id={node_id} "
            "no scoped doc_ids — skip global rag_chunks search"
        )
        return

    allow_primary = primary_ids if scope is not None else None
    rows = store.search_rag_chunk_rows(
        query,
        limit=RAG_CHUNK_SEARCH_LIMIT,
        doc_gate_threshold=DOC_GATE_THRESHOLD,
        allowed_doc_ids=allow_primary,
        prefilter=True,
    )
    scanned = store.count_rag_chunks_in_scope(allow_primary)
    trace(
        f"LECTURE_RAG_PREFILTER | node_id={node_id} scope=primary "
        f"allowed_sources_count={len(primary_ids)} "
        f"total_chunks_scanned={scanned} returned={len(rows)}"
    )

    need_secondary = False
    if scope is not None and secondary_ids:
        max_cos = max((float(r.get("_cosine_chunk") or 0) for r in rows), default=0.0)
        if len(rows) < LECTURE_RAG_PREFILTER_MIN_PRIMARY_CHUNKS or max_cos < (
            LECTURE_RAG_SECONDARY_SCORE_FLOOR
        ):
            need_secondary = True

    if need_secondary:
        extra = store.search_rag_chunk_rows(
            query,
            limit=RAG_CHUNK_SEARCH_LIMIT,
            doc_gate_threshold=DOC_GATE_THRESHOLD,
            allowed_doc_ids=secondary_ids,
            prefilter=True,
            relevance_penalty=LECTURE_RAG_SCOPE_SECONDARY_PENALTY,
        )
        seen_row: set[str] = {str(r.get(COL_CHUNK_ID) or "") for r in rows}
        for row in extra:
            cid = str(row.get(COL_CHUNK_ID) or "")
            if cid and cid not in seen_row:
                rows.append(row)
                seen_row.add(cid)
        trace(
            f"LECTURE_RAG_PREFILTER | node_id={node_id} scope=secondary "
            f"allowed_sources_count={len(secondary_ids)} "
            f"added={len(extra)} penalty={LECTURE_RAG_SCOPE_SECONDARY_PENALTY:.2f}"
        )

    rows.sort(key=lambda r: float(r.get("_cosine_chunk") or 0.0), reverse=True)

    if not rows:
        return
    trace(f"LECTURE_RAG fine_chunks ▶ | lancedb_rows={len(rows)}")
    allowed_doc_set = set(primary_ids) | set(secondary_ids)
    for row in rows:
        cid = str(row.get(COL_CHUNK_ID) or "")
        if not cid or cid in seen_chunk_ids:
            continue
        doc_id = str(row.get(COL_DOC_ID) or "")
        if scope is not None and allowed_doc_set and doc_id not in allowed_doc_set:
            continue
        text = (row.get(COL_CHUNK_TEXT) or "").strip()
        if len(text) < 24:
            continue
        seen_chunk_ids.add(cid)
        url = (row.get(COL_URL) or "").strip()
        key = (
            _normalize_url(url)
            if url.startswith("http")
            else f"doc:{row.get(COL_DOC_ID)}"
        )
        doc_id = str(row.get(COL_DOC_ID) or key)
        candidates.append(
            LectureContextCandidate(
                label="rag_fine_chunk",
                formatted="",
                plain=text[:6000],
                url_key=key,
                source_id=doc_id,
                source_title=str(row.get(COL_TITLE) or url or doc_id)[:200],
                doc_id=doc_id,
                chunk_index=int(row.get(COL_CHUNK_INDEX) or 0),
                chunks_in_doc=int(row.get(COL_CHUNKS_IN_DOC) or 0),
                retrieval_score=float(row.get("_cosine_chunk") or 0.0),
                trust_score=float(row.get("_trust_score") or 1.0),
                vector_similarity=float(row.get("_cosine_raw") or 0.0),
                doc_meta_vector=_lance_vector(list(row.get(COL_DOC_META_VECTOR) or [])),
                chunk_vector=_lance_vector(list(row.get(COL_CHUNK_VECTOR) or [])),
            )
        )


def _collect_rerank_candidates_sync(
    curriculum_id: str,
    node: NodeDataInput,
    query: str,
) -> tuple[list[str], list[LectureContextCandidate], list[str]]:
    """
    Пул кандидатов для CE/MMR (whitelist foundation — pinned, не в пуле).
    """
    pool_limit = max(LECTURE_RAG_CANDIDATE_LIMIT, LECTURE_RAG_MMR_TOP_K + 2)
    legacy_k = max(2, min(LECTURE_RAG_TOP_K, 5))
    candidate_limit = max(pool_limit, legacy_k)

    pinned: list[str] = []
    candidates: list[LectureContextCandidate] = []
    seen_urls: set[str] = set()
    seen_chunk_ids: set[str] = set()

    foundation = format_primary_whitelist_foundation(node, curriculum_id)
    if foundation.strip():
        pinned.append(foundation.strip())

    route_links = _collect_route_link_candidates(curriculum_id, node)
    route_urls = [u for u, _, _ in route_links]

    from knowledge_engine.services.lecture_rag_source_scope import (
        build_lecture_rag_source_scope,
        mapped_doc_ids_for_node,
    )

    scope = build_lecture_rag_source_scope(curriculum_id, node, route_urls)
    mapped_doc_ids = mapped_doc_ids_for_node(curriculum_id, node)
    _append_mandatory_mapped_rag_chunks(
        query,
        candidates,
        seen_chunk_ids,
        mapped_doc_ids=mapped_doc_ids,
        node_id=node.node_id,
    )
    allowed_url_norm = {
        _normalize_url(u) for u in scope.primary_urls if u.startswith("http")
    }
    for u in scope.library_urls:
        if u.startswith("http"):
            allowed_url_norm.add(_normalize_url(u))

    _append_fine_rag_chunk_candidates(
        query,
        candidates,
        seen_chunk_ids,
        scope=scope,
        node_id=node.node_id,
    )

    if route_urls:
        url_docs = _summaries_for_urls_with_vectors(
            route_urls,
            min(len(route_urls), candidate_limit),
        )
        for i, (ds, lance_vec) in enumerate(url_docs, 1):
            key = _normalize_url(ds.url)
            seen_urls.add(key)
            formatted = _format_document_summary(
                ds, i, tag="Конспект по ссылке маршрута"
            )
            doc_vec = _lance_vector(lance_vec)
            candidates.append(
                LectureContextCandidate(
                    label="route_doc",
                    formatted=formatted,
                    plain=_plain_from_document_summary(ds),
                    url_key=key,
                    source_id=key,
                    source_title=(ds.title or "").strip(),
                    doc_meta_vector=doc_vec,
                )
            )

    for i, (url, title, snippet) in enumerate(route_links, 1):
        key = _normalize_url(url)
        if key in seen_urls:
            continue
        formatted = _format_registry_stub(i, url, title, snippet)
        plain = "\n".join(p for p in [title, snippet] if (p or "").strip())[:4000]
        candidates.append(
            LectureContextCandidate(
                label="registry_stub",
                formatted=formatted,
                plain=plain or formatted[:2000],
                url_key=key,
                source_id=key,
                source_title=(title or url).strip()[:200],
            )
        )

    if allowed_url_norm:
        docs = _summaries_for_urls_with_vectors(
            list(scope.primary_urls) or route_urls,
            min(len(route_urls) or len(scope.primary_urls), candidate_limit),
        )
    else:
        docs = _hybrid_document_summaries_with_vectors(query, candidate_limit)
    for ds, lance_vec in docs:
        key = _normalize_url(ds.url)
        if allowed_url_norm and key not in allowed_url_norm:
            continue
        if key in seen_urls:
            continue
        seen_urls.add(key)
        doc_idx = sum(1 for c in candidates if c.label == "hybrid_semantic")
        formatted = _format_document_summary(
            ds, doc_idx + 1, tag="Семантический конспект"
        )
        doc_vec = _lance_vector(lance_vec)
        candidates.append(
            LectureContextCandidate(
                label="hybrid_semantic",
                formatted=formatted,
                plain=_plain_from_document_summary(ds),
                url_key=key,
                source_id=key,
                source_title=(ds.title or "").strip(),
                doc_meta_vector=doc_vec,
            )
        )

    knode_limit = max(2, LECTURE_RAG_KNODE_CANDIDATE_LIMIT)
    knodes = _hybrid_knowledge_nodes_with_vectors(query, knode_limit)
    for i, (content, source_url, level, lance_vec) in enumerate(knodes, 1):
        url_key = _normalize_url(source_url or "")
        if allowed_url_norm and url_key and url_key not in allowed_url_norm:
            continue
        formatted = _format_knowledge_node(i, content, source_url, level)
        sid = url_key or f"knode:{i}"
        doc_vec = _lance_vector(lance_vec)
        candidates.append(
            LectureContextCandidate(
                label="knowledge_node",
                formatted=formatted,
                plain=(content or "")[:6000],
                url_key=url_key,
                source_id=sid,
                source_title=(source_url or f"Knowledge graph {i}")[:200],
                doc_meta_vector=doc_vec,
                chunk_vector=doc_vec,
            )
        )

    return pinned, candidates, route_urls


def _minimal_collect_fallback(
    curriculum_id: str,
    node: NodeDataInput,
) -> tuple[list[str], list[LectureContextCandidate], list[str]]:
    """После таймаута collect — только whitelist foundation + route URL (без LanceDB)."""
    trace("LECTURE_RAG minimal fallback ▶ | whitelist + route URLs only")
    pinned: list[str] = []
    foundation = format_primary_whitelist_foundation(node, curriculum_id)
    if foundation.strip():
        pinned.append(foundation.strip())
    route_urls = [u for u, _, _ in _collect_route_link_candidates(curriculum_id, node)]
    return pinned, [], route_urls


async def _collect_rerank_candidates_timed(
    curriculum_id: str,
    node: NodeDataInput,
    query: str,
) -> tuple[list[str], list[LectureContextCandidate], list[str]]:
    try:
        return await run_blocking_timed(
            pool_rag_io(),
            LECTURE_RAG_COLLECT_TIMEOUT_SEC,
            _collect_rerank_candidates_sync,
            curriculum_id,
            node,
            query,
        )
    except asyncio.TimeoutError:
        trace(
            f"LECTURE_RAG collect timeout | {LECTURE_RAG_COLLECT_TIMEOUT_SEC}s "
            f"(фоновый collect может ещё идти) → minimal fallback"
        )
        return _minimal_collect_fallback(curriculum_id, node)


def _pinned_blocks_for_stats(pinned: list[str]) -> list[LectureContextCandidate]:
    out: list[LectureContextCandidate] = []
    for block in pinned or []:
        text = (block or "").strip()
        if not text:
            continue
        out.append(
            LectureContextCandidate(
                label="pinned_foundation",
                formatted=text,
                plain=text,
                url_key="",
            )
        )
    return out


async def _light_rag_hits_timed(rag: LightRAG, query: str, limit: int):
    try:
        return await asyncio.wait_for(
            _light_rag_hits(rag, query, limit),
            timeout=LECTURE_RAG_LIGHT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        trace(f"LECTURE_RAG light_rag timeout | {LECTURE_RAG_LIGHT_TIMEOUT_SEC}s")
        return []


async def _light_rag_hits(rag: LightRAG, query: str, limit: int):
    return await rag.vector_search(
        query,
        limit,
        kinds=frozenset({"profile", "fact"}),
        min_cosine=LIGHT_RAG_MIN_COSINE_SIM,
    )


async def _apply_rerank_mmr(
    query: str,
    candidates: list[LectureContextCandidate],
    *,
    node_title: str = "",
    core_concepts: list[str] | None = None,
) -> list[LectureContextCandidate]:
    if not candidates:
        return []
    from knowledge_engine.src.retrieval.academic_rerank import (
        presort_lecture_candidates,
    )

    # Hybrid academic pre-sort (sim×trust×cites×recency) before CE/MMR.
    candidates = presort_lecture_candidates(candidates)
    if LECTURE_CHUNK_CA_ENABLED:
        try:
            selected = await run_blocking_timed(
                pool_rag_ce(),
                LECTURE_RAG_RERANK_TIMEOUT_SEC,
                cross_attention_select_lecture_candidates_sync,
                query,
                candidates,
                node_title=node_title,
                core_concepts=core_concepts,
            )
            if selected:
                return selected
        except Exception as exc:
            trace(f"LECTURE_CHUNK_CA fallback | {exc}")
    try:
        selected = await run_blocking_timed(
            pool_rag_ce(),
            LECTURE_RAG_RERANK_TIMEOUT_SEC,
            diversify_lecture_candidates_sync,
            query,
            candidates,
        )
        return selected
    except Exception as exc:
        trace(f"LECTURE_RAG rerank/mmr fallback | {exc}")
        return fallback_dedupe_candidates(
            candidates, max(LECTURE_RAG_MMR_TOP_K, LECTURE_CHUNK_CA_TOP_K)
        )


def _legacy_concat_chunks(
    pinned: list[str],
    candidates: list[LectureContextCandidate],
) -> list[str]:
    """Склейка без CE/MMR (откат)."""
    deduped = fallback_dedupe_candidates(
        candidates,
        max(LECTURE_RAG_MMR_TOP_K, LECTURE_RAG_TOP_K),
    )
    return pinned + [c.formatted for c in deduped]


def _rerank_focus_query(
    node: NodeDataInput,
    user_query: str,
    curriculum_goal: str,
) -> str:
    focus = (user_query or "").strip()
    if focus:
        return focus[:2000]
    return _build_search_query(node, "", curriculum_goal)[:2000]


def _append_light_rag_candidates(
    candidates: list[LectureContextCandidate],
    hits: list[tuple[float, str, dict]],
) -> None:
    for sim, text, meta_h in hits:
        body = (text or "").strip()
        if len(body) < 24:
            continue
        node_tag = (meta_h.get("node_id") or "").strip()
        category = (meta_h.get("category") or "").strip()
        header = f"[LightRAG score={sim:.2f}"
        if node_tag:
            header += f" node={node_tag}"
        if category:
            header += f" category={category}"
        header += "]"
        formatted = f"{header}\n{body[:6000]}"
        candidates.append(
            LectureContextCandidate(
                label="light_rag",
                formatted=formatted,
                plain=body[:6000],
                url_key="",
                source_id=f"light_rag:{node_tag or category or len(candidates)}",
                source_title=f"LightRAG {node_tag or category or 'hit'}".strip(),
            )
        )


def _hybrid_document_summaries_with_vectors(
    query: str, limit: int
) -> list[tuple[DocumentSummary, list[float]]]:
    store = VectorStore()
    try:
        return store.hybrid_search_with_vectors(query, limit=limit)
    except RuntimeError:
        return []
    except Exception as exc:
        trace(f"LECTURE_RAG hybrid_search skip | {exc}")
        return []


def _summaries_for_urls_with_vectors(
    urls: list[str], limit: int
) -> list[tuple[DocumentSummary, list[float]]]:
    store = VectorStore()
    try:
        return store.fetch_summaries_by_urls_with_vectors(urls, limit=limit)
    except Exception as exc:
        trace(f"LECTURE_RAG fetch_by_url skip | {exc}")
        return []


def _hybrid_knowledge_nodes_with_vectors(
    query: str, limit: int
) -> list[tuple[str, str | None, str, list[float]]]:
    store = VectorStore()
    try:
        hits = store.hybrid_search_nodes_with_vectors(query, limit=limit)
        return [
            (n.content, n.source_url, n.level, vec)
            for n, vec in hits
            if (n.content or "").strip()
        ]
    except Exception as exc:
        trace(f"LECTURE_RAG knowledge_nodes skip | {exc}")
        return []


def _hybrid_document_summaries(query: str, limit: int) -> list[DocumentSummary]:
    store = VectorStore()
    try:
        return store.hybrid_search(query, limit=limit)
    except RuntimeError:
        return []
    except Exception as exc:
        trace(f"LECTURE_RAG hybrid_search skip | {exc}")
        return []


def _summaries_for_urls(urls: list[str], limit: int) -> list[DocumentSummary]:
    store = VectorStore()
    try:
        return store.fetch_summaries_by_urls(urls, limit=limit)
    except Exception as exc:
        trace(f"LECTURE_RAG fetch_by_url skip | {exc}")
        return []


def _hybrid_knowledge_nodes(
    query: str, limit: int
) -> list[tuple[str, str | None, str]]:
    store = VectorStore()
    try:
        nodes = store.hybrid_search_nodes(query, limit=limit)
        return [
            (n.content, n.source_url, n.level)
            for n in nodes
            if (n.content or "").strip()
        ]
    except Exception as exc:
        trace(f"LECTURE_RAG knowledge_nodes skip | {exc}")
        return []


def _collect_route_link_candidates(
    curriculum_id: str,
    node: NodeDataInput,
) -> list[tuple[str, str, str]]:
    """URL + title + snippet из маршрута, сессии ноды и архива ссылок."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(url: str, title: str = "", snippet: str = "") -> None:
        u = (url or "").strip()
        if not u.startswith("http"):
            return
        key = _normalize_url(u)
        if key in seen:
            return
        seen.add(key)
        out.append((u, (title or "").strip(), (snippet or "").strip()))

    cid = (curriculum_id or "").strip()
    nid = node.node_id

    graph = get_curriculum_graph(cid) if cid else None
    route_by_id: dict[str, dict] = {}
    if graph:
        for rs in graph.get("route_sources") or []:
            if isinstance(rs, dict) and rs.get("source_id"):
                route_by_id[str(rs["source_id"])] = rs
        for raw in graph.get("nodes") or []:
            if str(raw.get("node_id") or "") != nid:
                continue
            pid = str(raw.get("primary_source_id") or "").strip()
            if pid and pid in route_by_id:
                rs = route_by_id[pid]
                add(
                    str(rs.get("url") or ""),
                    str(rs.get("source_name") or ""),
                    str(rs.get("why_read") or ""),
                )
            for u in raw.get("resource_urls") or []:
                add(str(u))
            for lr in raw.get("learning_resources") or []:
                if isinstance(lr, dict):
                    add(
                        str(lr.get("url") or ""),
                        str(lr.get("title") or ""),
                        str(lr.get("why_read") or ""),
                    )

    if cid:
        session = get_session(cid, nid)
        for ref in session.content.references or []:
            add(
                ref.url,
                ref.title or ref.source_name,
                "\n".join(x for x in [ref.why_read, ref.key_focus] if x),
            )
        from knowledge_engine.src.node_deep_dive.session_store import (
            get_all_sessions_for_curriculum,
        )

        blob = get_all_sessions_for_curriculum(cid).get(nid) or {}
        for reg in blob.get("source_registry") or []:
            if isinstance(reg, dict):
                add(
                    str(reg.get("url") or ""),
                    str(reg.get("title") or ""),
                    str(reg.get("snippet") or ""),
                )

    meta = get_curriculum_meta(cid) if cid else None
    goal = (meta.get("target_goal") or "") if meta else ""
    if goal:
        try:
            archive = get_source_link_archive()
            for u in archive.get_reusable_urls(
                f"{goal} {node.title}",
                explored=set(),
                limit=6,
                min_trust=0.35,
            ):
                add(u, "archive", goal[:200])
        except Exception:
            pass

    return out[:16]


async def retrieve_lecture_rag_context(
    node: NodeDataInput,
    user_query: str,
    curriculum_id: str = "",
) -> LectureRagContextResult:
    try:
        return await _retrieve_lecture_rag_context_impl(node, user_query, curriculum_id)
    except Exception as exc:
        from knowledge_engine.ui.errors import trace_exception

        trace_exception(exc, "LECTURE_RAG retrieve")
        return empty_lecture_rag_context_result()


async def _retrieve_lecture_rag_context_impl(
    node: NodeDataInput,
    user_query: str,
    curriculum_id: str = "",
) -> LectureRagContextResult:
    """
    LanceDB + LightRAG → chunk cross-attention/MMR (или CE+MMR) → склейка для dense_material.
    """
    trace(
        f"LECTURE_RAG ▶ retrieve | node={node.node_id} "
        f"curriculum={(curriculum_id or '')[:32]}"
    )

    def _prelude() -> tuple[NodeDataInput, str, str]:
        enriched = enrich_node_learning_materials_from_graph(node, curriculum_id)
        meta = get_curriculum_meta(curriculum_id.strip()) if curriculum_id else None
        goal = (meta.get("target_goal") or "") if meta else ""
        q = _build_search_query(enriched, user_query, goal)
        return enriched, q, goal

    node, query, curriculum_goal = await run_blocking(pool_light(), _prelude)
    if not query:
        return _empty_rag_result()

    pool_limit = max(LECTURE_RAG_CANDIDATE_LIMIT, LECTURE_RAG_MMR_TOP_K + 2)
    trace(
        f"LECTURE_RAG ▶ collect | node={node.node_id} query_len={len(query)} "
        f"pool_limit={pool_limit}"
    )

    try:
        pinned, candidates, route_urls = await _collect_rerank_candidates_timed(
            curriculum_id,
            node,
            query,
        )
        trace(
            f"LECTURE_RAG collect ✓ | candidates={len(candidates)} "
            f"pinned_blocks={len(pinned)} route_urls={len(route_urls)}"
        )
        rag = LightRAG()
        hits = await _light_rag_hits_timed(rag, query, pool_limit)
        _append_light_rag_candidates(candidates, hits)

        trace(
            f"LECTURE_RAG pool ▶ | candidates={len(candidates)} pinned={len(pinned)} "
            f"route_urls={len(route_urls)} light_hits={len(hits)}"
        )

        if not candidates and not pinned:
            trace(f"LECTURE_RAG ⊘ empty | query_len={len(query)}")
            return _empty_rag_result()

        selected: list[LectureContextCandidate] = []
        if candidates:
            focus = _rerank_focus_query(node, user_query, curriculum_goal)
            mandatory_rag = [c for c in candidates if c.label == _PINNED_RAG_LABEL]
            pool = [c for c in candidates if c.label != _PINNED_RAG_LABEL]
            # Early exit BEFORE CE / cross-attention / MMR (heavy) and prompt stitch
            from knowledge_engine.src.services.openalex_evaluator import (
                filter_candidates_trust_hard_cutoff,
            )

            pool, dropped_pool = filter_candidates_trust_hard_cutoff(pool)
            mandatory_rag, dropped_pin = filter_candidates_trust_hard_cutoff(
                mandatory_rag
            )
            if dropped_pool or dropped_pin:
                trace(
                    f"LECTURE_RAG hard_cutoff early_exit ⊘ | "
                    f"pool_drop={dropped_pool} pinned_drop={dropped_pin} "
                    f"(before CE/MMR)"
                )
            if pool:
                selected = await _apply_rerank_mmr(
                    focus,
                    pool,
                    node_title=node.title,
                    core_concepts=list(node.core_concepts or []),
                )
            selected = _merge_mandatory_rag_after_rerank(mandatory_rag, selected)
            selected = _finalize_lecture_citation_candidates(selected)
            rag_body = _stitch_candidates_by_url(pinned, selected)
            citation_block = build_rag_chunk_citation_registry(selected)
        else:
            rag_body = _stitch_candidates_by_url(list(pinned), [])
            citation_block = ""

        stats = build_lecture_rag_stats(
            _pinned_blocks_for_stats(pinned), selected, route_urls
        )
        trace(
            f"LECTURE_RAG ✓ | out_len={len(rag_body)} mmr_selected={len(selected)} "
            f"local_sources={stats.local_sources_count} "
            f"avg_score={stats.local_avg_score:.3f} "
            f"rag_citation_ids={len({c.source_index for c in selected if c.source_index})}"
        )
        inspector = tuple(build_rag_chunk_inspector_payload(selected))
        return LectureRagContextResult(
            context=rag_body,
            stats=stats,
            citation_registry_block=citation_block,
            inspector_chunks=inspector,
        )
    except Exception as exc:
        trace(f"LECTURE_RAG full fallback | {exc}")
        try:
            pinned, candidates, route_urls = await _collect_rerank_candidates_timed(
                curriculum_id,
                node,
                query,
            )
            rag = LightRAG()
            hits = await _light_rag_hits_timed(
                rag, query, max(2, min(LECTURE_RAG_TOP_K, 5))
            )
            _append_light_rag_candidates(candidates, hits)
            chunks = _legacy_concat_chunks(pinned, candidates)
            if not chunks:
                return _empty_rag_result()
            body = "\n\n---\n\n".join(chunks)
            stats = build_lecture_rag_stats(
                _pinned_blocks_for_stats(pinned), [], route_urls
            )
            return LectureRagContextResult(context=body, stats=stats)
        except Exception as exc2:
            trace(f"LECTURE_RAG fallback failed | {exc2}")
            return _empty_rag_result()


def build_lecture_generation_payload(
    node: NodeDataInput,
    memory_rag_profile: str,
    user_query: str,
    rag_context: str,
    concepts_matrix: str,
    rolling_summary: str,
    curriculum_id: str = "",
    verified_sources_block: str = "",
    external_search_delta: bool = False,
    node_content: NodeContentBlock | None = None,
    memory: SessionMemory | None = None,
    rag_citation_registry: str = "",
    coverage_payload: str = "",
    lecture_scope: str = "",
    focus_text: str = "",
) -> str:
    """
    User payload: BLOCK 2 (semi-static node) → BLOCK 3 (dynamic session/RAG/query).
    BLOCK 1 (rules, JSON schema, citation) — только в system (`build_dense_system`).
    """
    from knowledge_engine.src.node_deep_dive.dialog_context import (
        build_shared_session_context_block,
        build_tutor_source_registry_pinned_block,
    )
    from knowledge_engine.src.node_deep_dive.interaction_prompt_layout import (
        BLOCK_DYNAMIC_HEADER,
        BLOCK_RAG_TAG,
        BLOCK_SEMI_STATIC_HEADER,
        BLOCK_USER_QUERY_TAG,
        PINNED_REGISTRY_TAG,
    )

    topic = (user_query or "").strip() or node.title
    rag_body = (rag_context or "").strip() or LECTURE_RAG_FALLBACK
    concepts = "\n".join(f"- {c}" for c in node.core_concepts)
    foundation = format_primary_whitelist_foundation(node, curriculum_id)
    scope = (lecture_scope or "").strip()
    focus = (focus_text or "").strip()

    # --- BLOCK 2: semi-static ---
    block2_parts: list[str] = [BLOCK_SEMI_STATIC_HEADER]

    registry_block = ""
    if (curriculum_id or "").strip():
        registry_block = build_tutor_source_registry_pinned_block(
            curriculum_id, node, node_content
        )
    if registry_block:
        block2_parts.append(f"{PINNED_REGISTRY_TAG}\n{registry_block}")

    if foundation.strip():
        block2_parts.append(foundation.strip())

    if (curriculum_id or "").strip():
        diagrams_block = build_lecture_pinned_diagrams_block(node, curriculum_id)
        if diagrams_block.strip():
            block2_parts.append(diagrams_block.strip())

    from knowledge_engine.src.node_deep_dive.node_materials_context import (
        format_available_node_materials_block,
    )

    materials_block = format_available_node_materials_block(node, node_content)
    if materials_block.strip():
        block2_parts.append(materials_block.strip())

    from knowledge_engine.src.node_deep_dive.tutor_diagram_citations import (
        build_diagram_catalog,
        format_diagram_catalog_block,
    )

    catalog = format_diagram_catalog_block(build_diagram_catalog(node_content))
    if catalog.strip():
        block2_parts.append(catalog.strip())

    lg = (getattr(node, "learning_goal", None) or "").strip()
    meta_lines = [
        "### node_metadata",
        f"title: {node.title}",
        f"layer: {node.layer}",
        f"category: {node.category}",
        f"brief_summary: {node.brief_summary}",
        f"core_concepts:\n{concepts}",
    ]
    if lg:
        meta_lines.insert(1, f"learning_goal: {lg}")
    block2_parts.append("\n".join(meta_lines))

    if memory is not None and memory.sub_concepts:
        from knowledge_engine.src.node_deep_dive.concept_map_state import (
            format_concept_map_for_tutor,
        )

        cmap = format_concept_map_for_tutor(
            memory,
            include_evaluator_transparency=False,
        )
        if cmap.strip():
            block2_parts.append(cmap.strip())

    block2 = "\n\n".join(p for p in block2_parts if p.strip())

    # --- BLOCK 3: dynamic ---
    block3_parts: list[str] = [BLOCK_DYNAMIC_HEADER]

    if memory is not None:
        session_block = build_shared_session_context_block(
            memory,
            user_message="",
            include_sliding_window=not external_search_delta,
        )
        if session_block:
            block3_parts.append(session_block)

    if (memory_rag_profile or "").strip():
        block3_parts.append(
            f"### directional_rag_profile\n{memory_rag_profile.strip()[:4000]}"
        )
    if (concepts_matrix or "").strip():
        block3_parts.append(f"### concepts_matrix\n{concepts_matrix.strip()}")
    if (rolling_summary or "").strip():
        block3_parts.append(f"### rolling_summary\n{rolling_summary.strip()[:4000]}")
    if (coverage_payload or "").strip():
        block3_parts.append(coverage_payload.strip())
    if (verified_sources_block or "").strip():
        block3_parts.append(verified_sources_block.strip())

    rag_parts: list[str] = [BLOCK_RAG_TAG]
    if (rag_citation_registry or "").strip():
        rag_parts.append(rag_citation_registry.strip())
    rag_parts.append("=== НАЧАЛО МАТЕРИАЛА ===")
    rag_parts.append(rag_body)
    rag_parts.append("=== КОНЕЦ МАТЕРИАЛА ===")
    block3_parts.append("\n".join(rag_parts))

    query_lines = [BLOCK_USER_QUERY_TAG, f"### user_query\n{topic}"]
    if scope == "targeted_lecture" and focus:
        query_lines.extend(
            [
                f"### lecture_scope\n{scope}",
                f"### user_focus\n{focus}",
                "INSTRUCTION: lecture EXCLUSIVELY on user_focus / active_subconcept_id; "
                "not a full-node overview; do NOT choose the topic from chat_history.",
            ]
        )
    if external_search_delta:
        query_lines.append(
            "ЗАДАЧА: ДОГЕНЕРАЦИЯ (дельта). Новые источники в VERIFIED_EXTERNAL_SOURCES. "
            "Только раздел «Дополнение к основному материалу по новым источникам:»."
        )
    block3_parts.append("\n".join(query_lines))

    block3 = "\n\n".join(p for p in block3_parts if p.strip())
    return f"{block2}\n\n{block3}".strip()
