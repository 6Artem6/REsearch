"""Cross-Encoder rerank + MMR для lecture RAG (без LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knowledge_engine.config import (
    EMBED_MODEL,
    LECTURE_RAG_CE_MIN_SCORE,
    LECTURE_RAG_MMR_LAMBDA,
    LECTURE_RAG_MMR_TOP_K,
    OLLAMA_BASE_URL,
)
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.ui.run_log import trace


@dataclass
class LectureContextCandidate:
    """Один фрагмент для rerank/MMR и финальной склейки."""

    label: str
    formatted: str
    plain: str
    url_key: str = ""
    source_id: str = ""
    source_title: str = ""
    source_index: int = 0
    chunk_index: int = 0
    chunks_in_doc: int = 0
    retrieval_score: float = 0.0
    trust_score: float = 1.0
    vector_similarity: float = 0.0
    doc_id: str = ""
    chunk_vector: np.ndarray | None = field(default=None, repr=False)
    doc_meta_vector: np.ndarray | None = field(default=None, repr=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embed_texts_sync(texts: list[str]) -> list[np.ndarray]:
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    out: list[np.ndarray] = []
    for t in texts:
        v = np.asarray(emb.embed_query((t or "")[:8000]), dtype=np.float64)
        out.append(v)
    return out


def _mmr_select_indices(
    relevance: list[float],
    embeddings: list[np.ndarray],
    k: int,
    lambda_: float,
) -> list[int]:
    if not relevance:
        return []
    k = max(1, min(k, len(relevance)))
    selected: list[int] = []
    pool = list(range(len(relevance)))
    while len(selected) < k and pool:
        best_i: int | None = None
        best_score = -1e9
        for i in pool:
            if not selected:
                score = relevance[i]
            else:
                max_sim = max(_cosine(embeddings[i], embeddings[j]) for j in selected)
                score = lambda_ * relevance[i] - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        pool.remove(best_i)
    return selected


def _preview_label(c: LectureContextCandidate, max_len: int = 56) -> str:
    head = (c.plain or c.formatted or "").replace("\n", " ").strip()
    if len(head) <= max_len:
        return head
    return head[: max_len - 1] + "…"


def diversify_lecture_candidates_sync(
    query: str,
    candidates: list[LectureContextCandidate],
) -> list[LectureContextCandidate]:
    """
    CE rerank → threshold → MMR. Синхронно (вызывать под run_under_uma_lock).
    """
    if not candidates:
        return []

    crit = (query or "").strip()
    plains = [(c.plain or c.formatted or "").strip() for c in candidates]
    valid = [i for i, p in enumerate(plains) if len(p) >= 24]
    if not valid:
        trace("LECTURE_RAG rerank ⊘ | no plain text >= 24 chars")
        return candidates[:LECTURE_RAG_MMR_TOP_K]

    trace(
        f"LECTURE_RAG rerank ▶ | candidates={len(candidates)} "
        f"valid_plain={len(valid)} ce_min={LECTURE_RAG_CE_MIN_SCORE:.2f}"
    )

    valid_plains = [plains[i] for i in valid]
    scores = score_relevance_pairs(crit, valid_plains)

    passed: list[int] = []
    dropped_ce: list[str] = []
    for local_i, global_i in enumerate(valid):
        sc = scores[local_i] if local_i < len(scores) else 0.0
        if sc >= LECTURE_RAG_CE_MIN_SCORE:
            passed.append(global_i)
        else:
            dropped_ce.append(
                f"{candidates[global_i].label} score={sc:.3f} "
                f"«{_preview_label(candidates[global_i], 40)}»"
            )

    trace(f"LECTURE_RAG ce_filter ✓ | kept={len(passed)} " f"dropped={len(dropped_ce)}")
    if dropped_ce:
        for line in dropped_ce[:8]:
            trace(f"LECTURE_RAG ce_drop | {line}")
        if len(dropped_ce) > 8:
            trace(f"LECTURE_RAG ce_drop | … +{len(dropped_ce) - 8} more")

    if not passed:
        trace("LECTURE_RAG ce_filter ⊘ | all below threshold — keep top by CE score")
        ranked = sorted(valid, key=lambda gi: scores[valid.index(gi)], reverse=True)
        passed = ranked[: max(LECTURE_RAG_MMR_TOP_K, 3)]

    passed_plains = [plains[i] for i in passed]
    relevance = [scores[valid.index(i)] if i in valid else 0.0 for i in passed]
    embeddings = _embed_texts_sync(passed_plains)

    mmr_indices = _mmr_select_indices(
        relevance,
        embeddings,
        LECTURE_RAG_MMR_TOP_K,
        LECTURE_RAG_MMR_LAMBDA,
    )
    selected_globals = [passed[i] for i in mmr_indices]

    trace(
        f"LECTURE_RAG mmr ✓ | lambda={LECTURE_RAG_MMR_LAMBDA:.2f} "
        f"top_k={LECTURE_RAG_MMR_TOP_K} selected={len(selected_globals)}"
    )
    for rank, gi in enumerate(selected_globals, 1):
        sc = relevance[passed.index(gi)] if gi in passed else 0.0
        c = candidates[gi]
        trace(
            f"LECTURE_RAG mmr_pick #{rank} | {c.label} ce={sc:.3f} "
            f"url={c.url_key[:48] or '—'} «{_preview_label(c)}»"
        )

    return [candidates[i] for i in selected_globals]


def fallback_dedupe_candidates(
    candidates: list[LectureContextCandidate],
    limit: int,
) -> list[LectureContextCandidate]:
    """URL + exact formatted dedup (legacy поведение)."""
    out: list[LectureContextCandidate] = []
    seen_urls: set[str] = set()
    seen_text: set[str] = set()
    for c in candidates:
        key = (c.url_key or "").strip().lower()
        if key and key in seen_urls:
            continue
        text_key = (c.formatted or "").strip()
        if text_key and text_key in seen_text:
            continue
        if key:
            seen_urls.add(key)
        if text_key:
            seen_text.add(text_key)
        out.append(c)
        if len(out) >= limit:
            break
    trace(
        f"LECTURE_RAG fallback_dedupe ✓ | in={len(candidates)} out={len(out)} limit={limit}"
    )
    return out


def _title_from_candidate(c: LectureContextCandidate) -> str:
    title = (c.source_title or "").strip()
    if title:
        return title
    plain = (c.plain or c.formatted or "").strip()
    if not plain:
        return (c.label or "source").strip()
    first = plain.split("\n", 1)[0].strip()
    if first.startswith("###"):
        first = first.lstrip("#").strip()
    return first[:240] or c.label


def _doc_meta_embed_text(c: LectureContextCandidate) -> str:
    return _title_from_candidate(c)[:2000]


def cross_attention_select_lecture_candidates_sync(
    topic_query: str,
    candidates: list[LectureContextCandidate],
    *,
    node_title: str = "",
    core_concepts: list[str] | None = None,
) -> list[LectureContextCandidate]:
    """Topic embedding + cross-attention/MMR; output with [R_N] formatting."""
    from knowledge_engine.config import (
        DOC_GATE_THRESHOLD,
        LECTURE_CHUNK_CA_ALPHA,
        LECTURE_CHUNK_CA_BETA,
        LECTURE_CHUNK_CA_GAMMA,
        LECTURE_CHUNK_CA_MAX_PER_SOURCE,
        LECTURE_CHUNK_CA_TOP_K,
        MAX_CHUNKS_PER_DOC,
        RAG_KNEE_DROP_RATIO,
        RAG_SCORE_MIN_FLOOR,
    )
    from knowledge_engine.services.chunk_cross_attention_mmr import (
        ChunkCandidate,
        select_diverse_chunks_with_cross_attention,
    )

    if not candidates:
        return []

    concepts_line = " ".join(core_concepts or [])[:2000]
    topic_text = "\n".join(
        x
        for x in [
            (node_title or "").strip(),
            concepts_line,
            (topic_query or "").strip(),
        ]
        if x
    )[:8000]
    if not topic_text.strip():
        topic_text = (topic_query or "lecture context")[:8000]

    trace(
        f"LECTURE_CHUNK_CA ▶ | candidates={len(candidates)} top_k={LECTURE_CHUNK_CA_TOP_K} "
        f"alpha={LECTURE_CHUNK_CA_ALPHA:.2f} beta={LECTURE_CHUNK_CA_BETA:.2f} "
        f"gamma={LECTURE_CHUNK_CA_GAMMA:.2f} max_per_src={LECTURE_CHUNK_CA_MAX_PER_SOURCE}"
    )

    topic_vec = _embed_texts_sync([topic_text])[0]

    chunk_records: list[ChunkCandidate] = []
    pending_chunk: list[tuple[int, str]] = []
    pending_doc: list[tuple[int, str]] = []

    for i, c in enumerate(candidates):
        plain = (c.plain or c.formatted or "").strip()
        sid = (c.source_id or c.url_key or f"{c.label}:{i}").strip()
        title = _title_from_candidate(c)
        cc = ChunkCandidate(
            text=plain,
            source_id=sid,
            source_title=title,
            meta={
                "origin_i": str(i),
                "doc_id": (c.doc_id or sid).strip(),
                "chunk_index": str(c.chunk_index or ""),
                "chunks_in_doc": str(c.chunks_in_doc or ""),
            },
        )
        if c.chunk_vector is not None:
            cc.chunk_vector = np.asarray(c.chunk_vector, dtype=np.float64)
        else:
            pending_chunk.append((len(chunk_records), plain[:8000]))
        if c.doc_meta_vector is not None:
            cc.doc_meta_vector = np.asarray(c.doc_meta_vector, dtype=np.float64)
        else:
            pending_doc.append((len(chunk_records), _doc_meta_embed_text(c)))
        chunk_records.append(cc)

    if pending_chunk:
        chunk_indices: list[int] = []
        chunk_texts: list[str] = []
        for ri, t in pending_chunk:
            plain = (t or "").strip()
            if len(plain) < 8:
                continue
            chunk_indices.append(ri)
            chunk_texts.append(plain[:8000])
        if chunk_texts:
            vecs = _embed_texts_sync(chunk_texts)
            for ri, v in zip(chunk_indices, vecs):
                chunk_records[ri].chunk_vector = v

    if pending_doc:
        doc_indices: list[int] = []
        doc_texts: list[str] = []
        for ri, t in pending_doc:
            if chunk_records[ri].doc_meta_vector is not None:
                continue
            txt = (t or "").strip()
            if not txt:
                continue
            doc_indices.append(ri)
            doc_texts.append(txt[:2000])
        if doc_texts:
            vecs = _embed_texts_sync(doc_texts)
            for ri, v in zip(doc_indices, vecs):
                chunk_records[ri].doc_meta_vector = v

    for cc in chunk_records:
        if cc.doc_meta_vector is None and cc.chunk_vector is not None:
            cc.doc_meta_vector = cc.chunk_vector

    def _fetch_anchor_doc_chunks(doc_id: str) -> list[ChunkCandidate]:
        from knowledge_engine.db.rag_chunks_schema import (
            COL_CHUNK_INDEX,
            COL_CHUNK_TEXT,
            COL_CHUNK_VECTOR,
            COL_CHUNKS_IN_DOC,
            COL_DOC_META_VECTOR,
            COL_TITLE,
            COL_URL,
        )
        from knowledge_engine.services.vector_store import VectorStore

        rows = VectorStore().fetch_rag_chunks_by_doc_id(doc_id)
        out_chunks: list[ChunkCandidate] = []
        for row in rows:
            text = (row.get(COL_CHUNK_TEXT) or "").strip()
            if len(text) < 24:
                continue
            vec = row.get(COL_CHUNK_VECTOR)
            if vec is None:
                continue
            url = str(row.get(COL_URL) or "").strip()
            title = str(row.get(COL_TITLE) or url or doc_id)[:200]
            out_chunks.append(
                ChunkCandidate(
                    text=text,
                    source_id=doc_id,
                    source_title=title,
                    chunk_vector=np.asarray(vec, dtype=np.float64),
                    doc_meta_vector=(
                        np.asarray(row.get(COL_DOC_META_VECTOR), dtype=np.float64)
                        if row.get(COL_DOC_META_VECTOR) is not None
                        else None
                    ),
                    meta={
                        "origin_i": "-1",
                        "doc_id": doc_id,
                        "chunk_index": str(int(row.get(COL_CHUNK_INDEX) or 0)),
                        "chunks_in_doc": str(int(row.get(COL_CHUNKS_IN_DOC) or 0)),
                        "url": url,
                    },
                )
            )
        return out_chunks

    max_per_doc = max(1, int(MAX_CHUNKS_PER_DOC or LECTURE_CHUNK_CA_MAX_PER_SOURCE))
    selected = select_diverse_chunks_with_cross_attention(
        topic_vec,
        chunk_records,
        top_k=LECTURE_CHUNK_CA_TOP_K,
        alpha=LECTURE_CHUNK_CA_ALPHA,
        beta=LECTURE_CHUNK_CA_BETA,
        gamma=LECTURE_CHUNK_CA_GAMMA,
        max_chunks_per_source=max_per_doc,
        doc_gate_threshold=DOC_GATE_THRESHOLD,
        min_floor=RAG_SCORE_MIN_FLOOR,
        knee_drop_ratio=RAG_KNEE_DROP_RATIO,
        fetch_doc_chunks=_fetch_anchor_doc_chunks,
    )
    if not selected:
        trace("LECTURE_CHUNK_CA ⊘ | empty selection — CE/MMR fallback path may run")
        return []

    out: list[LectureContextCandidate] = []
    for sc in selected:
        oi = sc.origin_candidate_index
        base = candidates[oi] if 0 <= oi < len(candidates) else None
        out.append(
            LectureContextCandidate(
                label=base.label if base else "chunk_ca",
                formatted=sc.formatted,
                plain=sc.text,
                url_key=(base.url_key if base else (sc.url_key or sc.source_id)),
                source_id=sc.source_id,
                source_title=sc.source_title,
                source_index=sc.source_index,
                chunk_index=int(sc.chunk_index or (base.chunk_index if base else 0)),
                chunks_in_doc=int(
                    sc.chunks_in_doc or (base.chunks_in_doc if base else 0)
                ),
                retrieval_score=float(sc.score),
                doc_id=(sc.doc_id or (base.doc_id if base else sc.source_id)),
                chunk_vector=base.chunk_vector if base else None,
                doc_meta_vector=base.doc_meta_vector if base else None,
            )
        )
    return out
