"""Cross-attention selection, relative gap threshold, anchor RAG, positional reorder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from knowledge_engine.ui.run_log import trace


@dataclass
class ChunkCandidate:
    """One retrievable fragment from LanceDB / local store."""

    text: str
    source_id: str
    source_title: str = ""
    chunk_vector: np.ndarray | None = None
    doc_meta_vector: np.ndarray | None = None
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectedChunk:
    source_index: int
    source_id: str
    source_title: str
    text: str
    formatted: str
    score: float
    origin_candidate_index: int = -1
    chunk_index: int = 0
    chunks_in_doc: int = 0
    url_key: str = ""
    doc_id: str = ""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_rows(vectors: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for v in vectors:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(arr))
        out.append(arr / n if n > 0 else arr)
    return out


@dataclass
class _PoolEntry:
    origin_i: int
    ch: ChunkCandidate
    chunk_vec: np.ndarray
    relevance: float


def filter_pool_by_knee_cutoff(
    entries: list[_PoolEntry],
    *,
    min_floor: float = 0.30,
    knee_drop_ratio: float = 0.12,
) -> list[_PoolEntry]:
    """
    Фильтрует пул чанков по точке относительного перепада релевантности (Knee Cutoff).

    1. Сортирует entries по убыванию relevance.
    2. Если top_score < min_floor — возвращает лучший элемент.
    3. drop_i = (s_i - s_{i+1}) / top_score; первый i с drop_i >= knee_drop_ratio — срез.
    4. Оставшийся срез фильтруется по e.relevance >= min_floor.
    """
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: e.relevance, reverse=True)
    top_score = sorted_entries[0].relevance

    if top_score < min_floor:
        trace(
            f"LECTURE_CHUNK_KNEE ⊘ | top={top_score:.3f} < floor={min_floor:.2f} "
            f"→ keep best only"
        )
        return [sorted_entries[0]]

    cutoff_index = len(sorted_entries)
    knee_at: int | None = None
    for i in range(len(sorted_entries) - 1):
        drop = (
            sorted_entries[i].relevance - sorted_entries[i + 1].relevance
        ) / top_score
        if drop >= knee_drop_ratio:
            cutoff_index = i + 1
            knee_at = i
            break

    kept = [e for e in sorted_entries[:cutoff_index] if e.relevance >= min_floor]

    if not kept:
        kept = [sorted_entries[0]]

    trace(
        f"LECTURE_CHUNK_KNEE ✓ | top={top_score:.3f} knee_i={knee_at} "
        f"cutoff_n={cutoff_index} kept={len(kept)}/{len(entries)} "
        f"ratio={knee_drop_ratio:.2f}"
    )
    return kept


def semantic_dedup_entries(
    entries: list[_PoolEntry],
    *,
    threshold: float,
) -> list[_PoolEntry]:
    if len(entries) < 2:
        return list(entries)
    order = sorted(entries, key=lambda e: e.relevance, reverse=True)
    kept: list[_PoolEntry] = []
    kept_vecs: list[np.ndarray] = []
    dropped = 0
    for e in order:
        dup = False
        for kv in kept_vecs:
            if _cosine(e.chunk_vec, kv) > threshold:
                dup = True
                dropped += 1
                break
        if dup:
            continue
        kept.append(e)
        kept_vecs.append(e.chunk_vec)
    if dropped:
        trace(f"LECTURE_CHUNK_DEDUP ✓ | removed={dropped} sim>{threshold:.2f}")
    return kept


def positional_reorder_entries(entries: list[_PoolEntry]) -> list[_PoolEntry]:
    """Lost-in-the-middle: best at start and end, middling scores in the center."""
    if len(entries) <= 2:
        return list(entries)
    ranked = sorted(entries, key=lambda e: e.relevance, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1]
    middle = ranked[2:]
    return [top1] + middle + [top2]


def _format_chunk(rank: int, ch: ChunkCandidate, body: str) -> str:
    title = (ch.source_title or ch.source_id or "Source").strip()[:200]
    ci = (ch.meta.get("chunk_index") or "").strip()
    cd = (ch.meta.get("chunks_in_doc") or "").strip()
    if ci and cd:
        return f"[R{rank}] (Source: {title}, Chunk {ci}/{cd})\n{body}"
    return f"[R{rank}] (Source: {title}): {body}"


def _entry_to_selected(rank: int, e: _PoolEntry) -> SelectedChunk:
    ch = e.ch
    sid = (ch.meta.get("doc_id") or ch.source_id or "").strip()
    body = (ch.text or "").strip()
    try:
        origin_i = int(ch.meta.get("origin_i", "-1"))
    except (TypeError, ValueError):
        origin_i = -1
    try:
        ci = int(ch.meta.get("chunk_index") or 0)
    except (TypeError, ValueError):
        ci = 0
    try:
        cd = int(ch.meta.get("chunks_in_doc") or 0)
    except (TypeError, ValueError):
        cd = 0
    url = (ch.meta.get("url") or "").strip()
    return SelectedChunk(
        source_index=rank,
        source_id=sid or ch.source_id,
        source_title=(ch.source_title or sid or "Source").strip()[:200],
        text=body,
        formatted=_format_chunk(rank, ch, body),
        score=e.relevance,
        origin_candidate_index=origin_i,
        chunk_index=ci,
        chunks_in_doc=cd,
        url_key=url,
        doc_id=sid,
    )


def _greedy_mmr_select(
    entries: list[_PoolEntry],
    *,
    top_k: int,
    gamma: float,
    max_per_source: int,
) -> list[_PoolEntry]:
    pool = list(entries)
    if not pool:
        return []
    k = min(top_k, len(pool))
    per_source: dict[str, int] = {}
    selected: list[_PoolEntry] = []
    selected_pi: set[int] = set()
    selected_vecs: list[np.ndarray] = []

    if len(entries) > 1:
        scores = [e.relevance for e in entries]
        std_dev = float(np.std(scores))
        effective_gamma = float(np.clip(gamma + (0.10 - std_dev) * 1.5, 0.35, 0.75))
        trace(
            f"LECTURE_CHUNK_MMR gamma | base={gamma:.3f} std={std_dev:.4f} "
            f"effective={effective_gamma:.3f}"
        )
    else:
        effective_gamma = gamma

    while len(selected) < k:
        best: _PoolEntry | None = None
        best_pi = -1
        best_final = -1e9
        for pi, e in enumerate(pool):
            if pi in selected_pi:
                continue
            sid = (e.ch.meta.get("doc_id") or e.ch.source_id or "").strip()
            if per_source.get(sid, 0) >= max_per_source:
                continue
            redundancy = 0.0
            if selected_vecs:
                redundancy = max(_cosine(e.chunk_vec, sv) for sv in selected_vecs)
            final = e.relevance - effective_gamma * redundancy
            if final > best_final:
                best_final = final
                best = e
                best_pi = pi
        if best is None:
            break
        selected_pi.add(best_pi)
        selected.append(best)
        selected_vecs.append(best.chunk_vec)
        sid = (best.ch.meta.get("doc_id") or best.ch.source_id or "").strip()
        per_source[sid] = per_source.get(sid, 0) + 1
    return selected


def _anchor_mode_select(
    filtered: list[_PoolEntry],
    *,
    anchor_doc_id: str,
    top_k: int,
    supplement_max: int,
    fetch_doc_chunks: Callable[[str], list[ChunkCandidate]] | None,
    topic: np.ndarray,
    alpha: float,
    beta: float,
) -> list[_PoolEntry]:
    anchor_id = anchor_doc_id.strip()
    anchor_entries: list[_PoolEntry] = []

    if fetch_doc_chunks:
        fetched = fetch_doc_chunks(anchor_id)
        for ch in fetched:
            if ch.chunk_vector is None:
                continue
            cv = np.asarray(ch.chunk_vector, dtype=np.float64).reshape(-1)
            cn = float(np.linalg.norm(cv))
            cv_n = cv / cn if cn > 0 else cv
            dv = cv_n
            if ch.doc_meta_vector is not None:
                dv = np.asarray(ch.doc_meta_vector, dtype=np.float64).reshape(-1)
                dn = float(np.linalg.norm(dv))
                dv = dv / dn if dn > 0 else dv
            rel = alpha * _cosine(cv_n, topic) + beta * _cosine(dv, topic)
            try:
                oi = int(ch.meta.get("origin_i", "-1"))
            except (TypeError, ValueError):
                oi = -1
            anchor_entries.append(
                _PoolEntry(origin_i=oi, ch=ch, chunk_vec=cv_n, relevance=rel)
            )
        anchor_entries.sort(
            key=lambda e: int(e.ch.meta.get("chunk_index") or 0),
        )

    if not anchor_entries:
        anchor_entries = [
            e
            for e in filtered
            if (e.ch.meta.get("doc_id") or e.ch.source_id or "").strip() == anchor_id
        ]
        anchor_entries.sort(
            key=lambda e: int(e.ch.meta.get("chunk_index") or 0),
        )

    supplements: list[_PoolEntry] = []
    for e in sorted(filtered, key=lambda x: x.relevance, reverse=True):
        sid = (e.ch.meta.get("doc_id") or e.ch.source_id or "").strip()
        if sid == anchor_id:
            continue
        if e in anchor_entries:
            continue
        supplements.append(e)
        if len(supplements) >= supplement_max:
            break

    combined = anchor_entries + supplements
    if top_k > 0 and len(combined) > top_k:
        # Preserve full anchor sequence; trim supplements only
        if len(anchor_entries) >= top_k:
            combined = anchor_entries[:top_k]
        else:
            slots = top_k - len(anchor_entries)
            combined = anchor_entries + supplements[:slots]
    trace(
        f"LECTURE_CHUNK_ANCHOR ✓ | doc={anchor_id[:16]} "
        f"anchor_chunks={len(anchor_entries)} supplement={len(supplements)} "
        f"out={len(combined)}"
    )
    return combined


def select_diverse_chunks_with_cross_attention(
    topic_vector: np.ndarray,
    candidate_chunks: list[ChunkCandidate],
    *,
    top_k: int = 8,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.55,
    max_chunks_per_source: int = 2,
    doc_gate_threshold: float = 0.0,
    min_floor: float = 0.30,
    knee_drop_ratio: float = 0.12,
    anchor_threshold: float = 0.70,
    semantic_dedup_threshold: float = 0.85,
    anchor_supplement_max: int = 2,
    fetch_doc_chunks: Callable[[str], list[ChunkCandidate]] | None = None,
) -> list[SelectedChunk]:
    """
    Doc-gate → relevance scores → relative gap filter → anchor or MMR → dedup → reorder.
    """
    if not candidate_chunks or top_k <= 0:
        return []

    from knowledge_engine.config import (
        RAG_ANCHOR_SUPPLEMENT_MAX,
        RAG_ANCHOR_THRESHOLD,
        RAG_CHUNK_SEMANTIC_DEDUP,
        RAG_KNEE_DROP_RATIO,
        RAG_SCORE_MIN_FLOOR,
    )

    min_floor = float(min_floor if min_floor else RAG_SCORE_MIN_FLOOR)
    knee_drop_ratio = float(knee_drop_ratio if knee_drop_ratio else RAG_KNEE_DROP_RATIO)
    anchor_threshold = float(
        anchor_threshold if anchor_threshold else RAG_ANCHOR_THRESHOLD
    )
    semantic_dedup_threshold = float(
        semantic_dedup_threshold or RAG_CHUNK_SEMANTIC_DEDUP
    )
    supplement_max = int(anchor_supplement_max or RAG_ANCHOR_SUPPLEMENT_MAX or 2)

    topic = np.asarray(topic_vector, dtype=np.float64).reshape(-1)
    topic_n = float(np.linalg.norm(topic))
    if topic_n > 0:
        topic = topic / topic_n

    raw_pool: list[tuple[int, ChunkCandidate, np.ndarray, np.ndarray]] = []
    for idx, ch in enumerate(candidate_chunks):
        text = (ch.text or "").strip()
        if len(text) < 24:
            continue
        if ch.chunk_vector is None:
            continue
        cv = np.asarray(ch.chunk_vector, dtype=np.float64).reshape(-1)
        if ch.doc_meta_vector is not None:
            dv = np.asarray(ch.doc_meta_vector, dtype=np.float64).reshape(-1)
        else:
            dv = cv
        if doc_gate_threshold > 0:
            dn = float(np.linalg.norm(dv))
            doc_cos = float(np.dot(dv / dn, topic)) if dn > 0 else 0.0
            if doc_cos < doc_gate_threshold:
                continue
        try:
            origin_i = int(ch.meta.get("origin_i", str(idx)))
        except (TypeError, ValueError):
            origin_i = idx
        raw_pool.append((origin_i, ch, cv, dv))

    if not raw_pool:
        return []

    chunk_vecs = _normalize_rows([p[2] for p in raw_pool])
    doc_vecs = _normalize_rows([p[3] for p in raw_pool])

    scored: list[_PoolEntry] = []
    for (origin_i, ch, _, _), cv, dv in zip(raw_pool, chunk_vecs, doc_vecs):
        rel = alpha * _cosine(cv, topic) + beta * _cosine(dv, topic)
        scored.append(_PoolEntry(origin_i=origin_i, ch=ch, chunk_vec=cv, relevance=rel))

    filtered = filter_pool_by_knee_cutoff(
        scored, min_floor=min_floor, knee_drop_ratio=knee_drop_ratio
    )
    top_score = max(e.relevance for e in filtered)
    top_entry = max(filtered, key=lambda e: e.relevance)
    anchor_doc = (
        top_entry.ch.meta.get("doc_id") or top_entry.ch.source_id or ""
    ).strip()

    if top_score >= anchor_threshold and anchor_doc:
        trace(f"LECTURE_CHUNK_MODE anchor | top={top_score:.3f} doc={anchor_doc[:20]}")
        picked_entries = _anchor_mode_select(
            filtered,
            anchor_doc_id=anchor_doc,
            top_k=top_k,
            supplement_max=supplement_max,
            fetch_doc_chunks=fetch_doc_chunks,
            topic=topic,
            alpha=alpha,
            beta=beta,
        )
    else:
        trace(
            f"LECTURE_CHUNK_MODE multi_source | top={top_score:.3f} "
            f"thr_anchor={anchor_threshold:.2f}"
        )
        max_per = max(1, int(max_chunks_per_source))
        picked_entries = _greedy_mmr_select(
            filtered,
            top_k=top_k,
            gamma=gamma,
            max_per_source=max_per,
        )

    picked_entries = semantic_dedup_entries(
        picked_entries, threshold=semantic_dedup_threshold
    )
    picked_entries = positional_reorder_entries(picked_entries)

    out: list[SelectedChunk] = []
    for rank, e in enumerate(picked_entries, 1):
        out.append(_entry_to_selected(rank, e))

    trace(
        f"LECTURE_CHUNK_CA ✓ | pool={len(scored)} knee_kept={len(filtered)} "
        f"selected={len(out)} top_k={top_k} doc_gate={doc_gate_threshold:.2f}"
    )
    for sc in out[: min(8, len(out))]:
        trace(
            f"LECTURE_CHUNK_CA pick R{sc.source_index} | score={sc.score:.3f} "
            f"src={sc.source_id[:48]} «{sc.source_title[:40]}»"
        )
    return out
