"""Cross-Encoder rerank + MMR для lecture RAG (без LLM)."""

from __future__ import annotations

from dataclasses import dataclass

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

    trace(
        f"LECTURE_RAG ce_filter ✓ | kept={len(passed)} "
        f"dropped={len(dropped_ce)}"
    )
    if dropped_ce:
        for line in dropped_ce[:8]:
            trace(f"LECTURE_RAG ce_drop | {line}")
        if len(dropped_ce) > 8:
            trace(f"LECTURE_RAG ce_drop | … +{len(dropped_ce) - 8} more")

    if not passed:
        trace("LECTURE_RAG ce_filter ⊘ | all below threshold — keep top by CE score")
        ranked = sorted(valid, key=lambda gi: scores[valid.index(gi)], reverse=True)
        passed = ranked[:max(LECTURE_RAG_MMR_TOP_K, 3)]

    passed_plains = [plains[i] for i in passed]
    relevance = [
        scores[valid.index(i)] if i in valid else 0.0 for i in passed
    ]
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
