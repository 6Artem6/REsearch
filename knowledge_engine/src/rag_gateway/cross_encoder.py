"""Локальный Cross-Encoder реранкер (без LLM)."""

from __future__ import annotations

import math
import threading
from typing import List

import numpy as np

from knowledge_engine.config import RAG_CROSS_ENCODER_MODEL
from knowledge_engine.ui.run_log import trace

_lock = threading.Lock()
_cross_encoder: object | None = None
_use_cosine_fallback = False


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_cross_encoder() -> object | None:
    global _cross_encoder, _use_cosine_fallback
    with _lock:
        if _use_cosine_fallback:
            return None
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            trace(f"RAG_CE ▶ загрузка {RAG_CROSS_ENCODER_MODEL}")
            _cross_encoder = CrossEncoder(RAG_CROSS_ENCODER_MODEL)
            return _cross_encoder
        except Exception as exc:
            _use_cosine_fallback = True
            trace(f"RAG_CE ⚠ CrossEncoder недоступен, cosine fallback | {exc}")
            return None


def _cosine_fallback_scores(criterion: str, texts: List[str]) -> List[float]:
    """Детерминированный fallback: косинус эмбеддингов Ollama (не cross-encoder)."""
    from langchain_ollama import OllamaEmbeddings

    from knowledge_engine.config import EMBED_MODEL, OLLAMA_BASE_URL

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    qv = np.asarray(emb.embed_query(criterion[:8000]), dtype=np.float64)
    nq = np.linalg.norm(qv)
    if nq == 0:
        return [0.0 for _ in texts]
    scores: List[float] = []
    for t in texts:
        tv = np.asarray(emb.embed_query(t[:8000]), dtype=np.float64)
        nt = np.linalg.norm(tv)
        if nt == 0:
            scores.append(0.0)
            continue
        cos = float(np.dot(qv, tv) / (nq * nt))
        scores.append(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
    return scores


def score_relevance_pairs(criterion: str, texts: List[str]) -> List[float]:
    """
    Парные оценки (relevance_criteria <-> chunk). Возврат в шкале 0..1.
    """
    if not texts:
        return []
    crit = (criterion or "").strip()
    if not crit:
        return [0.0 for _ in texts]

    model = _load_cross_encoder()
    if model is None:
        return _cosine_fallback_scores(crit, texts)

    pairs = [(crit, t[:2000]) for t in texts]
    try:
        raw = model.predict(pairs, batch_size=min(16, len(pairs)))
        out: List[float] = []
        for r in raw:
            val = float(r)
            out.append(_sigmoid(val))
        return out
    except Exception as exc:
        trace(f"RAG_CE ⚠ predict fallback | {exc}")
        return _cosine_fallback_scores(crit, texts)
