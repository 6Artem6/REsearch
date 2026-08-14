"""Локальный Cross-Encoder реранкер (без LLM)."""

from __future__ import annotations

import gc
import math
import threading
import time
from typing import List

import numpy as np

from knowledge_engine.config import (
    RAG_CE_AUTO_UNLOAD,
    RAG_CE_AUTO_UNLOAD_IDLE_SEC,
    RAG_CE_TORCH_DTYPE,
    RAG_CROSS_ENCODER_MODEL,
)
from knowledge_engine.ui.run_log import trace

_lock = threading.Lock()
_cross_encoder: object | None = None
_use_cosine_fallback = False
_last_ce_use_monotonic: float = 0.0
_idle_unload_timer: threading.Timer | None = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _resolve_torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_torch_dtype(device: str) -> object:
    import torch

    mode = (RAG_CE_TORCH_DTYPE or "auto").strip().lower()
    if mode == "float32":
        return torch.float32
    if mode == "bfloat16":
        return torch.bfloat16
    if mode == "float16":
        return torch.float16
    # auto
    if device in ("mps", "cuda"):
        if device == "mps" and hasattr(torch, "bfloat16"):
            # MPS: fp16 обычно стабильнее для cross-encoder
            return torch.float16
        return torch.float16
    return torch.float32


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload_cross_encoder() -> None:
    """Сброс модели и PyTorch allocator cache (для idle worker)."""
    global _cross_encoder, _idle_unload_timer
    with _lock:
        if _idle_unload_timer is not None:
            _idle_unload_timer.cancel()
            _idle_unload_timer = None
        if _cross_encoder is None:
            return
        _cross_encoder = None
    _release_torch_cache()
    trace("RAG_CE ✓ unloaded (weights released)")


def _idle_unload_callback() -> None:
    with _lock:
        if _cross_encoder is None:
            return
        idle = time.monotonic() - _last_ce_use_monotonic
        if idle < RAG_CE_AUTO_UNLOAD_IDLE_SEC * 0.95:
            return
    unload_cross_encoder()


def _schedule_idle_unload() -> None:
    global _idle_unload_timer
    if not RAG_CE_AUTO_UNLOAD:
        return
    with _lock:
        if _idle_unload_timer is not None:
            _idle_unload_timer.cancel()
        _idle_unload_timer = threading.Timer(
            RAG_CE_AUTO_UNLOAD_IDLE_SEC,
            _idle_unload_callback,
        )
        _idle_unload_timer.daemon = True
        _idle_unload_timer.start()


def _touch_ce_use() -> None:
    global _last_ce_use_monotonic
    _last_ce_use_monotonic = time.monotonic()
    _schedule_idle_unload()


def _create_cross_encoder() -> object:
    import torch
    from sentence_transformers import CrossEncoder

    device = _resolve_torch_device()
    dtype = _resolve_torch_dtype(device)
    trace(
        f"RAG_CE ▶ загрузка {RAG_CROSS_ENCODER_MODEL} | device={device} "
        f"dtype={dtype}"
    )
    automodel_args = {"dtype": dtype}
    try:
        model = CrossEncoder(
            RAG_CROSS_ENCODER_MODEL,
            device=device,
            automodel_args=automodel_args,
        )
        trace("RAG_CE ✓ loaded")
        return model
    except Exception as exc:
        if dtype != torch.float32:
            trace(f"RAG_CE ⚠ low-precision load failed, fp32 | {exc}")
            model = CrossEncoder(
                RAG_CROSS_ENCODER_MODEL,
                device=device,
            )
            trace("RAG_CE ✓ loaded (fp32 fallback)")
            return model
        raise


def _load_cross_encoder() -> object | None:
    global _cross_encoder, _use_cosine_fallback
    with _lock:
        if _use_cosine_fallback:
            return None
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            _cross_encoder = _create_cross_encoder()
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
        import torch

        with torch.inference_mode():
            raw = model.predict(pairs, batch_size=min(16, len(pairs)))
        out: List[float] = []
        for r in raw:
            val = float(r)
            out.append(_sigmoid(val))
        _touch_ce_use()
        _release_torch_cache()
        return out
    except Exception as exc:
        trace(f"RAG_CE ⚠ predict fallback | {exc}")
        _release_torch_cache()
        return _cosine_fallback_scores(crit, texts)
