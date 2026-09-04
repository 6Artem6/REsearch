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
    RAG_CROSS_ENCODER_REVISION,
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


def _identity_activation() -> object:
    import torch.nn as nn

    return nn.Identity()


def _create_cross_encoder() -> object:
    from knowledge_engine.services.ml_runtime import assert_ml_weights_allowed

    assert_ml_weights_allowed("Cross-Encoder reranker")
    import torch
    from sentence_transformers import CrossEncoder

    device = _resolve_torch_device()
    dtype = _resolve_torch_dtype(device)
    from knowledge_engine.services.hf_model_cache import resolve_hf_snapshot

    local = resolve_hf_snapshot(
        RAG_CROSS_ENCODER_MODEL,
        revision=RAG_CROSS_ENCODER_REVISION,
    )
    trace(
        f"RAG_CE ▶ загрузка {RAG_CROSS_ENCODER_MODEL} | device={device} "
        f"dtype={dtype} src=local_cache"
    )
    automodel_args = {"dtype": dtype}
    try:
        model = CrossEncoder(
            local,
            device=device,
            automodel_args=automodel_args,
            default_activation_function=_identity_activation(),
        )
        trace("RAG_CE ✓ loaded")
        return model
    except Exception as exc:
        if dtype != torch.float32:
            trace(f"RAG_CE ⚠ low-precision load failed, fp32 | {exc}")
            model = CrossEncoder(
                local,
                device=device,
                default_activation_function=_identity_activation(),
            )
            trace("RAG_CE ✓ loaded (fp32 fallback)")
            return model
        raise


def ensure_cross_encoder_loaded() -> None:
    """Public warmup entry point (see ``ml_memory_guard.warmup_pipeline_async``)
    — блокирующая, вызывать из ``asyncio.to_thread``. Идемпотентна: если
    модель уже загружена (или уже переключилась на cosine fallback),
    `_load_cross_encoder()` вернёт мгновенно."""
    _load_cross_encoder()


def _load_cross_encoder() -> object | None:
    global _cross_encoder, _use_cosine_fallback
    with _lock:
        if _use_cosine_fallback:
            return None
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            _cross_encoder = _create_cross_encoder()
            from knowledge_engine.services.ml_memory_guard import register_model

            register_model("cross_encoder", unload_cross_encoder)
            return _cross_encoder
        except Exception as exc:
            _use_cosine_fallback = True
            trace(f"RAG_CE ⚠ CrossEncoder недоступен, cosine fallback | {exc}")
            return None


def _cosine_fallback_scores(criterion: str, texts: List[str]) -> List[float]:
    """Deterministic fallback: BGE-M3 cosine mapped to [0, 1] (not Cross-Encoder)."""
    from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

    payload = [criterion[:8000], *[t[:8000] for t in texts]]
    vecs = embed_texts_bge_m3(payload)
    if len(vecs) != len(payload):
        return [0.0 for _ in texts]
    qv = np.asarray(vecs[0], dtype=np.float64)
    nq = np.linalg.norm(qv)
    if nq == 0:
        return [0.0 for _ in texts]
    scores: List[float] = []
    for tv_raw in vecs[1:]:
        tv = np.asarray(tv_raw, dtype=np.float64)
        nt = np.linalg.norm(tv)
        if nt == 0:
            scores.append(0.0)
            continue
        cos = float(np.dot(qv, tv) / (nq * nt))
        scores.append(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
    return scores


def score_relevance_pairs(criterion: str, texts: List[str]) -> List[float]:
    """
    Pair scores (relevance_criteria <-> chunk) in [0, 1].

    ``BAAI/bge-reranker-v2-m3`` emits raw logits. Sentence-Transformers
    ``predict()`` would apply Sigmoid by default when num_labels=1 — we force
    Identity and apply σ once so scores are calibrated, not double-squashed.
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
        import torch.nn as nn

        with torch.inference_mode():
            raw = model.predict(
                pairs,
                batch_size=min(16, len(pairs)),
                activation_fct=nn.Identity(),
            )
        out: List[float] = []
        for r in raw:
            out.append(max(0.0, min(1.0, _sigmoid(float(r)))))
        _touch_ce_use()
        from knowledge_engine.services.ml_memory_guard import guard_after_use

        guard_after_use("cross_encoder")
        return out
    except Exception as exc:
        trace(f"RAG_CE ⚠ predict fallback | {exc}")
        _release_torch_cache()
        return _cosine_fallback_scores(crit, texts)
