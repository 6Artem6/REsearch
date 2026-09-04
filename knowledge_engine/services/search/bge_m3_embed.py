"""System-wide BAAI/bge-m3 bi-encoder (LanceDB + RAG).

Cross-Encoder ``BAAI/bge-reranker-v2-m3`` is NOT used here (Inbound Gate / RAG
rerank only — see ``src/rag_gateway/cross_encoder.py``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

from knowledge_engine.db.embed_model_guard import CANONICAL_BI_ENCODER
from knowledge_engine.logging_setup import get_logger
from knowledge_engine.ui.run_log import trace

_perf_logger = get_logger(__name__)

EmbedBatchFn = Callable[[Sequence[str]], list[list[float]]]

_lock = threading.Lock()
_model: object | None = None
_injected: EmbedBatchFn | None = None


def set_embed_fn_for_tests(fn: EmbedBatchFn | None) -> None:
    """Replace the bi-encoder in unit tests (no HuggingFace download)."""
    global _injected
    _injected = fn


def set_domain_registry_embed_fn_for_tests(fn: EmbedBatchFn | None) -> None:
    """Alias kept for domain_registry tests."""
    set_embed_fn_for_tests(fn)


def _assert_bi_encoder_name(name: str) -> str:
    n = (name or "").strip() or CANONICAL_BI_ENCODER
    lowered = n.lower()
    if "reranker" in lowered:
        raise ValueError(
            f"bi-encoder must be {CANONICAL_BI_ENCODER}, not cross-encoder {n}"
        )
    if "nomic" in lowered:
        raise ValueError(
            f"nomic-embed-text is removed; use bi-encoder {CANONICAL_BI_ENCODER}"
        )
    return n


def _resolve_torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _canonical_embed_model() -> str:
    from knowledge_engine.config import EMBED_MODEL

    return _assert_bi_encoder_name(EMBED_MODEL)


def unload_bge_m3_model() -> None:
    """Освободить модель и MPS allocator cache — вызывается либо явно, либо
    ml_memory_guard'ом при превышении порога RAM (см. модуль-докстринг там:
    bge-m3 сама по себе держит driver_allocated_memory≈3GB, стабильно не
    растёт от вызова к вызову, и torch.mps.empty_cache() эту память НЕ
    отдаёт, пока модель не выгружена явно — только unload реально снижает
    footprint)."""
    global _model
    with _lock:
        if _model is None:
            return
        _model = None
    from knowledge_engine.services.ml_memory_guard import release_mps_cache

    release_mps_cache()
    trace("EMBED ✓ unloaded (weights released)")


def ensure_bge_m3_loaded() -> None:
    """Public warmup entry point (see ``ml_memory_guard.warmup_pipeline_async``)
    — блокирующая, вызывать из ``asyncio.to_thread``. Идемпотентна: если
    модель уже загружена, `_get_model()` вернёт её мгновенно."""
    _get_model()


def _get_model() -> object:
    from knowledge_engine.services.ml_runtime import assert_ml_weights_allowed

    assert_ml_weights_allowed("BGE-M3 embeddings")
    global _model
    with _lock:
        if _model is not None:
            return _model
        name = _canonical_embed_model()
        from sentence_transformers import SentenceTransformer

        from knowledge_engine.config import EMBED_MODEL_REVISION
        from knowledge_engine.services.hf_model_cache import resolve_hf_snapshot
        from knowledge_engine.services.ml_memory_guard import register_model

        device = _resolve_torch_device()
        local = resolve_hf_snapshot(name, revision=EMBED_MODEL_REVISION)
        trace(f"EMBED load ▶ | model={name} device={device} src=local_cache")
        _model = SentenceTransformer(local, device=device)
        register_model("bge_m3", unload_bge_m3_model)
        return _model


def embed_texts_bge_m3(texts: Sequence[str]) -> list[list[float]]:
    """L2-normalized BGE-M3 vectors."""
    import time

    payload = [str(t or "").strip() for t in texts]
    if _injected is not None:
        return _injected(payload)
    if not payload:
        return []
    t0 = time.perf_counter()
    model = _get_model()
    # Concurrent callers (asyncio.to_thread ingest workers) share this one
    # SentenceTransformer instance; unlike _get_model()'s lazy-init guard,
    # nothing previously serialized the actual .encode() calls, so concurrent
    # threads contended for the same MPS/GIL resources and inflated latency
    # (avg ~337ms under 4-way concurrency vs ~30-280ms single-threaded).
    with _lock:
        vecs = model.encode(
            payload,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    _perf_logger.debug(
        "PERF bge_m3 embed | n=%d chars=%d | %.0fms",
        len(payload),
        sum(len(p) for p in payload),
        (time.perf_counter() - t0) * 1000,
    )
    from knowledge_engine.services.ml_memory_guard import guard_after_use

    guard_after_use("bge_m3")
    return [list(map(float, row)) for row in vecs]


def embed_query_bge_m3(text: str) -> list[float]:
    vecs = embed_texts_bge_m3([(text or "")[:8000]])
    return vecs[0] if vecs else []


class BgeM3Embeddings:
    """LangChain-shaped adapter so VectorStore / LightRAG keep ``embed_query``."""

    def embed_query(self, text: str) -> list[float]:
        return embed_query_bge_m3(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return embed_texts_bge_m3([(t or "")[:8000] for t in texts])
