"""Offline probe embeddings for VectorEdgeCaseLexicon unit tests."""

from __future__ import annotations

import hashlib

import numpy as np

from knowledge_engine.src.node_deep_dive.edge_case_lexicon import (
    EDGE_CASE_REFERENCE_PHRASES,
)

_DIM = 48
_LABELS = ("edge_case", "bottleneck", "trade_off", "other")


def _axis(label: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float64)
    if label in _LABELS:
        v[_LABELS.index(label)] = 1.0
    return v


def _noise(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 0.04, size=_DIM)


def edge_case_probe_embed(text: str) -> list[float]:
    """Deterministic stand-in for Ollama when testing edge-case digest ranking."""
    raw = (text or "").strip().lower()
    for label, phrases in EDGE_CASE_REFERENCE_PHRASES.items():
        for p in phrases:
            pl = p.lower()
            if pl in raw or raw in pl:
                v = _axis(label) + _noise(raw)
                n = float(np.linalg.norm(v)) or 1.0
                return (v / n).tolist()
    # Test-only cue tokens mirroring seed phrase themes (not production matching).
    if any(
        tok in raw
        for tok in (
            "timeout",
            "таймаут",
            "gather",
            "каскад",
            "latency",
            "edge",
            "зависа",
            "race",
            "deadlock",
            "bottleneck",
            "узк",
        )
    ):
        label = "edge_case"
    elif any(tok in raw for tok in ("trade-off", "trade_off", "компромисс", "cancel")):
        label = "trade_off"
    elif any(tok in raw for tok in ("токен", "token", "лимит", "rate limit")):
        label = "bottleneck"
    else:
        v = _noise(raw) * 0.2
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()
    v = _axis(label) + _noise(raw)
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()
