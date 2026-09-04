"""Offline lexical probe embeddings for VectorIntentRouter unit tests.

Cue lists and overlay order come from ``intent_definitions.INTENT_RULES`` —
the same registry production uses for LanceDB reference phrases.
"""

from __future__ import annotations

import hashlib

import numpy as np

from knowledge_engine.src.node_deep_dive.intent_definitions import (
    INTENT_NAMES,
    INTENT_RULES,
    PROBE_WHOLE_MESSAGE_FALLBACK,
    probe_cues,
)

# Intent axes (one-hot-ish) + noise dims — order matches INTENT_RULES.
_INTENTS = INTENT_NAMES
_DIM = 32

# Short UI chips (1–3 tokens) keep full attractor weight; longer text dilutes.
# Slope is calibrated so incidental chip stems in ~12–15 token answers
# stay below VectorIntentRouter threshold 0.82 (cosine vs short reference phrases).
_DILUTION_FULL_MAGNITUDE = 4.0
_DILUTION_LEN_OFFSET = 3
_DILUTION_SLOPE = 0.22


def _axis(intent: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float64)
    if intent in _INTENTS:
        v[_INTENTS.index(intent)] = 1.0
    return v


def _noise(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    v = rng.normal(0.0, 0.05, size=_DIM)
    return v


def _intent_magnitude(n_words: int, *, has_explicit_mode: bool) -> float:
    """Chip-axis weight: full for ``[mode:]`` / short UI; diluted for long answers."""
    if has_explicit_mode:
        return _DILUTION_FULL_MAGNITUDE
    return _DILUTION_FULL_MAGNITUDE / (
        1.0 + _DILUTION_SLOPE * max(0, n_words - _DILUTION_LEN_OFFSET)
    )


def lexical_probe_embed(text: str) -> list[float]:
    """
    Deterministic vector space simulation for unit tests using semantic dilution.

    Short UI chip phrases land strongly on their intent axis.
    Long technical free-text answers naturally dilute intent cues below
    the activation threshold. No domain-specific wordlists (tech_markers).
    """
    t = (text or "").strip().lower()
    words = t.split()
    if not words:
        v = _noise("empty")
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else v.tolist()

    # Per-token noise; position is mixed in so repeated words do not collapse
    # to n * the same vector (which would drown an explicit [mode:] axis).
    v = np.zeros(_DIM, dtype=np.float64)
    for i, w in enumerate(words):
        v += _noise(f"{i}:{w}")

    has_explicit_mode = "[mode:" in t or "mode:" in t

    # Overlay L4/L5 precede generic deep_analysis because INTENT_RULES is ordered.
    matched = False
    for rule in INTENT_RULES:
        if any(c in t for c in probe_cues(rule)):
            magnitude = _intent_magnitude(len(words), has_explicit_mode=has_explicit_mode)
            v += magnitude * _axis(rule.intent)
            matched = True
            break
    if not matched:
        fallback = PROBE_WHOLE_MESSAGE_FALLBACK.get(t)
        if fallback:
            v += 3.0 * _axis(fallback)

    n = np.linalg.norm(v)
    if n < 1e-12:
        v = _noise(t + "|empty")
        n = np.linalg.norm(v)
    return (v / n).tolist()
