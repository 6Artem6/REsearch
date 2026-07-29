"""Политики сбора источников: hybrid | practical_only | academic_only."""

from __future__ import annotations

from typing import Literal

SourcePolicy = Literal["hybrid", "practical_only", "academic_only"]

_VALID: frozenset[str] = frozenset({"hybrid", "practical_only", "academic_only"})


def normalize_source_policy(raw: str | None, *, default: str = "hybrid") -> str:
    m = (raw or "").strip().lower().replace("-", "_")
    aliases = {
        "practical": "practical_only",
        "practice": "practical_only",
        "blogs": "practical_only",
        "academic": "academic_only",
        "consensus": "academic_only",
        "papers": "academic_only",
        "full": "hybrid",
        "both": "hybrid",
    }
    m = aliases.get(m, m)
    if m in _VALID:
        return m
    d = aliases.get(default, default)
    return d if d in _VALID else "hybrid"


def resolve_source_policy(
    source_policy: str | None,
    generation_mode: str | None,
    *,
    default: str,
) -> str:
    """Явный source_policy; иначе legacy generation_mode (consensus → academic)."""
    if (source_policy or "").strip():
        return normalize_source_policy(source_policy, default=default)
    gm = (generation_mode or "").strip().lower()
    if gm in ("consensus", "deep"):
        return "academic_only"
    return normalize_source_policy(None, default=default)


def depth_for_source_policy(policy: str) -> str:
    if policy in ("hybrid", "academic_only"):
        return "Deep Mechanics"
    return "Standard"
