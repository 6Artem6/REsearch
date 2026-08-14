"""Матрица квот источников: layer × node_risk_kind → academic / practical caps."""

from __future__ import annotations

from typing import NamedTuple

from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS


class SourceQuota(NamedTuple):
    academic_max: int
    practical_max: int
    total_max: int = CURRICULUM_DEEP_NODE_MAX_HITS


def get_source_quota(layer: str, node_risk_kind: str) -> SourceQuota:
    """Максимум академических и практических источников для ноды."""
    layer_normalized = (layer or "foundation").strip().lower()
    is_deep = (node_risk_kind or "BASE").strip().upper() == "DEEP"

    if layer_normalized == "sota":
        academic = 3 if is_deep else 2
        return SourceQuota(academic_max=academic, practical_max=1, total_max=4)

    if layer_normalized == "advanced":
        return SourceQuota(academic_max=2, practical_max=2, total_max=4)

    # foundation (default)
    if is_deep:
        return SourceQuota(academic_max=1, practical_max=3, total_max=4)
    return SourceQuota(academic_max=0, practical_max=2, total_max=2)
