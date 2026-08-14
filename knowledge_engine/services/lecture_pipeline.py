"""Условия dense lecture: достаточность локального RAG vs LECTURE_SEARCH."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from knowledge_engine.config import (
    LECTURE_LOCAL_QUALITY_THRESHOLD,
    LECTURE_MIN_LOCAL_SOURCES,
    LOCAL_QUALITY_THRESHOLD,
)
from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)

_MIN_PLAIN_CHARS = 24


@dataclass(frozen=True)
class LectureRagStats:
    local_sources_count: int
    mmr_selected: int
    pinned_count: int
    route_url_count: int
    has_quality_pinned: bool
    local_avg_score: float = 0.0
    local_sum_score: float = 0.0


def build_lecture_rag_stats(
    pinned: list[LectureContextCandidate],
    selected: list[LectureContextCandidate],
    route_urls: list[str],
) -> LectureRagStats:
    """Качественные фрагменты после CE/MMR + pinned whitelist + route URLs."""
    quality_fragments = 0
    has_pinned = False
    scores: list[float] = []
    for c in pinned or []:
        if (c.plain or "").strip():
            has_pinned = True
            break
    for c in list(pinned or []) + list(selected or []):
        plain = (c.plain or "").strip()
        if len(plain) >= _MIN_PLAIN_CHARS:
            quality_fragments += 1
            scores.append(float(c.retrieval_score or 0.0))
    route_n = len(
        [u for u in (route_urls or []) if (u or "").strip().startswith("http")]
    )
    local_count = quality_fragments
    local_sum = float(sum(scores)) if scores else 0.0
    local_avg = (local_sum / len(scores)) if scores else 0.0
    return LectureRagStats(
        local_sources_count=local_count,
        mmr_selected=len(selected or []),
        pinned_count=len(pinned or []),
        route_url_count=route_n,
        has_quality_pinned=has_pinned,
        local_avg_score=local_avg,
        local_sum_score=local_sum,
    )


def _quality_threshold() -> float:
    return float(LOCAL_QUALITY_THRESHOLD or LECTURE_LOCAL_QUALITY_THRESHOLD or 0.50)


def should_bypass_primary_external_search(stats: LectureRagStats) -> bool:
    """
    Skip Exa / SS / Consensus before lecture generation (not tool fallback).

    External search runs ONLY when:
      local_sources < MIN_LOCAL_SOURCES AND avg local score < LOCAL_QUALITY_THRESHOLD
    (unless quality-pinned whitelist context is already present).
    """
    if stats.has_quality_pinned:
        return True
    thr = _quality_threshold()
    count_ok = stats.local_sources_count >= LECTURE_MIN_LOCAL_SOURCES
    quality_ok = stats.local_sources_count > 0 and float(stats.local_avg_score) >= thr
    return count_ok or quality_ok


def needs_primary_external_search(stats: LectureRagStats) -> bool:
    return not should_bypass_primary_external_search(stats)


def log_external_search_bypass(stats: LectureRagStats) -> None:
    logger.info(
        "Local RAG context is sufficient (%d sources). Skipping external search.",
        stats.local_sources_count,
    )
    trace(
        "[LECTURE_PIPELINE] External search bypassed: sufficient local RAG context "
        f"found (count={stats.local_sources_count} "
        f"avg_score={stats.local_avg_score:.3f} "
        f"thr={_quality_threshold():.3f})"
    )


def log_external_search_run(stats: LectureRagStats) -> None:
    thr = _quality_threshold()
    trace(
        "[LECTURE_PIPELINE] External search required: local RAG below threshold "
        f"(count={stats.local_sources_count} < min={LECTURE_MIN_LOCAL_SOURCES} "
        f"and avg_score={stats.local_avg_score:.3f} < thr={thr:.3f})"
    )
