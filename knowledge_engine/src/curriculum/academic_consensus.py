"""Политика Consensus v0.8 для DEEP-нод и академического сбора."""

from __future__ import annotations

from knowledge_engine.config import CURRICULUM_USE_V08_CONSENSUS
from knowledge_engine.src.curriculum.schemas import CurriculumNode, CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

_SOTA_MARKERS: tuple[str, ...] = (
    "sota",
    "state-of-the-art",
    "state of the art",
    "benchmark",
    "бенчмарк",
    "arxiv",
    "paper",
    "статья",
    "research",
    "исследован",
    "formal",
    "формальн",
    "algorithm",
    "алгоритм",
    "ragas",
    "metrics",
    "evaluation framework",
    "научн",
    "академ",
    "theoretical",
    "теоретич",
)


def is_sota_rd_node(node: CurriculumNode) -> bool:
    """SotA / R&D: layer=sota или маркеры глубокой теории в тексте ноды."""
    if (node.layer or "").strip().lower() == "sota":
        return True
    blob = " ".join(
        [
            (node.title or ""),
            (node.brief_summary or ""),
            " ".join(node.core_concepts or []),
            (node.category or ""),
        ]
    ).lower()
    return any(m in blob for m in _SOTA_MARKERS)


def consensus_allowed_for_policy(source_policy: str) -> bool:
    p = (source_policy or "").strip().lower()
    return p in ("hybrid", "academic_only")


async def harvest_consensus_for_node(
    node: CurriculumNode,
    search_vector: str,
    anchor: str,
    reason: str,
    *,
    on_demand: bool = False,
    defer_ingest: bool = False,
    force_playwright: bool = False,
) -> list[CurriculumSearchHit]:
    """Playwright Consensus harvest с явным reason в логе."""
    if not CURRICULUM_USE_V08_CONSENSUS:
        trace(
            f"CURRICULUM consensus ⊘ | node={node.node_id} "
            f"reason={reason} (CURRICULUM_USE_V08_CONSENSUS=false)"
        )
        return []
    vec = (search_vector or "").strip()
    if len(vec) < 8:
        return []
    from knowledge_engine.src.curriculum.curriculum_v08_harvest import (
        harvest_curriculum_sources_v08,
    )

    trace(f"CURRICULUM consensus ▶ | node={node.node_id} reason={reason}")
    try:
        hits = await harvest_curriculum_sources_v08(
            vec,
            f"{anchor}:consensus:{node.node_id}",
            on_demand=on_demand,
            defer_ingest=defer_ingest,
            force_playwright=force_playwright,
        )
        trace(
            f"CURRICULUM consensus ✓ | node={node.node_id} reason={reason} "
            f"hits={len(hits)}"
        )
        return hits
    except Exception as exc:
        trace(f"CURRICULUM consensus ✗ | node={node.node_id} reason={reason} | {exc}")
        return []
