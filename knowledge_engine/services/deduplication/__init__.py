"""Claim consensus / entity-guided deduplication (opt-in)."""

from knowledge_engine.services.deduplication.entity_consensus_engine import (
    LocalFactDeduplicator,
    apply_anti_bloat_anchors,
    apply_entity_consensus_to_atoms,
    build_token_bounded_batches,
    claim_dedup_is_enabled,
    group_facts_by_entity,
)

__all__ = [
    "LocalFactDeduplicator",
    "apply_anti_bloat_anchors",
    "apply_entity_consensus_to_atoms",
    "build_token_bounded_batches",
    "claim_dedup_is_enabled",
    "group_facts_by_entity",
]
