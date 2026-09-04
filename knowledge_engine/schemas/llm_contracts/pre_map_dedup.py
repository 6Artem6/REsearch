"""Pre-MAP dedup bulk gate — Gemini Flash Lite contract.

canonical_map is modeled as a list of objects (Array of Pairs), not a
dynamic-key dict: Gemini's structured-output schema rejects dynamic-key
objects outright (`additionalProperties is not supported in the Gemini
API`) — the exact same constraint already documented in
knowledge_engine/ingest/tiered_code_pruner.py for a different contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CanonicalMapping(BaseModel):
    canonical_id: str = Field(description="Id of the kept (non-duplicate) candidate.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Ids of candidates that duplicate canonical_id.",
    )


class CanonicalMapContract(BaseModel):
    mappings: list[CanonicalMapping] = Field(
        default_factory=list,
        description="One entry per group of real duplicates found; a candidate "
        "without duplicates is simply absent from this list.",
    )

    def to_dict(self) -> dict[str, list[str]]:
        """canonical_id -> [alias_id, ...], for callers built around the old
        dict-shaped canonical_map."""
        return {m.canonical_id: list(m.aliases) for m in self.mappings}
