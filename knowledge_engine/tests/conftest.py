"""Shared test fixtures.

Pre-MAP Dedup's Context Extraction (src/deduplication/pre_map_deduplicator.py)
Triages every TEXT/CODE unit through a real Flash Lite call before
MMR/AST-capping — as of the Group Batching change, deduplicate_before_map_
reduce() calls _flash_lite_triage_core_units_batch() (ALL candidates in one
call), which internally falls back to _flash_lite_triage_core_units() (one
call per candidate) only if the batch itself fails. Code candidates are
compared via the isolated code_deduplicator.deduplicate_code_candidates()
(README/tree fetch + its own Flash Lite call), not the generic Bulk Gate.
The many existing fast, isolated unit tests for this module predate all of
these and don't mock any of them out — without a default pass-through here
they would all silently start making real network/Gemini calls (slow, costs
quota, requires GEMINI_API_KEY, may hit GitHub). Tests that specifically
want to exercise the real calls live in files with "live" in their
name/path and are exempted below.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_triage: test exercises _flash_lite_triage_core_units or "
        "_flash_lite_triage_core_units_batch directly and must NOT have "
        "them replaced by the pass-through autouse fixture.",
    )


@pytest.fixture(autouse=True)
def _pre_map_dedup_triage_passthrough_by_default(request, monkeypatch):
    if "live" in str(request.node.fspath).lower():
        return
    if request.node.get_closest_marker("real_triage"):
        return
    try:
        import knowledge_engine.src.deduplication.pre_map_deduplicator as pmd
    except Exception:
        return

    async def _passthrough(units, **kwargs):
        return list(units)

    async def _passthrough_batch(units_by_id, **kwargs):
        return dict(units_by_id)

    monkeypatch.setattr(pmd, "_flash_lite_triage_core_units", _passthrough)
    monkeypatch.setattr(pmd, "_flash_lite_triage_core_units_batch", _passthrough_batch)

    try:
        import knowledge_engine.src.deduplication.code_deduplicator as cd
    except Exception:
        return

    async def _code_dedup_passthrough(candidates, **kwargs):
        return {}

    monkeypatch.setattr(cd, "deduplicate_code_candidates", _code_dedup_passthrough)
