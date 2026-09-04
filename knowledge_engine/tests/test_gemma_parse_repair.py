"""Truncated-list-tail repair in _parse_structured (LATENCY REDUCTION &
PYDANTIC VALIDATION FIX task): a max_tokens cutoff mid-object inside a
trailing list item must be salvaged by dropping the broken tail instead of
failing the whole response and paying for a full extra Gemma HTTP retry."""

from __future__ import annotations

import json

from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    MapWindowResponse,
)
from knowledge_engine.services.llm.gemma_client import (
    _drop_truncated_list_tail,
    _is_thought_only_empty_response,
    _parse_structured,
)


def _atom(statement: str) -> dict:
    return {
        "scope": "MECHANIC",
        "statement": statement,
        "context_quote": "...",
        "source_chunk_ids": [],
    }


def test_parse_structured_repairs_truncated_trailing_atom() -> None:
    """Regression: real observed failure — knowledge_atoms[5] missing
    'statement' because Gemma's output hit max_tokens mid-object. Must
    salvage atoms[0:5] instead of returning None and forcing a retry."""
    payload = {
        "window_role": "role",
        "window_summary": "summary",
        "knowledge_atoms": [_atom(f"fact {i}") for i in range(5)]
        + [{"scope": "PRINCIPLE"}],  # truncated: missing 'statement'
        "required_diagrams": [],
    }
    raw = json.dumps(payload)
    result = _parse_structured(raw, MapWindowResponse)
    assert result is not None
    assert len(result.knowledge_atoms) == 5
    assert result.knowledge_atoms[-1].statement.startswith("fact 4")


def test_parse_structured_returns_none_when_scalar_field_truncated() -> None:
    """A truncated REQUIRED scalar field (not inside a list) can't be
    salvaged by dropping list items — must still return None."""
    raw = '{"window_role": "role", "knowledge_atoms": []'  # invalid JSON, cut off
    assert _parse_structured(raw, MapWindowResponse) is None


def test_drop_truncated_list_tail_picks_earliest_offending_index() -> None:
    data = {"knowledge_atoms": [_atom("a"), _atom("b"), {"scope": "X"}]}
    errors = [
        {"loc": ("knowledge_atoms", 2, "statement"), "type": "missing"},
    ]
    repaired = _drop_truncated_list_tail(data, errors)
    assert repaired is not None
    repaired_data, dropped = repaired
    assert dropped == 1
    assert len(repaired_data["knowledge_atoms"]) == 2


def test_drop_truncated_list_tail_no_match_returns_none() -> None:
    data = {"knowledge_atoms": []}
    errors = [{"loc": ("window_summary",), "type": "missing"}]
    assert _drop_truncated_list_tail(data, errors) is None


def test_is_thought_only_empty_response_detects_unclosed_thought() -> None:
    """Regression: real observed production failure (ConsensusBatchResponse,
    perf_debug.log) — model emits ONLY an unclosed <thought> block, no JSON
    at all. Retrying the identical prompt reproduces this reliably; the
    detector lets the caller skip straight to fallback instead of paying
    for a doomed same-prompt retry."""
    assert _is_thought_only_empty_response(
        "<thought>*   Input: A set of facts to deduplicate...\n"
        "*   f19, f20 are unique claims in this batch."
    )
    assert _is_thought_only_empty_response("")
    assert _is_thought_only_empty_response("   ")


def test_is_thought_only_empty_response_false_when_json_present() -> None:
    assert not _is_thought_only_empty_response('{"knowledge_atoms": []}')
    assert not _is_thought_only_empty_response(
        '<thought>reasoning here</thought>\n{"knowledge_atoms": []}'
    )
    assert not _is_thought_only_empty_response("```json\n{\"a\": 1}\n```")
