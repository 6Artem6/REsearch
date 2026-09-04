"""TriageDecisionResponse accepts Gemma's keep-only JSON (no prefixItems tuples)."""

from __future__ import annotations

from knowledge_engine.services.article_ingestion.triage_schemas import (
    TriageDecisionResponse,
)


def test_triage_accepts_keep_ranges_without_reasons():
    parsed = TriageDecisionResponse.model_validate(
        {"keep_paragraph_ranges": [["P_2", "P_123"]]}
    )
    assert parsed.keep_paragraph_ranges == [["P_2", "P_123"]]
    assert parsed.pruned_sections_reason == []


def test_triage_json_schema_has_no_tuple_prefix_items():
    schema = TriageDecisionResponse.model_json_schema()
    dumped = str(schema)
    assert "prefixItems" not in dumped
    ranges = schema["properties"]["keep_paragraph_ranges"]
    assert ranges["type"] == "array"
    assert ranges["items"]["type"] == "array"
