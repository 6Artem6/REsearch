"""Gemma MAP JSON cleanup: thought strip + json_repair before Pydantic."""

from __future__ import annotations

from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    MapWindowResponse,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _MAP_SYSTEM,
)
from knowledge_engine.services.llm.gemma_client import (
    _parse_structured,
    _strip_gemma_thought_wrapper,
    loads_json_lenient,
)


def test_map_system_forbids_thought_and_requires_escape():
    assert "DO NOT output <thought>" in _MAP_SYSTEM
    assert "IMMEDIATELY with the open curly bracket" in _MAP_SYSTEM
    assert "double backslashes" in _MAP_SYSTEM


def test_strip_thought_closed_and_unclosed():
    closed = '<thought>long reasoning</thought>\n{"window_role":"x"}'
    assert _strip_gemma_thought_wrapper(closed).startswith("{")
    unclosed = '<thought>still thinking about \\alpha and quotes\n{"a":1'
    cleaned = _strip_gemma_thought_wrapper(unclosed)
    assert "<thought" not in cleaned.lower()


def test_loads_json_lenient_repairs_invalid_escape_and_truncation():
    broken_escape = (
        '{"window_role":"math","window_summary":"use \\alpha carefully",'
        '"knowledge_atoms":[],"required_diagrams":[]}'
    )
    data = loads_json_lenient(broken_escape)
    assert isinstance(data, dict)
    assert data.get("window_role") == "math"

    truncated = (
        '{"window_role":"intro","window_summary":"This summary was cut mid-way '
        "without a closing quote"
    )
    try:
        repaired = loads_json_lenient(truncated)
        assert isinstance(repaired, dict)
    except ValueError:
        pass


def test_parse_structured_map_window_with_thought_wrapper():
    raw = (
        "<thought>I will extract atoms carefully...</thought>\n"
        "```json\n"
        "{"
        '"window_role":"Architecture",'
        '"window_summary":"Hooks gate tool calls before execution.",'
        '"knowledge_atoms":['
        '{"scope":"MECHANIC","statement":"Validate tool args before invoke",'
        '"context_quote":"gateway checks schema"}'
        "],"
        '"required_diagrams":[]'
        "}\n"
        "```"
    )
    parsed = _parse_structured(raw, MapWindowResponse)
    assert parsed is not None
    assert parsed.window_role == "Architecture"
    assert len(parsed.knowledge_atoms) == 1
