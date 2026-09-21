"""JSON-syntax repair: one targeted Gemini call given the parser error,
instead of blindly re-running the original generation request. See
gemini_stateless.py::_parse_structured_with_repair.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from knowledge_engine.src.adapters.llm_providers import gemini_stateless


class _Sample(BaseModel):
    name: str
    value: int


def test_valid_json_never_calls_repair(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        gemini_stateless, "_generate_once", lambda *a, **k: calls.append("called")
    )
    out = gemini_stateless._parse_structured_with_repair(
        '{"name": "a", "value": 1}', _Sample, "test", "model-x"
    )
    assert out == _Sample(name="a", value=1)
    assert calls == []


def test_json_syntax_error_triggers_one_repair_call(monkeypatch) -> None:
    broken = '{"name": "a" "value": 1}'  # missing comma
    seen: dict[str, str] = {}

    def fake_generate_once(model, payload, system_instruction, response_schema, label):
        seen["model"] = model
        seen["payload"] = payload
        seen["system_instruction"] = system_instruction
        assert response_schema is _Sample
        return json.dumps({"name": "a", "value": 1})

    monkeypatch.setattr(gemini_stateless, "_generate_once", fake_generate_once)
    out = gemini_stateless._parse_structured_with_repair(
        broken, _Sample, "test_label", "model-x"
    )
    assert out == _Sample(name="a", value=1)
    assert seen["model"] == "model-x"
    assert "PARSER_ERROR" in seen["payload"]
    assert broken in seen["payload"]


def test_pydantic_validation_error_is_not_repaired(monkeypatch) -> None:
    """Valid JSON but wrong shape (e.g. missing required field) must raise
    immediately — engine.py's own retry loops (e.g. star_guard) already
    handle these with a targeted, context-aware hint; this generic repair
    has no such context and must not intercept them."""
    calls: list[str] = []
    monkeypatch.setattr(
        gemini_stateless, "_generate_once", lambda *a, **k: calls.append("called")
    )
    with pytest.raises(RuntimeError):
        gemini_stateless._parse_structured_with_repair(
            '{"name": "a"}', _Sample, "test", "model-x"
        )
    assert calls == []


def test_repair_call_itself_failing_raises_original_error(monkeypatch) -> None:
    broken = '{"name": "a" "value": 1}'

    def still_broken(*a, **k):
        return broken

    monkeypatch.setattr(gemini_stateless, "_generate_once", still_broken)
    with pytest.raises(RuntimeError) as exc_info:
        gemini_stateless._parse_structured_with_repair(
            broken, _Sample, "test", "model-x"
        )
    assert "test" in str(exc_info.value)


def test_no_repair_model_raises_immediately(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        gemini_stateless, "_generate_once", lambda *a, **k: calls.append("called")
    )
    with pytest.raises(RuntimeError):
        gemini_stateless._parse_structured_with_repair(
            '{"name": "a" "value": 1}', _Sample, "test", ""
        )
    assert calls == []
