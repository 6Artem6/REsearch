"""HTTP timeouts must be retryable and fall through to the next model."""

from __future__ import annotations

from knowledge_engine.services.gemini_stateless import (
    _call_with_model_fallback,
    _is_http_timeout,
    _is_retryable,
)


class ReadTimeout(Exception):
    """Local stand-in for httpx.ReadTimeout (matched by type name)."""


def test_read_timeout_is_retryable() -> None:
    exc = ReadTimeout("The read operation timed out")
    assert _is_http_timeout(exc)
    assert _is_retryable(exc)


def test_nested_read_timeout_is_detected() -> None:
    root = RuntimeError("wrapper")
    root.__cause__ = ReadTimeout("The read operation timed out")
    assert _is_http_timeout(root)
    assert _is_retryable(root)


def test_timeout_falls_through_to_next_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_quota_store.quota_tracking_enabled",
        lambda: False,
    )
    calls: list[str] = []

    def generate(model: str) -> str:
        calls.append(model)
        if model == "model-a":
            raise ReadTimeout("The read operation timed out")
        return '{"ok": true}'

    text = _call_with_model_fallback(
        "test / timeout_fallback",
        generate,
        rpm_pause=False,
        models=["model-a", "model-b"],
    )
    assert text == '{"ok": true}'
    assert calls == ["model-a", "model-b"]
