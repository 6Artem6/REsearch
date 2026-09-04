"""logging_setup.py: LOG_LEVEL/LOG_TO_FILE/LOG_FILE_PATH config + trace() file mirror."""

from __future__ import annotations

import logging
import logging.handlers

import pytest

import knowledge_engine.logging_setup as logging_setup
from knowledge_engine.logging_setup import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean 'knowledge_engine' logger and unconfigured module state."""
    monkeypatch.setattr(logging_setup, "_configured", False)
    monkeypatch.setattr(logging_setup, "_trace_mirror_checked", False)
    monkeypatch.setattr(logging_setup, "_trace_mirror_logger", None)
    logger = logging.getLogger("knowledge_engine")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


def test_configure_logging_sets_level_from_override():
    logger = configure_logging(level="DEBUG", force=True)
    assert logger.level == logging.DEBUG


def test_configure_logging_defaults_to_info():
    logger = configure_logging(force=True)
    assert logger.level == logging.INFO


def test_configure_logging_is_idempotent_no_duplicate_handlers():
    configure_logging(force=True)
    n1 = len(logging.getLogger("knowledge_engine").handlers)
    configure_logging()  # no force — should be a no-op
    n2 = len(logging.getLogger("knowledge_engine").handlers)
    assert n1 == n2 == 1  # just the console StreamHandler (LOG_TO_FILE off)


def test_get_logger_returns_propagating_child():
    logger = get_logger("knowledge_engine.some_module")
    assert logger.name == "knowledge_engine.some_module"
    assert logger.propagate is True


def test_log_to_file_creates_file_and_writes_debug_entry(tmp_path):
    log_path = tmp_path / "app.log"
    configure_logging(level="DEBUG", to_file=True, file_path=log_path, force=True)
    logger = get_logger("knowledge_engine.test_debug")

    logger.debug("hello debug world")

    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "hello debug world" in content
    assert "[DEBUG]" in content
    assert "knowledge_engine.test_debug" in content


def test_log_level_filters_below_threshold(tmp_path):
    log_path = tmp_path / "app.log"
    configure_logging(level="WARNING", to_file=True, file_path=log_path, force=True)
    logger = get_logger("knowledge_engine.test_filter")

    logger.info("should be filtered out")
    logger.warning("should appear")

    content = log_path.read_text(encoding="utf-8")
    assert "should be filtered out" not in content
    assert "should appear" in content


def test_trace_mirror_logger_none_when_log_to_file_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(logging_setup._config, "LOG_TO_FILE", False)
    assert logging_setup.trace_mirror_logger() is None


def test_trace_mirror_logger_file_only_when_enabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "trace_mirror.log"
    monkeypatch.setattr(logging_setup._config, "LOG_TO_FILE", True)
    monkeypatch.setattr(logging_setup._config, "LOG_FILE_PATH", log_path)
    monkeypatch.setattr(logging_setup._config, "LOG_LEVEL", "INFO")

    mirror = logging_setup.trace_mirror_logger()
    assert mirror is not None
    assert mirror.propagate is False
    assert len(mirror.handlers) == 1
    assert isinstance(mirror.handlers[0], logging.handlers.RotatingFileHandler)

    mirror.info("trace mirror smoke line")
    content = log_path.read_text(encoding="utf-8")
    assert "trace mirror smoke line" in content


def test_trace_function_writes_to_mirror_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: ui.run_log.trace() reaches the file when LOG_TO_FILE is on."""
    import time

    from knowledge_engine.ui.run_log import trace

    log_path = tmp_path / "trace_via_run_log.log"
    monkeypatch.setattr(logging_setup._config, "LOG_TO_FILE", True)
    monkeypatch.setattr(logging_setup._config, "LOG_FILE_PATH", log_path)
    monkeypatch.setattr(logging_setup._config, "LOG_LEVEL", "INFO")

    trace("distinctive trace-to-file marker 12345")
    for _ in range(50):
        if log_path.exists() and "distinctive trace-to-file marker 12345" in log_path.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.05)

    assert log_path.is_file()
    assert "distinctive trace-to-file marker 12345" in log_path.read_text(encoding="utf-8")
