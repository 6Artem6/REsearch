"""Регресс-тест на баг sync_env_catalog.py: активная (не закомментированная)
строка ВНУТРИ старого catalog-блока раньше молча уничтожалась при
--merge-env — не переносилась ни в тело файла, ни обратно в блок (т.к. была
активна, значит не в miss_env). Нашли на реальном .env: пользователь вручную
раскомментировал CURRICULUM_TWO_PASS_MODEL_FIRST_ENABLED прямо внутри блока,
следующий --merge-env стёр бы это значение обратно на закомментированный
дефолт без единого предупреждения."""

from __future__ import annotations

from knowledge_engine.scripts.sync_env_catalog import (
    _MARKER_END,
    _MARKER_START,
    strip_catalog_block,
)


def test_strip_catalog_block_preserves_manually_activated_override():
    text = (
        "SOME_HAND_SET_KEY=1\n"
        f"{_MARKER_START}\n"
        "# ACADEMIC_FAST_FETCH_TIMEOUT_SEC=8.0\n"
        "CURRICULUM_TWO_PASS_MODEL_FIRST_ENABLED=true\n"
        "# CURRICULUM_URL_VALIDATE_TIMEOUT_SEC=10\n"
        f"{_MARKER_END}\n"
    )
    base, preserved = strip_catalog_block(text)

    assert "SOME_HAND_SET_KEY=1" in base
    assert _MARKER_START not in base
    assert preserved == ["CURRICULUM_TWO_PASS_MODEL_FIRST_ENABLED=true"]


def test_strip_catalog_block_no_active_lines_returns_empty_preserved():
    text = (
        "SOME_HAND_SET_KEY=1\n"
        f"{_MARKER_START}\n"
        "# ACADEMIC_FAST_FETCH_TIMEOUT_SEC=8.0\n"
        "# CURRICULUM_URL_VALIDATE_TIMEOUT_SEC=10\n"
        f"{_MARKER_END}\n"
    )
    base, preserved = strip_catalog_block(text)

    assert preserved == []
    assert "SOME_HAND_SET_KEY=1" in base


def test_strip_catalog_block_no_marker_returns_text_unchanged():
    text = "SOME_HAND_SET_KEY=1\nOTHER_KEY=2\n"
    base, preserved = strip_catalog_block(text)

    assert base == text.rstrip()
    assert preserved == []
