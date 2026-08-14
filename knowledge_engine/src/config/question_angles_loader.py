"""Загрузка Question Angles из JSON (кеш в RAM)."""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

_CONFIG_PATH = Path(__file__).resolve().parent / "question_angles.json"
_cache_lock = threading.Lock()


class QuestionAngleConfig(BaseModel):
    angle_id: str = Field(min_length=2, max_length=48)
    keywords: list[str] = Field(min_length=1, max_length=48)
    description: str = Field(min_length=4, max_length=400)


class QuestionAnglesFile(BaseModel):
    version: int = 1
    angles: list[QuestionAngleConfig] = Field(min_length=1, max_length=16)


def _load_file_uncached() -> QuestionAnglesFile:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return QuestionAnglesFile.model_validate(raw)


@lru_cache(maxsize=1)
def get_question_angles_file() -> QuestionAnglesFile:
    """Кеш конфигурации углов (перезагрузка — invalidate_question_angles_cache)."""
    with _cache_lock:
        return _load_file_uncached()


def invalidate_question_angles_cache() -> None:
    get_question_angles_file.cache_clear()


def get_question_angles() -> list[QuestionAngleConfig]:
    return list(get_question_angles_file().angles)


def get_angle_keywords_map() -> dict[str, tuple[str, ...]]:
    return {
        a.angle_id: tuple(k.lower() for k in a.keywords if k.strip())
        for a in get_question_angles()
    }


def get_angle_description(angle_id: str) -> str:
    aid = (angle_id or "").strip()
    for a in get_question_angles():
        if a.angle_id == aid:
            return a.description
    return aid


def format_question_angle_matrix_rules() -> str:
    """Блок матрицы углов для QUESTION_FORMATION_RULES (без хардкода id в .py)."""
    lines = [
        "3. МАТРИЦА УГЛОВ ВОПРОСА (QUESTION ANGLE MATRIX) — ротация внутри ноды:",
        "   - Углы (выбирай один на вопрос, не повторяй подряд тот же):",
    ]
    for i, angle in enumerate(get_question_angles(), start=1):
        letter = chr(ord("a") + i - 1)
        lines.append(f"     {letter}) {angle.angle_id} — {angle.description};")
    lines.append(
        "   - Если предыдущий вопрос был в одном угле — следующий ОБЯЗАН быть "
        "из другого, даже если та же sub_concept."
    )
    lines.append(
        "   - ЗАПРЕЩЕНО два подряд вопроса с одним и тем же углом "
        "(см. last_tutor_question_angle)."
    )
    return "\n".join(lines)
