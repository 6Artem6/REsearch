"""Загрузка .env в os.environ — лист-модуль без внутренних зависимостей.

Вынесено из config.py (единственного места, где раньше жила эта логика) для
db/pg_settings.py: pydantic_settings.BaseSettings должен читать уже
заполненный os.environ, но импортировать сам config.py напрямую означало бы
цикл (config.py в конце импортирует pg_settings.py). Оба модуля независимо
подключаются к этому файлу — цикла нет, .env парсится один раз (idempotent
через os.environ.setdefault).
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# Ключи из .env всегда перекрывают export в shell (типичный случай: GRAPH_VERSION=0.8)
DOTENV_FORCE_OVERRIDE_KEYS = frozenset(
    {
        "GRAPH_VERSION",
        "SEMANTIC_SCHOLAR_ENABLED",
    }
)

_loaded = False


def load_dotenv_once() -> None:
    """Подхват .env из корня репо и knowledge_engine/ (idempotent)."""
    global _loaded
    if _loaded:
        return
    candidates = [
        _PACKAGE_ROOT.parent / ".env",
        _PACKAGE_ROOT / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            if key in DOTENV_FORCE_OVERRIDE_KEYS:
                os.environ[key] = val
            else:
                os.environ.setdefault(key, val)
    _loaded = True
