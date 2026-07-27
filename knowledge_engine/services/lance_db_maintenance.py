"""Сброс LanceDB при несовместимости версий lance (encodings21 vs PageLayout)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from knowledge_engine.config import LANCE_DB_PATH
from knowledge_engine.ui.run_log import trace


def is_lance_format_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "rustpanic",
        "encodings21",
        "pagelayout",
        "decodeerror",
        "lance.encodings",
        "unknown error",
    )
    return any(n in msg for n in needles)


def reset_lance_directory(reason: str) -> Path:
    """Переносит .lancedb в backup и создаёт пустую директорию."""
    trace(f"LANCE ▶ reset | {reason[:200]}")
    LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
    if any(LANCE_DB_PATH.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = LANCE_DB_PATH.parent / f".lancedb.bak-{stamp}"
        shutil.move(str(LANCE_DB_PATH), str(backup))
        trace(f"LANCE backup → {backup}")
    LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
    return LANCE_DB_PATH
