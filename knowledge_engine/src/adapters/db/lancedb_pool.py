"""Process-wide LanceDB connection pool (one connect() per filesystem path)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_pool: dict[str, Any] = {}


def lancedb_pool_key(db_path: Path | str) -> str:
    return str(Path(db_path).expanduser().resolve())


def get_lancedb_connection(db_path: Path | str) -> Any:
    """Return a shared ``lancedb.connect`` handle for ``db_path``."""
    key = lancedb_pool_key(db_path)
    with _lock:
        db = _pool.get(key)
        if db is not None:
            return db
        import lancedb

        Path(key).mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(key)
        _pool[key] = db
        return db


def lancedb_pool_size() -> int:
    with _lock:
        return len(_pool)


def reset_lancedb_pool_for_tests() -> None:
    """Drop pool entries (tests). Open handles are not closed."""
    with _lock:
        _pool.clear()
