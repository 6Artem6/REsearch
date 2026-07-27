"""SQLite кэш репутации доменов (Domain Trust Engine v0.5)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.schemas import DomainTrustResult

_DEFAULT_DB = (PACKAGE_ROOT / ".domain_trust" / "domains.sqlite").resolve()


class DomainTrustStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = (db_path or _DEFAULT_DB).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS domain_trust_store (
                        domain TEXT PRIMARY KEY,
                        trust_score REAL NOT NULL,
                        category TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get_domain(self, domain: str) -> Optional[DomainTrustResult]:
        key = domain.strip().lower()
        if not key:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT domain, trust_score, category, reason, created_at "
                    "FROM domain_trust_store WHERE domain = ?",
                    (key,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return None
        return DomainTrustResult(
            domain=row["domain"],
            trust_score=float(row["trust_score"]),
            category=row["category"],
            reason=row["reason"],
            created_at=row["created_at"],
            from_cache=True,
        )

    def save_domain(self, result: DomainTrustResult) -> None:
        key = result.domain.strip().lower()
        created = result.created_at or datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO domain_trust_store (domain, trust_score, category, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        trust_score = excluded.trust_score,
                        category = excluded.category,
                        reason = excluded.reason,
                        created_at = excluded.created_at
                    """,
                    (key, result.trust_score, result.category, result.reason, created),
                )
                conn.commit()
            finally:
                conn.close()


_store: Optional[DomainTrustStore] = None
_store_lock = threading.Lock()


def get_domain_trust_store() -> DomainTrustStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                from knowledge_engine.config import DOMAIN_TRUST_DB_PATH

                _store = DomainTrustStore(DOMAIN_TRUST_DB_PATH)
    return _store
