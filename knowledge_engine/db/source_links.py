"""Архив всех найденных ссылок (повторный анализ, cache-first discovery)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowledge_engine.config import PACKAGE_ROOT

_DEFAULT_DB = (PACKAGE_ROOT / ".source_archive" / "links.sqlite").resolve()


class SourceLinkArchive:
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
                    CREATE TABLE IF NOT EXISTS source_links (
                        url TEXT PRIMARY KEY,
                        domain TEXT NOT NULL,
                        trust_score REAL,
                        category TEXT,
                        status TEXT NOT NULL DEFAULT 'discovered',
                        rejection_reason TEXT,
                        discovery_query TEXT,
                        fetch_method TEXT,
                        first_seen_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source_links_domain ON source_links(domain)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source_links_trust ON source_links(trust_score)"
                )
                conn.commit()
            finally:
                conn.close()

    def upsert(
        self,
        url: str,
        domain: str,
        trust_score: Optional[float] = None,
        category: Optional[str] = None,
        status: str = "discovered",
        rejection_reason: Optional[str] = None,
        discovery_query: Optional[str] = None,
        fetch_method: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO source_links (
                        url, domain, trust_score, category, status,
                        rejection_reason, discovery_query, fetch_method,
                        first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        domain = excluded.domain,
                        trust_score = COALESCE(excluded.trust_score, source_links.trust_score),
                        category = COALESCE(excluded.category, source_links.category),
                        status = excluded.status,
                        rejection_reason = COALESCE(excluded.rejection_reason, source_links.rejection_reason),
                        discovery_query = COALESCE(excluded.discovery_query, source_links.discovery_query),
                        fetch_method = COALESCE(excluded.fetch_method, source_links.fetch_method),
                        last_seen_at = excluded.last_seen_at
                    """,
                    (
                        url,
                        domain,
                        trust_score,
                        category,
                        status,
                        rejection_reason,
                        discovery_query,
                        fetch_method,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def mark_explored(self, url: str, fetch_ok: bool, fetch_method: str = "") -> None:
        from knowledge_engine.services.domain_profiler import normalize_domain

        status = "fetched_ok" if fetch_ok else "fetch_empty"
        self.upsert(
            url=url,
            domain=normalize_domain(url),
            status=status,
            fetch_method=fetch_method or None,
        )

    def get_reusable_urls(
        self,
        problem: str,
        explored: set[str],
        limit: int = 12,
        min_trust: float = 0.4,
        high_trust_only: bool = False,
    ) -> list[str]:
        """Ссылки из архива для cache-first (не rejected, не explored)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT url, trust_score, category, status FROM source_links
                    WHERE status NOT IN ('rejected_low_trust', 'fetch_empty')
                      AND (trust_score IS NULL OR trust_score >= ?)
                    ORDER BY (trust_score IS NULL), trust_score DESC, last_seen_at DESC
                    LIMIT 200
                    """,
                    (min_trust,),
                ).fetchall()
            finally:
                conn.close()

        from knowledge_engine.services.domain_profiler import is_high_trust_score

        out: list[str] = []
        problem_lower = (problem or "").lower()
        for row in rows:
            url = row["url"]
            if url in explored:
                continue
            score = row["trust_score"]
            cat = row["category"] or ""
            if high_trust_only and score is not None:
                if not is_high_trust_score(float(score), cat):
                    continue
            # лёгкий матч по домену/URL и задаче (опционально)
            if problem_lower and len(problem_lower) > 8:
                blob = f"{url} {cat}".lower()
                tokens = [t for t in problem_lower.split() if len(t) > 4][:6]
                if tokens and not any(t in blob for t in tokens):
                    # высокий trust — всё равно включаем
                    if score is None or float(score) < 0.75:
                        continue
            out.append(url)
            if len(out) >= limit:
                break
        return out


_archive: Optional[SourceLinkArchive] = None
_archive_lock = threading.Lock()


def get_source_link_archive() -> SourceLinkArchive:
    global _archive
    if _archive is None:
        with _archive_lock:
            if _archive is None:
                from knowledge_engine.config import SOURCE_ARCHIVE_DB_PATH

                _archive = SourceLinkArchive(SOURCE_ARCHIVE_DB_PATH)
    return _archive
