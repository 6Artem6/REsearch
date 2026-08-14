"""SQLite registry of domains blocked by anti-bot (for Exa exclude_domains)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

from knowledge_engine.config import PACKAGE_ROOT

_BLOCKLIST_PATH = (PACKAGE_ROOT / ".runs" / "domain_blocklist.db").resolve()
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_db() -> None:
    _BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_BLOCKLIST_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blocked_domains (
                domain TEXT PRIMARY KEY,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def extract_domain_from_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw.startswith("http"):
        return ""
    try:
        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def get_blocked_domains() -> list[str]:
    """Unique blocked hostnames for Exa `exclude_domains`."""
    with _lock:
        _ensure_db()
        with sqlite3.connect(_BLOCKLIST_PATH) as conn:
            rows = conn.execute(
                "SELECT domain FROM blocked_domains ORDER BY domain"
            ).fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for (dom,) in rows:
        k = (dom or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def add_blocked_domain(url: str, reason: str) -> str:
    """Persist domain from URL; returns normalized domain or empty string."""
    domain = extract_domain_from_url(url)
    if not domain:
        return ""
    r = (reason or "").strip()[:240]
    with _lock:
        _ensure_db()
        with sqlite3.connect(_BLOCKLIST_PATH) as conn:
            conn.execute(
                """
                INSERT INTO blocked_domains (domain, reason, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    reason = excluded.reason,
                    created_at = excluded.created_at
                """,
                (domain, r, _now_iso()),
            )
            conn.commit()
    return domain


def is_domain_blocked(url: str, blocked_domains: set[str] | None = None) -> bool:
    dom = extract_domain_from_url(url)
    if not dom:
        return False
    if blocked_domains is not None:
        return dom in blocked_domains
    return dom in set(get_blocked_domains())


def load_blocked_domain_set() -> set[str]:
    return set(get_blocked_domains())


def remove_blocked_domain(domain: str) -> bool:
    """Remove one hostname from blocklist (re-test Exa on that domain)."""
    k = (domain or "").strip().lower()
    if k.startswith("www."):
        k = k[4:]
    if not k:
        return False
    with _lock:
        _ensure_db()
        with sqlite3.connect(_BLOCKLIST_PATH) as conn:
            cur = conn.execute("DELETE FROM blocked_domains WHERE domain = ?", (k,))
            conn.commit()
            return cur.rowcount > 0
