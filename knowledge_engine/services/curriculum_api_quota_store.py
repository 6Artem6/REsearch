"""Локальные дневные лимиты curriculum API (Google CSE, Semantic Scholar)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from knowledge_engine.config import (
    CURRICULUM_API_QUOTA_TRACK,
    GOOGLE_CSE_DAILY_LIMIT,
    PACKAGE_ROOT,
    SEMANTIC_SCHOLAR_DAILY_LIMIT,
)
from knowledge_engine.ui.run_log import trace

_STATE_PATH = (PACKAGE_ROOT / ".runs" / "curriculum_api_quota_state.json").resolve()
_LOCK = threading.RLock()


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> dict[str, Any]:
    return {
        "day_utc": _utc_day(),
        "updated_at": _now_iso(),
        "google_cse": {
            "requests_today": 0,
            "daily_limit": GOOGLE_CSE_DAILY_LIMIT,
            "blocked_until_day": None,
            "last_status": "",
            "last_at": None,
        },
        "semantic_scholar": {
            "requests_today": 0,
            "daily_limit": SEMANTIC_SCHOLAR_DAILY_LIMIT,
            "blocked_until_day": None,
            "last_status": "",
            "last_at": None,
        },
    }


def _load() -> dict[str, Any]:
    if not _STATE_PATH.is_file():
        return _empty_state()
    try:
        raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _empty_state()
        return raw
    except Exception:
        return _empty_state()


def _save(state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _roll_day(state: dict[str, Any]) -> None:
    today = _utc_day()
    if state.get("day_utc") == today:
        return
    state["day_utc"] = today
    for key in ("google_cse", "semantic_scholar"):
        row = state.get(key)
        if isinstance(row, dict):
            row["requests_today"] = 0
            row["blocked_until_day"] = None


def _bucket(state: dict[str, Any], name: str) -> dict[str, Any]:
    row = state.get(name)
    if not isinstance(row, dict):
        row = {}
        state[name] = row
    if name == "google_cse":
        row.setdefault("daily_limit", GOOGLE_CSE_DAILY_LIMIT)
    else:
        row.setdefault("daily_limit", SEMANTIC_SCHOLAR_DAILY_LIMIT)
    row.setdefault("requests_today", 0)
    row.setdefault("blocked_until_day", None)
    row.setdefault("last_status", "")
    row.setdefault("last_at", None)
    return row


def quota_tracking_enabled() -> bool:
    return CURRICULUM_API_QUOTA_TRACK


def _is_blocked(row: dict[str, Any]) -> bool:
    blocked_day = row.get("blocked_until_day")
    if blocked_day and blocked_day >= _utc_day():
        return True
    limit = int(row.get("daily_limit") or 0)
    used = int(row.get("requests_today") or 0)
    return limit > 0 and used >= limit


def can_use_google_cse() -> tuple[bool, str]:
    if not quota_tracking_enabled():
        return True, "tracking_off"
    with _LOCK:
        state = _load()
        _roll_day(state)
        row = _bucket(state, "google_cse")
        if _is_blocked(row):
            used = row.get("requests_today", 0)
            lim = row.get("daily_limit", GOOGLE_CSE_DAILY_LIMIT)
            return False, f"local_daily_limit {used}/{lim}"
        return True, "ok"


def record_google_cse_result(
    *,
    ok: bool,
    http_status: int = 0,
    quota_exhausted: bool = False,
) -> None:
    if not quota_tracking_enabled():
        return
    with _LOCK:
        state = _load()
        _roll_day(state)
        row = _bucket(state, "google_cse")
        row["requests_today"] = int(row.get("requests_today") or 0) + 1
        row["last_at"] = _now_iso()
        if quota_exhausted or http_status == 429:
            row["blocked_until_day"] = _utc_day()
            row["last_status"] = "quota_429"
            trace(
                f"CURRICULUM quota google_cse ✗ | API 429 — блок до UTC midnight "
                f"(local {row['requests_today']}/{row['daily_limit']})"
            )
        elif ok:
            row["last_status"] = "ok"
        else:
            row["last_status"] = f"http_{http_status or 'err'}"
        _save(state)


def can_use_semantic_scholar() -> tuple[bool, str]:
    if not quota_tracking_enabled():
        return True, "tracking_off"
    with _LOCK:
        state = _load()
        _roll_day(state)
        row = _bucket(state, "semantic_scholar")
        if _is_blocked(row):
            used = row.get("requests_today", 0)
            lim = row.get("daily_limit", SEMANTIC_SCHOLAR_DAILY_LIMIT)
            return False, f"local_daily_limit {used}/{lim}"
        return True, "ok"


def record_semantic_scholar_result(
    *,
    ok: bool,
    http_status: int = 0,
    quota_exhausted: bool = False,
) -> None:
    if not quota_tracking_enabled():
        return
    with _LOCK:
        state = _load()
        _roll_day(state)
        row = _bucket(state, "semantic_scholar")
        row["requests_today"] = int(row.get("requests_today") or 0) + 1
        row["last_at"] = _now_iso()
        if quota_exhausted or http_status in (429, 503):
            row["blocked_until_day"] = _utc_day()
            row["last_status"] = f"quota_{http_status}"
            trace(
                f"CURRICULUM quota semantic_scholar ✗ | {http_status} — "
                f"local {row['requests_today']}/{row['daily_limit']}"
            )
        elif ok:
            row["last_status"] = "ok"
            row["blocked_until_day"] = None
        else:
            row["last_status"] = f"http_{http_status or 'err'}"
        _save(state)


def get_quota_summary() -> dict[str, Any]:
    with _LOCK:
        state = _load()
        _roll_day(state)
        _save(state)
        return dict(state)
