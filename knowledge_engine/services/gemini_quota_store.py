"""Локальные лимиты Gemini (RPD) и блокировки до запроса в API."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from knowledge_engine.config import GEMINI_RPM_BLOCK_SEC, PACKAGE_ROOT
from knowledge_engine.ui.run_log import trace

_STATE_PATH = (PACKAGE_ROOT / ".runs" / "gemini_quota_state.json").resolve()
_LOCK = threading.RLock()
_PACIFIC = ZoneInfo("America/Los_Angeles")

# Free tier (Google AI Studio), ориентиры — обновляются из 429 limit: N
_DEFAULT_RPD_BY_SUBSTRING: tuple[tuple[str, int], ...] = (
    ("flash-lite", 500),
    ("flash_lite", 500),
    ("gemma-4", 14400),
    ("flash-preview", 20),
    ("flash", 20),
)


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _pacific_day() -> str:
    return datetime.now(_PACIFIC).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_daily_limit_rpd(model: str) -> int:
    m = (model or "").lower()
    for needle, limit in _DEFAULT_RPD_BY_SUBSTRING:
        if needle in m:
            return limit
    return 20


def _empty_state() -> dict[str, Any]:
    return {
        "day_pacific": _pacific_day(),
        "day_utc": _utc_day(),
        "updated_at": _now_iso(),
        "models": {},
    }


def _load_unlocked() -> dict[str, Any]:
    if not _STATE_PATH.is_file():
        return _empty_state()
    try:
        raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _empty_state()
        if "models" not in raw:
            raw["models"] = {}
        return raw
    except Exception:
        return _empty_state()


def _save_unlocked(state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _roll_day(state: dict[str, Any]) -> None:
    """RPD: сброс по Pacific midnight (как в Gemini API docs)."""
    today = _pacific_day()
    if state.get("day_pacific") == today:
        return
    state["day_pacific"] = today
    state["day_utc"] = _utc_day()
    state.pop("quotas_collected_day_utc", None)
    for row in state.get("models", {}).values():
        if isinstance(row, dict):
            row["local_requests_today"] = 0
            row["daily_blocked"] = False
            row["block_source"] = None
            if row.get("block_source") == "probe":
                row["unavailable"] = False
                row["unavailable_reason"] = ""


def _ensure_model(state: dict[str, Any], model: str) -> dict[str, Any]:
    models = state.setdefault("models", {})
    row = models.get(model)
    if not isinstance(row, dict):
        row = {}
        models[model] = row
    if "daily_limit_rpd" not in row:
        row["daily_limit_rpd"] = default_daily_limit_rpd(model)
    row.setdefault("local_requests_today", 0)
    row.setdefault("daily_blocked", False)
    row.setdefault("rpm_blocked_until", None)
    row.setdefault("unavailable", False)
    row.setdefault("unavailable_reason", "")
    row.setdefault("last_status", "")
    row.setdefault("last_request_at", None)
    row.setdefault("last_probe_at", None)
    row.setdefault("block_source", None)
    return row


def extract_quota_fields_from_blob(blob: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    low = blob.lower()
    m = re.search(r"limit:\s*(\d+)", blob, re.I)
    if m:
        out["limit"] = int(m.group(1))
    m = re.search(r"quotaId['\"]?\s*[:=]\s*['\"]?([^'\"\\s,}]+)", blob, re.I)
    if m:
        out["quota_id"] = m.group(1)
    m = re.search(r"quotaMetric['\"]?\s*[:=]\s*['\"]?([^'\"\\s,}]+)", blob, re.I)
    if m:
        out["quota_metric"] = m.group(1)
    if "perminute" in low.replace("_", "").replace("-", "") or "per minute" in low:
        out["quota_class_hint"] = "per_minute"
    elif "perday" in low.replace("_", "").replace("-", "") or "per day" in low:
        out["quota_class_hint"] = "per_day"
    code_m = re.search(r"\b(429|403|404|500|503)\b", blob)
    if code_m:
        out["http_like_code"] = int(code_m.group(1))
    return out


def _parse_rpm_until(blob: str) -> str | None:
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", blob, re.I)
    if not m:
        m = re.search(
            r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", blob, re.I
        )
    if not m:
        return None
    sec = min(float(m.group(1)) + 0.5, 180.0)
    until = datetime.now(timezone.utc).timestamp() + sec
    return datetime.fromtimestamp(until, tz=timezone.utc).isoformat()


def _rpm_block_until_default() -> str:
    sec = max(30.0, min(GEMINI_RPM_BLOCK_SEC, 60.0))
    until = datetime.now(timezone.utc) + timedelta(seconds=sec)
    return until.isoformat()


def _is_daily_quota_blob(blob: str) -> bool:
    norm = blob.lower().replace("_", "").replace("-", "")
    return "requestsperdayperprojectpermodel" in norm or "generaterequestsperday" in norm


def _classify_429(blob: str, details: dict[str, Any]) -> str:
    """rpm | rpd | unknown"""
    if _is_daily_quota_blob(blob):
        return "rpd"
    if details.get("quota_class_hint") == "per_day":
        return "rpd"
    norm = blob.lower()
    if "perday" in norm or "generaterequestsperday" in norm.replace("_", ""):
        return "rpd"
    if details.get("quota_class_hint") == "per_minute":
        return "rpm"
    if "perminute" in norm.replace("_", "").replace("-", ""):
        return "rpm"
    if _parse_rpm_until(blob):
        return "rpm"
    return "unknown"


def model_usable(model: str, state: dict[str, Any] | None = None) -> tuple[bool, str]:
    with _LOCK:
        st = state if state is not None else _load_unlocked()
        _roll_day(st)
        row = _ensure_model(st, model)
        if row.get("unavailable"):
            if row.get("block_source") == "probe":
                collected = st.get("quotas_collected_day_utc")
                if collected != st.get("day_pacific"):
                    pass
                else:
                    reason = (row.get("unavailable_reason") or "unavailable").strip()
                    return False, f"модель недоступна в API ({reason[:120]})"
            else:
                reason = (row.get("unavailable_reason") or "unavailable").strip()
                return False, f"модель недоступна в API ({reason[:120]})"
        if row.get("daily_blocked"):
            if row.get("block_source") != "probe":
                limit = row.get("daily_limit_rpd", 20)
                return False, (
                    f"дневная квота исчерпана (лимит ~{limit} RPD, "
                    f"Pacific {st.get('day_pacific')})"
                )
            collected = st.get("quotas_collected_day_utc")
            if collected == st.get("day_pacific"):
                limit = row.get("daily_limit_rpd", 20)
                return False, f"дневная квота (probe {collected}): ~{limit} RPD"
        rpm_until = row.get("rpm_blocked_until")
        if rpm_until:
            try:
                until_dt = datetime.fromisoformat(str(rpm_until).replace("Z", "+00:00"))
                if until_dt > datetime.now(timezone.utc):
                    return False, f"RPM пауза до {rpm_until}"
                row["rpm_blocked_until"] = None
            except Exception:
                row["rpm_blocked_until"] = None
        local = int(row.get("local_requests_today") or 0)
        limit = int(row.get("daily_limit_rpd") or default_daily_limit_rpd(model))
        if local >= limit:
            return False, (
                f"локальный счётчик {local}/{limit} RPD "
                f"(Pacific день {st.get('day_pacific')})"
            )
        if state is None:
            _save_unlocked(st)
        return True, "ok"


def filter_models_for_quota(models: list[str]) -> list[str]:
    """Локальный RPD + RPM из store. Не дергает API (сбор квот — раз в день: check_gemini_quotas --save)."""
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        out: list[str] = []
        for model in models:
            ok, _reason = model_usable(model, state)
            if ok:
                out.append(model)
        if out != models:
            _save_unlocked(state)
        return out


def record_gemini_success(model: str, count_local: bool = True) -> None:
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        row = _ensure_model(state, model)
        if count_local:
            row["local_requests_today"] = int(row.get("local_requests_today") or 0) + 1
        row["last_status"] = "ok"
        row["last_request_at"] = _now_iso()
        _save_unlocked(state)


def record_gemini_error(model: str, exc: BaseException) -> None:
    blob = _error_blob(exc)
    details = extract_quota_fields_from_blob(blob)
    kind_429 = _classify_429(blob, details)
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        row = _ensure_model(state, model)
        row["last_status"] = "error"
        row["last_request_at"] = _now_iso()
        row["block_source"] = "api_error"
        if details.get("limit"):
            row["daily_limit_rpd"] = int(details["limit"])
        code = details.get("http_like_code")
        if code == 404:
            row["unavailable"] = True
            row["unavailable_reason"] = blob[:400]
        elif kind_429 == "rpd":
            row["daily_blocked"] = True
        elif code == 429 or kind_429 in ("rpm", "unknown"):
            until = _parse_rpm_until(blob) or _rpm_block_until_default()
            row["rpm_blocked_until"] = until
            row["daily_blocked"] = False
        _save_unlocked(state)


def _error_blob(exc: BaseException) -> str:
    parts: list[str] = [str(exc)]
    for attr in ("message", "response", "details"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(str(val))
    return "\n".join(parts)


def apply_probe_result(row: dict[str, Any], count_probe: bool = False) -> None:
    """Обновить store из строки check_gemini_quotas."""
    model = str(row.get("model") or "").strip()
    if not model:
        return
    status = row.get("status")
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        state["quotas_collected_day_utc"] = state.get("day_pacific")
        mrow = _ensure_model(state, model)
        details = row.get("quota_details") or {}
        blob = (row.get("error_preview") or "")[:1200]
        if details.get("limit"):
            mrow["daily_limit_rpd"] = int(details["limit"])
        mrow["last_probe_at"] = _now_iso()
        mrow["block_source"] = "probe"
        if status == "ok":
            mrow["last_status"] = "ok"
            mrow["unavailable"] = False
            mrow["unavailable_reason"] = ""
            mrow["daily_blocked"] = False
            if count_probe:
                mrow["local_requests_today"] = int(mrow.get("local_requests_today") or 0) + 1
        else:
            mrow["last_status"] = "error"
            code = details.get("http_like_code")
            if code == 404:
                mrow["unavailable"] = True
                preview = (row.get("error_preview") or "")[:400]
                mrow["unavailable_reason"] = preview
            if row.get("likely_daily_per_model"):
                mrow["daily_blocked"] = True
            elif code == 429:
                kind = _classify_429(blob, details)
                if kind == "rpd":
                    mrow["daily_blocked"] = True
                else:
                    mrow["rpm_blocked_until"] = (
                        _parse_rpm_until(blob) or _rpm_block_until_default()
                    )
            retry = row.get("retry_after_sec")
            if retry is not None and not row.get("likely_daily_per_model"):
                until = datetime.now(timezone.utc).timestamp() + float(retry)
                mrow["rpm_blocked_until"] = datetime.fromtimestamp(
                    until, tz=timezone.utc
                ).isoformat()
        _save_unlocked(state)


def clear_stale_quota_blocks() -> int:
    """При старте worker: истёкшие RPM, устаревшие probe/daily блоки."""
    cleared = 0
    now = datetime.now(timezone.utc)
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        today = state.get("day_pacific")
        for name, row in state.get("models", {}).items():
            if not isinstance(row, dict):
                continue
            rpm_until = row.get("rpm_blocked_until")
            if rpm_until:
                try:
                    until_dt = datetime.fromisoformat(
                        str(rpm_until).replace("Z", "+00:00")
                    )
                    if until_dt <= now:
                        row["rpm_blocked_until"] = None
                        cleared += 1
                except Exception:
                    row["rpm_blocked_until"] = None
                    cleared += 1
            if row.get("block_source") == "probe":
                if state.get("quotas_collected_day_utc") != today:
                    if row.get("daily_blocked"):
                        row["daily_blocked"] = False
                        cleared += 1
                    if row.get("unavailable"):
                        row["unavailable"] = False
                        row["unavailable_reason"] = ""
                        cleared += 1
            if row.get("daily_blocked"):
                local = int(row.get("local_requests_today") or 0)
                limit = int(row.get("daily_limit_rpd") or default_daily_limit_rpd(name))
                # Ложный daily_blocked после RPM 429 (Studio RPD ещё не исчерпан)
                if local < limit:
                    row["daily_blocked"] = False
                    row["block_source"] = None
                    cleared += 1
            if row.get("unavailable") and name.endswith("-it"):
                wrong = name.replace("-it", "")
                if state.get("models", {}).get(wrong, {}).get("unavailable"):
                    pass  # keep separate entries
        state.setdefault("day_pacific", today)
        state.setdefault("day_utc", _utc_day())
        _save_unlocked(state)
        if cleared:
            trace(f"GEMINI quota store | очищено устаревших блокировок: {cleared}")
    return cleared


def get_quota_summary() -> dict[str, Any]:
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        models_out: dict[str, Any] = {}
        for name, row in sorted(state.get("models", {}).items()):
            if not isinstance(row, dict):
                continue
            ok, reason = model_usable(name, state)
            models_out[name] = {
                "usable": ok,
                "reason": reason,
                "daily_limit_rpd": row.get("daily_limit_rpd"),
                "local_requests_today": row.get("local_requests_today"),
                "daily_blocked": row.get("daily_blocked"),
                "rpm_blocked_until": row.get("rpm_blocked_until"),
                "unavailable": row.get("unavailable"),
                "last_status": row.get("last_status"),
                "last_request_at": row.get("last_request_at"),
                "last_probe_at": row.get("last_probe_at"),
                "block_source": row.get("block_source"),
            }
        return {
            "state_path": str(_STATE_PATH),
            "day_pacific": state.get("day_pacific"),
            "day_utc": state.get("day_utc"),
            "quotas_collected_day_utc": state.get("quotas_collected_day_utc"),
            "updated_at": state.get("updated_at"),
            "models": models_out,
        }


def quota_tracking_enabled() -> bool:
    import os

    raw = os.getenv("GEMINI_QUOTA_TRACK", "true").lower()
    return raw in ("1", "true", "yes")
