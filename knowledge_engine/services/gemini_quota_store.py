"""Локальные лимиты Gemini (RPD) и блокировки до запроса в API."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from knowledge_engine.config import (
    GEMINI_QUOTA_SAFETY_RATIO,
    GEMINI_RPM_BLOCK_SEC,
    PACKAGE_ROOT,
)
from knowledge_engine.ui.run_log import trace

_STATE_PATH = (PACKAGE_ROOT / ".runs" / "gemini_quota_state.json").resolve()
_LOCK = threading.RLock()
_PACIFIC = ZoneInfo("America/Los_Angeles")

# Free tier (Google AI Studio) — flash-lite RPD overridden by GEMINI_FLASH_LITE_MAX_RPD
_DEFAULT_RPD_BY_SUBSTRING: tuple[tuple[str, int], ...] = (
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
    from knowledge_engine.config import GEMINI_FLASH_LITE_MAX_RPD

    m = (model or "").lower()
    if "flash-lite" in m or "flash_lite" in m:
        return max(1, int(GEMINI_FLASH_LITE_MAX_RPD))
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
    # Before daily_limit_source existed, a generic ``limit: 15`` from an RPM
    # 429 could be persisted as RPD. Rebuild legacy caps from configuration.
    if "daily_limit_source" not in row:
        row["daily_limit_rpd"] = default_daily_limit_rpd(model)
        row["daily_limit_source"] = "config"
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
    return (
        "requestsperdayperprojectpermodel" in norm or "generaterequestsperday" in norm
    )


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


class _GeminiMinuteGuard:
    """Per-model 60s sliding RPM/TPM guard.

    Hard RPM never exceeds default_rpm_limit_for_model (Flash Lite ≤14).
    Soft cap (GEMINI_QUOTA_SAFETY_RATIO) is used only by filter_models_for_quota
    to fail over early; try_reserve enforces the hard ceiling atomically.
    """

    def __init__(self, model: str) -> None:
        from knowledge_engine.services.gemini_stateless import (
            default_rpm_limit_for_model,
            default_tpm_limit_for_model,
        )

        ratio = max(0.5, min(1.0, float(GEMINI_QUOTA_SAFETY_RATIO)))
        self._model = (model or "").strip()
        hard_rpm = max(1, int(default_rpm_limit_for_model(self._model)))
        hard_tpm = max(100, int(default_tpm_limit_for_model(self._model)))
        self._hard_rpm = hard_rpm
        self._hard_tpm = hard_tpm
        # Soft caps for early chain switch (must stay ≤ hard)
        self._soft_rpm = max(1, min(hard_rpm, int(hard_rpm * ratio)))
        self._soft_tpm = max(100, min(hard_tpm, int(hard_tpm * ratio)))
        self._max_rpm = hard_rpm  # alias for diagnostics
        self._max_tpm = hard_tpm
        self._window = 60.0
        self._req_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._req_times and self._req_times[0] <= cutoff:
            self._req_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _would_exceed(self, now: float, est: int, *, soft: bool) -> bool:
        self._evict(now)
        rpm_cap = self._soft_rpm if soft else self._hard_rpm
        tpm_cap = self._soft_tpm if soft else self._hard_tpm
        if len(self._req_times) >= rpm_cap:
            return True
        tpm = sum(t for _, t in self._token_events)
        return tpm + est > tpm_cap

    def try_reserve(self, estimated_tokens: int) -> bool:
        """Atomic hard-cap reserve before HTTP (prevents 15→16 races)."""
        est = max(1, int(estimated_tokens))
        with _LOCK:
            now = time.time()
            if self._would_exceed(now, est, soft=False):
                return False
            self._req_times.append(now)
            self._token_events.append((now, est))
            return True

    def confirm_actual(self, total_tokens: int) -> None:
        """Reconcile last reservation TPM; does not add another RPM slot."""
        actual = max(1, int(total_tokens))
        with _LOCK:
            now = time.time()
            self._evict(now)
            if self._token_events:
                ts, _old = self._token_events[-1]
                self._token_events[-1] = (ts, actual)

    def release_last(self) -> None:
        """Drop last reservation if the HTTP call never left (rare)."""
        with _LOCK:
            if self._req_times:
                self._req_times.pop()
            if self._token_events:
                self._token_events.pop()

    def record_event(self, total_tokens: int) -> None:
        """Legacy append (prefer try_reserve + confirm_actual)."""
        if self.try_reserve(total_tokens):
            self.confirm_actual(total_tokens)

    def record_actual(self, total_tokens: int) -> None:
        self.confirm_actual(total_tokens)

    def rpm_used(self) -> int:
        with _LOCK:
            self._evict(time.time())
            return len(self._req_times)


_minute_guards: dict[str, _GeminiMinuteGuard] = {}


def _minute_guard(model: str) -> _GeminiMinuteGuard:
    m = (model or "").strip()
    g = _minute_guards.get(m)
    if g is None:
        g = _GeminiMinuteGuard(m)
        _minute_guards[m] = g
    return g


def reserve_gemini_minute_slot(
    model: str,
    estimated_tokens: int = 800,
) -> bool:
    """Reserve one hard RPM slot before an API call. False → at hard ceiling."""
    ok = _minute_guard(model).try_reserve(estimated_tokens)
    if not ok:
        g = _minute_guard(model)
        trace(
            f"GEMINI RPM hard_cap ⊘ | model={(model or '').strip()} "
            f"used={g.rpm_used()}/{g._hard_rpm} (refuse >{g._hard_rpm}/min)"
        )
    return ok


def confirm_gemini_minute_slot(model: str, total_tokens: int) -> None:
    _minute_guard(model).confirm_actual(total_tokens)


def release_gemini_minute_slot(model: str) -> None:
    _minute_guard(model).release_last()


def record_gemini_minute_usage(model: str, total_tokens: int) -> None:
    """Fallback when caller did not reserve (still hard-capped)."""
    g = _minute_guard(model)
    if not g.try_reserve(total_tokens):
        trace(
            f"GEMINI RPM hard_cap drop | model={(model or '').strip()} "
            f"— usage not recorded (already at {g._hard_rpm}/min)"
        )
        return
    g.confirm_actual(total_tokens)


def model_minute_guard_ok(model: str, estimated_tokens: int = 800) -> bool:
    """Soft-cap check for chain filter (early failover before hard ceiling)."""
    g = _minute_guard(model)
    with _LOCK:
        return not g._would_exceed(
            time.time(), max(1, int(estimated_tokens)), soft=True
        )


def filter_models_for_quota(models: list[str]) -> list[str]:
    """Локальный RPD + RPM из store. Не дергает API (сбор квот — раз в день: check_gemini_quotas --save)."""
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        out: list[str] = []
        for model in models:
            ok, _reason = model_usable(model, state)
            if not ok:
                continue
            if not model_minute_guard_ok(model):
                continue
            out.append(model)
        if out != models:
            _save_unlocked(state)
        return out


def set_model_daily_limit_rpd(model: str, limit: int) -> None:
    """Синхронизировать локальный RPD cap с VLM_GEMINI_MAX_RPD из .env."""
    m = (model or "").strip()
    if not m:
        return
    cap = max(1, int(limit))
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        row = _ensure_model(state, m)
        row["daily_limit_rpd"] = cap
        row["daily_limit_source"] = "config_override"
        _save_unlocked(state)


def record_gemini_success(
    model: str,
    count_local: bool = True,
    *,
    total_tokens: int | None = None,
    minute_already_reserved: bool = False,
) -> None:
    with _LOCK:
        state = _load_unlocked()
        _roll_day(state)
        row = _ensure_model(state, model)
        if count_local:
            row["local_requests_today"] = int(row.get("local_requests_today") or 0) + 1
        row["last_status"] = "ok"
        row["last_request_at"] = _now_iso()
        _save_unlocked(state)
    tok = total_tokens if total_tokens is not None else 800
    if minute_already_reserved:
        confirm_gemini_minute_slot(model, tok)
    else:
        record_gemini_minute_usage(model, tok)


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
            row["last_reported_quota_limit"] = int(details["limit"])
            row["last_reported_quota_class"] = kind_429
            if kind_429 == "rpd":
                row["reported_daily_limit_rpd"] = int(details["limit"])
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
            quota_kind = (
                "rpd"
                if row.get("likely_daily_per_model")
                else _classify_429(blob, details)
            )
            mrow["last_reported_quota_limit"] = int(details["limit"])
            mrow["last_reported_quota_class"] = quota_kind
            if quota_kind == "rpd":
                mrow["reported_daily_limit_rpd"] = int(details["limit"])
        mrow["last_probe_at"] = _now_iso()
        mrow["block_source"] = "probe"
        if status == "ok":
            mrow["last_status"] = "ok"
            mrow["unavailable"] = False
            mrow["unavailable_reason"] = ""
            mrow["daily_blocked"] = False
            if count_probe:
                mrow["local_requests_today"] = (
                    int(mrow.get("local_requests_today") or 0) + 1
                )
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
    from knowledge_engine.config import GEMINI_QUOTA_TRACK

    return GEMINI_QUOTA_TRACK
