"""Flash Lite RPM hard cap (≤14) and atomic minute reserve."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from knowledge_engine.services import gemini_quota_store
from knowledge_engine.services.gemini_quota_store import (
    _minute_guards,
    reserve_gemini_minute_slot,
)
from knowledge_engine.services.gemini_stateless import default_rpm_limit_for_model


def test_flash_lite_rpm_hard_cap_is_at_most_14():
    assert default_rpm_limit_for_model("gemini-3.5-flash-lite") <= 14
    assert default_rpm_limit_for_model("gemini-3.1-flash-lite") <= 14


def test_flash_lite_shared_rpd_and_tpm_caps():
    from knowledge_engine.config import (
        GEMINI_FLASH_LITE_MAX_RPD,
        GEMINI_FLASH_LITE_MAX_TPM,
    )
    from knowledge_engine.services.gemini_quota_store import default_daily_limit_rpd
    from knowledge_engine.services.gemini_stateless import default_tpm_limit_for_model

    assert GEMINI_FLASH_LITE_MAX_RPD == 490
    assert GEMINI_FLASH_LITE_MAX_TPM == 250000
    assert default_daily_limit_rpd("gemini-3.5-flash-lite") == 490
    assert default_daily_limit_rpd("gemini-3.1-flash-lite") == 490
    assert default_tpm_limit_for_model("gemini-3.5-flash-lite") == 250000
    assert default_tpm_limit_for_model("gemini-3.1-flash-lite") == 250000


def test_rpm_error_limit_is_not_persisted_as_rpd(monkeypatch, tmp_path):
    state_path = tmp_path / "gemini_quota_state.json"
    monkeypatch.setattr(gemini_quota_store, "_STATE_PATH", state_path)

    gemini_quota_store.record_gemini_error(
        "gemini-3.5-flash-lite",
        RuntimeError(
            "429 RESOURCE_EXHAUSTED quotaMetric: generate_requests_per_minute "
            "limit: 15 retry in 30s"
        ),
    )

    row = json.loads(state_path.read_text(encoding="utf-8"))["models"][
        "gemini-3.5-flash-lite"
    ]
    assert row["daily_limit_rpd"] == 490
    assert row["last_reported_quota_limit"] == 15
    assert row["last_reported_quota_class"] == "rpm"


def test_legacy_rpm_value_is_rebuilt_from_config(monkeypatch, tmp_path):
    state_path = tmp_path / "gemini_quota_state.json"
    monkeypatch.setattr(gemini_quota_store, "_STATE_PATH", state_path)
    state = gemini_quota_store._empty_state()
    state["models"]["gemini-3.5-flash-lite"] = {
        "daily_limit_rpd": 15,
        "local_requests_today": 85,
        "block_source": "api_error",
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    usable, reason = gemini_quota_store.model_usable("gemini-3.5-flash-lite")

    assert usable, reason
    row = json.loads(state_path.read_text(encoding="utf-8"))["models"][
        "gemini-3.5-flash-lite"
    ]
    assert row["daily_limit_rpd"] == 490
    assert row["daily_limit_source"] == "config"


def test_minute_reserve_never_exceeds_hard_cap(monkeypatch):
    model = "gemini-3.5-flash-lite-test-rpm"
    _minute_guards.pop(model, None)

    # Force known hard cap via default_rpm path used by guard ctor
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.default_rpm_limit_for_model",
        lambda m: 14 if "flash-lite" in m else 10,
    )
    _minute_guards.pop(model, None)

    ok_flags: list[bool] = []

    def _one(_: int) -> bool:
        return reserve_gemini_minute_slot(model, 100)

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok_flags = list(pool.map(_one, range(20)))

    assert sum(1 for x in ok_flags if x) == 14
    assert sum(1 for x in ok_flags if not x) == 6
    g = _minute_guards[model]
    assert g.rpm_used() == 14
    assert g.rpm_used() <= g._hard_rpm
