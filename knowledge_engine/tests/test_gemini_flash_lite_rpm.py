"""Flash Lite RPM hard cap (≤14) and atomic minute reserve."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
