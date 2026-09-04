"""AdaptiveTokenRateLimiter: cache-aware EMA adjustment on top of
TokenRateGovernor's sliding-window RPM/TPM bucket."""

from __future__ import annotations

from knowledge_engine.services.token_rate_governor import (
    AdaptiveTokenRateLimiter,
    get_governor,
    reset_governors_for_tests,
)


def test_get_governor_returns_adaptive_limiter():
    reset_governors_for_tests()
    g = get_governor("gemini-3.5-flash-lite")
    assert isinstance(g, AdaptiveTokenRateLimiter)
    assert g.effective_cache_factor == 1.0


def test_reconcile_returns_overestimated_tokens_to_window():
    """ТЗ тест 1: отправили 10k с прогнозом 10k, пришёл
    cached_content_token_count=9000 — лимитер должен вернуть 9000 токенов
    в окно доступности (real_uncached_prompt=1000)."""
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=100, max_tpm=100_000)

    waited = limiter.reserve_tokens(10_000, 0)
    assert waited == 0.0
    assert limiter.snapshot().tpm_used == 10_000

    limiter.reconcile_tokens(
        estimated_prompt_tokens=10_000,
        actual_prompt_tokens=10_000,
        cached_content_tokens=9_000,
        actual_completion_tokens=0,
    )

    # 10_000 projected -> 1_000 real: 9_000 freed back into the window.
    assert limiter.snapshot().tpm_used == 1_000


def test_reserve_tokens_applies_cache_factor_to_prompt_only():
    """Cache factor must scale the PROMPT estimate only — never the output
    estimate (caching never discounts generation)."""
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=100, max_tpm=100_000)
    limiter.effective_cache_factor = 0.5

    limiter.reserve_tokens(10_000, 2_000)

    # 10_000 * 0.5 + 2_000 = 7_000
    assert limiter.snapshot().tpm_used == 7_000


def test_effective_cache_factor_adapts_toward_real_hit_rate():
    """ТЗ тест 2: серия повторяющихся запросов с одинаковым фактическим
    hit-rate (~90% cached, alpha=0.1 каждый раз) должна тянуть
    effective_cache_factor от холодного старта (1.0) к реальному alpha,
    монотонно приближаясь на каждой итерации (EMA 0.8/0.2)."""
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=1000, max_tpm=10_000_000)

    prev = limiter.effective_cache_factor
    assert prev == 1.0
    for _ in range(25):
        limiter.reserve_tokens(10_000, 0)
        limiter.reconcile_tokens(
            estimated_prompt_tokens=10_000,
            actual_prompt_tokens=10_000,
            cached_content_tokens=9_000,  # alpha = 1000/10000 = 0.1 every time
            actual_completion_tokens=0,
        )
        assert limiter.effective_cache_factor < prev  # monotonic convergence
        prev = limiter.effective_cache_factor

    # factor_n - 0.1 = 0.8^n * (1.0 - 0.1); after 25 iterations that decays
    # to ~0.0034 — comfortably inside a 0.01 tolerance band around the real
    # steady-state hit rate (0.1).
    assert abs(limiter.effective_cache_factor - 0.1) < 0.01


def test_reserve_tokens_is_warmup_bypasses_bucket_entirely():
    """Warmup-запросы не должны списывать TPM реальных пользователей."""
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=100, max_tpm=100_000)

    waited = limiter.reserve_tokens(50_000, 0, is_warmup=True)

    assert waited == 0.0
    assert limiter.snapshot().tpm_used == 0
    assert limiter.snapshot().rpm_used == 0


def test_reconcile_tokens_is_warmup_noop():
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=100, max_tpm=100_000)
    limiter.reserve_tokens(1_000, 0)
    factor_before = limiter.effective_cache_factor

    limiter.reconcile_tokens(
        estimated_prompt_tokens=1_000,
        actual_prompt_tokens=1_000,
        cached_content_tokens=900,
        actual_completion_tokens=0,
        is_warmup=True,
    )

    # Untouched: no cache-factor update, no bucket reconciliation.
    assert limiter.effective_cache_factor == factor_before
    assert limiter.snapshot().tpm_used == 1_000


def test_reconcile_tokens_handles_fully_cached_prompt():
    """Edge case: real_uncached_prompt == 0 (100% cache hit) must not be
    dropped by TokenRateGovernor.confirm()'s `actual_tokens <= 0` guard —
    real_tpm_cost floors at 1, not 0."""
    limiter = AdaptiveTokenRateLimiter("gemini-test", max_rpm=100, max_tpm=100_000)
    limiter.reserve_tokens(5_000, 0)

    limiter.reconcile_tokens(
        estimated_prompt_tokens=5_000,
        actual_prompt_tokens=5_000,
        cached_content_tokens=5_000,
        actual_completion_tokens=0,
    )

    assert limiter.snapshot().tpm_used == 1
