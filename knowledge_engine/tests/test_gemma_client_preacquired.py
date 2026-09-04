"""RateLimitedLLMClient: preacquired/prefer_slot call paths must not
double-gate against the global GemmaTokenBudgetManager on top of their own
real per-model admission (acquire_parallel_wave's dual-basket wave, or
_pick_slot's slot.limiter.try_acquire)."""

from __future__ import annotations

import asyncio

from knowledge_engine.services.llm.gemma_client import (
    GemmaCloudClient,
    GemmaModelSlot,
    RateLimitedLLMClient,
)
from knowledge_engine.services.llm.rate_limiter import AsyncRateLimiter


class _FakeSchema:
    pass


def _make_slot(label: str, model: str) -> GemmaModelSlot:
    client = GemmaCloudClient(api_key="test-key", model=model)
    limiter = AsyncRateLimiter(max_rpm=1000, max_tpm=1_000_000)
    return GemmaModelSlot(label=label, model=model, client=client, limiter=limiter)


def _patch_complete_structured(monkeypatch):
    captured: dict = {}

    async def fake_complete_structured(self, *args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(
        GemmaCloudClient, "complete_structured", fake_complete_structured
    )
    return captured


def test_post_structured_preacquired_disables_global_token_budget(monkeypatch):
    """Regression: after acquire_parallel_wave already admits a MAP item onto
    a specific per-model slot, post_structured_preacquired must not run the
    GLOBAL GemmaTokenBudgetManager check a second time — that shared,
    single-process budget (sized for ONE model, ~15200 TPM) silently capped
    combined primary+fallback throughput at roughly half of what each
    model's own independent quota allows (confirmed live: both Gemma models
    stuck around 40-50% of their own 16K TPM ceiling on the AI Studio
    dashboard)."""
    captured = _patch_complete_structured(monkeypatch)
    slot = _make_slot("primary", "gemma-test-31b")
    client = RateLimitedLLMClient(slots=[slot])

    asyncio.run(
        client.post_structured_preacquired(
            slot, "system", "prompt", _FakeSchema, label="map/test/w0"
        )
    )

    assert captured.get("use_token_budget") is False


def test_post_structured_prefer_slot_disables_global_token_budget(monkeypatch):
    """REDUCE's priority fast path (prefer_slot) — same double-gate bug."""
    captured = _patch_complete_structured(monkeypatch)
    slot = _make_slot("primary", "gemma-test-31b")
    client = RateLimitedLLMClient(slots=[slot])

    result = asyncio.run(
        client.post_structured(
            "system", "prompt", _FakeSchema, label="reduce_synth/test", slot=slot
        )
    )

    assert result == "ok"
    assert captured.get("use_token_budget") is False


def test_post_structured_fallback_loop_disables_global_token_budget(monkeypatch):
    """The non-prefer_slot path (_pick_slot picks a slot fresh each try) also
    already ran its own real per-model acquire via slot.limiter.try_acquire()
    — must skip the global budget too."""
    captured = _patch_complete_structured(monkeypatch)
    slot = _make_slot("primary", "gemma-test-31b")
    client = RateLimitedLLMClient(slots=[slot])

    result = asyncio.run(
        client.post_structured(
            "system", "prompt", _FakeSchema, label="consensus_batch/test"
        )
    )

    assert result == "ok"
    assert captured.get("use_token_budget") is False


def test_post_structured_preacquired_returns_real_usage_not_estimate(monkeypatch):
    """Regression: reconcile_batch_usage() used to be fed a fresh RE-ESTIMATE
    (gemma_rl.estimate_request_tokens) instead of the REAL usage_total the
    API actually returned — the internal TPM ledger never converged toward
    real Gemma billing, staying inflated at the original estimate forever.
    That's why 'wave full'/'busy' fired while the AI Studio dashboard still
    showed both models under 50% of their own TPM ceiling.
    post_structured_preacquired must surface the real number via on_usage,
    not silently recompute a (likely larger) estimate."""

    async def fake_complete_structured(self, *args, on_usage=None, **kwargs):
        if on_usage is not None:
            on_usage(321)  # the "real" API-reported usage_total
        return "ok"

    monkeypatch.setattr(
        GemmaCloudClient, "complete_structured", fake_complete_structured
    )
    slot = _make_slot("primary", "gemma-test-31b")
    client = RateLimitedLLMClient(slots=[slot])

    out, usage = asyncio.run(
        client.post_structured_preacquired(
            slot,
            "system",
            "a much longer prompt than the tiny real usage above, so a "
            "re-estimate would clearly differ from 321",
            _FakeSchema,
            label="map/test/w0",
        )
    )

    assert out == "ok"
    assert usage == 321


def test_post_structured_preacquired_falls_back_to_estimate_without_usage(
    monkeypatch,
):
    """Defensive fallback: if the API response carries no usage at all (should
    not happen in practice), still return SOME positive number rather than 0
    — a 0-token reconcile would look like a free call and never account for
    real cost."""

    async def fake_complete_structured(self, *args, **kwargs):
        return "ok"

    monkeypatch.setattr(
        GemmaCloudClient, "complete_structured", fake_complete_structured
    )
    slot = _make_slot("primary", "gemma-test-31b")
    client = RateLimitedLLMClient(slots=[slot])

    out, usage = asyncio.run(
        client.post_structured_preacquired(
            slot, "system", "prompt", _FakeSchema, label="map/test/w0"
        )
    )

    assert out == "ok"
    assert usage > 0
