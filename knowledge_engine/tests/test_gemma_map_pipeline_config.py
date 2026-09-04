"""MAP pipeline: unified fixed limits for every provider/model."""

from __future__ import annotations

import asyncio
import inspect

import knowledge_engine.config as cfg
import knowledge_engine.services.article_ingestion.blog_spatial_summarizer as spatial_mod
from knowledge_engine.config import (
    BLOG_SPATIAL_MAP_CONCURRENCY,
    BLOG_SPATIAL_MAP_MAX_TOKENS,
    GEMMA_CONCURRENCY,
    GEMMA_MAP_MAX_OUTPUT_TOKENS,
    MAX_CONCURRENT_MAP_REQUESTS,
    gemma_map_concurrency_live,
    map_pipeline_concurrency,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    MapWindowResponse,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _MAP_SYSTEM,
    MapReduceArticleJob,
    _prompt_for_window,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.llm.gemma_client import (
    resolve_gemma_map_max_output_tokens,
)


def test_map_config_defaults():
    assert BLOG_SPATIAL_MAP_MAX_TOKENS == 2800
    # GEMMA_MAP_MAX_OUTPUT_TOKENS is env-driven (config.py:
    # int(os.getenv("GEMMA_MAP_MAX_OUTPUT_TOKENS", "4096"))) — don't assert a
    # hardcoded literal here, it goes stale the moment someone sets the env
    # var (as .env now does, on purpose, to cut MAP generation latency).
    assert GEMMA_MAP_MAX_OUTPUT_TOKENS > 0
    # MAX_CONCURRENT_MAP_REQUESTS is env-driven (config.py:
    # int(os.getenv("MAX_CONCURRENT_MAP_REQUESTS", "8"))) — don't assert a
    # hardcoded literal, it goes stale the moment the default or .env changes
    # (LATENCY REDUCTION task raised it 4->8 to widen the dual-basket wave
    # candidate pool). Assert consistency across the aliases instead.
    assert MAX_CONCURRENT_MAP_REQUESTS > 0
    # Single concurrency story for all MAP backends / models.
    assert map_pipeline_concurrency() == MAX_CONCURRENT_MAP_REQUESTS
    assert gemma_map_concurrency_live() == MAX_CONCURRENT_MAP_REQUESTS
    assert GEMMA_CONCURRENCY == MAX_CONCURRENT_MAP_REQUESTS
    assert BLOG_SPATIAL_MAP_CONCURRENCY == MAX_CONCURRENT_MAP_REQUESTS


def test_resolve_max_output_tokens_follows_config(monkeypatch):
    """Regression: this used to hardcode `return 4096` unconditionally,
    silently ignoring GEMMA_MAP_MAX_OUTPUT_TOKENS from config/.env — every MAP
    call ran with max_tokens=4096 regardless of what was configured."""
    import knowledge_engine.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "GEMMA_MAP_MAX_OUTPUT_TOKENS", 1024)
    assert resolve_gemma_map_max_output_tokens(None) == 1024
    assert resolve_gemma_map_max_output_tokens(1_000) == 1024
    assert resolve_gemma_map_max_output_tokens(50_000) == 1024

    monkeypatch.setattr(cfg_mod, "GEMMA_MAP_MAX_OUTPUT_TOKENS", 2048)
    assert resolve_gemma_map_max_output_tokens(1_000) == 2048
    # bare input_tokens (no explicit projected_out) still doesn't branch —
    # only an explicit, content-aware projected_out from the caller does.
    assert resolve_gemma_map_max_output_tokens(1_000) == resolve_gemma_map_max_output_tokens(
        50_000
    )


def test_resolve_max_output_tokens_uses_projected_out_when_given(monkeypatch):
    """Step 1 of the MAP-latency audit: dynamic_target_facts() already
    computes a content-aware projected_out (proportional to how many
    knowledge_atoms a window realistically needs) but it was logged and then
    discarded — every window, however little it had to say, was allowed to
    generate up to the flat GEMMA_MAP_MAX_OUTPUT_TOKENS cap, and the model
    took proportionally longer the more of that unused headroom it used.
    When the caller supplies projected_out, the returned cap must shrink
    toward it (plus a margin for JSON/window_role overhead) instead of
    always returning the flat ceiling — while still never exceeding it."""
    import knowledge_engine.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "GEMMA_MAP_MAX_OUTPUT_TOKENS", 4096)

    small = resolve_gemma_map_max_output_tokens(1_000, projected_out=440)
    assert 440 < small < 4096

    large = resolve_gemma_map_max_output_tokens(1_000, projected_out=1_760)
    assert small < large < 4096

    # A huge projected_out must still be capped at the configured ceiling.
    assert resolve_gemma_map_max_output_tokens(1_000, projected_out=100_000) == 4096


def test_map_concurrency_semaphore_limit():
    """2N concurrent workers with Semaphore(N) → peak in-flight == N."""

    async def _run() -> int:
        sem = asyncio.Semaphore(MAX_CONCURRENT_MAP_REQUESTS)
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def fake_http_call(_i: int) -> None:
            nonlocal in_flight, max_in_flight
            async with sem:
                async with lock:
                    in_flight += 1
                    if in_flight > max_in_flight:
                        max_in_flight = in_flight
                await asyncio.sleep(0.04)
                async with lock:
                    in_flight -= 1

        n = MAX_CONCURRENT_MAP_REQUESTS * 2
        await asyncio.gather(*[fake_http_call(i) for i in range(n)])
        return max_in_flight

    peak = asyncio.run(_run())
    assert peak == MAX_CONCURRENT_MAP_REQUESTS


def test_map_pipeline_uses_unified_semaphore_constant():
    """Summarizer binds map_sem to MAX_CONCURRENT_MAP_REQUESTS (not per-model)."""
    src = inspect.getsource(spatial_mod)
    assert "asyncio.Semaphore(MAX_CONCURRENT_MAP_REQUESTS)" in src
    assert "map_pipeline_concurrency()" in src
    # No leftover provider fork for concurrency.
    assert "if use_gemma else BLOG_SPATIAL_MAP_CONCURRENCY" not in src
    assert map_pipeline_concurrency() == MAX_CONCURRENT_MAP_REQUESTS
    assert cfg.map_pipeline_concurrency() == cfg.gemma_map_concurrency_live()


def test_map_gemma_rl_uses_independent_limiters_by_default(monkeypatch):
    """Regression: ``map_reduce_jobs_pooled_async`` built ``gemma_rl`` with
    ``map_parallel_streams=GEMMA_MAP_FORCE_PER_MODEL_LIMITS and
    GEMMA_MAP_FIXED_MINUTE_PACING`` — with the documented default
    (fixed_minute=False, the modern sliding-window dispatcher), that AND
    collapsed to False regardless of GEMMA_MAP_FORCE_PER_MODEL_LIMITS,
    silently handing both MAP model slots the SAME shared rate limiter.
    ``acquire_parallel_wave``'s dual-basket split then always skipped the
    second slot (its ``seen_limiters`` dedup guard treats a shared limiter
    as already tried), capping every MAP wave at one in-flight request
    instead of two. Confirmed live via ``perf_debug.log``: the fallback
    model never once appeared in a wave-reserve trace across three separate
    node-init runs, each fully sequential (1875s / 1325s vs a ~480s
    baseline for the same node)."""
    import knowledge_engine.config as cfg_mod
    from knowledge_engine.services.llm.gemma_client import RateLimitedLLMClient

    # Pin the exact construction site: must not re-introduce the AND-gate.
    src = inspect.getsource(spatial_mod.map_reduce_jobs_pooled_async)
    call_args = src.split("map_parallel_streams=", 1)[1].split(")", 1)[0]
    assert "GEMMA_MAP_FIXED_MINUTE_PACING" not in call_args
    assert "GEMMA_MAP_FORCE_PER_MODEL_LIMITS" in call_args

    monkeypatch.setattr(cfg_mod, "GEMMA_MAP_FORCE_PER_MODEL_LIMITS", True)
    monkeypatch.setattr(cfg_mod, "GEMMA_MAP_FIXED_MINUTE_PACING", False)

    client = RateLimitedLLMClient(
        map_parallel_streams=cfg_mod.GEMMA_MAP_FORCE_PER_MODEL_LIMITS
    )
    assert len(client._slots) == 2
    assert id(client._slots[0].limiter) != id(client._slots[1].limiter)


def test_wave_admission_estimate_uses_projected_out_not_flat_cap():
    """DEEP AUDIT FIX: the wave-admission estimate built inside
    _run_gemma_map_waves (private closure of map_reduce_jobs_pooled_async)
    used to call estimate_request_tokens()/estimate_budget() with no
    max_output_tokens, silently falling back to the FLAT
    resolve_gemma_map_max_output_tokens(inp) — the pre-Step-1 worst-case cap
    — even though the ACTUAL dispatched call (_map_gemma_chunk_preacquired)
    was already sized off dynamic_target_facts()'s projected_out. Live logs
    showed the exact fingerprint: est_tpm=8248 == input(~4152) + flat
    cap(4096), while the real cloud dashboard showed only 23-29% TPM
    utilization for the same calls — the wave admission reserved against a
    number the real call could no longer produce, filling the local ledger
    long before real usage ever approached it ("wave full" / lockout
    paradox). The admission estimate must use the same projected_out-aware
    cap as the real call."""
    src = inspect.getsource(spatial_mod.map_reduce_jobs_pooled_async)
    # Isolate _run_gemma_map_waves's body (up to the next top-level nested def
    # at the same indentation) rather than the whole enclosing function.
    start = src.index("async def _run_gemma_map_waves")
    end = src.index("\n        map_tasks:", start)
    waves_src = src[start:end]
    assert "dynamic_target_facts(" in waves_src
    assert "resolve_gemma_map_max_output_tokens(" in waves_src
    assert "projected_out=projected_out" in waves_src
    # Must not silently fall back to the flat, no-cap estimate anymore (the
    # old CODE call site, not this docstring's own mention of the bug it
    # replaces).
    assert "gemma_rl.estimate_request_tokens(" not in waves_src


def test_prompt_structure_for_caching():
    """``_MAP_SYSTEM`` is a stable module-level prefix; chunk text only in user."""
    assert isinstance(_MAP_SYSTEM, str)
    assert spatial_mod._MAP_SYSTEM is _MAP_SYSTEM
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        _MAP_SYSTEM as again,
    )

    assert again is _MAP_SYSTEM
    assert "DO NOT output <thought>" in _MAP_SYSTEM

    job = MapReduceArticleJob(
        job_id="https://example.com/a",
        title="Stable Title",
        url="https://example.com/a",
        windows=[],
    )
    w0 = TokenWindowChunk(window_index=0, body="[P_1] First unique window body AAA.")
    w1 = TokenWindowChunk(window_index=1, body="[P_2] Second unique window body BBB.")
    u0 = _prompt_for_window(job, w0)
    u1 = _prompt_for_window(job, w1)
    assert "First unique window body AAA" in u0
    assert "Second unique window body BBB" in u1
    assert "First unique window body AAA" not in _MAP_SYSTEM
    assert "Second unique window body BBB" not in _MAP_SYSTEM
    assert spatial_mod._MAP_SYSTEM is _MAP_SYSTEM
    assert MapWindowResponse.__name__ == "MapWindowResponse"
