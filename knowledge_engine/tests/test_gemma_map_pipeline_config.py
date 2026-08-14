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
    assert GEMMA_MAP_MAX_OUTPUT_TOKENS == 4096
    assert MAX_CONCURRENT_MAP_REQUESTS == 4
    # Single concurrency story for all MAP backends / models.
    assert map_pipeline_concurrency() == 4
    assert gemma_map_concurrency_live() == 4
    assert GEMMA_CONCURRENCY == 4
    assert BLOG_SPATIAL_MAP_CONCURRENCY == MAX_CONCURRENT_MAP_REQUESTS


def test_resolve_max_output_tokens_returns_4096():
    assert resolve_gemma_map_max_output_tokens(None) == 4096
    assert resolve_gemma_map_max_output_tokens(1_000) == 4096
    assert resolve_gemma_map_max_output_tokens(8_000) == 4096
    assert resolve_gemma_map_max_output_tokens(50_000) == 4096
    # No branching: source is a constant return.
    src = inspect.getsource(resolve_gemma_map_max_output_tokens)
    assert "return 4096" in src
    assert "if " not in src.split(":", 1)[1]


def test_map_concurrency_semaphore_limit():
    """8 concurrent workers with Semaphore(4) → peak in-flight == 4."""

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

        await asyncio.gather(*[fake_http_call(i) for i in range(8)])
        return max_in_flight

    peak = asyncio.run(_run())
    assert peak == MAX_CONCURRENT_MAP_REQUESTS
    assert peak <= 4


def test_map_pipeline_uses_unified_semaphore_constant():
    """Summarizer binds map_sem to MAX_CONCURRENT_MAP_REQUESTS (not per-model)."""
    src = inspect.getsource(spatial_mod)
    assert "asyncio.Semaphore(MAX_CONCURRENT_MAP_REQUESTS)" in src
    assert "map_pipeline_concurrency()" in src
    # No leftover provider fork for concurrency.
    assert "if use_gemma else BLOG_SPATIAL_MAP_CONCURRENCY" not in src
    assert map_pipeline_concurrency() == MAX_CONCURRENT_MAP_REQUESTS
    assert cfg.map_pipeline_concurrency() == cfg.gemma_map_concurrency_live()


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
