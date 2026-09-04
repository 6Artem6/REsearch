"""Сжатие длинных RAG-фактов через Gemma (Directional RAG Gateway)."""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel, Field

from knowledge_engine.config import (
    GEMMA_MAP_MAX_OUTPUT_TOKENS,
    RAG_FACT_COMPRESS_GEMMA_TIMEOUT_SEC,
    gemma_cloud_api_key_available,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.llm.gemma_client import RateLimitedLLMClient
from knowledge_engine.src.rag_gateway.fact_text import (
    FACT_MAX_CHARS,
    GEMMA_SUMMARY_MAX_CHARS,
    truncate_fact_at_word_boundary,
)
from knowledge_engine.ui.run_log import trace

_GEMMA_INPUT_MAX_CHARS = 8_000
_COMPRESS_MAX_OUTPUT_TOKENS = 640

_SUMMARY_SYSTEM = (
    "You compress engineering fragments from a tutoring system's memory. "
    "Preserve facts, terminology, constraints, and code snippets. No preamble.\n"
    f"{RUSSIAN_OUTPUT_RULE}"
)
"""
RU (пояснение): сжатие RAG-фактов Gemma при превышении FACT_MAX_CHARS —
сохранять термины/ограничения/код, без вводных фраз.
"""


class _FactSummaryContract(BaseModel):
    summary: str = Field(min_length=8, max_length=FACT_MAX_CHARS)


def _resolve_gemma_timeout(gemma_timeout_sec: float | None) -> float:
    cap = RAG_FACT_COMPRESS_GEMMA_TIMEOUT_SEC
    if gemma_timeout_sec is None:
        return cap
    return max(0.0, min(float(gemma_timeout_sec), cap))


async def _summarize_with_gemma(
    fact_text: str,
    context_topic: str,
    *,
    timeout_sec: float,
) -> str | None:
    if not gemma_cloud_api_key_available():
        trace("RAG_GATEWAY fact compress ⊘ | Gemma API key not configured")
        return None
    if timeout_sec < 1.0:
        return None

    topic = (context_topic or "учебная нода").strip()[:200] or "учебная нода"
    body = fact_text.strip()[:_GEMMA_INPUT_MAX_CHARS]
    prompt = (
        f"Исходный контекст:\n{body}\n\n"
        f"Сократи инженерный контекст по теме '{topic}', строго сохраняя все "
        "технические детали, ключевые термины, архитектурные ограничения и код. "
        f"Выходной текст НЕ должен превышать {GEMMA_SUMMARY_MAX_CHARS} символов."
    )
    http_timeout = httpx.Timeout(timeout_sec)
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        rl = RateLimitedLLMClient()
        out = await asyncio.wait_for(
            rl.post_structured(
                _SUMMARY_SYSTEM,
                prompt,
                _FactSummaryContract,
                label="rag_gateway/fact_compress",
                client=client,
                max_tokens=min(
                    _COMPRESS_MAX_OUTPUT_TOKENS, GEMMA_MAP_MAX_OUTPUT_TOKENS
                ),
            ),
            timeout=timeout_sec,
        )
    if out is None or not (out.summary or "").strip():
        return None
    return out.summary.strip()


async def compress_fact_if_needed(
    fact_text: str,
    context_topic: str,
    *,
    gemma_timeout_sec: float | None = None,
) -> str:
    """
    Возвращает fact_text без изменений, если он укладывается в лимит схемы.
    Иначе — суммаризация Gemma (короткий таймаут); при сбое — обрезка по границе слова.
    """
    text = (fact_text or "").strip()
    if len(text) <= FACT_MAX_CHARS:
        return text

    trace(
        f"RAG_GATEWAY ⚠ fact over budget | len={len(text)} "
        f"topic={(context_topic or '')[:60]}"
    )
    timeout = _resolve_gemma_timeout(gemma_timeout_sec)
    if timeout < 1.0:
        trace("RAG_GATEWAY fact compress ⊘ | no Gemma time budget — word fallback")
        return truncate_fact_at_word_boundary(text, FACT_MAX_CHARS)

    try:
        compressed = await _summarize_with_gemma(
            text,
            context_topic,
            timeout_sec=timeout,
        )
        if compressed and len(compressed) >= 8:
            if len(compressed) > FACT_MAX_CHARS:
                compressed = truncate_fact_at_word_boundary(compressed, FACT_MAX_CHARS)
            trace(
                f"RAG_GATEWAY fact compress ✓ | {len(text)}→{len(compressed)} "
                f"gemma_timeout={timeout:.0f}s topic={(context_topic or '')[:40]}"
            )
            return compressed
    except asyncio.TimeoutError:
        trace(
            f"RAG_GATEWAY fact compress ✗ | Gemma timeout ({timeout:.0f}s) — word fallback"
        )
    except Exception as exc:
        trace(f"RAG_GATEWAY fact compress ✗ | {type(exc).__name__}: {exc}")

    fallback = truncate_fact_at_word_boundary(text, FACT_MAX_CHARS)
    trace(f"RAG_GATEWAY fact compress ⊘ fallback | {len(text)}→{len(fallback)}")
    return fallback
