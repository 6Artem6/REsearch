"""Gemma 4 repair для Mermaid, не прошедшего детерминированную валидацию."""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel, Field

from knowledge_engine.config import (
    GEMMA_MAP_MAX_OUTPUT_TOKENS,
    gemma_cloud_api_key_available,
)
from knowledge_engine.services.llm.gemma_client import RateLimitedLLMClient
from knowledge_engine.ui.run_log import trace

# Gemma Mermaid sanitizer — isolated system prompt (English only).
MERMAID_GEMMA_SANITIZER_PROMPT = (
    "You are a strict Mermaid Syntax Fixer.\n"
    "Your ONLY job is to take broken Mermaid code and output VALID, compile-ready "
    "Mermaid code without markdown wrappers or conversational filler.\n\n"
    "FIX RULES:\n"
    "1. DIAGRAM TYPE SEPARATION:\n"
    "   - If the code mixes `flowchart` and `xychart-beta`, separate them. "
    "`xychart-beta` MUST start on its own line and cannot be inside node definitions.\n"
    "   - Prefer a single coherent diagram: for benchmarks keep pure `xychart-beta`; "
    "for architecture keep pure `flowchart` / `sequenceDiagram`.\n"
    "2. SYNTAX CLEANUP:\n"
    '   - Fix unmatched quotes: ensure every label quote `"` is properly closed.\n'
    '   - Remove duplicate brackets like `"]"]"` or `"]"]`.\n'
    '   - Clean broken node IDs containing spaces (e.g., convert `Node[ "Text"]` '
    'to `Node["Text"]`).\n'
    "   - Fix broken array syntax in charts: `bar [10, 20, 30]`.\n"
    "   - Node IDs: alphanumeric + underscore only; labels with special chars in "
    'double quotes: `ID["Label with <br/>"]`.\n'
    "   - CRITICAL: `end` that closes a subgraph/alt/loop MUST be alone on its line. "
    "Never write `end A --> B`; put `end` then connections on following lines.\n"
    "   - Close every `subgraph` with a matching standalone `end`.\n"
    "   - One statement per line: never splice a node declaration onto edges "
    '(`N["lab"] A --> B`) or two separate edges (`A --> B B --> C`). '
    "Chains `A --> B --> C` are OK.\n"
    "3. PRESERVE SEMANTICS:\n"
    "- Do not invent nodes/edges. Do NOT include %%{init:...}%% (server adds styling).\n"
    "- sequenceDiagram: each message/Note on ONE line; "
    'use A->>B: "label" with quotes.\n'
    "- When the user message lists `Line N: ...` lint errors, fix those lines first.\n\n"
    "OUTPUT FORMAT: Return ONLY the corrected Mermaid code "
    "(JSON field `mermaid`, diagram body only, no ``` fences).\n"
)

# Back-compat alias used by repair call sites.
_REPAIR_SYSTEM = MERMAID_GEMMA_SANITIZER_PROMPT


class MermaidRepairResponse(BaseModel):
    mermaid: str = Field(
        description="Corrected Mermaid diagram body (no ``` fences, no init directive)",
    )


async def _repair_invalid_mermaid_async(
    broken_code: str,
    error_msg: str,
    *,
    client: httpx.AsyncClient | None = None,
    rl: RateLimitedLLMClient | None = None,
) -> str:
    if not gemma_cloud_api_key_available():
        trace("MERMAID_GEMMA_REPAIR ⊘ | GEMINI_API_KEY not set")
        return ""
    broken = (broken_code or "").strip()[:11000]
    if not broken:
        return ""
    prompt = (
        f"Validation error: {error_msg}\n\n"
        "Broken Mermaid:\n"
        f"{broken}\n\n"
        "Return JSON with field mermaid containing the fixed diagram."
    )
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    llm = rl or RateLimitedLLMClient()
    try:
        out = await llm.post_structured(
            _REPAIR_SYSTEM,
            prompt,
            MermaidRepairResponse,
            label="mermaid_repair/gemma",
            client=http,
            max_tokens=min(4096, GEMMA_MAP_MAX_OUTPUT_TOKENS),
        )
    finally:
        if owns_client:
            await http.aclose()
    if out is None or not (out.mermaid or "").strip():
        trace("MERMAID_GEMMA_REPAIR ✗ | empty response")
        return ""
    trace(f"MERMAID_GEMMA_REPAIR ✓ | out_len={len(out.mermaid)}")
    return out.mermaid.strip()


def repair_invalid_mermaid(broken_code: str, error_msg: str = "") -> str:
    """Синхронная обёртка для ingest pipeline."""
    msg = (error_msg or "validate_mermaid_syntax failed").strip()
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                _repair_invalid_mermaid_async(broken_code, msg),
            ).result()
    except RuntimeError:
        return asyncio.run(_repair_invalid_mermaid_async(broken_code, msg))
