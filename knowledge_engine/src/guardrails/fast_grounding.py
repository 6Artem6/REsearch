"""Pre-guardrails SearXNG micro-search for term grounding (no query rewriting)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional

from knowledge_engine.config import SEARXNG_DISCOVERY_CATEGORIES, SEARXNG_ENABLED
from knowledge_engine.services.searxng_client import searxng_search_json
from knowledge_engine.ui.run_log import trace

SearchClient = Callable[..., Awaitable[List[dict[str, Any]]]]

_MAX_SNIPPET_CHARS = 400
_MAX_GROUNDING_WORDS = 300


async def _default_search_client(
    query: str,
    limit: int = 3,
    **kwargs: Any,
) -> List[dict[str, Any]]:
    categories = kwargs.get("categories")
    if categories is None:
        categories = list(SEARXNG_DISCOVERY_CATEGORIES) or None
    raw = await searxng_search_json(
        query,
        limit=limit,
        categories=categories,
    )
    if raw and raw[0].get("error"):
        return []
    return raw


def _format_grounding_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    parts: list[str] = []
    word_count = 0
    for idx, item in enumerate(results[:3], start=1):
        if item.get("error"):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        url = str(item.get("url") or "").strip()
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS] + "…"
        block = f"[{idx}] {title}\nURL: {url}\n{snippet}".strip()
        block_words = len(block.split())
        if word_count + block_words > _MAX_GROUNDING_WORDS:
            remaining = _MAX_GROUNDING_WORDS - word_count
            if remaining <= 0:
                break
            block = " ".join(block.split()[:remaining])
        parts.append(block)
        word_count += len(block.split())
    return "\n\n".join(parts)


async def get_term_grounding_context(
    raw_query: str,
    search_client: Optional[SearchClient] = None,
) -> str:
    """
    Raw SearXNG lookup on untouched user query; top 2–3 snippets as grounding text.
    Returns empty string on failure (pipeline continues).
    """
    query = (raw_query or "").strip()
    if not query or not SEARXNG_ENABLED:
        trace("GROUNDING ⊘ SearXNG disabled or empty query")
        return ""

    client = search_client or _default_search_client
    trace(f"GROUNDING ▶ SearXNG raw pre-check | {query[:140]}")

    try:
        results = await client(query, limit=3)
    except Exception as exc:
        trace(f"GROUNDING ✗ SearXNG error | {exc}")
        return ""

    clean = [r for r in results if r.get("url") and not r.get("error")]
    if not clean:
        trace("GROUNDING ⊘ no snippets")
        return ""

    block = _format_grounding_block(clean)
    if block:
        trace(f"GROUNDING ✓ snippets={len(clean)} | ~{len(block.split())} words")
        if "mcp" in query.lower():
            mcp_in_snippets = (
                "mcp" in block.lower() or "model context protocol" in block.lower()
            )
            trace(f"GROUNDING MCP probe | in_snippets={mcp_in_snippets}")
    return block
