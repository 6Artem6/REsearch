"""Stage 1/2: Exa → (optional) academic search → VERIFIED_EXTERNAL_SOURCES."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from knowledge_engine.config import (
    LECTURE_EXTERNAL_SEARCH_ENABLED,
    LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
    LECTURE_EXTERNAL_SEARCH_TOP_K,
    MAX_EXTERNAL_SOURCES,
)
from knowledge_engine.services.blocking_pools import pool_net_sync, run_blocking_timed
from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
)
from knowledge_engine.services.node_source_registry import is_disallowed_source_url
from knowledge_engine.services.search.exa_client import (
    ExaNotConfiguredError,
    ExaSearchClient,
)
from knowledge_engine.services.search.exa_transform import (
    postprocess_exa_hits_for_external_recall,
)
from knowledge_engine.services.search.providers import (
    ConsensusSearchProvider,
    SemanticScholarProvider,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.ui.run_log import trace
from knowledge_engine.utils.link_sanitizer import (
    extract_urls_from_text,
    normalize_lecture_url,
)

logger = logging.getLogger(__name__)

_SEARCH_TOOL_RE = re.compile(
    r'\{\s*"action"\s*:\s*"search_external_materials"\s*,\s*"query"\s*:\s*"(?P<q>(?:\\.|[^"\\])*)"\s*\}',
    re.I | re.S,
)
_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")


@dataclass(frozen=True)
class VerifiedExternalSource:
    url: str
    title: str
    snippet: str
    provider: str = ""
    score: float = 0.0


def external_source_limit(top_k: int | None = None) -> int:
    if top_k is not None:
        return max(1, int(top_k))
    return max(1, int(MAX_EXTERNAL_SOURCES or LECTURE_EXTERNAL_SEARCH_TOP_K or 3))


def build_external_search_query(
    node: NodeDataInput,
    subtopic: str,
    *,
    query_override: str = "",
) -> str:
    override = (query_override or "").strip()
    if override:
        return override[:500]
    title = (node.title or "").strip()
    sub = (subtopic or "").strip()
    parts = [title, sub, "technical documentation paper architecture"]
    return " ".join(p for p in parts if p)[:500]


def query_needs_en_translation(text: str) -> bool:
    """True when the focus/query contains Cyrillic (RU) for academic indexes."""
    raw = (text or "").strip()
    if not raw:
        return False
    # Any meaningful Cyrillic → translate; pure English stays as-is.
    return len(_CYRILLIC_RE.findall(raw)) >= 3


def translate_to_en_query(focus: str) -> str:
    """
    Translate lecture focus / node topic to English for Consensus / Semantic Scholar.

    Already-English queries are returned unchanged. Russian → Gemini Lite academic sanitize
    (same path as Consensus prep); on failure keep the original focus.
    """
    raw = (focus or "").strip()
    if not raw:
        return ""
    if not query_needs_en_translation(raw):
        return raw[:500]
    try:
        from knowledge_engine.src.processors.consensus_query_prep import (
            extract_preserved_terms_for_consensus,
        )
        from knowledge_engine.src.processors.validator import (
            sanitize_query_for_consensus,
        )

        terms = extract_preserved_terms_for_consensus(raw)
        out = sanitize_query_for_consensus(
            raw,
            f"lecture_external:{raw[:80]}",
            "",
            terms,
        )
        en = (getattr(out, "academic_query_en", None) or "").strip()
        if en:
            trace(
                f"LECTURE_SEARCH translate_en ✓ | in_len={len(raw)} out_len={len(en)}"
            )
            return en[:500]
    except Exception as exc:
        trace(f"LECTURE_SEARCH translate_en skip | {exc}")
    return raw[:500]


def _exa_sources_sync(query: str, limit: int) -> list[VerifiedExternalSource]:
    client = ExaSearchClient()
    if not client.is_configured():
        return []
    try:
        resp = client.search(query, num_results=max(3, limit))
    except (ExaNotConfiguredError, ValueError) as exc:
        trace(f"LECTURE_EXA skip | {exc}")
        return []
    except Exception as exc:
        trace(f"LECTURE_EXA error | {exc}")
        return []
    out: list[VerifiedExternalSource] = []
    processed = postprocess_exa_hits_for_external_recall(
        list(resp.hits),
        cap=max(3, limit),
    )
    for i, hit in enumerate(processed):
        exa_score = getattr(hit, "exa_relevance_score", None)
        if exa_score is None:
            score = max(0.0, 1.0 - 0.05 * i)
        else:
            try:
                score = float(exa_score)
            except (TypeError, ValueError):
                score = max(0.0, 1.0 - 0.05 * i)
        out.append(
            VerifiedExternalSource(
                url=hit.url,
                title=(hit.title or hit.url)[:400],
                snippet=(hit.snippet or "").strip()[:1200],
                provider="exa",
                score=score,
            )
        )
    return out


async def _provider_sources(
    provider_name: str,
    query: str,
    limit: int,
) -> list[VerifiedExternalSource]:
    if provider_name == "semantic_scholar":
        provider = SemanticScholarProvider()
    elif provider_name == "consensus":
        provider = ConsensusSearchProvider()
    else:
        return []
    try:
        rows = await provider.search(query, limit=limit)
    except Exception as exc:
        trace(f"LECTURE_SEARCH {provider_name} skip | {exc}")
        return []
    out: list[VerifiedExternalSource] = []
    for i, row in enumerate(rows):
        url = str(row.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        raw_score = row.get("score")
        try:
            score = (
                float(raw_score) if raw_score is not None else max(0.0, 0.55 - 0.04 * i)
            )
        except (TypeError, ValueError):
            score = max(0.0, 0.55 - 0.04 * i)
        out.append(
            VerifiedExternalSource(
                url=url,
                title=str(row.get("title") or url)[:400],
                snippet=str(row.get("snippet") or "")[:1200],
                provider=str(row.get("source") or provider_name),
                score=score,
            )
        )
    return out


def _merge_sources(
    batches: list[list[VerifiedExternalSource]],
    top_k: int,
) -> list[VerifiedExternalSource]:
    """Dedupe by URL; keep highest score; rank by relevance score (desc)."""
    best: dict[str, VerifiedExternalSource] = {}
    for batch in batches:
        for src in batch:
            key = normalize_lecture_url(src.url)
            if not key:
                continue
            prev = best.get(key)
            if prev is None or float(src.score) > float(prev.score):
                best[key] = src
    ranked = sorted(
        best.values(),
        key=lambda s: float(s.score),
        reverse=True,
    )
    return ranked[: max(1, top_k)]


async def fetch_verified_external_sources(
    node: NodeDataInput,
    subtopic: str,
    curriculum_id: str = "",
    *,
    query_override: str = "",
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    """Waterfall: Exa first (early exit) → EN academic Consensus/SS fallback."""
    try:
        return await _fetch_verified_external_sources_impl(
            node,
            subtopic,
            curriculum_id,
            query_override=query_override,
            top_k=top_k,
        )
    except Exception as exc:
        from knowledge_engine.ui.errors import trace_exception

        trace_exception(exc, "LECTURE_SEARCH")
        return []


async def _exa_batch(query: str, per_provider: int) -> list[VerifiedExternalSource]:
    try:
        return await run_blocking_timed(
            pool_net_sync(),
            LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
            _exa_sources_sync,
            query,
            per_provider,
        )
    except asyncio.TimeoutError:
        trace(f"LECTURE_EXA skip | timeout {LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC}s")
        return []
    except Exception as exc:
        trace(f"LECTURE_EXA skip | {exc}")
        return []


async def _fetch_verified_external_sources_impl(
    node: NodeDataInput,
    subtopic: str,
    curriculum_id: str = "",
    *,
    query_override: str = "",
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    trace("LECTURE_SEARCH ▶ fetch_verified_external_sources (waterfall)")
    if not LECTURE_EXTERNAL_SEARCH_ENABLED:
        return []
    node = enrich_node_learning_materials_from_graph(node, curriculum_id)
    query = build_external_search_query(node, subtopic, query_override=query_override)
    if not query:
        return []
    cap = external_source_limit(top_k)
    per_provider = max(2, cap)

    # --- Step 1: Exa (early exit) ---
    exa_batch = await _exa_batch(query, per_provider)
    if len(exa_batch) >= cap:
        merged = _merge_sources([exa_batch], cap)
        logger.info(
            "Exa satisfied external source limit (%d/%d). Skipping Consensus/SS.",
            len(exa_batch),
            cap,
        )
        trace(
            f"LECTURE_SEARCH ✓ early_exit exa | query_len={len(query)} "
            f"sources={len(merged)} exa={len(exa_batch)}"
        )
        return merged

    # --- Step 2: Academic fallback (English query) ---
    focus_for_academic = (query_override or subtopic or node.title or query).strip()
    academic_query = await asyncio.to_thread(translate_to_en_query, focus_for_academic)
    if not academic_query:
        academic_query = query
    need = max(1, cap - len(exa_batch))
    trace(
        f"LECTURE_SEARCH ▶ academic fallback | need={need} "
        f"exa={len(exa_batch)} q_en_len={len(academic_query)}"
    )

    async def _provider_batch(name: str, limit: int) -> list[VerifiedExternalSource]:
        try:
            return await _provider_sources(name, academic_query, limit)
        except Exception as exc:
            trace(f"LECTURE_SEARCH {name} skip | {exc}")
            return []

    ss_batch, consensus_batch = await asyncio.gather(
        _provider_batch("semantic_scholar", need),
        _provider_batch("consensus", need),
    )
    merged = _merge_sources([exa_batch, ss_batch, consensus_batch], cap)
    trace(
        f"LECTURE_SEARCH ✓ waterfall | query_len={len(query)} sources={len(merged)} "
        f"exa={len(exa_batch)} ss={len(ss_batch)} cons={len(consensus_batch)}"
    )
    return merged


def format_verified_external_sources_block(
    sources: list[VerifiedExternalSource],
) -> str:
    if not sources:
        return ""
    lines = [
        "=== ИСТИННЫЕ ПРОВЕРЕННЫЕ ИСТОЧНИКИ (Используй ТОЛЬКО эти ссылки) ===",
        "VERIFIED_EXTERNAL_SOURCES:",
    ]
    for i, src in enumerate(sources, 1):
        snippet = (src.snippet or "").strip().replace("\n", " ")
        if snippet:
            snippet = f'"{snippet[:900]}"'
        else:
            snippet = (
                "(нет сниппета — не выдумывай содержание; упомяни title без ссылки)"
            )
        lines.append(f"- Source [{i}]:")
        lines.append(f"  URL: {src.url}")
        lines.append(f"  Title: {src.title}")
        lines.append(f"  Snippet: {snippet}")
        lines.append("")
    lines.append(
        "В тексте лекции — теги [S1]… по реестру; URL только в JSON used_sources "
        "(копировать из списка выше). Не вставляй http в lecture_body."
    )
    return "\n".join(lines).strip()


def merge_verified_sources(
    existing: list[VerifiedExternalSource],
    new: list[VerifiedExternalSource],
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    cap = external_source_limit(top_k)
    return _merge_sources([existing, new], cap)


def collect_lecture_allowed_urls(
    verified: list[VerifiedExternalSource],
    rag_context: str,
    node: NodeDataInput,
    curriculum_id: str = "",
    *,
    skip_graph_enrich: bool = False,
) -> set[str]:
    allowed: set[str] = set()
    for src in verified:
        key = normalize_lecture_url(src.url)
        if key and not is_disallowed_source_url(src.url):
            allowed.add(key)
    allowed |= {
        u
        for u in extract_urls_from_text(rag_context)
        if u and not is_disallowed_source_url(u)
    }
    if not skip_graph_enrich:
        node = enrich_node_learning_materials_from_graph(node, curriculum_id)
    for u in getattr(node, "resource_urls", None) or []:
        raw = str(u)
        key = normalize_lecture_url(raw)
        if key and not is_disallowed_source_url(raw):
            allowed.add(key)
    for lr in getattr(node, "learning_resources", None) or []:
        if isinstance(lr, dict):
            raw = str(lr.get("url") or "")
            key = normalize_lecture_url(raw)
            if key and not is_disallowed_source_url(raw):
                allowed.add(key)
    ref = node.source_ref
    if ref is not None:
        raw = ref.url or ""
        key = normalize_lecture_url(raw)
        if key and not is_disallowed_source_url(raw):
            allowed.add(key)
    return {u for u in allowed if u and not is_disallowed_source_url(u)}


def parse_search_external_materials_request(text: str) -> str | None:
    t = (text or "").strip()
    if not t or len(t) > 900:
        return None
    if "search_external_materials" not in t:
        return None
    m = _SEARCH_TOOL_RE.search(t)
    if m:
        raw = m.group("q").replace('\\"', '"').strip()
        return raw or None
    if t.startswith("{") and t.endswith("}"):
        try:
            data = json.loads(t)
        except json.JSONDecodeError:
            return None
        if str(data.get("action") or "").strip().lower() == "search_external_materials":
            q = str(data.get("query") or "").strip()
            return q or None
    return None


def is_search_tool_only_response(text: str) -> bool:
    t = (text or "").strip()
    if not parse_search_external_materials_request(t):
        return False
    stripped = _SEARCH_TOOL_RE.sub("", t).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            pass
    return len(stripped) < 40
