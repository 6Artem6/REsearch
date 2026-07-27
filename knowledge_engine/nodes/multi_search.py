"""Загрузка первоисточников, vision, summarizer → LanceDB."""

from __future__ import annotations

import re
from typing import Any

import httpx

from knowledge_engine.config import (
    GEMINI_PRIMARY,
    MAX_FETCH_URLS,
    MAX_LANCE_INDEX_URLS,
    MULTI_SEARCH_SKIP_VISION,
    SKIP_GEMINI,
)
from knowledge_engine.schemas import EngineGraphState, EngineState
from knowledge_engine.services.search.browser_search import fetch_page_html
from knowledge_engine.services.search.horizons import (
    SearchHorizon,
    build_horizon_queries,
)
from knowledge_engine.services.search.registry import default_registry
from knowledge_engine.services.search.url_filter import rank_and_cap_urls
from knowledge_engine.services.summarizer import (
    summarize_article,
    summarize_gemini_bundle,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.services.vision import analyze_page_diagrams
from knowledge_engine.ui.errors import trace_exception
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.run_log import node_end, node_start

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", html)).strip()


def _fetch_text(url: str) -> str:
    if "arxiv.org" in url or url.endswith(".pdf"):
        try:
            with httpx.Client(timeout=45.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text[:15000]
        except httpx.HTTPError:
            return ""
    try:
        html = fetch_page_html(url, headless=True)
        return _strip_html(html)[:15000]
    except Exception:
        return ""


def multi_search_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("multi_search_node (horizons + summarizer + LanceDB)")
    parsed = EngineState.model_validate(state)
    set_status("[Multi-search] SearchRegistry: SearXNG + академические API…")

    set_status("[Multi-search] три горизонта: SOTA / Infra / Prod…")

    registry = default_registry()
    horizon_queries_map = build_horizon_queries(
        parsed.user_problem,
        parsed.context_constraints,
        parsed.abstractions,
    )
    api_hits, horizon_queries = registry.multi_search_horizons_sync(
        parsed.user_problem,
        parsed.context_constraints,
        parsed.abstractions,
        limit_per_provider=2,
        horizon_queries=horizon_queries_map,
    )
    per_horizon = {
        h.value: sum(1 for x in api_hits if x.horizon == h.value) for h in SearchHorizon
    }
    set_status(
        f"[Multi-search] URL: sota={per_horizon['sota']} "
        f"infra={per_horizon['infra']} prod={per_horizon['prod']}"
    )

    gemini_primary = (
        not SKIP_GEMINI
        and GEMINI_PRIMARY
        and bool(parsed.external_ai_dialogue_history)
        and parsed.is_facts_sufficient
    )
    skip_playwright_url_fetch = not SKIP_GEMINI
    if skip_playwright_url_fetch:
        set_status(
            "[Multi-search] Frugal: без Playwright URL (один браузер — только Gemini heavy)"
        )
    fetch_cap = MAX_FETCH_URLS
    if gemini_primary:
        fetch_cap = MAX_LANCE_INDEX_URLS
        set_status(
            "[Multi-search] Gemini-primary: горизонты API + "
            f"≤{fetch_cap} URL в LanceDB (без 7B×N парсинга)"
        )

    urls = rank_and_cap_urls(
        list(parsed.collected_urls) + [h.url for h in api_hits], fetch_cap
    )

    summaries = list(parsed.found_summaries)
    seen = {s.url for s in summaries}
    store = VectorStore()
    new_facts = list(parsed.found_facts)

    for hit in api_hits:
        snippet = (hit.snippet or hit.title or "").strip()
        if snippet and snippet not in new_facts:
            new_facts.append(snippet[:500])

    if gemini_primary and not any(s.url == "gemini-research-bundle" for s in summaries):
        snippets = [h.snippet or h.title for h in api_hits if h.snippet or h.title]
        try:
            bundle = summarize_gemini_bundle(
                parsed.user_problem,
                parsed.external_ai_dialogue_history,
                snippets,
            )
            store.save_summary(bundle)
            summaries.append(bundle)
            seen.add(bundle.url)
            for t in bundle.key_takeaways:
                if t not in new_facts:
                    new_facts.append(t)
        except Exception as exc:
            detail = trace_exception(exc, "Summarizer/Gemini bundle")
            set_status(f"[Summarizer] Gemini bundle: {detail}")

    if not gemini_primary and not skip_playwright_url_fetch:
        for url in urls:
            if url in seen:
                continue
            set_status(f"[Multi-search] парсинг {url[:70]}…")
            raw = _fetch_text(url)
            if not raw:
                continue
            title = url
            for hit in api_hits:
                if hit.url == url:
                    title = hit.title
                    break

            diagram_desc: list[str] = []
            if not MULTI_SEARCH_SKIP_VISION and not url.endswith(".pdf"):
                try:
                    diagram_desc = analyze_page_diagrams(url)
                    set_status(f"[Vision] {len(diagram_desc)} описаний схем")
                except Exception as exc:
                    detail = trace_exception(exc, "Vision")
                    set_status(f"[Vision] пропуск: {detail}")

            try:
                summary = summarize_article(
                    title, url, raw, diagram_descriptions=diagram_desc
                )
                store.save_summary(summary)
                summaries.append(summary)
                seen.add(url)
                for t in summary.key_takeaways:
                    if t not in new_facts:
                        new_facts.append(t)
            except Exception as exc:
                detail = trace_exception(exc, f"Summarizer/{url[:40]}")
                set_status(f"[Summarizer] ошибка для {url}: {detail}")

    merged_search_queries = list(parsed.search_queries)
    for q in horizon_queries.values():
        if q not in merged_search_queries:
            merged_search_queries.append(q)

    node_end(
        "multi_search_node (horizons + summarizer + LanceDB)",
        f"summaries={len(summaries)}, urls={len(urls)}",
    )
    return {
        "found_summaries": [s.model_dump() for s in summaries],
        "found_facts": new_facts,
        "is_facts_sufficient": parsed.is_facts_sufficient or len(summaries) >= 1,
        "collected_urls": urls,
        "search_queries": merged_search_queries,
        "search_horizon_queries": horizon_queries,
    }
