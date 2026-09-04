"""Lightweight async fetch + Trafilatura extraction + BGE-M3/MMR passage
selection for the lecture external-search waterfall.

Deliberately a thinner sibling of ``src/curriculum/pre_flight_triage.py``:
same core primitives (Trafilatura paragraph extraction, BGE-M3 embeddings,
greedy MMR), but WITHOUT Code Preservation Policy/AST collapsing, TOC-based
triage, or Zero-Waste Handover — that machinery exists to feed a full
MAP-REDUCE ingest document and is not needed for a short lecture snippet.

No Gemma/LLM pass over any fetched document here — selection is pure vector
math (BGE-M3 cosine + greedy MMR). The one LLM call in the lecture waterfall
stays outside this module, in the Flash Lite Content Quality Gate applied
to the passages this module already picked (``_lite_rerank_exa_hits`` /
``_BATCH_SYSTEM``, ``lite_search_pipeline.py``).
"""

from __future__ import annotations

import asyncio

import httpx

from knowledge_engine.config import (
    LECTURE_DEDUP_COSINE_THRESHOLD,
    LECTURE_PASSAGE_FETCH_CONCURRENCY,
    LECTURE_PASSAGE_FETCH_TIMEOUT_SEC,
    LECTURE_PASSAGE_MIN_CHARS,
    LECTURE_PASSAGE_MMR_LAMBDA,
    LECTURE_PASSAGE_MMR_TOP_K,
)
from knowledge_engine.services.web_extract import github_blob_to_raw_fetch_url
from knowledge_engine.src.curriculum.pre_flight_triage import (
    _extract_paragraphs,
    stage3_mmr_paragraphs_batch,
)
from knowledge_engine.ui.run_log import trace

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


async def fetch_html(url: str, *, timeout_sec: float) -> str:
    """Один GET, короткий таймаут — намеренно без Playwright-фолбэка
    (smart_fetch_page_html): тот рассчитан на полноценный ingest и может
    занимать секунды/десятки секунд, здесь бюджет — единицы секунд на URL,
    зависшие/медленные ответы просто отбрасываются (см. докстринг модуля)."""
    fetch_url = github_blob_to_raw_fetch_url(url)
    try:
        async with httpx.AsyncClient(
            timeout=timeout_sec,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "ru,en;q=0.9"},
        ) as client:
            resp = await client.get(fetch_url)
            if resp.status_code >= 400:
                return ""
            return resp.text
    except httpx.HTTPError as exc:
        trace(f"LECTURE_PASSAGE fetch ✗ | {url[:60]} | {type(exc).__name__}")
        return ""
    except Exception as exc:  # noqa: BLE001 - внешний сетевой вызов
        trace(f"LECTURE_PASSAGE fetch error | {url[:60]} | {exc}")
        return ""


async def fetch_and_extract_passages(
    urls: list[str],
    *,
    core_theme: str,
    top_k: int | None = None,
    timeout_sec: float | None = None,
    concurrency: int | None = None,
    min_chars: int | None = None,
) -> dict[str, list[str]]:
    """URL → лучшие (разнообразные, релевантные core_theme) абзацы этого
    URL. URL, для которых фетч не удался или текст оказался слишком тонким
    (< min_chars на абзац, весь документ отфильтрован) — просто отсутствуют
    в результате; вызывающий код должен фолбэкнуться на снипет/highlight
    Exa для таких источников, а не терять их целиком."""
    uniq = [u for u in dict.fromkeys(urls) if (u or "").strip().startswith("http")]
    if not uniq:
        return {}
    timeout = (
        timeout_sec if timeout_sec is not None else LECTURE_PASSAGE_FETCH_TIMEOUT_SEC
    )
    conc = max(
        1,
        concurrency if concurrency is not None else LECTURE_PASSAGE_FETCH_CONCURRENCY,
    )
    chars_min = min_chars if min_chars is not None else LECTURE_PASSAGE_MIN_CHARS
    sem = asyncio.Semaphore(conc)

    async def _one(url: str) -> tuple[str, list[str]]:
        async with sem:
            html = await fetch_html(url, timeout_sec=timeout)
        if not html:
            return url, []
        paragraphs = await asyncio.to_thread(
            _extract_paragraphs, html, url, min_chars=chars_min
        )
        return url, paragraphs

    results = await asyncio.gather(*[_one(u) for u in uniq])
    with_paragraphs = [(u, p) for u, p in results if p]
    trace(
        f"LECTURE_PASSAGE fetch ✓ | urls={len(uniq)} with_text={len(with_paragraphs)}"
    )
    if not with_paragraphs:
        return {}

    k = top_k if top_k is not None else LECTURE_PASSAGE_MMR_TOP_K
    selected = await asyncio.to_thread(
        stage3_mmr_paragraphs_batch,
        core_theme,
        [p for _, p in with_paragraphs],
        top_k=k,
        lambda_param=LECTURE_PASSAGE_MMR_LAMBDA,
    )
    out = {
        url: passages
        for (url, _), passages in zip(with_paragraphs, selected)
        if passages
    }
    trace(f"LECTURE_PASSAGE mmr ✓ | with_passages={len(out)}/{len(with_paragraphs)}")
    return out


async def find_near_duplicate_urls(
    passages_by_url: dict[str, list[str]],
    *,
    threshold: float | None = None,
    anchor: str = "lecture_dedup",
) -> dict[str, str]:
    """URL → URL канонического источника, если этот URL — почти дубликат
    (BGE-M3 Union-Find кластеризация + Flash Lite Bulk Gate — та же
    кластеризация и тот же промпт, что src/deduplication/
    pre_map_deduplicator.py гоняет ПЕРЕД MAP+REDUCE в DEEP; здесь без
    Triage-шага (PaperStructureAnalyzer) — passages_by_url уже прошли MMR,
    дополнительная классификация CORE/CONTEXT/DROP не нужна).

    В отличие от DEEP (где дубликат помечается ALIAS и просто теряет слот
    в финальном наборе — там нет MAP+REDUCE-бюджета, который надо беречь,
    но и нет резерва на этом этапе, см. разбор), здесь URL из этого
    результата ожидается, что вызывающий код ЗАМЕНИТ на другой источник из
    резерва — то и другое реализовано в _exa_sources_multi_vector."""
    urls_with_passages = {u: p for u, p in passages_by_url.items() if p}
    if len(urls_with_passages) < 2:
        return {}

    from knowledge_engine.src.deduplication.pre_map_deduplicator import (
        _cluster_text_candidates,
        _pool_vector,
        _run_bulk_gate,
        _sanitize_canonical_map,
    )

    doc_vectors: dict[str, list[float]] = {}
    for url, passages in urls_with_passages.items():
        vec = await asyncio.to_thread(_pool_vector, passages)
        if vec is not None:
            doc_vectors[url] = vec
    if len(doc_vectors) < 2:
        return {}

    lam = threshold if threshold is not None else LECTURE_DEDUP_COSINE_THRESHOLD
    groups = _cluster_text_candidates(doc_vectors, threshold=lam)
    suspect_groups = [g for g in groups if len(g) > 1]
    trace(
        f"LECTURE_PASSAGE dedup cluster ✓ | candidates={len(doc_vectors)} "
        f"suspect_groups={len(suspect_groups)} threshold={lam:.2f}"
    )
    if not suspect_groups:
        return {}

    raw_map = await _run_bulk_gate(
        suspect_groups, [], urls_with_passages, anchor=anchor
    )
    clean_map = _sanitize_canonical_map(raw_map, set(urls_with_passages.keys()))
    alias_of_url = {
        alias_url: canonical_url
        for canonical_url, aliases in clean_map.items()
        for alias_url in aliases
    }
    trace(
        f"LECTURE_PASSAGE dedup ✓ | groups={len(suspect_groups)} "
        f"canonical_groups={len(clean_map)} aliases={len(alias_of_url)}"
    )
    return alias_of_url
