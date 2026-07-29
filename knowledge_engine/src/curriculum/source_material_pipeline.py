"""Обогащение поисковых hits выдержками (LanceDB / v0.8 summarizer)."""

from __future__ import annotations

import re

from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.ui.run_log import trace

_CHUNK_SPLIT = re.compile(r"\n\s*\n|(?<=[.!?])\s+")


def _split_extracts(text: str, max_chunks: int = 6, chunk_len: int = 600) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for block in _CHUNK_SPLIT.split(raw):
        s = block.strip()
        if len(s) < 40:
            continue
        if len(s) > chunk_len:
            s = s[:chunk_len].rsplit(" ", 1)[0] + "…"
        parts.append(s)
        if len(parts) >= max_chunks:
            break
    return parts


def _extracts_from_document_summary(ds) -> list[str]:
    """Полноценные выжимки из LanceDB (Consensus summarizer)."""
    return _deep_extract_blocks(
        list(ds.key_takeaways or []),
        list(ds.failure_modes or []),
        [],
        min_words=150,
        max_words=300,
    )


def _extract_word_total(extracts: list[str]) -> int:
    return sum(len((e or "").split()) for e in extracts)


def enrich_search_hits_with_extracts(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """key_extracts из LanceDB (takeaways + failure_modes) или v0.8 harvest."""
    if not hits:
        return hits

    urls = [h.url for h in hits if h.url.startswith("http")]
    by_url: dict[str, object] = {}
    try:
        store = VectorStore()
        for ds in store.fetch_summaries_by_urls(urls, limit=len(urls) + 2):
            key = (ds.url or "").strip().rstrip("/").lower()
            if key:
                by_url[key] = ds
    except Exception as exc:
        trace(f"CURRICULUM extracts LanceDB skip | {exc}")

    enriched: list[CurriculumSearchHit] = []
    for i, hit in enumerate(hits, start=1):
        key_lower = hit.url.strip().rstrip("/").lower()
        ds = by_url.get(key_lower)
        extracts: list[str] = list(hit.key_extracts or [])

        if ds:
            lance_extracts = _extracts_from_document_summary(ds)
            if _extract_word_total(lance_extracts) > _extract_word_total(extracts):
                extracts = lance_extracts
            if not hit.title and ds.title:
                hit = hit.model_copy(update={"title": ds.title[:400]})

        if not extracts or _extract_word_total(extracts) < 120:
            if ds:
                extracts = _extracts_from_document_summary(ds) or extracts
            if not extracts and hit.snippet:
                extracts = _deep_extract_blocks([], [], [hit.snippet], 150, 300)
            if not extracts and hit.snippet:
                extracts = _split_extracts(hit.snippet, max_chunks=2, chunk_len=800)

        seen: set[str] = set()
        deduped: list[str] = []
        for e in extracts:
            e = e.strip()
            if not e or e in seen:
                continue
            seen.add(e)
            deduped.append(e[:2000])

        enriched.append(
            hit.model_copy(
                update={
                    "source_id": hit.source_id or f"src_{i}",
                    "key_extracts": deduped[:8],
                }
            )
        )

    deep = sum(1 for h in enriched if _extract_word_total(h.key_extracts) >= 150)
    trace(
        f"CURRICULUM extracts enrich ✓ | hits={len(enriched)} "
        f"deep_context={deep}/{len(enriched)}"
    )
    return enriched


def summarize_whitelist_blog_hits(
    hits: list[CurriculumSearchHit],
    target_goal: str = "",
) -> list[CurriculumSearchHit]:
    """Lite/Summarizer для whitelist-блогов (второй tier после Consensus)."""
    from knowledge_engine.services.summarizer import summarize_article
    from knowledge_engine.services.vector_store import VectorStore
    from knowledge_engine.services.web_extract import smart_fetch_page_text
    from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks

    store = VectorStore()
    out: list[CurriculumSearchHit] = []
    for h in hits:
        if (h.source_tier or "") != "whitelist_blog":
            out.append(h)
            continue
        if h.key_extracts and sum(len(e.split()) for e in h.key_extracts) >= 120:
            out.append(h)
            continue
        try:
            text, _method = smart_fetch_page_text(h.url)
            if len((text or "").strip()) < 200:
                out.append(h)
                continue
            summary = summarize_article(
                h.title or h.url,
                h.url,
                text[:14000],
            )
            store.save_summary(summary)
            extracts = _deep_extract_blocks(
                list(summary.key_takeaways or []),
                list(summary.failure_modes or []),
                [],
            )
            if not extracts:
                extracts = _deep_extract_blocks([], [], [h.snippet or text[:2000]], 80, 300)
            out.append(
                h.model_copy(
                    update={
                        "title": (summary.title or h.title)[:400],
                        "key_extracts": extracts[:8],
                    }
                )
            )
        except Exception as exc:
            trace(f"CURRICULUM blog summarizer skip | {h.url[:50]} | {exc}")
            out.append(h)
    blog_deep = sum(
        1
        for h in out
        if h.source_tier == "whitelist_blog"
        and sum(len(e.split()) for e in h.key_extracts) >= 120
    )
    trace(f"CURRICULUM blog summarizer ✓ | deep_blogs={blog_deep}")
    return out
