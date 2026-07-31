"""Reuse академических hits из LanceDB / registry без Playwright."""

from __future__ import annotations

from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit, CurriculumSourceRegistryEntry
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.run_log import trace


def _extracts_from_summary(summary) -> list[str]:
    from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks

    takeaways = list(summary.key_takeaways or [])
    failures = list(summary.failure_modes or [])
    blocks = _deep_extract_blocks(takeaways, failures, [], min_words=80, max_words=300)
    if blocks:
        return blocks[:8]
    title = (summary.title or "").strip()
    if len(title) >= 40:
        return [title[:800]]
    return []


def _tier_for_url(url: str, entry_tier: str = "") -> str:
    t = (entry_tier or "").strip().lower()
    if t:
        return t[:24]
    low = (url or "").lower()
    if "arxiv" in low:
        return "arxiv"
    if "doi.org" in low or "openreview" in low:
        return "consensus"
    return "academic_open"


def hits_from_registry_entries(
    entries: list[CurriculumSourceRegistryEntry | dict],
    *,
    cap: int,
    exclude_url_keys: set[str],
) -> list[CurriculumSearchHit]:
    out: list[CurriculumSearchHit] = []
    for raw in entries:
        if isinstance(raw, CurriculumSourceRegistryEntry):
            e = raw
        else:
            e = CurriculumSourceRegistryEntry.model_validate(raw)
        url = (e.url or "").strip()
        if not url.startswith("http"):
            continue
        key = _normalize_url_key(url)
        if not key or key in exclude_url_keys:
            continue
        extracts = list(e.key_extracts or [])
        if not extracts and (e.snippet or e.why_read):
            text = (e.snippet or e.why_read or "").strip()
            if len(text) >= 80:
                extracts = [text[:800]]
        if not extracts:
            continue
        out.append(
            CurriculumSearchHit(
                url=url[:2000],
                title=(e.title or url)[:400],
                snippet=(e.snippet or extracts[0])[:1200],
                key_extracts=extracts[:8],
                source_tier=_tier_for_url(url, e.source_tier),
                skip_ollama_summary=True,
            )
        )
        if len(out) >= cap:
            break
    return out


def hits_from_lancedb_goal(
    goal: str,
    *,
    cap: int,
    exclude_url_keys: set[str],
) -> list[CurriculumSearchHit]:
    q = (goal or "").strip()
    if len(q) < 8 or cap <= 0:
        return []
    store = VectorStore()
    try:
        summaries = store.hybrid_search(q[:1200], limit=max(cap + 4, 6))
    except Exception as exc:
        trace(f"CURRICULUM on_demand LanceDB ⊘ | hybrid_search | {exc}")
        return []
    out: list[CurriculumSearchHit] = []
    for summary in summaries:
        url = (summary.url or "").strip()
        if not url.startswith("http"):
            continue
        key = _normalize_url_key(url)
        if not key or key in exclude_url_keys:
            continue
        extracts = _extracts_from_summary(summary)
        if not extracts:
            continue
        out.append(
            CurriculumSearchHit(
                url=url[:2000],
                title=(summary.title or url)[:400],
                snippet=extracts[0][:1200],
                key_extracts=extracts,
                source_tier=_tier_for_url(url),
                skip_ollama_summary=True,
            )
        )
        if len(out) >= cap:
            break
    return out


def merge_on_demand_reuse_hits(
    goal: str,
    registry_entries: list,
    *,
    cap: int,
    exclude_url_keys: set[str],
) -> list[CurriculumSearchHit]:
    """Registry (с extracts) + LanceDB hybrid — без сетевого harvest."""
    seen: set[str] = set(exclude_url_keys)
    merged: list[CurriculumSearchHit] = []

    reg_hits = hits_from_registry_entries(
        registry_entries,
        cap=cap,
        exclude_url_keys=seen,
    )
    for h in reg_hits:
        k = _normalize_url_key(h.url)
        if k:
            seen.add(k)
        merged.append(h)

    if len(merged) < cap:
        need = cap - len(merged)
        for h in hits_from_lancedb_goal(
            goal,
            cap=need + 2,
            exclude_url_keys=seen,
        ):
            k = _normalize_url_key(h.url)
            if k:
                seen.add(k)
            merged.append(h)
            if len(merged) >= cap:
                break

    if merged:
        trace(
            f"CURRICULUM on_demand reuse ✓ | hits={len(merged)} "
            f"(registry+LanceDB, cap={cap})"
        )
    return merged[:cap]
