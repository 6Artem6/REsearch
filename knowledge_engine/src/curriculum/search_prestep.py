"""Search-First: реальный поиск материалов перед генерацией маршрута."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse

import httpx

from knowledge_engine.config import (
    CURRICULUM_OPEN_SEARCH_QUERY_CONCURRENCY,
    CURRICULUM_SEARCH_MIN_HITS,
    CURRICULUM_SEARCH_PROBE_URLS,
    CURRICULUM_SEARCH_TARGET_HITS,
)
from knowledge_engine.services.search.registry import default_registry
from knowledge_engine.services.search.url_filter import (
    is_blocked_url,
    url_priority_score,
)
from knowledge_engine.src.curriculum.curriculum_search_sites import (
    CURRICULUM_PRIORITY_ENGINEERING_SITES,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

_HTTP_URL_RE = re.compile(r"^https?://", re.I)


def _normalize_url_key(url: str) -> str:
    u = (url or "").strip()
    if not _HTTP_URL_RE.match(u):
        return ""
    parsed = urlparse(u)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").rstrip("/") or ""
    scheme = (parsed.scheme or "https").lower()
    return urlunparse((scheme, host, path, "", "", ""))


def build_curriculum_search_queries(target_goal: str) -> list[str]:
    goal = (target_goal or "").strip()
    if len(goal) < 8:
        return []
    from knowledge_engine.src.source_evaluator.whitelist import (
        APPROVED_SOURCES_WHITELIST,
    )

    queries = [
        f"{goal} system design best practices articles",
        f"{goal} architecture patterns engineering blog",
        f"{goal} deep dive production case study",
    ]
    for site in CURRICULUM_PRIORITY_ENGINEERING_SITES:
        queries.append(f"site:{site} {goal[:120]}")
    for entries in APPROVED_SOURCES_WHITELIST.values():
        for raw in entries[:2]:
            host = (raw or "").split("/")[0].strip()
            if host and host not in CURRICULUM_PRIORITY_ENGINEERING_SITES:
                queries.append(f"site:{host} {goal[:100]}")
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:12]


def _probe_url_reachable(url: str, timeout: float = 8.0) -> bool:
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "REsearch-CurriculumSearch/1.0"},
        ) as client:
            resp = client.head(url)
            if resp.status_code >= 400 or resp.status_code == 405:
                resp = client.get(url, headers={"Range": "bytes=0-2048"})
            return resp.status_code < 400
    except Exception:
        return False


def _probe_urls_parallel(urls: list[str], max_workers: int = 6) -> set[str]:
    ok: set[str] = set()
    if not urls:
        return ok
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_probe_url_reachable, u): u for u in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            if fut.result():
                ok.add(_normalize_url_key(url))
    return ok


def collect_curriculum_source_hits(
    target_goal: str,
    limit_per_provider: int = 4,
    depth_level: str = "",
    generation_mode: str = "fast",
    source_policy: str | None = None,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.curriculum.source_material_pipeline import (
        collect_sources_by_policy,
    )
    from knowledge_engine.src.curriculum.source_policy import resolve_source_policy

    policy = resolve_source_policy(
        source_policy,
        generation_mode,
        default="practical_only",
    )
    return collect_sources_by_policy(
        target_goal,
        source_policy=policy,
        limit_per_provider=limit_per_provider,
    )


def collect_curriculum_search_hits(
    target_goal: str,
    limit_per_provider: int = 4,
    depth_level: str = "",
    generation_mode: str = "fast",
) -> list[CurriculumSearchHit]:
    return collect_curriculum_source_hits(
        target_goal,
        limit_per_provider=limit_per_provider,
        depth_level=depth_level,
        generation_mode=generation_mode,
    )


def _collect_whitelist_blog_hits(
    target_goal: str,
    limit_per_provider: int = 4,
    max_hits: int = 8,
    exclude_url_keys: set[str] | None = None,
) -> list[CurriculumSearchHit]:
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_collectible_article_url,
    )

    exclude = exclude_url_keys or set()
    raw = _collect_search_registry_hits(target_goal, limit_per_provider)
    out: list[CurriculumSearchHit] = []
    for h in raw:
        if len(out) >= max_hits:
            break
        key = _normalize_url_key(h.url)
        if not key or key in exclude:
            continue
        if not is_collectible_article_url(h.url):
            continue
        exclude.add(key)
        tier = (h.source_tier or "").strip() or "whitelist_blog"
        out.append(h.model_copy(update={"source_tier": tier}))
    trace(
        f"CURRICULUM open search blogs ✓ | from_search={len(raw)} accepted={len(out)}"
    )
    return out


def _collect_search_registry_hits(
    target_goal: str,
    limit_per_provider: int = 4,
) -> list[CurriculumSearchHit]:
    queries = build_curriculum_search_queries(target_goal)
    if not queries:
        return []

    registry = default_registry()
    conc = max(1, CURRICULUM_OPEN_SEARCH_QUERY_CONCURRENCY)
    trace(
        f"CURRICULUM open search batch ▶ | queries={len(queries)} "
        f"concurrency={conc} limit_per_provider={limit_per_provider}"
    )
    raw_hits = registry.multi_search_queries_batch_sync(
        queries,
        limit_per_provider=limit_per_provider,
        concurrency=conc,
    )

    merged: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    for h in raw_hits:
        key = _normalize_url_key(h.url)
        if not key or key in seen or is_blocked_url(h.url):
            continue
        seen.add(key)
        merged.append(
            CurriculumSearchHit(
                url=h.url.strip()[:2000],
                title=(h.title or h.url).strip()[:400],
                snippet=(h.snippet or "").strip()[:1200],
                published_date=(h.published_date or "").strip()[:32],
                key_extracts=list(h.key_extracts or [])[:12],
                source_tier=(h.source or "whitelist_blog").strip()[:24],
                skip_ollama_summary=bool(h.skip_ollama_summary),
            )
        )

    merged.sort(key=lambda x: url_priority_score(x.url))
    cap = CURRICULUM_SEARCH_TARGET_HITS
    merged = merged[:cap]

    if CURRICULUM_SEARCH_PROBE_URLS and merged:
        reachable = _probe_urls_parallel([m.url for m in merged])
        probed = [m for m in merged if _normalize_url_key(m.url) in reachable]
        trace(
            f"CURRICULUM search probe | reachable={len(probed)}/{len(merged)} "
            f"min_required={CURRICULUM_SEARCH_MIN_HITS}"
        )
        if len(probed) >= CURRICULUM_SEARCH_MIN_HITS:
            merged = probed[:cap]
        elif probed:
            merged = probed

    trace(
        f"CURRICULUM search prestep ✓ | queries={len(queries)} hits={len(merged)} "
        f"goal={target_goal[:60]}…"
    )
    return merged


def search_hits_as_prompt_json(hits: list[CurriculumSearchHit]) -> str:
    import json

    rows = []
    for i, h in enumerate(hits, start=1):
        sid = (h.source_id or "").strip() or f"src_{i}"
        extracts = list(h.key_extracts or [])
        if not extracts and h.snippet:
            extracts = [h.snippet[:800]]
        rows.append(
            {
                "source_id": sid,
                "title": h.title,
                "url": h.url,
                "source_tier": h.source_tier or "",
                "published_date": (h.published_date or "").strip(),
                "key_extracts": extracts[:8],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def assign_source_ids(hits: list[CurriculumSearchHit]) -> list[CurriculumSearchHit]:
    out: list[CurriculumSearchHit] = []
    for i, h in enumerate(hits, start=1):
        sid = (h.source_id or "").strip() or f"src_{i}"
        out.append(h.model_copy(update={"source_id": sid[:16]}))
    return out


def search_hit_index(hits: list[CurriculumSearchHit]) -> dict[str, CurriculumSearchHit]:
    out: dict[str, CurriculumSearchHit] = {}
    for h in hits:
        key = _normalize_url_key(h.url)
        if key:
            out[key] = h
    return out
