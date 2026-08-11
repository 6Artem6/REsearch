"""Lite: на основе уже найденных материалов — ещё 2+ сайта и поиск site:."""

from __future__ import annotations

from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteSiteSuggestionsContract,
)
from knowledge_engine.services.search.registry import default_registry
from knowledge_engine.services.search.url_filter import is_blocked_url
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.search_prestep import _normalize_url_key
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
    normalize_site_host,
)
from knowledge_engine.ui.run_log import trace


def _suggest_extra_sites(
    target_goal: str,
    seeds: list[CurriculumSearchHit],
    anchor: str,
) -> list[str]:
    from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    goal = (target_goal or "").strip()
    seed_block = "\n".join(
        f"- {(h.title or h.url)[:200]} | {h.url}" for h in seeds[:10]
    )
    system = (
        f"{RUSSIAN_OUTPUT_RULE}\n\n"
        "Ты помогаешь расширить поиск инженерных статей для учебного маршрута.\n"
        "На основе цели и уже найденных материалов предложи **минимум 2** домены "
        "(engineering blog, official docs, research lab) где вероятны глубокие статьи.\n"
        "Не SEO-агрегаторы и не generic tutorial farms.\n"
        "JSON: sites (2–4 hostname без схемы), rationale (1 предложение, русский)."
    )
    user = f"### learning_goal\n{goal[:1200]}\n\n### seed_materials\n{seed_block or '(пусто)'}\n"
    try:
        out = run_gemini_lite_structured(
            system,
            user,
            anchor,
            LiteSiteSuggestionsContract,
            "curriculum / lite_site_suggest",
        )
    except Exception as exc:
        trace(f"CURRICULUM lite site suggest skip | {exc}")
        return []

    hosts: list[str] = []
    seen: set[str] = set()
    for raw in out.sites or []:
        host = normalize_site_host(raw)
        if not host or host in seen or "." not in host:
            continue
        seen.add(host)
        hosts.append(host)
    if len(hosts) < 2:
        trace(
            f"CURRICULUM lite site suggest ⊘ | sites={len(hosts)} "
            f"| {(out.rationale or '')[:80]}"
        )
        return []
    trace(
        f"CURRICULUM lite site suggest ✓ | sites={hosts[:4]} | "
        f"{(out.rationale or '')[:100]}"
    )
    return hosts[:4]


def collect_lite_suggested_site_hits(
    target_goal: str,
    seeds: list[CurriculumSearchHit],
    *,
    exclude_url_keys: set[str] | None = None,
    limit_per_provider: int = 4,
    max_hits: int = 8,
    anchor: str = "",
) -> list[CurriculumSearchHit]:
    """SearXNG site: на домены, предложенные Lite по уже найденным hits."""
    goal = (target_goal or "").strip()
    if len(goal) < 8 or max_hits <= 0:
        return []
    sites = _suggest_extra_sites(
        goal, seeds, anchor or f"curriculum_sites:{goal[:400]}"
    )
    if not sites:
        return []

    exclude = exclude_url_keys or set()
    registry = default_registry()
    out: list[CurriculumSearchHit] = []
    seen: set[str] = set()

    for host in sites:
        if len(out) >= max_hits:
            break
        query = f"site:{host} {goal[:120]}"
        try:
            hits = registry.multi_search_sync(
                query,
                limit_per_provider=limit_per_provider,
            )
        except Exception as exc:
            trace(f"CURRICULUM lite site search skip | {host} | {exc}")
            continue
        for h in hits:
            if len(out) >= max_hits:
                break
            url = (h.url or "").strip()
            key = _normalize_url_key(url)
            if not key or key in exclude or key in seen:
                continue
            if is_blocked_url(url) or not is_collectible_article_url(url):
                continue
            seen.add(key)
            exclude.add(key)
            out.append(
                CurriculumSearchHit(
                    url=url,
                    title=(h.title or url)[:400],
                    snippet=(h.snippet or "")[:1200],
                    source_tier="lite_suggested_site",
                )
            )

    trace(f"CURRICULUM lite site search ✓ | new_hits={len(out)}")
    return out
