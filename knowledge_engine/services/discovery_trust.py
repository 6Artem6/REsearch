"""Сохранение ссылок в архив + приоритет доверенных доменов."""

from __future__ import annotations

from knowledge_engine.config import SOURCE_ARCHIVE_ENABLED
from knowledge_engine.db.source_links import get_source_link_archive
from knowledge_engine.schemas import DomainTrustResult
from knowledge_engine.services.domain_profiler import (
    filter_urls_by_domain_trust,
    is_high_trust_score,
    normalize_domain,
)
from knowledge_engine.services.searxng_client import (
    engine_trust_hint,
    is_priority_searxng_engine,
)


def enrich_trust_map_from_searxng(
    trust_map: dict[str, DomainTrustResult],
    urls: list[str],
    url_searxng_engines: dict[str, str],
) -> dict[str, DomainTrustResult]:
    out = dict(trust_map)
    for url in urls:
        engine = url_searxng_engines.get(url, "")
        hint = engine_trust_hint(engine)
        if not hint:
            continue
        dom = normalize_domain(url)
        score, category = hint
        existing = out.get(dom)
        if existing and existing.trust_score >= score:
            continue
        out[dom] = DomainTrustResult(
            domain=dom,
            trust_score=score,
            category=category,
            reason=f"searxng_engine:{engine}",
            is_valid_for_research=True,
            from_cache=False,
        )
    return out


def archive_urls_from_discovery(
    raw_urls: list[str],
    trust_by_domain: dict[str, DomainTrustResult],
    accepted: list[str],
    rejected: list[str],
    discovery_query: str,
) -> None:
    if not SOURCE_ARCHIVE_ENABLED:
        return
    archive = get_source_link_archive()
    seen: set[str] = set()
    for url in raw_urls:
        if url in seen:
            continue
        seen.add(url)
        dom = normalize_domain(url)
        trust = trust_by_domain.get(dom)
        if url in rejected:
            archive.upsert(
                url=url,
                domain=dom,
                trust_score=trust.trust_score if trust else None,
                category=trust.category if trust else None,
                status="rejected_low_trust",
                rejection_reason=trust.reason if trust else "REJECTED_LOW_TRUST_DOMAIN",
                discovery_query=discovery_query,
            )
        elif url in accepted:
            archive.upsert(
                url=url,
                domain=dom,
                trust_score=trust.trust_score if trust else None,
                category=trust.category if trust else None,
                status="accepted",
                discovery_query=discovery_query,
            )
        else:
            archive.upsert(
                url=url,
                domain=dom,
                trust_score=trust.trust_score if trust else None,
                category=trust.category if trust else None,
                status="discovered",
                discovery_query=discovery_query,
            )


def prioritize_trusted_urls(
    urls: list[str],
    trust_by_domain: dict[str, DomainTrustResult],
    url_searxng_engines: dict[str, str] | None = None,
) -> list[str]:
    """Доверенные домены и IT/science движки SearXNG — в начало очереди."""
    engine_prio: list[str] = []
    high: list[str] = []
    rest: list[str] = []
    engines = url_searxng_engines or {}
    for url in urls:
        engine = engines.get(url, "")
        dom = normalize_domain(url)
        trust = trust_by_domain.get(dom)
        if engine and is_priority_searxng_engine(engine):
            engine_prio.append(url)
        elif trust and is_high_trust_score(trust.trust_score, trust.category):
            high.append(url)
        else:
            rest.append(url)
    if engine_prio or high:
        from knowledge_engine.ui.run_log import trace

        trace(
            f"DOMAIN prioritize | searxng_engine={len(engine_prio)} "
            f"high_trust={len(high)} other={len(rest)}"
        )
    return engine_prio + high + rest


def apply_domain_trust_to_urls(
    raw_urls: list[str],
    discovery_query: str,
    url_searxng_engines: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, DomainTrustResult]]:
    accepted, rejected, trust_by_domain = filter_urls_by_domain_trust(raw_urls)
    if url_searxng_engines:
        trust_by_domain = enrich_trust_map_from_searxng(
            trust_by_domain,
            raw_urls,
            url_searxng_engines,
        )
    archive_urls_from_discovery(
        raw_urls, trust_by_domain, accepted, rejected, discovery_query
    )
    accepted = prioritize_trusted_urls(accepted, trust_by_domain, url_searxng_engines)
    return accepted, rejected, trust_by_domain
