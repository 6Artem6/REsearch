"""Domain Trust Engine v0.5 — Gemini batch profiler + SQLite кэш."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from knowledge_engine.config import (
    DOMAIN_TRUST_BATCH_SIZE,
    DOMAIN_TRUST_ENABLED,
    DOMAIN_TRUST_MIN_SCORE,
)
from knowledge_engine.db.domains import get_domain_trust_store
from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE
from knowledge_engine.schemas import (
    DomainProfilerBatchItem,
    DomainProfilerBatchResult,
    DomainTrustResult,
)
from knowledge_engine.schemas.llm_contracts.domain import DomainProfilerBatchContract
from knowledge_engine.services.gemini_stateless import (
    is_gemini_available,
    run_stateless_gemini,
)
from knowledge_engine.ui.run_log import trace

_HIGH_TRUST_CATEGORIES = frozenset(
    {
        "tech_blog",
        "academic",
        "official_docs",
        "documentation",
    }
)


def is_high_trust_score(score: float, category: str) -> bool:
    from knowledge_engine.config import DOMAIN_TRUST_HIGH_SCORE

    cat = (category or "").strip().lower()
    if cat in _HIGH_TRUST_CATEGORIES:
        return True
    return score >= DOMAIN_TRUST_HIGH_SCORE


_LOW_TRUST_CATEGORIES = frozenset(
    {
        "e_commerce",
        "seo_farm",
        "seo_spam",
    }
)

_STATIC_LOW_TRUST_HOSTS = frozenset(
    {
        "ikea.com",
        "amazon.com",
        "amazon.co.uk",
        "avito.ru",
        "aliexpress.com",
        "ebay.com",
        "images.google.com",
        "www.google.com",
    }
)


def normalize_domain(url_or_host: str) -> str:
    raw = (url_or_host or "").strip().lower()
    if not raw:
        return ""
    if "://" in raw:
        host = urlparse(raw).netloc
    else:
        host = raw.split("/")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _valid_for_research(score: float, category: str, gemini_flag: bool) -> bool:
    cat = category.strip().lower()
    if cat in _LOW_TRUST_CATEGORIES:
        return False
    if score < DOMAIN_TRUST_MIN_SCORE:
        return False
    return gemini_flag


def _static_reject(domain: str) -> Optional[DomainTrustResult]:
    if domain in _STATIC_LOW_TRUST_HOSTS:
        return DomainTrustResult(
            domain=domain,
            trust_score=0.05,
            category="e_commerce",
            reason="static deny-list (retail / search UI)",
            created_at=datetime.now(timezone.utc).isoformat(),
            from_cache=True,
            is_valid_for_research=False,
        )
    for suffix in (".ikea.com", ".amazon.com"):
        if domain.endswith(suffix) or domain == suffix.lstrip("."):
            return DomainTrustResult(
                domain=domain,
                trust_score=0.1,
                category="e_commerce",
                reason=f"static suffix match {suffix}",
                created_at=datetime.now(timezone.utc).isoformat(),
                from_cache=True,
                is_valid_for_research=False,
            )
    return None


def _neutral_result(domain: str, reason: str) -> DomainTrustResult:
    return DomainTrustResult(
        domain=domain,
        trust_score=0.5,
        category="general_news",
        reason=reason,
        created_at=datetime.now(timezone.utc).isoformat(),
        is_valid_for_research=True,
    )


def _trust_from_batch_item(
    key: str, item: DomainProfilerBatchItem
) -> DomainTrustResult:
    return DomainTrustResult(
        domain=key,
        trust_score=item.trust_score,
        category=(item.category or "general_news").strip().lower(),
        reason=item.reason,
        created_at=datetime.now(timezone.utc).isoformat(),
        from_cache=False,
        is_valid_for_research=_valid_for_research(
            item.trust_score, item.category, item.is_valid_for_research
        ),
    )


class DomainProfiler:
    def __init__(self) -> None:
        self._store = get_domain_trust_store()

    def _resolve_cached_or_static(self, key: str) -> Optional[DomainTrustResult]:
        static = _static_reject(key)
        if static is not None:
            return static
        cached = self._store.get_domain(key)
        if cached is not None:
            trace(
                f"DOMAIN cache ✓ {key} | score={cached.trust_score:.2f} | {cached.category}"
            )
            return cached
        return None

    async def _gemini_batch(self, keys: list[str]) -> dict[str, DomainTrustResult]:
        if not keys:
            return {}
        if not is_gemini_available():
            return {k: _neutral_result(k, "Gemini недоступен") for k in keys}

        listed = "\n".join(f"- {k}" for k in keys)
        system = (
            f"{GEMINI_RUSSIAN_ROLE} "
            "Оцени каждый домен для глубоких инженерных и научных исследований "
            "(Backend, AI, Distributed Systems, Architecture). "
            "Один JSON: domains[] с полями domain, trust_score, category, "
            "is_valid_for_research, reason (кратко, русский)."
        )
        user = (
            f"Домены ({len(keys)}):\n{listed}\n\n"
            "Верни оценку для каждого домена из списка (domain должен совпадать)."
        )
        trace(f"DOMAIN profiler ▶ Gemini batch | count={len(keys)}")
        raw = await asyncio.to_thread(
            run_stateless_gemini,
            system,
            user,
            f"Domain trust batch ({len(keys)} hosts)",
            DomainProfilerBatchContract,
            "domain_profiler / batch",
            True,
        )
        if not isinstance(raw, DomainProfilerBatchResult):
            raw = DomainProfilerBatchResult.model_validate(raw)

        by_domain: dict[str, DomainProfilerBatchItem] = {}
        for item in raw.domains:
            dk = normalize_domain(item.domain)
            if dk:
                by_domain[dk] = item

        out: dict[str, DomainTrustResult] = {}
        for key in keys:
            item = by_domain.get(key)
            if item is None:
                result = _neutral_result(key, "нет в ответе batch — нейтральный score")
                trace(f"DOMAIN profiler ? {key} | missing in batch response")
            else:
                result = _trust_from_batch_item(key, item)
                trace(
                    f"DOMAIN profiler ✓ {key} | score={result.trust_score:.2f} | "
                    f"{result.category} | valid={result.is_valid_for_research}"
                )
            self._store.save_domain(result)
            out[key] = result
        return out

    async def evaluate_domain(self, domain: str) -> DomainTrustResult:
        key = normalize_domain(domain)
        if not key:
            return DomainTrustResult(
                domain="",
                trust_score=0.0,
                category="seo_spam",
                reason="empty domain",
                is_valid_for_research=False,
            )
        resolved = self._resolve_cached_or_static(key)
        if resolved is not None:
            return resolved
        batch = await self._gemini_batch([key])
        return batch[key]

    async def evaluate_domains(
        self, domains: list[str]
    ) -> dict[str, DomainTrustResult]:
        keys = list(
            dict.fromkeys(normalize_domain(d) for d in domains if normalize_domain(d))
        )
        if not keys:
            return {}

        results: dict[str, DomainTrustResult] = {}
        pending: list[str] = []
        for key in keys:
            resolved = self._resolve_cached_or_static(key)
            if resolved is not None:
                results[key] = resolved
            else:
                pending.append(key)

        chunk = max(1, DOMAIN_TRUST_BATCH_SIZE)
        for i in range(0, len(pending), chunk):
            batch_keys = pending[i : i + chunk]
            batch_out = await self._gemini_batch(batch_keys)
            results.update(batch_out)

        return results


_profiler: Optional[DomainProfiler] = None


def get_domain_profiler() -> DomainProfiler:
    global _profiler
    if _profiler is None:
        _profiler = DomainProfiler()
    return _profiler


def should_reject_url(trust: DomainTrustResult) -> bool:
    return not trust.is_valid_for_research


def filter_urls_by_domain_trust(
    urls: list[str],
) -> tuple[list[str], list[str], dict[str, DomainTrustResult]]:
    """Синхронная обёртка: accept, reject, trust по домену."""
    if not DOMAIN_TRUST_ENABLED or not urls:
        return list(urls), [], {}

    profiler = get_domain_profiler()
    domains = [normalize_domain(u) for u in urls]
    unique_domains = list(dict.fromkeys(d for d in domains if d))

    async def _run() -> dict[str, DomainTrustResult]:
        return await profiler.evaluate_domains(unique_domains)

    loop = asyncio.new_event_loop()
    try:
        trust_map = loop.run_until_complete(_run())
    finally:
        loop.close()

    accepted: list[str] = []
    rejected: list[str] = []
    for url in urls:
        dom = normalize_domain(url)
        trust = trust_map.get(dom)
        if trust is None:
            accepted.append(url)
            continue
        if should_reject_url(trust):
            rejected.append(url)
            trace(
                f"DOMAIN ✗ {url[:55]} | REJECTED_LOW_TRUST_DOMAIN | "
                f"{dom} score={trust.trust_score:.2f} cat={trust.category}"
            )
        else:
            accepted.append(url)
            trace(f"DOMAIN ✓ {dom} | score={trust.trust_score:.2f} | {trust.category}")
    return accepted, rejected, trust_map
