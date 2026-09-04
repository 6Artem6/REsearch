"""HTTP liveness check for Flash Lite Exa domain hypotheses (Pass 1)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Sequence

import httpx

from knowledge_engine.services.search.exa_domains import clean_domain_for_exa
from knowledge_engine.ui.run_log import trace

EXA_DOMAIN_HTTP_TIMEOUT_SEC = 2.0
_PROBE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; KnowledgeEngine-domain-probe/1.0)"),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}
_HEAD_RETRY_GET_STATUSES = frozenset({403, 405, 501})
_MAX_CONCURRENT_PROBES = 8


def _unique_probe_hosts(domains: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in domains:
        host = clean_domain_for_exa(raw)
        if not host or "." not in host or host in seen:
            continue
        seen.add(host)
        out.append(host)
    return out


def _is_live_status(code: int) -> bool:
    return 200 <= int(code) < 400


async def _probe_one(client: httpx.AsyncClient, host: str) -> str | None:
    url = f"https://{host}"
    try:
        response = await client.head(url, follow_redirects=True)
        if response.status_code in _HEAD_RETRY_GET_STATUSES:
            response = await client.get(url, follow_redirects=True)
        code = int(response.status_code)
        if _is_live_status(code):
            trace(f"EXA domain HTTP ✓ | {host} | {code}")
            return host
        trace(f"EXA domain HTTP ⊘ | {host} | HTTP {code}")
        return None
    except httpx.TimeoutException:
        trace(f"EXA domain HTTP ⊘ | {host} | Timeout")
        return None
    except httpx.RequestError as exc:
        trace(f"EXA domain HTTP ⊘ | {host} | {type(exc).__name__}")
        return None


async def validate_exa_domains(domains: Sequence[str]) -> list[str]:
    """Keep hosts that answer HTTPS without DNS/timeout/4xx/5xx."""
    hosts = _unique_probe_hosts(domains)
    trace(f"EXA domain HTTP ▶ | n={len(hosts)}")
    if not hosts:
        trace("EXA domain HTTP done | live=0/0")
        return []

    timeout = httpx.Timeout(EXA_DOMAIN_HTTP_TIMEOUT_SEC)
    limits = httpx.Limits(max_connections=_MAX_CONCURRENT_PROBES)
    sem = asyncio.Semaphore(_MAX_CONCURRENT_PROBES)

    async def _guarded(client: httpx.AsyncClient, host: str) -> str | None:
        async with sem:
            return await _probe_one(client, host)

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=_PROBE_HEADERS,
        follow_redirects=True,
        limits=limits,
    ) as client:
        probed = await asyncio.gather(*[_guarded(client, h) for h in hosts])

    live = [h for h in probed if h]
    trace(
        f"EXA domain HTTP done | live={len(live)}/{len(hosts)}"
        + (f" | {', '.join(live)}" if live else "")
    )
    return live


def validate_exa_domains_blocking(domains: Sequence[str]) -> list[str]:
    """Sync wrapper for `ExaSearchClient.search_expanded`."""

    def _run() -> list[str]:
        return asyncio.run(validate_exa_domains(domains))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=60)


async def prepare_exa_pass1_domains(domains: Sequence[str]) -> list[str]:
    """HTTP-live hosts only. Does not assign OFFICIAL_DOCS."""
    return await validate_exa_domains(domains)


def prepare_exa_pass1_domains_blocking(domains: Sequence[str]) -> list[str]:
    """Sync HTTP probe. Does not assign OFFICIAL_DOCS."""
    return validate_exa_domains_blocking(domains)
