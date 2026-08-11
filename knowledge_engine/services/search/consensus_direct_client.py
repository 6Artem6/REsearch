"""Hybrid Consensus Direct API: session cache + curl_cffi paper_search.

См. knowledge_engine/docs/CONSENSUS_API_DIRECT.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from knowledge_engine.config import (
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_QUICK_OPEN_ACCESS,
)
from knowledge_engine.services.search.consensus_session_manager import (
    ConsensusSession,
    ConsensusSessionManager,
    block_unnecessary_requests,
    cookies_dict_to_header,
    filter_consensus_cookies,
    get_consensus_session_manager,
    session_jwt_from_cookies_dict,
    shutdown_consensus_session_manager,
)
from knowledge_engine.src.retrieval.consensus_capture import papers_from_json_text
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.ui.run_log import trace

try:
    from curl_cffi.requests import AsyncSession
except ImportError:  # pragma: no cover
    AsyncSession = None  # type: ignore[misc, assignment]

# Product alias (ScholarPaper shape from Consensus JSON).
ConsensusPaper = ScholarPaper

PAPER_SEARCH_URL = "https://consensus.app/api/paper_search/"
IMPERSONATE = "chrome124"


def build_search_payload(query: str, *, open_access: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": (query or "").strip(),
        "product_feature": "quick_search",
        "filters": {},
    }
    if open_access:
        payload["filters"]["open_access"] = "true"
    return payload


def papers_to_raw_text(papers: list[ScholarPaper], query: str) -> str:
    lines = [f"Consensus Direct API results for: {query.strip()}", ""]
    for i, p in enumerate(papers, 1):
        bit = f"{i}. {p.title}"
        if p.year:
            bit += f" ({p.year})"
        if p.source_url:
            bit += f"\n   {p.source_url}"
        abstract = (p.abstract or p.tldr or "").strip()
        if abstract:
            bit += f"\n   {abstract[:600]}"
        lines.append(bit)
    return "\n".join(lines)


class ConsensusDirectClient:
    """Session-manager warmup → curl_cffi POST /api/paper_search/ → in-page fetch fallback."""

    def __init__(
        self,
        *,
        headless: bool = CONSENSUS_BROWSER_HEADLESS,
        open_access: bool = CONSENSUS_QUICK_OPEN_ACCESS,
        impersonate: str = IMPERSONATE,
        session_manager: ConsensusSessionManager | None = None,
    ) -> None:
        self.headless = headless
        self.open_access = open_access
        self.impersonate = impersonate
        self._session_manager = session_manager
        self._lock = asyncio.Lock()
        self._active: Optional[ConsensusSession] = None

    async def _manager(self) -> ConsensusSessionManager:
        if self._session_manager is not None:
            return self._session_manager
        return await get_consensus_session_manager()

    async def ensure_session(self, *, force: bool = False) -> ConsensusSession:
        mgr = await self._manager()
        sess = await mgr.get_active_session(force=force)
        self._active = sess
        return sess

    async def ensure_warmup_async(self) -> None:
        mgr = await self._manager()
        await mgr.ensure_warmup_async()

    async def search_papers(self, query: str, limit: int = 20) -> list[ConsensusPaper]:
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(int(limit or 20), 50))
        t0 = time.perf_counter()
        async with self._lock:
            sess = await self.ensure_session()
            papers, via, status = await self._search_curl(q, limit, sess)
            if papers is None and status in (307, 401, 403, 0):
                trace(
                    f"Consensus Direct ⊘ curl status={status} → force session refresh"
                )
                mgr = await self._manager()
                mgr.invalidate()
                sess = await self.ensure_session(force=True)
                papers, via, status = await self._search_curl(q, limit, sess)
            if papers is None:
                trace(
                    f"Consensus Direct ⊘ curl_cffi status={status} → in-page fetch fallback"
                )
                papers, via, status = await self._search_inpage(q, limit)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            n = len(papers or [])
            trace(
                f"Consensus Direct ✓ via={via} status={status} "
                f"papers={n} {elapsed_ms:.0f}ms | q={q[:80]}"
            )
            return list(papers or [])[:limit]

    # alias
    async def search(self, query: str, limit: int = 20) -> list[ConsensusPaper]:
        return await self.search_papers(query, limit=limit)

    async def close(self) -> None:
        self._active = None
        if self._session_manager is not None:
            await self._session_manager.close()

    def _auth_headers(self, sess: ConsensusSession) -> dict[str, str]:
        headers = {
            "User-Agent": sess.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://consensus.app",
            "Referer": "https://consensus.app/",
        }
        cookie_header = cookies_dict_to_header(sess.cookies_dict)
        if cookie_header:
            headers["Cookie"] = cookie_header
        if sess.session_jwt:
            headers["Authorization"] = f"Bearer {sess.session_jwt}"
        return headers

    async def _search_curl(
        self,
        query: str,
        limit: int,
        sess: ConsensusSession,
    ) -> tuple[Optional[list[ScholarPaper]], str, int]:
        if AsyncSession is None:
            trace("Consensus Direct ⊘ curl_cffi not installed")
            return None, "curl_cffi_missing", 0
        payload = build_search_payload(query, open_access=self.open_access)
        headers = self._auth_headers(sess)
        # защита от раздутого Cookie
        if len(headers.get("Cookie", "")) > 12_000:
            trace("Consensus Direct ⊘ Cookie header too large — filter failed?")
            return None, "cookie_too_large", 400
        try:
            async with AsyncSession(impersonate=self.impersonate) as session:
                resp = await session.post(
                    PAPER_SEARCH_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    allow_redirects=False,
                )
                status = int(resp.status_code)
                body = resp.text or ""
                if status in (307, 401, 403):
                    return None, "curl_cffi", status
                if status != 200:
                    trace(f"Consensus Direct ⊘ curl HTTP {status} | {body[:160]}")
                    return None, "curl_cffi", status
                papers = papers_from_json_text(body)
                if not papers and body.lstrip().startswith(("{", "[")):
                    return [], "curl_cffi", status
                if limit and len(papers) > limit:
                    papers = papers[:limit]
                return papers, "curl_cffi", status
        except Exception as exc:
            trace(f"Consensus Direct ⊘ curl_cffi exc | {exc}")
            return None, "curl_cffi_error", 0

    async def _search_inpage(
        self, query: str, limit: int
    ) -> tuple[Optional[list[ScholarPaper]], str, int]:
        """Короткий browser + fetch(); статика блокируется."""
        from playwright.async_api import async_playwright

        from knowledge_engine.services.search.playwright_launch import (
            launch_persistent_context_async,
        )

        payload = build_search_payload(query, open_access=self.open_access)
        playwright = await async_playwright().start()
        context = None
        try:
            context = await launch_persistent_context_async(
                playwright, headless=self.headless
            )
            page = context.pages[0] if context.pages else await context.new_page()
            await page.route("**/*", block_unnecessary_requests)
            await page.goto(
                "https://consensus.app/",
                wait_until="commit",
                timeout=7_000,
            )
            # дать Clerk/CF минимально подняться для fetch credentials
            await asyncio.sleep(0.35)
            result = await page.evaluate(
                """async ({url, payload}) => {
                    const resp = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json',
                        },
                        body: JSON.stringify(payload),
                        credentials: 'include',
                    });
                    const text = await resp.text();
                    return {status: resp.status, body: text};
                }""",
                {"url": PAPER_SEARCH_URL, "payload": payload},
            )
            # обновить кэш сессии из jar после fetch
            raw = filter_consensus_cookies(await context.cookies())
            cookies_dict = {
                str(c.get("name") or ""): str(c.get("value") or "") for c in raw
            }
            jwt = session_jwt_from_cookies_dict(cookies_dict)
            if jwt:
                mgr = await self._manager()
                from knowledge_engine.services.search.consensus_session_manager import (
                    ConsensusSession as Sess,
                )

                mgr._cache = Sess(
                    cf_clearance=cookies_dict.get("cf_clearance", ""),
                    session_jwt=jwt,
                    cookies_dict=cookies_dict,
                    created_at=time.time(),
                )
                self._active = mgr._cache

            status = int((result or {}).get("status") or 0)
            body = str((result or {}).get("body") or "")
            if status != 200:
                return None, "inpage_fetch", status
            papers = papers_from_json_text(body)
            if limit and len(papers) > limit:
                papers = papers[:limit]
            return papers, "inpage_fetch", status
        except Exception as exc:
            trace(f"Consensus Direct ⊘ in-page fetch exc | {exc}")
            return None, "inpage_error", 0
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
            try:
                await playwright.stop()
            except Exception:
                pass


_shared_direct_lock = asyncio.Lock()
_shared_direct: Optional[ConsensusDirectClient] = None


async def acquire_consensus_direct_client() -> ConsensusDirectClient:
    global _shared_direct
    async with _shared_direct_lock:
        if _shared_direct is None:
            _shared_direct = ConsensusDirectClient()
        return _shared_direct


async def shutdown_consensus_direct_client() -> None:
    global _shared_direct
    async with _shared_direct_lock:
        if _shared_direct is not None:
            await _shared_direct.close()
            _shared_direct = None
        await shutdown_consensus_session_manager()
        trace("Consensus Direct ✓ client closed")
