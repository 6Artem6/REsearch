"""Stateful Playwright session with Consensus.app: локальный persistent profile, новый чат на прогон."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional
from urllib.parse import quote_plus, urlencode

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from knowledge_engine.config import (
    BROWSER_PROFILE_PATH,
    CONSENSUS_AUTH_RECOVERY_CYCLES,
    CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC,
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_CLOSE_AFTER_EACH_HARVEST,
    CONSENSUS_HAR_PATH,
    CONSENSUS_INPUT_SELECTOR,
    CONSENSUS_LOG_JSON_TRAFFIC,
    CONSENSUS_MIN_RESPONSE_CHARS,
    CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC,
    CONSENSUS_NEW_THREAD_EACH_RUN,
    CONSENSUS_PAPER_HARVEST_PASSES,
    CONSENSUS_PAPER_HARVEST_PAUSE_SEC,
    CONSENSUS_QUICK_BASE_URL,
    CONSENSUS_QUICK_LOAD_MORE_CLICKS,
    CONSENSUS_QUICK_OPEN_ACCESS,
    CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC,
    CONSENSUS_RECORD_HAR,
    CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC,
    CONSENSUS_RESPONSE_MAX_SEC,
    CONSENSUS_RESPONSE_SELECTOR,
    CONSENSUS_REUSE_BROWSER_SESSION,
    CONSENSUS_SEND_SELECTORS,
    CONSENSUS_START_URL,
    CONSENSUS_STREAM_POLL_SEC,
    CONSENSUS_STREAM_STABLE_ROUNDS,
    CONSENSUS_UI_POLL_SEC,
    CONSENSUS_USE_DIRECT_API,
    CONSENSUS_USE_QUICK_PAPER_SEARCH,
)
from knowledge_engine.services.search.playwright_launch import (
    launch_persistent_context_async,
)
from knowledge_engine.src.retrieval.consensus_capture import (
    papers_from_json_text_relaxed,
)
from knowledge_engine.src.retrieval.consensus_papers import (
    extract_paper_cards_from_page,
    extract_papers_from_text,
    merge_scholar_papers,
)
from knowledge_engine.src.retrieval.consensus_types import ConsensusMessageResult
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper
from knowledge_engine.logging_setup import get_logger
from knowledge_engine.ui.run_log import trace

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://[^\s\]<\"')]+")
_RESULTS_URL_RE = re.compile(
    r"consensus\.app/(search|threads|thread|p|chat|results|quick)(/|$)",
    re.I,
)
_LANDING_PATH_RE = re.compile(r"^/(home)?/?$", re.I)


class ConsensusLoginRequiredError(RuntimeError):
    """Consensus просит войти — нужен recovery (goto / new thread / restart browser)."""


_shared_session_lock = asyncio.Lock()
_shared_session: Optional["ConsensusSessionManager"] = None


async def acquire_consensus_session() -> ConsensusSessionManager:
    """Один браузер с локальным profile между прогонами (если CONSENSUS_REUSE_BROWSER_SESSION)."""
    global _shared_session
    async with _shared_session_lock:
        if CONSENSUS_REUSE_BROWSER_SESSION and _shared_session is not None:
            return _shared_session
        mgr = ConsensusSessionManager()
        if CONSENSUS_REUSE_BROWSER_SESSION:
            _shared_session = mgr
        return mgr


async def release_consensus_session(session: ConsensusSessionManager) -> None:
    """После прогона: закрыть браузер или оставить для reuse (CONSENSUS_REUSE_BROWSER_SESSION)."""
    global _shared_session
    async with _shared_session_lock:
        if CONSENSUS_CLOSE_AFTER_EACH_HARVEST:
            if _shared_session is session:
                _shared_session = None
            await session.close()
            trace("Consensus ✓ session closed after harvest")
            return
        if CONSENSUS_REUSE_BROWSER_SESSION and session is _shared_session:
            await session.release_for_next_run()
            return
        if _shared_session is session:
            _shared_session = None
        await session.close()


async def shutdown_shared_consensus_session() -> None:
    """Корректно закрыть браузер (сохранить cookies в profile) при остановке API."""
    global _shared_session
    async with _shared_session_lock:
        if _shared_session is not None:
            trace("Consensus ▶ shutdown | сохранение profile …")
            await _shared_session.close()
            _shared_session = None
    try:
        from knowledge_engine.services.search.consensus_direct_client import (
            shutdown_consensus_direct_client,
        )

        await shutdown_consensus_direct_client()
    except Exception as exc:
        trace(f"Consensus Direct shutdown ⊘ | {exc}")


class ConsensusSessionManager:
    """Держит живую страницу Consensus для первичного и уточняющих запросов (RETRY)."""

    def __init__(
        self,
        headless: bool = CONSENSUS_BROWSER_HEADLESS,
        *,
        record_har_path: Optional[str] = None,
        log_json_traffic: Optional[bool] = None,
    ) -> None:
        self.headless = headless
        if record_har_path is not None:
            self.record_har_path: Optional[str] = record_har_path or None
        elif CONSENSUS_RECORD_HAR:
            self.record_har_path = str(CONSENSUS_HAR_PATH)
        else:
            self.record_har_path = None
        self.log_json_traffic = (
            CONSENSUS_LOG_JSON_TRAFFIC
            if log_json_traffic is None
            else bool(log_json_traffic)
        )
        if self.record_har_path:
            self.log_json_traffic = True
        self._playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._started = False
        self._on_thread = False
        self._api_papers: list[ScholarPaper] = []
        self._json_traffic_log: list[dict[str, Any]] = []

    async def _on_network_response(self, response) -> None:
        try:
            if response.status != 200:
                return
            url = response.url or ""
            ct = (response.headers.get("content-type") or "").lower()
            is_json = (
                "application/json" in ct
                or "json" in ct
                or url.rstrip("/").endswith(".json")
            )
            if not is_json:
                return

            method = "GET"
            try:
                method = (response.request.method or "GET").upper()
            except Exception:
                pass

            body = ""
            try:
                body = await response.text()
            except Exception:
                body = ""

            if self.log_json_traffic:
                preview = (body or "").replace("\n", " ")[:200]
                line = (
                    f"Consensus JSON | {method} {response.status} {url[:180]} | "
                    f"{preview}"
                )
                trace(line)
                logger.debug(line)
                self._json_traffic_log.append(
                    {
                        "method": method,
                        "status": response.status,
                        "url": url,
                        "content_type": ct,
                        "preview": preview,
                    }
                )

            if "consensus" not in url.lower():
                return
            if not any(
                tok in url.lower()
                for tok in (
                    "api",
                    "graphql",
                    "paper",
                    "search",
                    "thread",
                    "message",
                    "citation",
                    "results",
                    "query",
                )
            ):
                return
            if len(body) < 40:
                return
            found = papers_from_json_text_relaxed(body)
            if found:
                self._api_papers = merge_scholar_papers(self._api_papers, found)
                trace(
                    f"Consensus ✓ API capture | +{len(found)} papers | "
                    f"total={len(self._api_papers)}"
                )
        except Exception:
            return

    def _wire_network_capture(self, page: Page) -> None:
        page.on(
            "response",
            lambda resp: asyncio.create_task(self._on_network_response(resp)),
        )

    def _page_is_alive(self) -> bool:
        page = self.page
        if page is None:
            return False
        try:
            return not page.is_closed()
        except Exception:
            return False

    async def _pick_work_page(self) -> Page:
        assert self.context is not None
        for p in self.context.pages:
            try:
                if p.is_closed():
                    continue
                if "consensus.app" in (p.url or ""):
                    return p
            except Exception:
                continue
        for p in self.context.pages:
            try:
                if not p.is_closed():
                    return p
            except Exception:
                continue
        return await self.context.new_page()

    async def _hard_close_browser(self) -> None:
        if self.context is not None:
            await self.context.close()
            self.context = None
            await asyncio.sleep(0.35)
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self.page = None
        self._started = False
        self._on_thread = False

    async def _goto_start_url(self) -> None:
        page = self.page
        assert page is not None
        trace(f"Consensus ▶ goto {CONSENSUS_START_URL}")
        await page.goto(
            CONSENSUS_START_URL, wait_until="domcontentloaded", timeout=90000
        )

    async def _goto_quick_base(self) -> None:
        page = self.page
        assert page is not None
        url = f"{CONSENSUS_QUICK_BASE_URL}/"
        trace(f"Consensus ▶ goto {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)

    def _quick_search_url(self, query: str) -> str:
        q = (query or "").strip()[:14000]
        params: dict[str, str] = {"q": q}
        if CONSENSUS_QUICK_OPEN_ACCESS:
            params["oa"] = "true"
        qs = urlencode(params, quote_via=quote_plus)
        return f"{CONSENSUS_QUICK_BASE_URL}/?{qs}"

    async def _handle_pro_limit_modal_if_present(
        self,
        page: Page,
        *,
        wait_visible_ms: int = 2500,
    ) -> bool:
        """Модал Pro message limit → «Find papers» (basic paper search)."""
        try:
            has_limit = await page.evaluate(
                """() => {
                    const t = (document.body?.innerText || '');
                    return /Pro message limit|No messages left|Use basic paper search/i.test(t);
                }"""
            )
        except Exception:
            has_limit = False
        if not has_limit:
            return False
        try:
            modal = page.locator("[data-testid='composed-modal']")
            await modal.wait_for(state="visible", timeout=wait_visible_ms)
        except Exception:
            pass
        trace("Consensus ▶ modal | Pro limit → Find papers")
        clicked = await page.evaluate(
            """() => {
                const mod = document.querySelector('[data-testid="composed-modal"]');
                const roots = mod ? [mod, document.body] : [document.body];
                for (const root of roots) {
                    const buttons = [...root.querySelectorAll('button')];
                    const btn = buttons.find((b) =>
                        /find papers/i.test((b.innerText || b.textContent || '').trim())
                    );
                    if (btn) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }"""
        )
        if not clicked:
            try:
                loc = page.get_by_role("button", name=re.compile(r"find papers", re.I))
                await loc.first.click(timeout=5000)
                clicked = True
            except Exception as exc:
                trace(f"Consensus ⊘ modal Find papers | {exc}")
                return False
        trace("Consensus ✓ modal | Find papers clicked")
        await asyncio.sleep(1.0)
        return True

    async def _probe_quick_results_state(self, page: Page) -> dict[str, Any]:
        try:
            return await page.evaluate(
                """() => {
                    const body = (document.body?.innerText || '');
                    const links = document.querySelectorAll(
                        'a[href*="doi.org"], a[href*="arxiv.org"], '
                        + 'a[href*="semanticscholar"], a[href*="consensus.app/papers"]'
                    ).length;
                    const loadMore = [...document.querySelectorAll('button')].some((b) =>
                        /load more results/i.test((b.innerText || '').trim())
                    );
                    const proModal = !!document.querySelector('[data-testid="composed-modal"]')
                        && /Pro message limit|Use basic paper search/i.test(body);
                    return {
                        url: location.href,
                        academic_links: links,
                        load_more: loadMore,
                        pro_modal: proModal,
                        body_len: body.length,
                    };
                }"""
            )
        except Exception:
            return {
                "url": page.url or "",
                "academic_links": 0,
                "load_more": False,
                "pro_modal": False,
                "body_len": 0,
            }

    async def _wait_quick_results_surface(self, page: Page) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC
        last_trace = 0.0
        trace("Consensus ▶ wait quick results …")
        while loop.time() < deadline:
            if await self._handle_pro_limit_modal_if_present(
                page, wait_visible_ms=1200
            ):
                await asyncio.sleep(0.8)
            state = await self._probe_quick_results_state(page)
            links = int(state.get("academic_links") or 0)
            load_more = bool(state.get("load_more"))
            if links >= 3 or load_more:
                trace(
                    f"Consensus ✓ quick results | links={links} load_more={load_more} "
                    f"url={str(state.get('url', ''))[:80]}"
                )
                self._on_thread = True
                return
            if bool(state.get("pro_modal")):
                await self._handle_pro_limit_modal_if_present(page)
            now = loop.time()
            if now - last_trace >= 4.0:
                last_trace = now
                trace(
                    f"Consensus ▶ quick results poll | links={links} "
                    f"load_more={load_more} pro_modal={state.get('pro_modal')}"
                )
            await asyncio.sleep(CONSENSUS_UI_POLL_SEC)
        trace(
            f"Consensus ⊘ quick results timeout {CONSENSUS_QUICK_RESULTS_MAX_WAIT_SEC}s"
        )

    async def _click_load_more_results(self, page: Page) -> int:
        clicks = max(0, CONSENSUS_QUICK_LOAD_MORE_CLICKS)
        done = 0
        for i in range(clicks):
            clicked = await page.evaluate(
                """() => {
                    const buttons = [...document.querySelectorAll('button')];
                    const btn = buttons.find((b) =>
                        /load more results/i.test((b.innerText || b.textContent || '').trim())
                    );
                    if (!btn || btn.disabled) return false;
                    btn.scrollIntoView({ block: 'center' });
                    btn.click();
                    return true;
                }"""
            )
            if not clicked:
                if i == 0:
                    trace("Consensus ⊘ Load more results | button not found")
                break
            done += 1
            trace(f"Consensus ✓ Load more results | click={done}/{clicks}")
            await asyncio.sleep(CONSENSUS_PAPER_HARVEST_PAUSE_SEC)
            await self._handle_pro_limit_modal_if_present(page, wait_visible_ms=800)
        if done:
            trace(f"Consensus ✓ Load more done | clicks={done}")
        return done

    async def _collect_quick_page_text(self, page: Page) -> str:
        try:
            return str(
                await page.evaluate(
                    """() => {
                        const main = document.querySelector('main') || document.body;
                        return (main?.innerText || '').slice(0, 14000);
                    }"""
                )
            ).strip()
        except Exception:
            return ""

    async def _send_direct_api_once(self, prompt_text: str) -> ConsensusMessageResult:
        """Hybrid Direct API: curl_cffi + Playwright warmup (без DOM/кликов)."""
        from knowledge_engine.services.search.consensus_direct_client import (
            acquire_consensus_direct_client,
            papers_to_raw_text,
        )

        q = (prompt_text or "").strip()
        trace(f"Consensus ▶ direct API search | q={q[:120]}")
        client = await acquire_consensus_direct_client()
        papers = await client.search_papers(q, limit=20)
        text = papers_to_raw_text(papers, q)
        self._api_papers = list(papers)
        from knowledge_engine.ui.llm_trace import trace_plain_io

        trace_plain_io("Consensus (direct API)", q, text[:8000])
        trace(f"Consensus ✓ direct API | papers={len(papers)} text={len(text)} sym")
        return ConsensusMessageResult(raw_text=text, papers=papers)

    async def _send_quick_paper_search_once(
        self, prompt_text: str
    ) -> ConsensusMessageResult:
        if CONSENSUS_USE_DIRECT_API:
            return await self._send_direct_api_once(prompt_text)
        page = self.page
        assert page is not None
        q = (prompt_text or "").strip()
        url = self._quick_search_url(q)
        trace(f"Consensus ▶ quick paper search | {url[:140]}")
        if await self._detect_login_wall(page):
            raise ConsensusLoginRequiredError("login wall before quick search")
        self._api_papers = []
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        await asyncio.sleep(0.5)
        await self._handle_pro_limit_modal_if_present(page)
        await self._wait_quick_results_surface(page)
        await self._click_load_more_results(page)
        await self.harvest_paper_pool()
        dom_papers = await extract_paper_cards_from_page(page)
        text = await self._collect_quick_page_text(page)
        text_papers = extract_papers_from_text(text)
        papers = merge_scholar_papers(
            merge_scholar_papers(self._api_papers, dom_papers),
            text_papers,
        )
        from knowledge_engine.ui.llm_trace import trace_plain_io

        trace_plain_io("Consensus (quick paper search)", q, text[:8000])
        trace(f"Consensus ✓ quick search | text={len(text)} sym | papers={len(papers)}")
        return ConsensusMessageResult(raw_text=text, papers=papers)

    async def _detect_login_wall(self, page: Page) -> bool:
        url = (page.url or "").lower()
        if any(x in url for x in ("login", "signin", "sign-in", "accounts.google")):
            trace("Consensus ⊘ login wall | URL")
            return True
        try:
            blocked = await page.evaluate(
                """() => {
                    const body = (document.body?.innerText || '').toLowerCase();
                    const signIn = body.includes('sign in') || body.includes('sign up')
                        || body.includes('войти') || body.includes('log in');
                    const input = document.querySelector(
                        '[data-testid="new-thread-input"], textarea[data-testid="new-thread-input"]'
                    );
                    const searchBtn = document.querySelector('[data-testid="search-button"]');
                    const modal = document.querySelector(
                        '[role="dialog"], [data-testid*="login"], [data-testid*="auth"]'
                    );
                    if (modal && signIn) return true;
                    if (signIn && input && !searchBtn) return true;
                    return false;
                }"""
            )
            if blocked:
                trace("Consensus ⊘ login wall | UI (Sign in без поиска)")
            return bool(blocked)
        except Exception:
            return False

    async def _probe_surface_state(self, page: Page) -> dict[str, Any]:
        try:
            return await page.evaluate(
                """() => {
                    const ni = document.querySelector(
                        'textarea[data-testid="new-thread-input"], [data-testid="new-thread-input"]'
                    );
                    const si = document.querySelector(
                        'textarea[data-testid="search-input"], [data-testid="search-input"]'
                    );
                    const vis = (el) => !!(el && (el.offsetParent || el.getClientRects().length));
                    return {
                        url: location.href,
                        new_thread_input: vis(ni),
                        search_input: vis(si),
                    };
                }"""
            )
        except Exception:
            return {
                "url": page.url or "",
                "new_thread_input": False,
                "search_input": False,
            }

    async def _wait_usable_input_surface(
        self,
        page: Page,
        *,
        max_sec: float | None = None,
        label: str = "input",
    ) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (
            max_sec if max_sec is not None else CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC
        )
        last_trace = 0.0
        while loop.time() < deadline:
            state = await self._probe_surface_state(page)
            if state.get("new_thread_input") or state.get("search_input"):
                trace(
                    f"Consensus ✓ {label} | "
                    f"new_thread={state.get('new_thread_input')} "
                    f"search={state.get('search_input')}"
                )
                return True
            now = loop.time()
            if now - last_trace >= 5.0:
                last_trace = now
                trace(
                    f"Consensus ▶ {label} poll | url={str(state.get('url', ''))[:85]} "
                    f"new_thread={state.get('new_thread_input')} "
                    f"search={state.get('search_input')}"
                )
            await asyncio.sleep(CONSENSUS_UI_POLL_SEC)
        trace(
            f"Consensus ⊘ {label} | timeout {max_sec or CONSENSUS_NEW_DIALOG_MAX_WAIT_SEC}s"
        )
        return False

    async def _wait_new_dialog_input(self) -> None:
        page = self.page
        assert page is not None
        if await self._wait_usable_input_surface(page, label="new dialog input"):
            return
        await self._wait_input_ready()

    async def _open_new_thread_ui(self) -> None:
        page = self.page
        if page is None:
            return
        selectors = (
            "button[data-testid='new-thread-button']",
            "[data-testid='new-thread-button']",
            "button[data-testid='new-thread']",
            "[data-testid='new-thread']",
            "a[href='/home']",
            "a[href='https://consensus.app/home']",
        )
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                await loc.wait_for(state="visible", timeout=1200)
                await loc.click(timeout=4000)
                trace(f"Consensus ✓ new thread | {sel}")
                await asyncio.sleep(0.35)
                self._on_thread = False
                return
            except Exception:
                continue
        try:
            clicked = await page.evaluate(
                """() => {
                    const nodes = [...document.querySelectorAll('button, a, [role="button"]')];
                    const hit = nodes.find((n) =>
                        /new\\s*thread/i.test((n.innerText || n.textContent || '').trim())
                    );
                    if (hit) {
                        hit.click();
                        return true;
                    }
                    return false;
                }"""
            )
            if clicked:
                trace("Consensus ✓ new thread | sidebar text")
                await asyncio.sleep(0.45)
                self._on_thread = False
                return
        except Exception:
            pass
        trace("Consensus ▶ new thread fallback | goto home")
        await self._goto_start_url()
        self._on_thread = False

    async def _open_new_dialog(self) -> None:
        """Новый пользовательский запрос — отдельный чат (RETRY остаётся в текущем треде)."""
        page = self.page
        if page is None:
            return
        url = page.url or ""
        on_thread = self._on_thread or _RESULTS_URL_RE.search(url) or "/threads/" in url

        if not on_thread:
            if await self._wait_usable_input_surface(
                page, max_sec=3.0, label="home input"
            ):
                self._on_thread = False
                return

        if on_thread:
            trace("Consensus ▶ new dialog | leave thread → /home")
            try:
                await self._goto_start_url()
            except Exception as exc:
                trace(f"Consensus ⊘ goto home | {exc}")
            if await self._wait_usable_input_surface(
                page, label="home after leave thread"
            ):
                self._on_thread = False
                return

        trace("Consensus ▶ new dialog | new-thread UI fallback")
        await self._open_new_thread_ui()
        if await self._wait_usable_input_surface(page, label="input after new-thread"):
            self._on_thread = False
            return
        await self._wait_input_ready()
        self._on_thread = False

    async def harvest_paper_pool(self) -> list[ScholarPaper]:
        """Прокрутка результатов + DOM/API capture для большого пула перед Lite-отбором."""
        page = self.page
        if page is None:
            return list(self._api_papers)
        from knowledge_engine.config import CURRICULUM_V08_PAPER_POOL_SIZE

        target = max(15, min(CURRICULUM_V08_PAPER_POOL_SIZE, 100))
        passes = max(1, CONSENSUS_PAPER_HARVEST_PASSES)
        trace(f"Consensus ▶ paper pool | passes={passes}")
        last_logged = len(self._api_papers)
        for i in range(passes):
            if len(self._api_papers) >= target:
                trace(
                    f"Consensus ✓ paper pool early | papers={len(self._api_papers)} "
                    f">= target={target}"
                )
                break
            try:
                await page.evaluate(
                    "() => window.scrollBy(0, Math.min(1400, window.innerHeight || 800))"
                )
            except Exception:
                pass
            dom = await extract_paper_cards_from_page(page)
            if dom:
                self._api_papers = merge_scholar_papers(self._api_papers, dom)
                if len(self._api_papers) != last_logged:
                    trace(f"Consensus ✓ DOM papers={len(self._api_papers)}")
                    last_logged = len(self._api_papers)
            if i < passes - 1:
                await asyncio.sleep(CONSENSUS_PAPER_HARVEST_PAUSE_SEC)
        trace(f"Consensus ✓ paper pool | papers={len(self._api_papers)}")
        return list(self._api_papers)

    async def _soft_auth_recovery(self) -> None:
        """Сохранить cookies/profile: home + новый чат без перезапуска браузера."""
        trace("Consensus ▶ auth recovery soft | goto + new thread")
        if self.page is None:
            return
        try:
            await self._goto_start_url()
        except Exception as exc:
            trace(f"Consensus ⊘ soft recovery goto | {exc}")
        await self._open_new_thread_ui()
        try:
            await self._wait_input_ready()
        except Exception as exc:
            trace(f"Consensus ⊘ soft recovery input | {exc}")

    async def _hard_auth_recovery(self, cycle: int) -> None:
        """Перезапуск Playwright с тем же user_data_dir (локальная сессия)."""
        trace(
            f"Consensus ▶ auth recovery hard | cycle {cycle}/{CONSENSUS_AUTH_RECOVERY_CYCLES}"
        )
        await self._hard_close_browser()
        await asyncio.sleep(0.5)
        await self._bootstrap_browser()
        if await self._detect_login_wall(self.page):
            trace("Consensus ⊘ login wall после hard recovery bootstrap")

    async def _auth_recovery_cycle(self, cycle: int) -> None:
        """Сначала soft (reuse session), затем hard restart с тем же profile."""
        if cycle <= 1:
            await self._soft_auth_recovery()
            if self.page is not None and not await self._detect_login_wall(self.page):
                return
        await self._hard_auth_recovery(cycle)

    async def _bootstrap_browser(self) -> None:
        if CONSENSUS_USE_DIRECT_API and CONSENSUS_USE_QUICK_PAPER_SEARCH:
            trace(
                "Consensus ✓ bootstrap | direct API mode — "
                "Playwright warmup inside ConsensusDirectClient"
            )
            self._started = True
            self._on_thread = False
            return
        trace(
            f"Consensus ▶ Playwright bootstrap | profile={BROWSER_PROFILE_PATH} "
            f"| headless={self.headless}"
        )
        self._playwright = await async_playwright().start()
        if self.record_har_path:
            trace(f"Consensus ▶ HAR record | {self.record_har_path}")
        self.context = await launch_persistent_context_async(
            self._playwright,
            headless=self.headless,
            record_har_path=self.record_har_path,
        )
        self.page = await self._pick_work_page()
        self._wire_network_capture(self.page)
        if CONSENSUS_USE_QUICK_PAPER_SEARCH:
            trace(
                "Consensus ✓ bootstrap | quick mode — страница откроется на send "
                "(/?q=&oa=true), без goto /quick/"
            )
            self._started = True
            self._on_thread = False
            trace(f"Consensus ✓ bootstrap | url={self.page.url[:80]}")
            return
        await self._goto_start_url()
        await self._ensure_search_surface_ready()
        self._started = True
        self._on_thread = _RESULTS_URL_RE.search(self.page.url or "") is not None
        trace(f"Consensus ✓ bootstrap | url={self.page.url[:80]}")

    async def _start_with_auth_recovery(self) -> None:
        if CONSENSUS_USE_DIRECT_API and CONSENSUS_USE_QUICK_PAPER_SEARCH:
            if not self._started:
                await self._bootstrap_browser()
            return
        if self._started and self._page_is_alive():
            return
        if self._started and not self._page_is_alive():
            trace("Consensus ⊘ page closed — re-bootstrap с сохранённым profile")
            self._started = False
            self.page = None
        last_err: Exception | None = None
        for cycle in range(CONSENSUS_AUTH_RECOVERY_CYCLES + 1):
            try:
                await self._bootstrap_browser()
                if await self._detect_login_wall(self.page):
                    raise ConsensusLoginRequiredError("login wall on start")
                return
            except ConsensusLoginRequiredError as exc:
                last_err = exc
                if cycle >= CONSENSUS_AUTH_RECOVERY_CYCLES:
                    break
                await self._auth_recovery_cycle(cycle + 1)
        raise ConsensusLoginRequiredError(
            "Consensus: требуется вход — recovery "
            f"исчерпан ({CONSENSUS_AUTH_RECOVERY_CYCLES} циклов). "
            "Один раз: остановите API и выполните "
            "`python -m knowledge_engine.main consensus-login` "
            f"(профиль {BROWSER_PROFILE_PATH}). "
            "browser-login открывает Gemini, не Consensus."
        ) from last_err

    async def start(self) -> None:
        await self._start_with_auth_recovery()

    async def begin_new_run(self) -> None:
        """Новый анализ: новый диалог Consensus, тот же локальный browser profile."""
        if not CONSENSUS_NEW_THREAD_EACH_RUN:
            self._api_papers = []
            return
        self._api_papers = []
        if not self._started or self.page is None:
            return
        if CONSENSUS_USE_QUICK_PAPER_SEARCH:
            trace("Consensus ▶ begin new run (quick — новый URL на send)")
            return
        trace("Consensus ▶ begin new run (new dialog)")
        await self._open_new_dialog()

    async def release_for_next_run(self) -> None:
        """Не закрывать браузер — следующий прогон reuse session + begin_new_run."""
        trace("Consensus ✓ session released (browser kept for reuse)")

    async def _send_message_once(self, prompt_text: str) -> ConsensusMessageResult:
        if CONSENSUS_USE_QUICK_PAPER_SEARCH:
            return await self._send_quick_paper_search_once(prompt_text)
        page = self.page
        assert page is not None
        trace(f"Consensus ▶ send | {len(prompt_text)} sym")
        if await self._detect_login_wall(page):
            raise ConsensusLoginRequiredError("login wall before send")
        form = page.locator("[data-testid='search-input-form']")
        try:
            await form.wait_for(state="visible", timeout=15000)
        except Exception:
            pass
        input_box = await self._find_input(page)
        await self._fill_input(page, input_box, prompt_text[:14_000])
        await asyncio.sleep(0.25)
        await self._submit(page)
        await asyncio.sleep(0.6)
        text = await self._wait_for_response(page)
        if await self._detect_login_wall(page) and self._is_landing(page):
            raise ConsensusLoginRequiredError("login wall after submit (stuck on home)")
        await self.harvest_paper_pool()
        dom_papers = await extract_paper_cards_from_page(page)
        text_papers = extract_papers_from_text(text)
        papers = merge_scholar_papers(
            merge_scholar_papers(self._api_papers, dom_papers),
            text_papers,
        )
        from knowledge_engine.ui.llm_trace import trace_plain_io

        trace_plain_io(
            "Consensus (Playwright UI)",
            prompt_text,
            text,
        )
        trace(f"Consensus ✓ response | {len(text)} sym | papers={len(papers)}")
        return ConsensusMessageResult(raw_text=text, papers=papers)

    async def send_message(self, prompt_text: str) -> ConsensusMessageResult:
        """Отправить сообщение; при login wall — recovery без входа (до N циклов)."""
        if CONSENSUS_USE_DIRECT_API and CONSENSUS_USE_QUICK_PAPER_SEARCH:
            if not self._started:
                await self._start_with_auth_recovery()
            try:
                return await self._send_direct_api_once(prompt_text)
            except Exception as exc:
                msg = str(exc).lower()
                if "login" in msg or "__session" in msg or "consensus-login" in msg:
                    raise ConsensusLoginRequiredError(str(exc)) from exc
                raise

        last_err: Exception | None = None
        for cycle in range(CONSENSUS_AUTH_RECOVERY_CYCLES + 1):
            try:
                if not self._started:
                    await self._start_with_auth_recovery()
                return await self._send_message_once(prompt_text)
            except ConsensusLoginRequiredError as exc:
                last_err = exc
                trace(f"Consensus ⊘ login required | recovery pending | {exc}")
                if cycle >= CONSENSUS_AUTH_RECOVERY_CYCLES:
                    break
                await self._auth_recovery_cycle(cycle + 1)
        raise ConsensusLoginRequiredError(
            "Consensus: не удалось продолжить без входа после "
            f"{CONSENSUS_AUTH_RECOVERY_CYCLES} recovery-циклов "
            "(soft: goto+new thread → hard: restart browser с тем же profile)."
        ) from last_err

    def _is_landing(self, page: Page) -> bool:
        try:
            path = page.url.split("consensus.app", 1)[-1].split("?", 1)[0]
        except Exception:
            path = page.url
        return bool(_LANDING_PATH_RE.match(path or "/"))

    async def _dump_ui_probe(self, page: Page, label: str) -> None:
        try:
            data = await page.evaluate(
                """() => {
                    const testids = [...document.querySelectorAll('[data-testid]')]
                        .map((e) => e.getAttribute('data-testid'))
                        .filter(Boolean);
                    const inputs = [...document.querySelectorAll('textarea, input[type="text"], [contenteditable="true"]')]
                        .slice(0, 12)
                        .map((e) => ({
                            tag: e.tagName,
                            testid: e.getAttribute('data-testid') || '',
                            visible: !!(e.offsetParent || e.getClientRects().length),
                            disabled: !!e.disabled,
                        }));
                    const buttons = [...document.querySelectorAll('button')]
                        .slice(0, 20)
                        .map((b) => ({
                            testid: b.getAttribute('data-testid') || '',
                            aria: b.getAttribute('aria-label') || '',
                            disabled: b.disabled,
                            visible: !!(b.offsetParent || b.getClientRects().length),
                        }));
                    return {
                        url: location.href,
                        title: document.title,
                        body_snippet: (document.body?.innerText || '').slice(0, 400),
                        testids: testids.slice(0, 80),
                        inputs,
                        buttons,
                    };
                }"""
            )
            trace(
                f"Consensus probe {label} | url={str(data.get('url', ''))[:100]} | "
                f"testids={len(data.get('testids') or [])}"
            )
            for tid in (data.get("testids") or [])[:25]:
                trace(f"Consensus probe testid | {tid}")
            for inp in (data.get("inputs") or [])[:6]:
                trace(f"Consensus probe input | {inp}")
            for btn in (data.get("buttons") or [])[:8]:
                if btn.get("testid") or "search" in (btn.get("aria") or "").lower():
                    trace(f"Consensus probe button | {btn}")
        except Exception as exc:
            trace(f"Consensus probe failed | {label} | {exc}")

    async def _ensure_search_surface_ready(self) -> None:
        """Home / new-thread: поле ввода и форма поиска (после SPA hydrate)."""
        page = self.page
        assert page is not None
        try:
            ni_ms = 15000 if self.headless else 45000
            await page.wait_for_load_state("networkidle", timeout=ni_ms)
        except Exception:
            trace("Consensus ⊘ networkidle timeout — продолжаем с domcontentloaded")
        if await self._detect_login_wall(page):
            raise ConsensusLoginRequiredError("login wall before input ready")
        per_sel = max(8.0, CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC / 4.0)
        selectors = [
            s.strip() for s in CONSENSUS_INPUT_SELECTOR.split(",") if s.strip()
        ]
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=int(per_sel * 1000))
                trace(f"Consensus ✓ input ready | {sel}")
                return
            except Exception:
                trace(f"Consensus ⊘ input wait | {sel}")
        trace("Consensus ▶ input missing — try new thread UI")
        await self._open_new_thread_ui()
        await self._wait_new_dialog_input()
        trace("Consensus ✓ input after new-thread flow")

    async def _wait_input_ready(self) -> None:
        page = self.page
        assert page is not None
        selectors = [
            s.strip() for s in CONSENSUS_INPUT_SELECTOR.split(",") if s.strip()
        ]
        per_sel = max(
            8.0, CONSENSUS_BOOTSTRAP_INPUT_TIMEOUT_SEC / max(1, len(selectors))
        )
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=int(per_sel * 1000))
                return
            except Exception:
                continue
        await self._dump_ui_probe(page, "input_not_found")
        raise RuntimeError(
            "Consensus: поле ввода не найдено. Задайте CONSENSUS_INPUT_SELECTOR "
            "или один раз войдите через consensus-login на consensus.app."
        )

    async def _find_input(self, page: Page):
        if not self._on_thread:
            for sel in (
                "textarea[data-testid='new-thread-input']",
                "[data-testid='new-thread-input']",
            ):
                try:
                    el = await page.query_selector(sel)
                    if el:
                        return el
                except Exception:
                    continue
        for sel in [
            s.strip() for s in CONSENSUS_INPUT_SELECTOR.split(",") if s.strip()
        ]:
            els = await page.query_selector_all(sel)
            if not els:
                continue
            # На треде — обычно последнее поле; на home — первое
            return els[-1] if self._on_thread and len(els) > 1 else els[0]
        return await page.wait_for_selector(
            CONSENSUS_INPUT_SELECTOR.split(",")[0], timeout=15000
        )

    async def _fill_input(self, page: Page, input_box, prompt: str) -> None:
        await input_box.scroll_into_view_if_needed()
        await input_box.click(force=True)
        await asyncio.sleep(0.2)
        text = prompt[:14_000]
        try:
            await input_box.fill(text)
        except Exception:
            pass
        try:
            await input_box.evaluate(
                """(el, t) => {
                    el.focus();
                    if (el.isContentEditable) {
                        el.innerText = t;
                    } else {
                        el.value = t;
                    }
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                text,
            )
        except Exception:
            await page.keyboard.insert_text(text)
        await asyncio.sleep(0.35)
        # React: убедиться, что value в textarea совпадает
        try:
            current = await input_box.evaluate("(el) => el.value || el.innerText || ''")
            if len(str(current).strip()) < min(20, len(text.strip()) // 2):
                trace("Consensus ⊘ textarea empty after fill — retry insert_text")
                await input_box.click(force=True)
                await page.keyboard.insert_text(text)
        except Exception:
            pass

    async def _click_submit_near_input(self, page: Page) -> bool:
        """Кнопка Submit search (data-testid=search-button) в форме поиска."""
        try:
            btn = page.locator("button[data-testid='search-button']").first
            await btn.wait_for(state="visible", timeout=12000)
            for _ in range(40):
                try:
                    if not await btn.is_disabled():
                        break
                except Exception:
                    break
                await asyncio.sleep(0.25)
            await btn.click(timeout=12000)
            trace("Consensus ✓ submit | data-testid=search-button")
            return True
        except Exception as exc:
            trace(f"Consensus ⊘ search-button click | {exc}")
            return False

    async def _submit(self, page: Page) -> None:
        trace("Consensus ▶ click submit")
        if await self._click_submit_near_input(page):
            return

        for sel in CONSENSUS_SEND_SELECTORS:
            if "search-button" in sel:
                continue
            try:
                loc = page.locator(sel).first
                await loc.wait_for(state="visible", timeout=3000)
                await loc.click(timeout=5000)
                trace(f"Consensus ✓ submit via {sel[:50]}")
                return
            except Exception:
                continue

        try:
            submitted = await page.evaluate(
                """() => {
                    const form = document.querySelector('[data-testid="search-input-form"] form')
                        || document.querySelector('form');
                    if (form && typeof form.requestSubmit === 'function') {
                        form.requestSubmit();
                        return true;
                    }
                    if (form) {
                        form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                        return true;
                    }
                    return false;
                }"""
            )
            if submitted:
                trace("Consensus ✓ submit via form.requestSubmit")
                return
        except Exception:
            pass

        await page.keyboard.press("Enter")
        trace("Consensus ⊘ submit fallback Enter")

    async def _wait_leave_landing(self, page: Page, timeout_sec: float = 90.0) -> None:
        if not self._is_landing(page):
            self._on_thread = True
            return
        trace("Consensus ▶ wait navigation off /home …")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        while loop.time() < deadline:
            if not self._is_landing(page) or _RESULTS_URL_RE.search(page.url):
                self._on_thread = True
                trace(f"Consensus ✓ results URL | {page.url[:90]}")
                return
            await asyncio.sleep(0.5)
        trace(
            "Consensus ⊘ still on landing after submit — проверьте кнопку поиска в UI"
        )

    async def _generating(self, page: Page) -> bool:
        markers = (
            "[data-testid*='loading']",
            "[aria-busy='true']",
            ".animate-spin",
            "[class*='loading']",
        )
        for sel in markers:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    return True
            except Exception:
                continue
        return False

    async def _read_answer_blocks(self, page: Page) -> list[str]:
        texts: list[str] = []
        if self._is_landing(page):
            return []

        for sel in [
            s.strip() for s in CONSENSUS_RESPONSE_SELECTOR.split(",") if s.strip()
        ]:
            try:
                nodes = await page.query_selector_all(sel)
                for n in nodes:
                    t = (await n.inner_text()).strip()
                    if t and len(t) > 80:
                        texts.append(t)
            except Exception:
                continue

        if not texts:
            try:
                main = await page.query_selector("main")
                if main:
                    t = (await main.inner_text()).strip()
                    if len(t) > 120:
                        texts.append(t)
            except Exception:
                pass
        return texts

    async def _wait_for_response(self, page: Page) -> str:
        await self._wait_leave_landing(page)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + CONSENSUS_RESPONSE_MAX_SEC
        first_deadline = loop.time() + CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC

        trace("Consensus ▶ wait answer stream …")
        while loop.time() < first_deadline:
            if await self._generating(page):
                await asyncio.sleep(CONSENSUS_STREAM_POLL_SEC)
                continue
            blocks = await self._read_answer_blocks(page)
            if blocks and len(blocks[-1]) >= 80:
                break
            await asyncio.sleep(CONSENSUS_STREAM_POLL_SEC)

        stable = 0
        last_len = 0
        while loop.time() < deadline:
            if await self._generating(page):
                stable = 0
                await asyncio.sleep(CONSENSUS_STREAM_POLL_SEC)
                continue
            blocks = await self._read_answer_blocks(page)
            if not blocks:
                await asyncio.sleep(CONSENSUS_STREAM_POLL_SEC)
                continue
            cur = blocks[-1]
            cur_len = len(cur)
            if cur_len == last_len and cur_len >= CONSENSUS_MIN_RESPONSE_CHARS:
                stable += 1
                if stable >= CONSENSUS_STREAM_STABLE_ROUNDS:
                    return cur
            else:
                stable = 0
                last_len = cur_len
            await asyncio.sleep(CONSENSUS_STREAM_POLL_SEC)

        blocks = await self._read_answer_blocks(page)
        if blocks:
            return blocks[-1]
        if self._is_landing(page):
            if await self._detect_login_wall(page):
                raise ConsensusLoginRequiredError("stuck on home with login wall")
            raise RuntimeError(
                "Consensus: поиск не запустился — страница осталась на /home. "
                "Проверьте CONSENSUS_SEND_SELECTORS."
            )
        return ""

    def extract_urls(self, text: str) -> list[str]:
        urls = _URL_RE.findall(text or "")
        out: list[str] = []
        for u in urls:
            u = u.rstrip(".,);]")
            if u not in out:
                out.append(u)
        return out

    async def close(self) -> None:
        await self._hard_close_browser()
        self._api_papers = []
        trace("Consensus ✓ session closed")
