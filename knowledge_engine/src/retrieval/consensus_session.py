"""Stateful Playwright session with Consensus.app: локальный persistent profile, новый чат на прогон."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from knowledge_engine.config import (
    BROWSER_PROFILE_PATH,
    CONSENSUS_AUTH_RECOVERY_CYCLES,
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_INPUT_SELECTOR,
    CONSENSUS_MIN_RESPONSE_CHARS,
    CONSENSUS_NEW_THREAD_EACH_RUN,
    CONSENSUS_RESPONSE_FIRST_TIMEOUT_SEC,
    CONSENSUS_RESPONSE_MAX_SEC,
    CONSENSUS_RESPONSE_SELECTOR,
    CONSENSUS_REUSE_BROWSER_SESSION,
    CONSENSUS_SEND_SELECTORS,
    CONSENSUS_START_URL,
    CONSENSUS_STREAM_POLL_SEC,
    CONSENSUS_STREAM_STABLE_ROUNDS,
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
from knowledge_engine.ui.run_log import trace

_URL_RE = re.compile(r"https?://[^\s\]<\"')]+")
_RESULTS_URL_RE = re.compile(
    r"consensus\.app/(search|threads|thread|p|chat|results)(/|$)",
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
    """После прогона: оставить браузер живым или полностью закрыть."""
    global _shared_session
    async with _shared_session_lock:
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
        if _shared_session is None:
            return
        trace("Consensus ▶ shutdown | сохранение profile …")
        await _shared_session.close()
        _shared_session = None


class ConsensusSessionManager:
    """Держит живую страницу Consensus для первичного и уточняющих запросов (RETRY)."""

    def __init__(self, headless: bool = CONSENSUS_BROWSER_HEADLESS) -> None:
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._started = False
        self._on_thread = False
        self._api_papers: list[ScholarPaper] = []

    async def _on_network_response(self, response) -> None:
        try:
            if response.status != 200:
                return
            url = response.url or ""
            if "consensus" not in url:
                return
            ct = (response.headers.get("content-type") or "").lower()
            if "json" not in ct and not url.rstrip("/").endswith(".json"):
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
                )
            ):
                return
            body = await response.text()
            if len(body) < 40:
                return
            found = papers_from_json_text_relaxed(body)
            if found:
                self._api_papers = merge_scholar_papers(self._api_papers, found)
                trace(
                    f"Consensus ✓ API capture | +{len(found)} papers | total={len(self._api_papers)}"
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

    async def _wait_new_dialog_input(self) -> None:
        page = self.page
        assert page is not None
        selectors = (
            "textarea[data-testid='new-thread-input']",
            "[data-testid='new-thread-input']",
        )
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=12000)
                trace("Consensus ✓ new dialog | new-thread-input")
                return
            except Exception:
                continue
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
                await loc.wait_for(state="visible", timeout=2500)
                await loc.click(timeout=8000)
                trace(f"Consensus ✓ new thread | {sel}")
                await asyncio.sleep(0.45)
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
        if on_thread:
            trace("Consensus ▶ new dialog | leave thread")
        await self._open_new_thread_ui()
        await self._wait_new_dialog_input()
        self._on_thread = False

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
        trace(
            f"Consensus ▶ Playwright bootstrap | profile={BROWSER_PROFILE_PATH} "
            f"| headless={self.headless}"
        )
        self._playwright = await async_playwright().start()
        self.context = await launch_persistent_context_async(
            self._playwright, headless=self.headless
        )
        self.page = await self._pick_work_page()
        self._wire_network_capture(self.page)
        await self._goto_start_url()
        await self._wait_input_ready()
        self._started = True
        self._on_thread = _RESULTS_URL_RE.search(self.page.url or "") is not None
        trace(f"Consensus ✓ bootstrap | url={self.page.url[:80]}")

    async def _start_with_auth_recovery(self) -> None:
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
        trace("Consensus ▶ begin new run (new dialog)")
        await self._open_new_dialog()

    async def release_for_next_run(self) -> None:
        """Не закрывать браузер — следующий прогон reuse session + begin_new_run."""
        trace("Consensus ✓ session released (browser kept for reuse)")

    async def _send_message_once(self, prompt_text: str) -> ConsensusMessageResult:
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
        await self._submit(page)
        await asyncio.sleep(0.4)
        text = await self._wait_for_response(page)
        if await self._detect_login_wall(page) and self._is_landing(page):
            raise ConsensusLoginRequiredError("login wall after submit (stuck on home)")
        dom_papers = await extract_paper_cards_from_page(page)
        text_papers = extract_papers_from_text(text)
        papers = merge_scholar_papers(
            merge_scholar_papers(self._api_papers, dom_papers),
            text_papers,
        )
        trace(f"Consensus ✓ response | {len(text)} sym | papers={len(papers)}")
        return ConsensusMessageResult(raw_text=text, papers=papers)

    async def send_message(self, prompt_text: str) -> ConsensusMessageResult:
        """Отправить сообщение; при login wall — recovery без входа (до N циклов)."""
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

    async def _wait_input_ready(self) -> None:
        page = self.page
        assert page is not None
        selectors = [
            s.strip() for s in CONSENSUS_INPUT_SELECTOR.split(",") if s.strip()
        ]
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=25000)
                return
            except Exception:
                continue
        raise RuntimeError(
            "Consensus: поле ввода не найдено. Задайте CONSENSUS_INPUT_SELECTOR "
            "или один раз войдите через browser-login на consensus.app."
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
            await btn.wait_for(state="visible", timeout=8000)
            await btn.click(timeout=10000)
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
