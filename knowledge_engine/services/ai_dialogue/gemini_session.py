"""Gemini через Playwright (async + sync-обёртка для LangGraph)."""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from knowledge_engine.config import (
    AI_CHAT_PROVIDER_NAME,
    AI_CHAT_START_URL,
    GEMINI_BROWSER_HEADLESS,
    GEMINI_INPUT_SELECTOR,
    GEMINI_MIN_RESPONSE_CHARS,
    GEMINI_PAYLOAD_MAX_CHARS,
    GEMINI_RESPONSE_FIRST_TIMEOUT_SEC,
    GEMINI_RESPONSE_MAX_SEC,
    GEMINI_RESPONSE_SELECTOR,
    GEMINI_SEND_SELECTORS,
    GEMINI_STREAM_POLL_SEC,
    GEMINI_STREAM_STABLE_ROUNDS,
)
from knowledge_engine.services.ai_dialogue.base import (
    BaseAIDialogueSession,
    DialogueTurn,
)
from knowledge_engine.services.search.playwright_launch import (
    launch_persistent_context_async,
)
from knowledge_engine.ui.logger import set_status

_URL_RE = re.compile(r"https?://[^\s\]<\"')]+")


class BrowserGeminiSession:
    """Persistent context → gemini.google.com/app."""

    def __init__(self, headless: bool = GEMINI_BROWSER_HEADLESS) -> None:
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def init_session(self) -> None:
        self._playwright = await async_playwright().start()
        self.context = await launch_persistent_context_async(
            self._playwright, headless=self.headless
        )
        self.page = await self.context.new_page()
        await self.page.goto(
            AI_CHAT_START_URL, wait_until="domcontentloaded", timeout=90000
        )
        await self._ensure_chat_ready()

    async def _ensure_chat_ready(self) -> None:
        """Чат доступен (гость или уже залогиненный persistent profile)."""
        page = self.page
        assert page is not None
        try:
            await page.wait_for_selector(GEMINI_INPUT_SELECTOR, timeout=20000)
            return
        except Exception:
            pass
        url = page.url
        try:
            snippet = (await page.inner_text("body"))[:2500]
        except Exception:
            snippet = ""
        if "accounts.google.com" in url or "Sign in" in snippet or "Войти" in snippet:
            raise RuntimeError(
                "Gemini: открылась страница входа Google. Один раз: "
                "python -m knowledge_engine.main browser-login — войти в Google "
                "или начать чат без аккаунта, если интерфейс позволяет. "
                "Сессия сохранится в .browser_state/<PLAYWRIGHT_BROWSER>/."
            )
        raise RuntimeError(
            "Gemini: поле ввода не найдено. Один раз откройте browser-login, "
            "дойдите до экрана чата (гость или Google), нажмите Enter."
        )

    async def _generating_visible(self, page: Page) -> bool:
        selectors = [
            "button[aria-label*='Stop']",
            "button[aria-label*='Останов']",
            "[data-test-id*='stop']",
            "mat-progress-bar",
        ]
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    return True
            except Exception:
                continue
        return False

    async def _read_last_response_text(self, page: Page) -> str:
        responses = await page.query_selector_all(GEMINI_RESPONSE_SELECTOR)
        if not responses:
            return ""
        last_response = responses[-1]
        return (await last_response.inner_text()).strip()

    async def _wait_for_complete_response(
        self, page: Page, min_response_chars: int = GEMINI_MIN_RESPONSE_CHARS
    ) -> str:
        """Ждать завершения стриминга Gemini (не снимать текст на первом чанке)."""
        set_status(f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] жду начало ответа…")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GEMINI_RESPONSE_MAX_SEC
        first_deadline = loop.time() + GEMINI_RESPONSE_FIRST_TIMEOUT_SEC

        while loop.time() < first_deadline:
            text = await self._read_last_response_text(page)
            if len(text) >= 20:
                break
            await asyncio.sleep(GEMINI_STREAM_POLL_SEC)
        else:
            raise RuntimeError(
                "Gemini: нет ответа в течение "
                f"{GEMINI_RESPONSE_FIRST_TIMEOUT_SEC}s — проверьте чат в браузере."
            )

        stable_rounds = 0
        last_len = 0
        while loop.time() < deadline:
            if await self._generating_visible(page):
                stable_rounds = 0
                text = await self._read_last_response_text(page)
                set_status(
                    f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] генерация… {len(text)} симв."
                )
                await asyncio.sleep(GEMINI_STREAM_POLL_SEC)
                continue

            text = await self._read_last_response_text(page)
            cur_len = len(text)
            if cur_len > 0 and cur_len == last_len:
                stable_rounds += 1
                if stable_rounds >= GEMINI_STREAM_STABLE_ROUNDS:
                    set_status(
                        f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] ответ готов ({cur_len} симв.)"
                    )
                    return text
            else:
                stable_rounds = 0
                last_len = cur_len
                if cur_len > 0:
                    set_status(
                        f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] генерация… {cur_len} симв."
                    )
            await asyncio.sleep(GEMINI_STREAM_POLL_SEC)

        text = await self._read_last_response_text(page)
        set_status(
            f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] таймаут {GEMINI_RESPONSE_MAX_SEC}s, "
            f"снимаю {len(text)} симв."
        )
        return text

    async def ask_gemini(self, payload: str) -> str:
        """Только подготовленный Sandwich payload (не сырой HTML)."""
        if len(payload) > GEMINI_PAYLOAD_MAX_CHARS:
            payload = payload[:GEMINI_PAYLOAD_MAX_CHARS]
        lowered = payload.lower()
        if "<html" in lowered or "<body" in lowered:
            raise ValueError(
                "Gemini: сырой HTML запрещён — используйте build_gemini_payload / DocumentSummary"
            )
        return await self._ask_gemini_text(payload)

    async def _fill_input(self, page: Page, input_box, prompt: str) -> None:
        await input_box.click()
        await input_box.fill("")
        try:
            await input_box.fill(prompt)
        except Exception:
            pass
        # contenteditable в Gemini/Firefox часто не шлёт Enter после fill
        try:
            await input_box.evaluate(
                """(el, text) => {
                    el.focus();
                    if (el.isContentEditable) {
                        el.innerText = text;
                    } else {
                        el.value = text;
                    }
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                prompt,
            )
        except Exception:
            await page.keyboard.insert_text(prompt[:12000])

    async def _submit_prompt(self, page: Page) -> None:
        for sel in GEMINI_SEND_SELECTORS:
            try:
                btn = await page.query_selector(sel)
                if btn and await btn.is_visible() and await btn.is_enabled():
                    await btn.click()
                    set_status(f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] Send (кнопка)")
                    return
            except Exception:
                continue
        await page.keyboard.press("Enter")
        await asyncio.sleep(0.25)
        # Meta+Enter в некоторых UI
        await page.keyboard.press("Meta+Enter")

    async def _ask_gemini_text(self, prompt: str) -> str:
        if not self.page:
            await self.init_session()

        set_status(f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] отправка запроса…")
        page = self.page
        assert page is not None

        input_box = await page.wait_for_selector(GEMINI_INPUT_SELECTOR, timeout=45000)
        await self._fill_input(page, input_box, prompt)
        await self._submit_prompt(page)

        min_chars = GEMINI_MIN_RESPONSE_CHARS if len(prompt) > 900 else 60
        text = await self._wait_for_complete_response(
            page, min_response_chars=min_chars
        )
        if len(text) < min_chars:
            set_status(
                f"[Dialogue: {AI_CHAT_PROVIDER_NAME}] короткий ответ ({len(text)} симв.) — "
                "возможно обрезан; увеличьте GEMINI_RESPONSE_MAX_SEC"
            )
        return text

    async def close(self) -> None:
        if self.context is not None:
            await self.context.close()
            self.context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self.page = None


class BrowserGeminiDialogueSession(BaseAIDialogueSession):
    """Sync API для ai_react_loop_node (внутри asyncio.run на каждый send — сессия живёт между вызовами)."""

    def __init__(self, headless: bool = GEMINI_BROWSER_HEADLESS) -> None:
        super().__init__()
        self._session = BrowserGeminiSession(headless=headless)
        self._loop = asyncio.new_event_loop()
        self._ready = False

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def ask_gemini(self, payload: str) -> str:
        """Heavy reasoner: только структурированный payload от context_manager."""
        return self.send(payload)

    def send(self, message: str) -> str:
        if not self._ready:
            self._run(self._session.init_session())
            self._ready = True
        answer = self._run(self._session.ask_gemini(message))
        self.history.append(DialogueTurn(role="user", content=message))
        self.history.append(DialogueTurn(role="assistant", content=answer))
        return answer

    def close(self) -> None:
        if self._ready:
            self._run(self._session.close())
            self._ready = False
        self._loop.close()

    def extract_reference_urls(self, text: str) -> List[str]:
        urls = _URL_RE.findall(text)
        cleaned: list[str] = []
        for u in urls:
            u = u.rstrip(".,);]")
            if u not in cleaned:
                cleaned.append(u)
        return cleaned
