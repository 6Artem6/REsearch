"""Playwright: persistent context, человеческие задержки, проверка сессии."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from knowledge_engine.config import AI_CHAT_START_URL
from knowledge_engine.services.search.playwright_launch import (
    launch_persistent_context_sync,
)


def wait_for_terminal_enter(message: str) -> None:
    """Один Enter в терминале (не typer.prompt — он зацикливается на пустом вводе)."""
    try:
        input(message)
    except EOFError:
        pass


def human_delay(min_s: float = 1.5, max_s: float = 3.5) -> None:
    """Случайная пауза против детекции ботов."""
    time.sleep(random.uniform(min_s, max_s))


@contextmanager
def persistent_browser(
    headless: bool = True,
) -> Iterator[tuple[Playwright, BrowserContext]]:
    """Persistent context — куки и сессия в `.browser_state/<engine>/`."""
    playwright = sync_playwright().start()
    context: Optional[BrowserContext] = None
    try:
        context = launch_persistent_context_sync(playwright, headless=headless)
        yield playwright, context
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        try:
            playwright.stop()
        except Exception:
            pass


def check_session_alive(
    context: BrowserContext,
    probe_url: str | None = None,
) -> bool:
    """
    Проверка, что браузерный контекст жив и страница отвечает.
    Не гарантирует авторизацию, но отсекает «мёртвую» сессию.
    """
    url = probe_url or AI_CHAT_START_URL
    page: Optional[Page] = None
    try:
        page = context.new_page()
        human_delay(0.5, 1.0)
        response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
        if response is None:
            return False
        return response.ok or response.status < 500
    except Exception:
        return False
    finally:
        if page is not None:
            page.close()


def fetch_page_html(url: str, headless: bool = True) -> str:
    """Загрузить HTML страницы через persistent Playwright."""
    with persistent_browser(headless=headless) as (_, context):
        if not check_session_alive(context):
            raise RuntimeError(
                "Браузерная сессия не отвечает — откройте setup и войдите вручную."
            )
        page = context.new_page()
        try:
            human_delay()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            human_delay(1.0, 2.0)
            return page.content()
        finally:
            page.close()
