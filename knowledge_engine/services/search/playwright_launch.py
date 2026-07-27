"""Запуск persistent context: chromium или firefox (Playwright-бинарники)."""

from __future__ import annotations

from typing import Any

from playwright.async_api import BrowserContext
from playwright.async_api import Playwright as AsyncPlaywright
from playwright.sync_api import BrowserContext as SyncBrowserContext
from playwright.sync_api import Playwright as SyncPlaywright

from knowledge_engine.config import BROWSER_PROFILE_PATH, PLAYWRIGHT_BROWSER
from knowledge_engine.services.search.playwright_browsers import (
    ensure_playwright_browsers_path,
)


def _persistent_kwargs(headless: bool) -> dict[str, Any]:
    BROWSER_PROFILE_PATH.mkdir(parents=True, exist_ok=True)
    return {
        "user_data_dir": str(BROWSER_PROFILE_PATH),
        "headless": headless,
        "viewport": {"width": 1280, "height": 900},
    }


def launch_persistent_context_sync(
    playwright: SyncPlaywright,
    headless: bool,
) -> SyncBrowserContext:
    ensure_playwright_browsers_path()
    kwargs = _persistent_kwargs(headless)
    if PLAYWRIGHT_BROWSER == "firefox":
        return playwright.firefox.launch_persistent_context(**kwargs)
    return playwright.chromium.launch_persistent_context(
        **kwargs,
        args=["--disable-blink-features=AutomationControlled"],
    )


async def launch_persistent_context_async(
    playwright: AsyncPlaywright,
    headless: bool,
) -> BrowserContext:
    ensure_playwright_browsers_path()
    kwargs = _persistent_kwargs(headless)
    if PLAYWRIGHT_BROWSER == "firefox":
        return await playwright.firefox.launch_persistent_context(**kwargs)
    return await playwright.chromium.launch_persistent_context(
        **kwargs,
        args=["--disable-blink-features=AutomationControlled"],
    )


def playwright_browser_label() -> str:
    return "Firefox" if PLAYWRIGHT_BROWSER == "firefox" else "Chromium"
