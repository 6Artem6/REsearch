"""Запуск persistent context: chromium или firefox (Playwright-бинарники)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from playwright.async_api import BrowserContext
from playwright.async_api import Playwright as AsyncPlaywright
from playwright.sync_api import BrowserContext as SyncBrowserContext
from playwright.sync_api import Playwright as SyncPlaywright

from knowledge_engine.config import BROWSER_PROFILE_PATH, PLAYWRIGHT_BROWSER
from knowledge_engine.services.search.playwright_browsers import (
    ensure_playwright_browsers_path,
)

PathLike = Union[str, Path]


def _persistent_kwargs(
    headless: bool,
    *,
    record_har_path: Optional[PathLike] = None,
) -> dict[str, Any]:
    BROWSER_PROFILE_PATH.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "user_data_dir": str(BROWSER_PROFILE_PATH),
        "headless": headless,
        "viewport": {"width": 1280, "height": 900},
    }
    if record_har_path:
        har = Path(record_har_path).expanduser()
        har.parent.mkdir(parents=True, exist_ok=True)
        kwargs["record_har_path"] = str(har)
        kwargs["record_har_content"] = "embed"
    return kwargs


def launch_persistent_context_sync(
    playwright: SyncPlaywright,
    headless: bool,
    *,
    record_har_path: Optional[PathLike] = None,
) -> SyncBrowserContext:
    ensure_playwright_browsers_path()
    kwargs = _persistent_kwargs(headless, record_har_path=record_har_path)
    if PLAYWRIGHT_BROWSER == "firefox":
        return playwright.firefox.launch_persistent_context(**kwargs)
    return playwright.chromium.launch_persistent_context(
        **kwargs,
        args=["--disable-blink-features=AutomationControlled"],
    )


async def launch_persistent_context_async(
    playwright: AsyncPlaywright,
    headless: bool,
    *,
    record_har_path: Optional[PathLike] = None,
) -> BrowserContext:
    ensure_playwright_browsers_path()
    kwargs = _persistent_kwargs(headless, record_har_path=record_har_path)
    if PLAYWRIGHT_BROWSER == "firefox":
        return await playwright.firefox.launch_persistent_context(**kwargs)
    return await playwright.chromium.launch_persistent_context(
        **kwargs,
        args=["--disable-blink-features=AutomationControlled"],
    )


def playwright_browser_label() -> str:
    return "Firefox" if PLAYWRIGHT_BROWSER == "firefox" else "Chromium"
