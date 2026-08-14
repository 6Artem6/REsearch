"""Единый путь к Playwright browsers (.venv/.local-browsers, не ~/Library/Caches)."""

from __future__ import annotations

import os
from pathlib import Path

from knowledge_engine.config import PLAYWRIGHT_BROWSERS_PATH


def venv_local_browsers_path() -> Path:
    import playwright

    return Path(playwright.__file__).resolve().parent / "driver/package/.local-browsers"


def ensure_playwright_browsers_path() -> str:
    """
    dev-native выставляет PLAYWRIGHT_BROWSERS_PATH; CLI без этого ищет ms-playwright в Cache.
    Подхватываем chromium из .venv, если он установлен через install-playwright.sh.
    """
    raw = PLAYWRIGHT_BROWSERS_PATH or os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if raw and raw != "0" and Path(raw).is_dir():
        return raw

    venv_browsers = venv_local_browsers_path()
    if venv_browsers.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(venv_browsers)
        return str(venv_browsers)

    if raw == "0":
        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)

    return os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
