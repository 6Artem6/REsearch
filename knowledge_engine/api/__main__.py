"""HTTP API entry (uvicorn)."""

from __future__ import annotations

import os

import uvicorn

import knowledge_engine  # noqa: F401 — Deprecation filter в __init__
from knowledge_engine.config import PACKAGE_ROOT


def main() -> None:
    host = os.getenv("KE_API_HOST", "127.0.0.1")
    port = int(os.getenv("KE_API_PORT", "8765"))
    reload = os.getenv("KE_API_RELOAD", "false").lower() in ("1", "true", "yes")
    reload_dirs = [
        str(PACKAGE_ROOT / "api"),
        str(PACKAGE_ROOT / "graph"),
        str(PACKAGE_ROOT / "services"),
        str(PACKAGE_ROOT / "nodes"),
        str(PACKAGE_ROOT / "ui"),
        str(PACKAGE_ROOT / "src"),
    ]
    uvicorn.run(
        "knowledge_engine.api.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=reload_dirs if reload else None,
        reload_includes=["*.py"] if reload else None,
        factory=False,
    )


if __name__ == "__main__":
    main()
