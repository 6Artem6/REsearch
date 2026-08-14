"""HTTP API entry (uvicorn)."""

from __future__ import annotations

import uvicorn

import knowledge_engine  # noqa: F401 — Deprecation filter в __init__
from knowledge_engine.config import (
    KE_API_HOST,
    KE_API_PORT,
    KE_API_RELOAD,
    PACKAGE_ROOT,
)


def main() -> None:
    host = KE_API_HOST
    port = KE_API_PORT
    reload = KE_API_RELOAD
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
