"""HTTP API entry (uvicorn)."""

from __future__ import annotations

import uvicorn

import knowledge_engine  # noqa: F401 — Deprecation filter в __init__
from knowledge_engine.src.config.settings import (
    KE_API_HOST,
    KE_API_PORT,
    KE_API_RELOAD,
    PACKAGE_ROOT,
)
from knowledge_engine.src.shared.ml_runtime.ml_runtime import mark_api_process


def main() -> None:
    mark_api_process()
    host = KE_API_HOST
    port = KE_API_PORT
    reload = KE_API_RELOAD
    reload_dirs = [
        str(PACKAGE_ROOT / "api"),
        str(PACKAGE_ROOT / "graph"),
        str(PACKAGE_ROOT / "services"),
        str(PACKAGE_ROOT / "nodes"),
        str(PACKAGE_ROOT / "schemas"),
        str(PACKAGE_ROOT / "ui"),
        str(PACKAGE_ROOT / "src"),
    ]
    uvicorn.run(
        "knowledge_engine.src.entrypoints.api.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=reload_dirs if reload else None,
        reload_includes=["*.py"] if reload else None,
        factory=False,
    )


if __name__ == "__main__":
    main()
