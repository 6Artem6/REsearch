"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from knowledge_engine.api.http_errors import api_error_payload, format_any_error
from knowledge_engine.api.routes import (
    analysis,
    analysis_wait,
    health,
    search,
    v07,
    v08,
)
from knowledge_engine.ui.run_log import trace

_WEB_STATIC = Path(__file__).resolve().parent.parent / "web" / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Engine API",
        description="REST API для анализа, поиска и unraveling (LangGraph + Gemini + Ollama).",
        version="0.8.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.on_event("startup")
    def _startup_log() -> None:
        import knowledge_engine.config as cfg
        from knowledge_engine.services.job_store import _JOB_STORE_PATH, job_store

        n = len(job_store.list_recent(limit=10_000))
        print(
            f"Knowledge Engine API | GRAPH_VERSION={cfg.GRAPH_VERSION} "
            f"| trace_stdout={cfg.KE_TRACE_STDOUT} | jobs_loaded={n} "
            f"| job_store={_JOB_STORE_PATH} "
            f"| consensus_profile={cfg.BROWSER_PROFILE_PATH}",
            flush=True,
        )

    @app.on_event("shutdown")
    async def _shutdown_consensus_browser() -> None:
        from knowledge_engine.src.retrieval.consensus_session import (
            shutdown_shared_consensus_session,
        )

        await shutdown_shared_consensus_session()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        trace(f"API HTTP {exc.status_code} | {detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail, "error": detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        detail = format_any_error(exc)
        trace(f"API 500 | {detail}")
        return JSONResponse(status_code=500, content=api_error_payload(exc))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount sub-apps pattern avoided — single router tree
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(analysis.router, prefix="/api/v1")
    app.include_router(analysis_wait.router, prefix="/api/v1")
    app.include_router(v07.router, prefix="/api/v1")
    app.include_router(v08.router, prefix="/api/v1")

    if _WEB_STATIC.is_dir():
        app.mount(
            "/app/static",
            StaticFiles(directory=str(_WEB_STATIC)),
            name="ke-web-static",
        )

    @app.get("/app", tags=["ui"])
    @app.get("/app/", tags=["ui"])
    def web_app() -> FileResponse:
        return FileResponse(_WEB_STATIC / "index.html")

    @app.get("/", tags=["health"])
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app", status_code=302)

    @app.get("/api", tags=["health"])
    def api_root() -> dict[str, str]:
        return {
            "service": "knowledge-engine",
            "web_ui": "/app",
            "docs": "/docs",
            "health": "/api/v1/health",
            "v07_runs": "/api/v1/v07/runs",
            "v08_explain": "/api/v1/v08/explain",
        }

    return app


app = create_app()
