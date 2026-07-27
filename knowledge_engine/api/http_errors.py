"""HTTP / CLI ошибки с файлом и строкой."""

from __future__ import annotations

import json
from typing import Any

import httpx

from knowledge_engine.ui.errors import format_error_location, format_error_with_cause


def format_http_status_error(exc: httpx.HTTPStatusError) -> str:
    base = format_error_location(exc)
    try:
        body = exc.response.json()
        if isinstance(body, dict) and body.get("detail"):
            detail = body["detail"]
            if isinstance(detail, str):
                return f"{base} | API: {detail}"
            return f"{base} | API: {json.dumps(detail, ensure_ascii=False)}"
    except Exception:
        pass
    text = (exc.response.text or "").strip()[:400]
    if text:
        return f"{base} | body: {text}"
    return base


def format_any_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return format_http_status_error(exc)
    return format_error_with_cause(exc)


def api_error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "detail": format_any_error(exc),
        "error": format_any_error(exc),
        "type": type(exc).__name__,
    }
