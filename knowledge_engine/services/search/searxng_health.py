"""Проверка доступности SearXNG."""

from __future__ import annotations

import httpx

from knowledge_engine.config import (
    SEARXNG_BASE_URL,
    SEARXNG_REQUEST_HEADERS,
    SEARXNG_TIMEOUT_SEC,
)


def check_searxng() -> tuple[bool, str]:
    """GET /search?format=json — True если есть results или сервис отвечает."""
    url = f"{SEARXNG_BASE_URL}/search"
    try:
        with httpx.Client(
            timeout=SEARXNG_TIMEOUT_SEC, headers=SEARXNG_REQUEST_HEADERS
        ) as client:
            resp = client.get(
                url,
                params={"q": "test", "format": "json", "engines": "bing"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return False, f"SearXNG недоступен ({SEARXNG_BASE_URL}): {exc}"

    errors = data.get("errors") or []
    results = data.get("results") or []
    if results:
        return True, f"OK: {len(results)} результатов, URL={SEARXNG_BASE_URL}"
    if errors:
        return False, f"SearXNG ответил, но без results. errors={errors[:3]}"
    return False, "SearXNG: пустой JSON (проверьте settings.yml → search.formats: json)"
