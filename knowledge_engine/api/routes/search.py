"""Поиск (аналог CLI test-search)."""

from __future__ import annotations

from fastapi import APIRouter

from knowledge_engine.api.schemas.requests import SearchTestRequest
from knowledge_engine.api.schemas.responses import SearchHit, SearchTestResponse
from knowledge_engine.services.search_service import search_flat, search_horizons

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/test", response_model=SearchTestResponse)
def test_search(body: SearchTestRequest) -> SearchTestResponse:
    if body.flat:
        raw = search_flat(body.query, limit_per_provider=body.limit_per_provider)
        hits = [SearchHit(**h) for h in raw]
        return SearchTestResponse(mode="flat", hits=hits)

    data = search_horizons(
        body.query,
        body.constraints,
        limit_per_provider=body.limit_per_provider,
    )
    all_hits: list[SearchHit] = []
    for horizon, items in data["results"].items():
        for item in items:
            all_hits.append(
                SearchHit(
                    source=item.get("source", ""),
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    horizon=horizon,
                )
            )
    return SearchTestResponse(mode="horizons", hits=all_hits, meta=data["meta"])
