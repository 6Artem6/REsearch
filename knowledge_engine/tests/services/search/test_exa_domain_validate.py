"""HTTP Pass 1 probe does not assign OFFICIAL_DOCS."""

from __future__ import annotations

from types import SimpleNamespace

from knowledge_engine.services.search.exa_domain_validate import (
    prepare_exa_pass1_domains_blocking,
)
from knowledge_engine.services.search.exa_domains import is_official_docs_host


def test_prepare_pass1_http_200_does_not_mark_official(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def head(self, url, **kwargs):
            return SimpleNamespace(status_code=200)

        async def get(self, url, **kwargs):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.httpx.AsyncClient",
        _Client,
    )
    live = prepare_exa_pass1_domains_blocking(["habr.com"])
    assert live == ["habr.com"]
    assert not is_official_docs_host("habr.com")
