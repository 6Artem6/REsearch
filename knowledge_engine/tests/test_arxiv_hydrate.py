"""arXiv id_list hydrate + ScholarPaper merge."""

from __future__ import annotations

import asyncio
from typing import Sequence

from knowledge_engine.services.search.arxiv_client import (
    ArxivClient,
    ArxivEntry,
    normalize_arxiv_id,
)
from knowledge_engine.src.retrieval.arxiv_hydrate import (
    extract_arxiv_id_from_paper,
    hydrate_scholar_papers,
    merge_arxiv_entry_into_paper,
)
from knowledge_engine.src.retrieval.semantic_scholar import ScholarPaper


def test_normalize_arxiv_id_strips_version_and_url():
    assert normalize_arxiv_id("https://arxiv.org/abs/2301.07041v2") == "2301.07041"
    assert normalize_arxiv_id("arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert normalize_arxiv_id("2301.07041") == "2301.07041"


def test_extract_arxiv_id_from_url_doi_and_field():
    p1 = ScholarPaper(source_url="https://arxiv.org/abs/2301.07041")
    assert extract_arxiv_id_from_paper(p1) == "2301.07041"

    p2 = ScholarPaper(doi="10.48550/arXiv.1706.03762")
    assert extract_arxiv_id_from_paper(p2) == "1706.03762"

    p3 = ScholarPaper(arxiv_id="1234.56789v3")
    assert extract_arxiv_id_from_paper(p3) == "1234.56789"

    p4 = ScholarPaper(paper_id="cs/9901002")
    assert extract_arxiv_id_from_paper(p4) == "cs/9901002"

    p5 = ScholarPaper(title="no id here")
    assert extract_arxiv_id_from_paper(p5) == ""


def test_merge_arxiv_entry_prefers_longer_abstract():
    paper = ScholarPaper(
        title="Short",
        abstract="tiny",
        source_url="https://www.semanticscholar.org/paper/abc",
        source="semantic_scholar",
        arxiv_id="2301.07041",
    )
    entry = ArxivEntry(
        arxiv_id="2301.07041",
        title="Attention Is All You Need (canonical)",
        abstract="A" * 400,
        entry_id="http://arxiv.org/abs/2301.07041",
        pdf_url="https://arxiv.org/pdf/2301.07041.pdf",
        published="2023-01-17T00:00:00Z",
    )
    merged = merge_arxiv_entry_into_paper(paper, entry)
    assert merged.abstract == "A" * 400
    assert "arxiv.org" in merged.source_url
    assert merged.pdf_url.endswith("2301.07041.pdf")
    assert merged.source == "semantic_scholar+arxiv"
    assert merged.year == 2023
    assert merged.arxiv_id == "2301.07041"


def test_hydrate_scholar_papers_batches_and_merges(monkeypatch):
    calls: list[list[str]] = []

    async def _fake_fetch(self, ids: Sequence[str], *, chunk_size=None):
        chunk = list(ids)
        calls.append(chunk)
        return [
            ArxivEntry(
                arxiv_id=normalize_arxiv_id(i),
                title=f"Title {i}",
                abstract=f"Abstract for {i} " + ("x" * 320),
                entry_id=f"http://arxiv.org/abs/{normalize_arxiv_id(i)}",
                pdf_url=f"https://arxiv.org/pdf/{normalize_arxiv_id(i)}.pdf",
                published="2024-01-01T00:00:00Z",
            )
            for i in chunk
        ]

    monkeypatch.setattr(ArxivClient, "fetch_by_ids", _fake_fetch)

    papers = [
        ScholarPaper(
            title="A",
            source_url="https://arxiv.org/abs/1111.11111",
            source="consensus",
        ),
        ScholarPaper(title="B", source="semantic_scholar"),
        ScholarPaper(
            title="C",
            arxiv_id="2222.22222",
            abstract="short",
            source="semantic_scholar",
        ),
    ]

    out = asyncio.run(hydrate_scholar_papers(papers, chunk=50))
    assert len(calls) == 1
    assert set(calls[0]) == {"1111.11111", "2222.22222"}
    assert out[0].source == "consensus+arxiv"
    assert out[1].source == "semantic_scholar"
    assert out[1].abstract == ""
    assert len(out[2].abstract) > 300
    assert out[2].source == "semantic_scholar+arxiv"


def test_fetch_by_ids_chunks(monkeypatch):
    entries_by_call: list[list[str]] = []

    async def _get_atom(self, params):
        id_list = str(params.get("id_list") or "")
        ids_local = [x for x in id_list.split(",") if x]
        entries_by_call.append(ids_local)
        return [
            ArxivEntry(
                arxiv_id=i,
                title=i,
                abstract="",
                entry_id=f"http://arxiv.org/abs/{i}",
                pdf_url=f"https://arxiv.org/pdf/{i}.pdf",
            )
            for i in ids_local
        ]

    monkeypatch.setattr(ArxivClient, "_get_atom", _get_atom)
    client = ArxivClient(id_list_chunk=20)
    ids = [f"1000.{i:05d}" for i in range(55)]
    out = asyncio.run(client.fetch_by_ids(ids, chunk_size=20))
    assert len(out) == 55
    assert [len(c) for c in entries_by_call] == [20, 20, 15]


def test_client_retries_on_503(monkeypatch):
    attempts = {"n": 0}

    class _FakeResp:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _FakeResp(503)
            atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.07041</id>
    <title>Retry OK</title>
    <summary>Abstract text</summary>
  </entry>
</feed>"""
            return _FakeResp(200, atom)

    async def _noop_acquire():
        return None

    async def _noop_pause(_wait):
        return None

    monkeypatch.setattr(
        "knowledge_engine.services.search.arxiv_client.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.arxiv_client.acquire_arxiv_slot_async",
        _noop_acquire,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.arxiv_client.arxiv_pause_before_retry_async",
        _noop_pause,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.arxiv_client.random.uniform",
        lambda _a, _b: 0.0,
    )

    client = ArxivClient(max_retries=2, backoff_base_sec=0.01)
    entries = asyncio.run(client.search(search_query="all:test", max_results=1))
    assert attempts["n"] == 2
    assert len(entries) == 1
    assert entries[0].arxiv_id == "2301.07041"
    assert entries[0].title == "Retry OK"
