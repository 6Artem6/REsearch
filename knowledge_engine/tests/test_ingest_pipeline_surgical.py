"""Surgical ingest fixes: GitHub raw fetch URL, MAP skip, window_summary prompt mix."""

from __future__ import annotations

from knowledge_engine.db.rag_chunks_schema import COL_CHUNK_TEXT, COL_WINDOW_SUMMARY
from knowledge_engine.services.lecture_rag_context import _chunk_plain_for_lecture
from knowledge_engine.services.web_extract import github_blob_to_raw_fetch_url
from knowledge_engine.src.curriculum.academic_source_fetch import (
    _stream_skip_practical_ingest,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def test_github_blob_rewrites_to_raw_for_fetch_only():
    blob = "https://github.com/python/cpython/blob/main/Doc/library/functions.rst"
    raw = github_blob_to_raw_fetch_url(blob)
    assert raw == (
        "https://raw.githubusercontent.com/python/cpython/main/Doc/library/functions.rst"
    )
    pep = "https://peps.python.org/pep-0634/"
    assert github_blob_to_raw_fetch_url(pep) == pep
    already = "https://raw.githubusercontent.com/a/b/main/x.md"
    assert github_blob_to_raw_fetch_url(already) == already


def test_stream_never_skips_practical_ingest_for_exa_highlights():
    hit = CurriculumSearchHit(
        url="https://peps.python.org/pep-0634/",
        title="PEP 634",
        snippet="x" * 200,
        key_extracts=["word " * 120],
        source_tier="exa",
        skip_ollama_summary=True,
    )
    assert _stream_skip_practical_ingest(hit) is False


def test_chunk_plain_for_lecture_prefixes_window_summary():
    row = {
        COL_CHUNK_TEXT: "full window body about switch statement lowering",
        COL_WINDOW_SUMMARY: "Window: compiler lowers switch to jump table.",
    }
    plain = _chunk_plain_for_lecture(row)
    assert plain.startswith("Window: compiler lowers switch")
    assert "full window body" in plain
    no_sum = _chunk_plain_for_lecture({COL_CHUNK_TEXT: "only body"})
    assert no_sum == "only body"
