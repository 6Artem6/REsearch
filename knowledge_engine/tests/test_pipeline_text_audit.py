"""Fetch→MAP volume: no 12k strip, GitHub docs skip gate, extracts keep source."""

from __future__ import annotations

from types import SimpleNamespace

import trafilatura

from knowledge_engine.services.article_ingestion.raw_source import is_code_or_raw_source
from knowledge_engine.services.parsers.html_annotator import build_annotated_article
from knowledge_engine.services.web_extract import _clean_html, _extract_readable_text
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.source_material_pipeline import (
    DEEP_BLOG_EXTRACT_WORDS,
    _summary_to_extracts_and_title,
)
from knowledge_engine.src.parsers.paper_structure_analyzer import (
    _filtered_body_discards_source,
)


def test_clean_html_probe_still_caps_but_is_not_ingest_body():
    html = "<p>" + ("word " * 5000) + "</p>"
    probe = _clean_html(html)
    assert len(probe) <= 12_000


def test_extract_readable_text_keeps_full_trafilatura_body(monkeypatch):
    payload = ("technical " * 4000).strip()
    monkeypatch.setattr(trafilatura, "extract", lambda *_a, **_k: payload)
    out = _extract_readable_text("<html><body><p>ignored</p></body></html>")
    assert len(out.split()) >= 3500
    assert len(out) > 12_000


def test_html_annotator_keeps_definition_lists():
    html = (
        "<html><body><article>"
        "<dl><dt>PyEval_InitThreads</dt>"
        "<dd>Initialize the GIL before creating threads. "
        "The lock must be held by the current thread.</dd></dl>"
        "</article></body></html>"
    )
    ann = build_annotated_article(html, "https://docs.python.org/3/c-api/threads.html")
    blob = ann.annotated_markdown
    assert "PyEval_InitThreads" in blob
    assert "Initialize the GIL" in blob


def test_github_markdown_blob_is_raw_source():
    md = "https://github.com/python/cpython/blob/f23a1837/InternalDocs/interpreter.md"
    assert is_code_or_raw_source(md)
    assert not is_code_or_raw_source("https://peps.python.org/pep-0703/")


def test_ingest_gate_fail_open_when_filter_keeps_stub():
    raw = ("mechanism " * 3000).strip()
    stub = ("noise " * 90).strip()
    assert _filtered_body_discards_source(raw, stub)
    assert not _filtered_body_discards_source(raw, ("mechanism " * 2000).strip())


def test_summary_extracts_backfill_from_source_when_takeaways_thin():
    hit = CurriculumSearchHit(
        url="https://docs.python.org/3/c-api/threads.html",
        title="Threads",
        snippet="gil",
    )
    summary = SimpleNamespace(
        title="Threads",
        executive_summary="Короткий паспорт.",
        key_takeaways=["[SCOPE: INSTANCE] GIL."],
        failure_modes=[],
    )
    source = ("interpreter lock mechanics " * 200).strip()
    extracts, _title = _summary_to_extracts_and_title(
        hit, summary, source_text=source
    )
    assert sum(len(e.split()) for e in extracts) >= DEEP_BLOG_EXTRACT_WORDS
    assert any("mechanics" in e for e in extracts)
