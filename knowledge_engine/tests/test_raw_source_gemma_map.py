"""Raw/code ingest uses Gemma Cloud MAP windows — never ChatOllama."""

from __future__ import annotations

from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    FinalArticleSummaryResponse,
    MapWindowResponse,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceJobOutcome,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.article_ingestion.raw_source import (
    is_code_or_raw_source,
    wrap_raw_source_as_annotated,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit

_CEVAL = """
/* CPython Python/ceval_gil.c */
#include "Python.h"
#include "pycore_gil.h"

static void take_gil(PyThreadState *tstate)
{
    struct gil_runtime_state *gil = tstate->interp->runtime->ceval.gil;
    MUTEX_LOCK(gil->mutex);
    while (_Py_atomic_load_relaxed(&gil->locked)) {
        COND_WAIT(gil->cond, gil->mutex);
    }
    _Py_atomic_store_relaxed(&gil->locked, 1);
    MUTEX_UNLOCK(gil->mutex);
}
"""
_CEVAL_BODY = (_CEVAL.strip() + "\n") * 12
_CEVAL_URL = (
    "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c"
)


def test_is_code_or_raw_source_detects_c_and_github_raw():
    assert is_code_or_raw_source(_CEVAL_URL)
    assert is_code_or_raw_source(
        "https://github.com/python/cpython/blob/main/Python/ceval_gil.c"
    )
    assert is_code_or_raw_source(
        "https://github.com/python/cpython/blob/f23a1837/InternalDocs/interpreter.md"
    )
    assert is_code_or_raw_source("https://example.com/x.py")
    assert not is_code_or_raw_source("https://peps.python.org/pep-0703/")


def test_wrap_raw_source_tags_paragraphs_without_html(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.ingest.tiered_code_pruner.maybe_prune_code_for_map",
        lambda text, page_url="": text,
    )
    ann = wrap_raw_source_as_annotated(_CEVAL_BODY, _CEVAL_URL)
    assert "[P_1]" in ann.annotated_markdown
    assert "take_gil" in ann.annotated_markdown
    assert not ann.fig_map


def test_ingest_raw_c_file_writes_map_windows_without_ollama(monkeypatch):
    from knowledge_engine.services.article_ingestion import blog_spatial_pipeline as pipe

    monkeypatch.setattr(
        "knowledge_engine.ingest.tiered_code_pruner.maybe_prune_code_for_map",
        lambda text, page_url="": text,
    )
    gate_calls: list[str] = []

    def _gate(*_a, **_k):
        gate_calls.append("gate")
        raise AssertionError("HTML ingest gate must not run for raw C")

    monkeypatch.setattr(pipe, "_apply_inbound_ingest_gate", _gate)
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.figure_registry_service.persist_figure_registry",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.figure_registry_service.run_vlm_on_registry",
        lambda *_a, **_k: 0,
    )

    def _map(md: str, **kwargs):
        assert kwargs.get("source_kind") == "source_code"
        assert "take_gil" in md
        windows = [TokenWindowChunk(window_index=0, body=md[:1200])]
        final = FinalArticleSummaryResponse(
            executive_summary="GIL take/drop loop on mutex and condvar.",
            key_takeaways=["[SCOPE: MECHANIC] take_gil waits while gil->locked"],
        )
        mapped = MapWindowResponse(
            window_role="Locking",
            window_summary="take_gil acquires the GIL mutex and waits on cond.",
        )
        return MapReduceJobOutcome(final=final, map_results=[mapped]), windows

    monkeypatch.setattr(pipe, "map_reduce_summarize_blog_outcome", _map)

    saved: dict[str, object] = {}

    class _FakeStore:
        def save_summary(self, summary, skip_rag_ingest=False, **_k):
            saved["skip_rag_ingest"] = skip_rag_ingest
            saved["summary"] = summary

        def upsert_rag_academic_map_windows(
            self, url, title, texts, summary, window_summaries=None
        ):
            saved["map_texts"] = list(texts)
            saved["window_summaries"] = window_summaries
            doc_id = VectorStore.doc_id_for_url(url)
            saved["chunk_ids"] = [
                map_window_chunk_id(doc_id, i) for i in range(len(texts))
            ]
            return len(texts)

        def upsert_knowledge_atoms(self, url, atoms, **_k):
            saved["atoms"] = list(atoms)
            return len(atoms)

    monkeypatch.setattr(pipe, "VectorStore", _FakeStore)

    ollama_calls: list[str] = []

    def _boom_ollama(*_a, **_k):
        ollama_calls.append("summarize_article")
        raise AssertionError("Ollama summarizer must not run")

    monkeypatch.setattr(
        "knowledge_engine.services.summarizer.summarize_article",
        _boom_ollama,
    )

    _ann, summary, _saved_vlm = pipe.ingest_blog_with_spatial_mapping(
        "ceval_gil.c",
        _CEVAL_URL,
        raw_html=_CEVAL_BODY,
        save_lancedb=True,
    )
    assert gate_calls == []
    assert ollama_calls == []
    assert summary is not None
    assert summary.executive_summary == "GIL take/drop loop on mutex and condvar."
    assert saved.get("skip_rag_ingest") is True
    chunk_ids = saved.get("chunk_ids") or []
    assert chunk_ids
    assert all("_map_" in cid for cid in chunk_ids)
    assert saved.get("map_texts")


def test_spatial_fallback_calls_gemma_map_not_naive_passport(monkeypatch):
    from knowledge_engine.src.curriculum import source_material_pipeline as smp

    hit = CurriculumSearchHit(
        url=_CEVAL_URL,
        title="ceval_gil.c",
        snippet="GIL",
        source_tier="exa",
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.ingest_blog_with_spatial_mapping",
        lambda *_a, **_k: (None, None, 0),
    )
    persist_calls: list[str] = []

    def _persist(title: str, url: str, text: str) -> DocumentSummary:
        persist_calls.append(url)
        assert "take_gil" in text or "GIL" in text or len(text) > 50
        return DocumentSummary(
            title=title,
            url=url,
            key_takeaways=["[SCOPE: MECHANIC] take_gil"],
        )

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.persist_gemma_cloud_map_fallback",
        _persist,
    )
    monkeypatch.setattr(
        smp,
        "smart_fetch_page_text",
        lambda _url: (_CEVAL_BODY, "http"),
    )
    naive: list[str] = []

    def _naive_save(*_a, **_k):
        naive.append("save_summary")
        raise AssertionError("naive save_summary without MAP must not run")

    monkeypatch.setattr(smp.VectorStore, "save_summary", _naive_save)

    extracts, title = smp._ingest_url_with_spatial_map_reduce(
        hit, _CEVAL_BODY, tier_label="blog"
    )
    assert persist_calls == [_CEVAL_URL]
    assert naive == []
    assert title
    assert extracts or title == hit.title


def test_llm_ssot_is_gemma_not_chat_ollama():
    import inspect

    import knowledge_engine.llm as llm

    src = inspect.getsource(llm)
    assert "langchain_ollama" not in src
    assert "ChatOllama(" not in src
    runnable = llm.structured_chat("qwen2.5-coder:7b", DocumentSummary)
    assert isinstance(runnable, llm.GemmaStructuredRunnable)
    assert runnable.model  # remapped off qwen at invoke via _client
