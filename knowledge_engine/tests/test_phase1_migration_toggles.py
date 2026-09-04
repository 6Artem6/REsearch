"""Phase 1 migration toggles: anchors, citation validator, AST factory fallback."""

from __future__ import annotations

from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    _prompt_for_window,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
    maybe_prepend_chunk_anchor,
    split_annotated_text_by_tokens,
)
from knowledge_engine.services.article_ingestion.raw_source import (
    wrap_raw_source_as_annotated,
    wrap_raw_source_linear,
)
from knowledge_engine.services.vector_store import VectorStore

_SAMPLE_MD = "[P_1]\nHello paragraph one.\n\n[P_2]\nHello paragraph two."
_SAMPLE_C = "static void take_gil(void) {\n    return;\n}\n"


def test_config_source_defaults_are_legacy():
    import inspect

    import knowledge_engine.config as ke_config

    src = inspect.getsource(ke_config)
    assert 'os.getenv("CODE_PARSER_MODE", "linear")' in src
    assert '_env_bool("CHUNK_ANCHOR_INJECTION", False)' in src
    assert '_env_bool("ANCHOR_REGEX_VALIDATE", False)' in src
    assert 'os.getenv("CLAIM_DEDUP_MODE", "none")' in src
    assert 'os.getenv("CLAIM_MMR_LAMBDA", "0.7")' in src
    assert '_env_bool("MIGRATION_USE_CONTEXT_CACHING", False)' in src
    assert 'os.getenv("INGEST_CACHE_TTL_SECONDS", "86400")' in src


def test_default_toggles_keep_legacy_chunk_bodies(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "CODE_PARSER_MODE", "linear")
    monkeypatch.setattr(ke_config, "CHUNK_ANCHOR_INJECTION", False)
    monkeypatch.setattr(ke_config, "ANCHOR_REGEX_VALIDATE", False)
    monkeypatch.setattr(
        "knowledge_engine.ingest.tiered_code_pruner.maybe_prune_code_for_map",
        lambda text, page_url="": text,
    )

    linear = wrap_raw_source_linear(_SAMPLE_C * 20, "https://example.com/x.c")
    wrapped = wrap_raw_source_as_annotated(_SAMPLE_C * 20, "https://example.com/x.c")
    assert wrapped.annotated_markdown == linear.annotated_markdown
    assert "[P_1]" in wrapped.annotated_markdown
    assert "[ANCHOR:" not in wrapped.annotated_markdown
    assert not wrapped.annotated_markdown.startswith("[A")

    windows = split_annotated_text_by_tokens(_SAMPLE_MD, title="T")
    assert windows
    assert all("[ANCHOR:" not in w.body for w in windows)
    assert all(not w.body.startswith("[A1]") for w in windows)
    assert maybe_prepend_chunk_anchor("plain body", 1) == "plain body"


def test_chunk_anchor_injection_prepends_ordinal(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "CHUNK_ANCHOR_INJECTION", True)

    out = maybe_prepend_chunk_anchor("plain body", 1)
    assert out.startswith("[A1]\n")
    assert out.endswith("plain body")
    assert maybe_prepend_chunk_anchor(out, 1) == out
    assert "[ANCHOR:" not in out

    url = "https://example.com/article"
    job = MapReduceArticleJob(
        job_id=url,
        title="T",
        url=url,
        windows=[
            TokenWindowChunk(window_index=0, body="[P_1]\nHello"),
            TokenWindowChunk(window_index=1, body="[P_2]\nWorld"),
        ],
    )
    cid0 = map_window_chunk_id(VectorStore.doc_id_for_url(url), 0)
    cid1 = map_window_chunk_id(VectorStore.doc_id_for_url(url), 1)
    assert job.windows[0].body.startswith("[A1]\n")
    assert job.windows[1].body.startswith("[A2]\n")
    assert job.anchor_index_map["A1"]["chunk_id"] == cid0
    assert job.anchor_index_map["A2"]["chunk_id"] == cid1
    prompt = _prompt_for_window(job, job.windows[0])
    assert "[A1]" in prompt
    assert "[ANCHOR:" not in prompt
    window_part = prompt.split("<window_text>", 1)[-1]
    assert "[A1]" in window_part
    assert cid0 not in window_part


def test_ast_mode_c_file_falls_back_to_linear(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "CODE_PARSER_MODE", "ast")
    monkeypatch.setattr(
        "knowledge_engine.ingest.tiered_code_pruner.maybe_prune_code_for_map",
        lambda text, page_url="": text,
    )

    def _boom(self, *_a, **_k):
        raise RuntimeError("forced AST failure")

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.ast_code_chunker.AstCodeChunker.wrap",
        _boom,
    )

    linear = wrap_raw_source_linear(_SAMPLE_C * 20, "https://example.com/x.c")
    wrapped = wrap_raw_source_as_annotated(_SAMPLE_C * 20, "https://example.com/x.c")
    assert wrapped.annotated_markdown == linear.annotated_markdown
    assert "[P_1]" in wrapped.annotated_markdown
