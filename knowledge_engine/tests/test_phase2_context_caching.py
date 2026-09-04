"""Phase 1.1 anchors + Phase 2 AST chunking and ingest Gemini cache."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.services.article_ingestion.ast_code_chunker import (
    AstCodeChunker,
    linear_chunk_code,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    _prompt_for_window,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.llm.ingest_context_cache_manager import (
    IngestContextCacheManager,
    ingest_cache_registry_key,
)
from knowledge_engine.services.validators.anchor_validator import (
    validate_and_annotate_anchors,
)
from knowledge_engine.services.vector_store import VectorStore

_TINY_DEFS = "\n\n".join(
    f"def tiny_{i}():\n    value = {i}\n    return value" for i in range(18)
)
_PY_SRC = f'''\
"""module docstring"""

def alpha():
    marker = "ALPHA_START"
    x = 1
    y = 2
    return "ALPHA_END"


def beta():
    marker = "BETA_START"
    a = 1
    b = 2
    return "BETA_END"


{_TINY_DEFS}


def mid_fn():
    start = "MID_START"
    acc = 0
    acc += 1
    acc += 2
    acc += 3
    acc += 4
    acc += 5
    acc += 6
    acc += 7
    acc += 8
    return "MID_END"
'''


def _paragraph_blocks(markdown: str) -> list[str]:
    return re.findall(r"\[P_\d+\]\n(.*?)(?=\n\[P_|\Z)", markdown, flags=re.S)


def test_ordinal_anchors_and_unverified_annotation(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "CHUNK_ANCHOR_INJECTION", True)
    monkeypatch.setattr(ke_config, "ANCHOR_REGEX_VALIDATE", True)

    url = "https://example.com/doc"
    job = MapReduceArticleJob(
        job_id=url,
        title="T",
        url=url,
        windows=[
            TokenWindowChunk(window_index=0, body="first window body"),
            TokenWindowChunk(window_index=1, body="second window body"),
        ],
    )
    assert list(job.anchor_index_map) == ["A1", "A2"]
    assert job.windows[0].body.startswith("[A1]\n")
    assert job.windows[1].body.startswith("[A2]\n")
    cid0 = map_window_chunk_id(VectorStore.doc_id_for_url(url), 0)
    assert job.anchor_index_map["A1"]["chunk_id"] == cid0
    prompt = _prompt_for_window(job, job.windows[1])
    assert "[A2]" in prompt
    assert "[ANCHOR:" not in prompt
    assert "[[A" not in prompt

    src = "See [S1] and [R1] plus arr[0] and [A1] vs [A99]."
    marked, unverified = validate_and_annotate_anchors(src, set(job.anchor_index_map))
    assert "[S1]" in marked
    assert "[R1]" in marked
    assert "arr[0]" in marked
    assert "[A1]" in marked
    assert "[A1 (? unverified)]" not in marked
    assert "[A99 (? unverified)]" in marked
    assert unverified == ["A99"]


def test_validate_and_annotate_anchors_noop_when_flag_off(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "ANCHOR_REGEX_VALIDATE", False)
    src = "cite [A99] and [S1]"
    marked, unverified = validate_and_annotate_anchors(src, {"A1"})
    assert marked == src
    assert unverified == []


def test_ast_code_chunker_keeps_python_functions_intact():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_python")

    ann = AstCodeChunker()._wrap_ast(_PY_SRC, "https://example.com/mod.py")
    blocks = _paragraph_blocks(ann.annotated_markdown)
    assert blocks
    def_count = len(re.findall(r"^def ", _PY_SRC, flags=re.M))
    assert len(blocks) < def_count
    for start, end in (
        ("ALPHA_START", "ALPHA_END"),
        ("BETA_START", "BETA_END"),
        ("MID_START", "MID_END"),
    ):
        owners = [b for b in blocks if start in b or end in b]
        assert owners, start
        for block in owners:
            assert start in block and end in block


def test_ast_code_chunker_syntax_error_falls_back_to_linear():
    broken = "def broken(\n    pass\n"
    linear = linear_chunk_code(broken, "https://example.com/bad.py")
    wrapped = AstCodeChunker().wrap(broken, "https://example.com/bad.py")
    assert wrapped.annotated_markdown == linear.annotated_markdown


def test_ast_oversized_function_falls_back_to_linear():
    body = "\n".join(f"    x{i} = {i}" for i in range(160))
    src = f"def huge():\n{body}\n    return x0\n"
    linear = linear_chunk_code(src, "https://example.com/huge.py")
    wrapped = AstCodeChunker().wrap(src, "https://example.com/huge.py")
    assert wrapped.annotated_markdown == linear.annotated_markdown


def test_ingest_cache_key_namespace_isolated():
    key = ingest_cache_registry_key("doc42", "summaries")
    assert key.startswith("ingest:doc42:")
    assert key == ingest_cache_registry_key("doc42", "summaries")
    assert key != ingest_cache_registry_key("doc42", "other")


def test_ingest_cache_manager_create_and_generate(monkeypatch, tmp_path: Path):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MIGRATION_USE_CONTEXT_CACHING", True)

    created: dict[str, object] = {}

    class _Caches:
        def create(self, **kwargs):
            created["kwargs"] = kwargs
            return SimpleNamespace(name="cachedContents/ingest-test")

        def get(self, name: str):
            return SimpleNamespace(name=name)

    class _Models:
        def generate_content(self, **kwargs):
            created["gen"] = kwargs
            payload = {
                "executive_summary": "cached reduce",
                "key_takeaways": ["[SCOPE: PRINCIPLE] fact"],
            }
            return SimpleNamespace(text=json.dumps(payload))

    client = SimpleNamespace(caches=_Caches(), models=_Models())
    mgr = IngestContextCacheManager(
        client,
        registry_path=tmp_path / "gemini_ingest_cache_registry.json",
        model="gemini-test",
        ttl_seconds=120,
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        FinalArticleSummaryResponse,
    )

    out = mgr.generate_structured(
        doc_id="doc1",
        system_instruction="REDUCE system",
        cache_content="## Window summaries\nhello",
        user_prompt="synthesize",
        schema=FinalArticleSummaryResponse,
        max_tokens=256,
    )
    assert out is not None
    assert out.executive_summary == "cached reduce"
    assert created["kwargs"]["model"] == "gemini-test"
    cfg = created["gen"]["config"]
    assert getattr(cfg, "cached_content", None) == "cachedContents/ingest-test"
    raw = json.loads((tmp_path / "gemini_ingest_cache_registry.json").read_text())
    assert any(k.startswith("ingest:doc1:") for k in raw)


def test_ingest_cache_manager_falls_back_on_create_error(monkeypatch, tmp_path: Path):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MIGRATION_USE_CONTEXT_CACHING", True)

    class _Boom(Exception):
        status_code = 404

    class _Caches:
        def create(self, **_kwargs):
            raise _Boom("404 cached content not found")

    client = SimpleNamespace(caches=_Caches(), models=SimpleNamespace())
    mgr = IngestContextCacheManager(
        client,
        registry_path=tmp_path / "reg.json",
        model="gemini-test",
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        FinalArticleSummaryResponse,
    )

    out = mgr.generate_structured(
        doc_id="doc1",
        system_instruction="sys",
        cache_content="matrix",
        user_prompt="go",
        schema=FinalArticleSummaryResponse,
    )
    assert out is None


def test_ingest_cache_manager_falls_back_on_generate_error(monkeypatch, tmp_path: Path):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "MIGRATION_USE_CONTEXT_CACHING", True)

    class _Caches:
        def create(self, **_kwargs):
            return SimpleNamespace(name="cachedContents/x")

        def get(self, name: str):
            return SimpleNamespace(name=name)

    class _Models:
        def generate_content(self, **_kwargs):
            err = TimeoutError("deadline exceeded")
            raise err

    client = SimpleNamespace(caches=_Caches(), models=_Models())
    mgr = IngestContextCacheManager(
        client,
        registry_path=tmp_path / "reg.json",
        model="gemini-test",
    )
    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        FinalArticleSummaryResponse,
    )

    out = mgr.generate_structured(
        doc_id="doc1",
        system_instruction="sys",
        cache_content="matrix",
        user_prompt="go",
        schema=FinalArticleSummaryResponse,
    )
    assert out is None


def test_ingest_cache_disabled_by_default(tmp_path: Path):
    mgr = IngestContextCacheManager(
        SimpleNamespace(),
        registry_path=tmp_path / "reg.json",
        model="gemini-test",
    )
    hit = mgr.get_or_create(
        doc_id="d",
        content="body",
        system_instruction="sys",
    )
    assert hit.mode == "disabled"
    assert not hit.is_active
