"""Registry языков tree-sitter: расширение URL → lazy parser."""

from __future__ import annotations

import pytest

from knowledge_engine.services.article_ingestion.ast_code_chunker import (
    EXTENSION_TO_LANGUAGE,
    _CUSTOM_LANGUAGE_LOADERS,
    _PARSER_CACHE,
    AstChunkError,
    AstCodeChunker,
    language_from_url,
    linear_chunk_code,
    parser_for_language,
)


def test_language_from_url_is_extension_dict_only():
    assert language_from_url("https://ex.com/src/mod.py") == "python"
    assert language_from_url("https://ex.com/a.ts") == "typescript"
    assert language_from_url("https://ex.com/a.tsx") == "tsx"
    assert language_from_url("https://ex.com/a.go") == "go"
    assert language_from_url("https://ex.com/a.rs") == "rust"
    assert language_from_url("https://ex.com/a.cpp") == "cpp"
    assert language_from_url("https://ex.com/a.c") == "c"
    assert language_from_url("https://ex.com/A.JAVA") == "java"
    assert language_from_url("") is None
    assert language_from_url("https://ex.com/README") is None
    assert language_from_url("https://ex.com/file.md") is None
    for suffix, lang in EXTENSION_TO_LANGUAGE.items():
        assert language_from_url(f"https://ex.com/x{suffix}") == lang


def test_no_text_heuristic_without_extension():
    py_looking = "from __future__ import annotations\n\ndef foo():\n    return 1\n"
    assert language_from_url("https://ex.com/raw") is None
    linear = linear_chunk_code(py_looking, "https://ex.com/raw")
    wrapped = AstCodeChunker().wrap(py_looking, "https://ex.com/raw")
    assert wrapped.annotated_markdown == linear.annotated_markdown


def test_missing_grammar_package_falls_back_to_linear():
    src = "package main\nfunc main() {}\n"
    linear = linear_chunk_code(src, "https://ex.com/main.go")
    wrapped = AstCodeChunker().wrap(src, "https://ex.com/main.go")
    # tree_sitter_go may be absent → linear; if present, AST is allowed.
    try:
        parser_for_language("go")
    except AstChunkError:
        assert wrapped.annotated_markdown == linear.annotated_markdown


def test_parser_registry_caches_and_uses_custom_ts_loader():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_python")
    _PARSER_CACHE.pop("python", None)
    p1 = parser_for_language("python")
    p2 = parser_for_language("python")
    assert p1 is p2
    assert "typescript" in _CUSTOM_LANGUAGE_LOADERS
    assert "tsx" in _CUSTOM_LANGUAGE_LOADERS
    pytest.importorskip("tree_sitter_typescript")
    _PARSER_CACHE.pop("tsx", None)
    tsx = parser_for_language("tsx")
    assert tsx is parser_for_language("tsx")
