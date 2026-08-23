"""Three-tier AST + Flash Lite code prune before Gemma MAP."""

from __future__ import annotations

from pathlib import Path

from knowledge_engine.ingest.dependency_resolver import (
    SUPPORTING_CONTEXT_MARK,
    format_blob_with_supporting_context,
)
from knowledge_engine.ingest.tiered_code_pruner import (
    TieredClassificationResult,
    assemble_tiered_context,
    classify_code_tiers_flash_lite,
    extract_ast_signatures_and_calls,
    extract_ast_signatures_and_calls_from_text,
    extract_functions_from_source,
    maybe_prune_code_for_map,
    split_combined_blob,
)

_CEVAL_FRAGMENT = """\
#include "Python.h"
#include "pycore_gil.h"

/* Acquire the GIL for this thread. */
static void take_gil(PyThreadState *tstate)
{
    struct gil_runtime_state *gil = tstate->interp->runtime->ceval.gil;
    MUTEX_LOCK(gil->mutex);
    gil_helper(gil);
    _Py_atomic_store_relaxed(&gil->locked, 1);
}

/* Drop the GIL and restore thread state. */
static void drop_gil(PyThreadState *tstate)
{
    take_gil(tstate);
    MUTEX_UNLOCK(tstate->interp->runtime->ceval.gil->mutex);
}

/* Thin wrapper around the lock helper. */
static void gil_helper(struct gil_runtime_state *gil)
{
    take_gil_check(gil);
    take_gil(NULL);
}

static void *mem_alloc(size_t n)
{
    return malloc(n);
}
"""

_DEP_HEADER = """\
void _Py_atomic_store_relaxed(void *p, int v);
void take_gil_check(struct gil_runtime_state *gil);
"""


def _write(tmp: Path, name: str, body: str) -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


def test_extract_functions_from_ceval_fragment():
    spans = extract_functions_from_source(_CEVAL_FRAGMENT, "c")
    names = [s.name for s in spans]
    assert "take_gil" in names
    assert "drop_gil" in names
    assert "gil_helper" in names
    assert "mem_alloc" in names
    take = next(s for s in spans if s.name == "take_gil")
    assert "MUTEX_LOCK" in take.body
    assert take.start_line >= 1
    assert take.end_line > take.start_line
    callees = {name for name, _line in take.calls}
    assert "gil_helper" in callees or "_Py_atomic_store_relaxed" in callees


def test_extract_ast_signatures_and_calls_deps_have_no_bodies(tmp_path: Path):
    target = _write(tmp_path, "ceval_gil.c", _CEVAL_FRAGMENT)
    dep = _write(tmp_path, "pycore_gil.h", _DEP_HEADER)
    graph = extract_ast_signatures_and_calls(str(target), [str(dep)])
    assert any(fn.name == "take_gil" for fn in graph.target_functions)
    assert graph.dep_signatures
    assert any("_Py_atomic_store_relaxed" in sig for _, sig in graph.dep_signatures)
    assert all("{" not in (sig or "") for _, sig in graph.dep_signatures)


def test_assemble_drops_low_keeps_high_body_and_medium_stub():
    graph = extract_ast_signatures_and_calls_from_text(
        _CEVAL_FRAGMENT,
        target_path="ceval_gil.c",
        dep_files=[("pycore_gil.h", _DEP_HEADER)],
    )
    classification = TieredClassificationResult(
        high_functions=["take_gil", "drop_gil"],
        medium_functions=["gil_helper"],
        low_functions=["mem_alloc"],
    )
    out = assemble_tiered_context(
        _CEVAL_FRAGMENT, classification, graph, lang="c"
    )
    assert "## AST External Signatures & Cross-Calls" in out
    assert "## High & Medium Priority Code Engine" in out
    assert "static void take_gil" in out
    assert "MUTEX_LOCK(gil->mutex)" in out
    assert "static void drop_gil" in out
    assert "static void gil_helper" in out
    assert "{ /* body omitted */ }" in out
    assert "Calls HIGH" in out
    assert "malloc(n)" not in out
    assert "mem_alloc" not in out
    assert "Calls HIGH: take_gil() at line" in out


def test_classify_uses_flash_lite_contract(monkeypatch):
    captured: dict[str, object] = {}

    def _lite(system, user, anchor, schema, label, **_k):
        captured["system"] = system
        captured["schema"] = schema
        captured["user"] = user
        return TieredClassificationResult(
            high_functions=["take_gil"],
            medium_functions=["gil_helper"],
            low_functions=["mem_alloc", "drop_gil"],
        )

    monkeypatch.setattr(
        "knowledge_engine.src.analytics.gemini_v07.run_gemini_lite_structured",
        _lite,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available",
        lambda: True,
    )
    names = ["take_gil", "drop_gil", "gil_helper", "mem_alloc"]
    result = classify_code_tiers_flash_lite(
        _CEVAL_FRAGMENT, function_names=names
    )
    assert captured["schema"] is TieredClassificationResult
    assert "HIGH" in str(captured["system"])
    assert result.high_functions == ["take_gil"]
    assert "gil_helper" in result.medium_functions
    assert "drop_gil" in result.low_functions
    assert "mem_alloc" in result.low_functions


def test_maybe_prune_code_for_map_uses_classification(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "knowledge_engine.ingest.tiered_code_pruner.classify_code_tiers_flash_lite",
        lambda *_a, **_k: TieredClassificationResult(
            high_functions=["take_gil", "drop_gil"],
            medium_functions=["gil_helper"],
            low_functions=["mem_alloc"],
        ),
    )
    combined = format_blob_with_supporting_context(
        _CEVAL_FRAGMENT, [("Include/pycore_gil.h", _DEP_HEADER)]
    )
    target, deps = split_combined_blob(combined)
    assert SUPPORTING_CONTEXT_MARK in combined
    assert "take_gil" in target
    assert deps and deps[0][0].endswith("pycore_gil.h")
    out = maybe_prune_code_for_map(
        combined,
        "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c",
    )
    assert "malloc(n)" not in out
    assert "MUTEX_LOCK" in out
    assert "mem_alloc" not in out
    assert "## High & Medium Priority Code Engine" in out
    assert "pycore_gil.h" in out


def test_maybe_prune_skips_when_gemini_unavailable(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available",
        lambda: False,
    )
    combined = format_blob_with_supporting_context(
        _CEVAL_FRAGMENT, [("Include/pycore_gil.h", _DEP_HEADER)]
    )
    out = maybe_prune_code_for_map(
        combined,
        "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c",
    )
    assert out == combined
