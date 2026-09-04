"""AST local-import expansion for GitHub blob ingest (no regex over source)."""

from __future__ import annotations

import ast
import inspect

import httpx
import pytest

from knowledge_engine.ingest.dependency_resolver import (
    MAX_LOCAL_DEPENDENCIES,
    DependencyResolver,
    extract_local_imports,
    format_blob_with_supporting_context,
    maybe_fetch_github_blob_with_deps,
    resolve_dependency_paths,
)
from knowledge_engine.services.article_ingestion.github_tree_loader import (
    parse_github_blob_url,
)


def test_python_ast_relative_imports_extracted():
    code = """
from . import utils
from ..config import settings
from ...pkg.core import x
"""
    specs = extract_local_imports(code, "python")
    assert ".utils" in specs
    assert "..config" in specs
    assert "...pkg.core" in specs


def test_python_ast_strips_external_packages():
    code = """
import os
import numpy
from torch import nn
import utils
from numpy.linalg import norm
"""
    specs = extract_local_imports(code, "python")
    joined = " ".join(specs)
    assert "numpy" not in joined
    assert "os" not in joined
    assert "torch" not in joined
    assert specs == []


def test_python_same_package_absolute_when_target_known():
    code = "from markupsafe.element import Markup\nimport markupsafe._speedups\n"
    specs = extract_local_imports(
        code, "python", target_path="src/markupsafe/_native.py"
    )
    assert "markupsafe.element" in specs
    assert "markupsafe._speedups" in specs


def test_resolve_python_relative_paths():
    tree = [
        "src/pkg/mod.py",
        "src/pkg/utils.py",
        "src/pkg/utils/__init__.py",
        "src/config.py",
        "src/config/__init__.py",
    ]
    resolved = resolve_dependency_paths(
        tree,
        "src/pkg/mod.py",
        [".utils", "..config"],
        language="python",
    )
    assert "src/pkg/utils.py" in resolved
    assert "src/config.py" in resolved


def test_resolve_caps_at_five_files():
    tree = [f"pkg/dep{i}.py" for i in range(8)] + ["pkg/mod.py"]
    specs = [f".dep{i}" for i in range(8)]
    resolved = resolve_dependency_paths(
        tree, "pkg/mod.py", specs, language="python"
    )
    assert len(resolved) == MAX_LOCAL_DEPENDENCIES
    assert resolved == [f"pkg/dep{i}.py" for i in range(5)]
    assert DependencyResolver(max_files=5).max_files == 5


def test_extract_does_not_import_regex():
    import knowledge_engine.ingest.dependency_resolver as mod

    source = inspect.getsource(mod)
    assert "import re\n" not in source
    assert "import re " not in source
    ast.parse(source)


def test_c_quote_include_not_angle(monkeypatch):
    pytest.importorskip("tree_sitter_c")
    code = """
#include <stdio.h>
#include "pycore_gil.h"
#include "Python/ceval.h"
"""
    specs = extract_local_imports(code, "c")
    assert "stdio.h" not in specs
    assert "pycore_gil.h" in specs
    assert "Python/ceval.h" in specs


def test_c_filename_match_in_tree():
    pytest.importorskip("tree_sitter_c")
    tree = [
        "Python/ceval_gil.c",
        "Include/internal/pycore_gil.h",
        "Include/Python.h",
    ]
    specs = extract_local_imports(
        '#include "pycore_gil.h"\n#include <stdio.h>\n',
        "c",
        target_path="Python/ceval_gil.c",
    )
    resolved = resolve_dependency_paths(
        tree, "Python/ceval_gil.c", specs, language="c"
    )
    assert resolved == ["Include/internal/pycore_gil.h"]


def test_js_relative_only():
    code = """
import fs from 'fs';
import helper from './helper.js';
const x = require('../lib/util');
"""
    specs = extract_local_imports(code, "javascript")
    assert any(s.startswith("./helper") for s in specs)
    assert any(s.startswith("../lib/util") for s in specs)
    assert not any(s == "fs" or s.endswith("/fs") for s in specs)


def test_supporting_context_wrapper():
    text = format_blob_with_supporting_context(
        "int main(void) { return 0; }\n",
        [("Include/foo.h", "#define FOO 1\n")],
    )
    assert "[Supporting Context: Include/foo.h]" in text
    assert "#define FOO 1" in text
    assert "int main" in text


def test_parse_github_blob_url_no_repo_root():
    assert parse_github_blob_url("https://github.com/python/cpython") is None
    got = parse_github_blob_url(
        "https://github.com/python/cpython/blob/main/Python/ceval_gil.c"
    )
    assert got == ("python", "cpython", "main", "Python/ceval_gil.c")


def test_maybe_fetch_blob_with_deps_uses_trees_loader(monkeypatch):
    import knowledge_engine.config as ke_config
    from knowledge_engine.services.article_ingestion import github_tree_loader as gtl

    monkeypatch.setattr(ke_config, "USE_GITHUB_TREES_API", True)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/git/trees/" in url:
            return httpx.Response(
                200,
                json={
                    "truncated": False,
                    "tree": [
                        {
                            "path": "pkg/mod.py",
                            "type": "blob",
                            "size": 80,
                            "sha": "a",
                        },
                        {
                            "path": "pkg/utils.py",
                            "type": "blob",
                            "size": 40,
                            "sha": "b",
                        },
                        {
                            "path": "pkg/__pycache__/x.pyc",
                            "type": "blob",
                            "size": 10,
                            "sha": "c",
                        },
                    ],
                },
            )
        if url.endswith("pkg/mod.py"):
            return httpx.Response(200, text="from . import utils\n")
        if url.endswith("pkg/utils.py"):
            return httpx.Response(200, text="VALUE = 1\n")
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = gtl.GitHubTreeLoader(client=client, token="t")
    monkeypatch.setattr(gtl, "GitHubTreeLoader", lambda *a, **k: loader)

    url = "https://github.com/acme/lib/blob/main/pkg/mod.py"
    got = maybe_fetch_github_blob_with_deps(url)
    assert got is not None
    text, method = got
    assert method == "github_blob"
    assert "from . import utils" in text
    assert "[Supporting Context: pkg/utils.py]" in text
    assert "VALUE = 1" in text


def test_maybe_fetch_blob_noop_when_flag_off(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "USE_GITHUB_TREES_API", False)
    assert (
        maybe_fetch_github_blob_with_deps(
            "https://github.com/acme/lib/blob/main/pkg/mod.py"
        )
        is None
    )
