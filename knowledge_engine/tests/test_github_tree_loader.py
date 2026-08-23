"""GitHub Trees API ingest: URL parse, tree filter, zip fallback."""

from __future__ import annotations

import httpx

from knowledge_engine.services.article_ingestion.github_tree_loader import (
    GitHubTreeLoader,
    is_github_tree_ingest_url,
    maybe_fetch_github_repo_corpus,
    parse_github_url,
)


def test_parse_github_url_owner_repo_ref():
    assert parse_github_url("https://github.com/owner/repo") == (
        "owner",
        "repo",
        "main",
    )
    assert parse_github_url("https://github.com/owner/repo.git") == (
        "owner",
        "repo",
        "main",
    )
    assert parse_github_url("https://github.com/owner/repo/tree/main") == (
        "owner",
        "repo",
        "main",
    )
    assert parse_github_url("https://www.github.com/acme/lib/tree/develop") == (
        "acme",
        "lib",
        "develop",
    )
    assert parse_github_url("https://github.com/o/r/commit/abc123") == (
        "o",
        "r",
        "abc123",
    )
    assert GitHubTreeLoader.parse_github_url("https://github.com/o/r/tree/v1") == (
        "o",
        "r",
        "v1",
    )


def test_is_github_tree_ingest_url_skips_blob_and_issues():
    assert is_github_tree_ingest_url("https://github.com/owner/repo")
    assert is_github_tree_ingest_url("https://github.com/owner/repo/tree/main")
    assert not is_github_tree_ingest_url(
        "https://github.com/owner/repo/blob/main/src/a.py"
    )
    assert not is_github_tree_ingest_url("https://github.com/owner/repo/issues/1")
    assert not is_github_tree_ingest_url("https://example.com/owner/repo")


def test_filter_tree_items_drops_vendor_and_oversize():
    loader = GitHubTreeLoader(max_file_size=102400)
    items = [
        {"path": "src/app.py", "type": "blob", "size": 80, "sha": "a"},
        {"path": "README.md", "type": "blob", "size": 40, "sha": "m"},
        {"path": "node_modules/pkg/index.js", "type": "blob", "size": 10, "sha": "n"},
        {"path": "vendor/lib.go", "type": "blob", "size": 10, "sha": "v"},
        {"path": "dist/bundle.js", "type": "blob", "size": 10, "sha": "d"},
        {"path": "heavy.py", "type": "blob", "size": 200000, "sha": "h"},
        {"path": "src/pkg/__pycache__/x.pyc", "type": "blob", "size": 10, "sha": "p"},
        {"path": "logo.png", "type": "blob", "size": 50, "sha": "i"},
        {"path": "src", "type": "tree", "sha": "t"},
    ]
    kept = loader.filter_tree_items(items)
    paths = {str(x["path"]) for x in kept}
    assert paths == {"src/app.py", "README.md"}


def test_load_repository_files_fetches_filtered_blobs(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "USE_GITHUB_TREES_API", True)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/git/trees/" in url and "recursive=1" in url:
            return httpx.Response(
                200,
                json={
                    "truncated": False,
                    "tree": [
                        {
                            "path": "pkg/mod.py",
                            "type": "blob",
                            "size": 24,
                            "sha": "s1",
                        },
                        {
                            "path": "node_modules/x.js",
                            "type": "blob",
                            "size": 8,
                            "sha": "s2",
                        },
                    ],
                },
            )
        if "raw.githubusercontent.com" in url and url.endswith("pkg/mod.py"):
            return httpx.Response(200, text="def hello():\n    return 1\n")
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = GitHubTreeLoader(client=client, token="t0ken")
    files = loader.load_repository_files("https://github.com/acme/lib")
    assert not loader.used_zip_fallback
    assert len(files) == 1
    assert files[0]["path"] == "pkg/mod.py"
    assert "def hello" in files[0]["content"]


def test_trees_http_403_falls_back_to_zip(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "/git/trees/" in str(request.url):
            return httpx.Response(403, json={"message": "API rate limit exceeded"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = GitHubTreeLoader(client=client)

    def _zip(_owner: str, _repo: str, _ref: str):
        return [
            {
                "path": "ok.py",
                "content": "x = 1\n",
                "size": 6,
                "sha": "",
                "url": "https://github.com/o/r/blob/main/ok.py",
            }
        ]

    monkeypatch.setattr(loader, "_load_via_zip", _zip)
    files = loader.load_repository_files("https://github.com/o/r")
    assert loader.used_zip_fallback is True
    assert files[0]["path"] == "ok.py"


def test_maybe_fetch_github_repo_corpus_noop_when_flag_off(monkeypatch):
    import knowledge_engine.config as ke_config

    monkeypatch.setattr(ke_config, "USE_GITHUB_TREES_API", False)
    assert maybe_fetch_github_repo_corpus("https://github.com/owner/repo") is None
