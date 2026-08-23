"""HuggingFace snapshot: cache-first, network only on miss."""

from __future__ import annotations

from knowledge_engine.services.hf_model_cache import resolve_hf_snapshot


def test_resolve_uses_local_snapshot_without_download(monkeypatch):
    calls: list[dict] = []

    def fake_snapshot_download(*, local_files_only=False, **kwargs):
        calls.append({"local_files_only": local_files_only, **kwargs})
        if local_files_only:
            return "/tmp/hf-cache/bge-m3"
        raise AssertionError("must not hit the Hub when cache exists")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    path = resolve_hf_snapshot("BAAI/bge-m3", revision="abc123")
    assert path == "/tmp/hf-cache/bge-m3"
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == "BAAI/bge-m3"
    assert calls[0]["revision"] == "abc123"


def test_resolve_downloads_once_on_cache_miss(monkeypatch):
    calls: list[dict] = []

    def fake_snapshot_download(*, local_files_only=False, **kwargs):
        calls.append({"local_files_only": local_files_only, **kwargs})
        if local_files_only:
            raise FileNotFoundError("not in cache")
        return "/tmp/hf-cache/downloaded"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    path = resolve_hf_snapshot("BAAI/bge-m3", revision="abc123")
    assert path == "/tmp/hf-cache/downloaded"
    assert [c["local_files_only"] for c in calls] == [True, False]
