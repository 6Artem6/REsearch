"""API process must not load local embedding / reranker weights."""

from __future__ import annotations

import pytest

from knowledge_engine.services.ml_runtime import (
    assert_ml_weights_allowed,
    ml_weights_allowed,
)
from knowledge_engine.services.search import bge_m3_embed
from knowledge_engine.src.rag_gateway import cross_encoder


def test_api_role_disallows_ml_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KE_PROCESS_ROLE", "api")
    assert ml_weights_allowed() is False
    with pytest.raises(RuntimeError, match="worker"):
        assert_ml_weights_allowed("BGE-M3 embeddings")
    with pytest.raises(RuntimeError, match="worker"):
        bge_m3_embed._get_model()
    with pytest.raises(RuntimeError, match="worker"):
        cross_encoder._create_cross_encoder()


def test_worker_role_allows_ml_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KE_PROCESS_ROLE", "worker")
    assert ml_weights_allowed() is True
    assert_ml_weights_allowed("BGE-M3 embeddings")


def test_unset_role_allows_ml_weights_for_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KE_PROCESS_ROLE", raising=False)
    assert ml_weights_allowed() is True
