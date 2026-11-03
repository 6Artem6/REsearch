"""knowledge_engine.llm: структурные вызовы primary-модели идут через пул
(fallback + circuit breaker), явная другая модель — одиночный клиент.

Живой случай: doc_triage (article_pruner) звал одиночный GemmaCloudClient
gemma-4-31b-it — при 500/503 primary обрезка разделов просто пропускалась,
хотя gemma-4-26b-a4b-it отвечала."""

from __future__ import annotations

from knowledge_engine import llm
from knowledge_engine.src.adapters.llm_providers.gemma_client import (
    GemmaCloudClient,
    RateLimitedLLMClient,
)
from knowledge_engine.src.config.settings import GEMMA_PRIMARY_MODEL


def test_primary_and_legacy_names_use_pool() -> None:
    for model in (None, "", GEMMA_PRIMARY_MODEL, "qwen2.5:7b"):
        assert isinstance(llm._structured_client(model), RateLimitedLLMClient)


def test_explicit_other_model_keeps_single_client() -> None:
    client = llm._structured_client("gemma-4-26b-a4b-it")
    assert isinstance(client, GemmaCloudClient)
    assert client._model == "gemma-4-26b-a4b-it"
