import pytest

from knowledge_engine.src.utils.fast_tokenizer import CACHE_DIR, token_counter

_TOKENIZER_VOCAB_OK = (CACHE_DIR / "qwen.tiktoken").is_file() and (
    CACHE_DIR / "gemma_gemini_tokenizer.json"
).is_file()

pytestmark = pytest.mark.skipif(
    not _TOKENIZER_VOCAB_OK,
    reason="Run: python knowledge_engine/src/utils/tokenizers_cache/download_tokenizers.py",
)


def test_token_counter_single():
    text = "Привет, мир! Testing Qwen and Gemini tokenization."

    qwen_count = token_counter.count_tokens(text, "qwen2.5-coder:7b")
    gemini_count = token_counter.count_tokens(text, "gemini-3.5-flash-lite")

    assert qwen_count > 0
    assert gemini_count > 0


def test_token_counter_batch_parallel():
    batch = ["Тестовый промпт " + str(i) for i in range(100)]
    counts = token_counter.count_tokens_batch(batch, "gemini-3.6-flash")

    assert len(counts) == 100
    assert all(c > 0 for c in counts)


def test_estimate_llm_tokens_uses_fast_counter():
    from knowledge_engine.services.gemini_stateless import estimate_llm_tokens

    text = "Привет, мир! Testing trace token estimate."
    assert estimate_llm_tokens(text, "gemini-2.5-flash") > 0
    assert estimate_llm_tokens(text, "qwen2.5-coder:7b") > 0
