# Оффлайн-модуль быстрого параллельного подсчёта токенов.
# Qwen 2.5 — tiktoken (локальный qwen.tiktoken); Gemma/Gemini — Rust tokenizers (GIL release).

from __future__ import annotations

from pathlib import Path
from typing import List

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer

CACHE_DIR = Path(__file__).parent / "tokenizers_cache"

_QWEN_PAT_STR = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

_QWEN_SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|" + "im_end" + "|>": 151645,
}


class FastTokenCounter:
    _instance: FastTokenCounter | None = None

    def __new__(cls) -> FastTokenCounter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tokenizers()
        return cls._instance

    def _init_tokenizers(self) -> None:
        qwen_file = CACHE_DIR / "qwen.tiktoken"
        if qwen_file.is_file():
            mergeable_ranks = load_tiktoken_bpe(str(qwen_file))
            self.qwen_enc = tiktoken.Encoding(
                name="qwen2.5",
                pat_str=_QWEN_PAT_STR,
                mergeable_ranks=mergeable_ranks,
                special_tokens=_QWEN_SPECIAL_TOKENS,
            )
        else:
            self.qwen_enc = None

        gemma_gemini_file = CACHE_DIR / "gemma_gemini_tokenizer.json"
        if gemma_gemini_file.is_file():
            self.google_enc = Tokenizer.from_file(str(gemma_gemini_file))
        else:
            self.google_enc = None

    def count_tokens(self, text: str, model_alias: str) -> int:
        if not text:
            return 0

        model_lower = model_alias.lower()

        if "qwen" in model_lower and self.qwen_enc:
            return len(self.qwen_enc.encode(text, disallowed_special=()))

        if any(m in model_lower for m in ("gemma", "gemini")) and self.google_enc:
            return len(self.google_enc.encode(text).ids)

        return self._fallback_heuristic(text)

    def count_tokens_batch(self, texts: List[str], model_alias: str) -> List[int]:
        if not texts:
            return []

        model_lower = model_alias.lower()

        if any(m in model_lower for m in ("gemma", "gemini")) and self.google_enc:
            encodings = self.google_enc.encode_batch(texts)
            return [len(enc.ids) for enc in encodings]

        if "qwen" in model_lower and self.qwen_enc:
            return [len(self.qwen_enc.encode(t, disallowed_special=())) for t in texts]

        return [self._fallback_heuristic(t) for t in texts]

    @staticmethod
    def _fallback_heuristic(text: str) -> int:
        cyr_len = len([c for c in text if "\u0400" <= c <= "\u04ff"])
        other_len = len(text) - cyr_len
        return int(cyr_len / 1.75 + other_len / 3.7)


token_counter = FastTokenCounter()
