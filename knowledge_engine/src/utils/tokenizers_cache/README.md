# Offline tokenizer vocabularies

Local copies for `fast_tokenizer.py` (no HuggingFace Hub at runtime).

**These files are not in git** (~26 MB). Download once after installing Python deps.

| File | Source | Notes |
|------|--------|--------|
| `qwen.tiktoken` | [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B) | BPE ranks shared across Qwen1.5–Qwen2.5 |
| `qwen2.5.json` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | HF `tokenizer.json` backup reference |
| `gemma_gemini_tokenizer.json` | [mlx-community/gemma-2-2b-it-4bit](https://huggingface.co/mlx-community/gemma-2-2b-it-4bit) | Gemma 2 SP/BPE; Gemma + Gemini aliases |

## First-time setup

From the repository root:

```bash
python knowledge_engine/src/utils/tokenizers_cache/download_tokenizers.py
```

Or with curl:

```bash
CACHE=knowledge_engine/src/utils/tokenizers_cache
mkdir -p "$CACHE"
curl -sfL -o "$CACHE/qwen.tiktoken" \
  https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen.tiktoken
curl -sfL -o "$CACHE/qwen2.5.json" \
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer.json
curl -sfL -o "$CACHE/gemma_gemini_tokenizer.json" \
  https://huggingface.co/mlx-community/gemma-2-2b-it-4bit/resolve/main/tokenizer.json
```

Refresh: run the same script or curl block again.
