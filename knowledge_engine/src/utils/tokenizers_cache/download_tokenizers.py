#!/usr/bin/env python3
"""Download offline tokenizer vocab files into this directory (not tracked in git)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent

FILES: dict[str, str] = {
    "qwen.tiktoken": (
        "https://huggingface.co/Qwen/Qwen-14B/resolve/main/qwen.tiktoken"
    ),
    "qwen2.5.json": (
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer.json"
    ),
    "gemma_gemini_tokenizer.json": (
        "https://huggingface.co/mlx-community/gemma-2-2b-it-4bit/resolve/main/tokenizer.json"
    ),
}


def download(name: str, url: str) -> None:
    dest = CACHE_DIR / name
    print(f"-> {name}")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)
    print("Done.", CACHE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
