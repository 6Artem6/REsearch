#!/usr/bin/env bash
# Ollama на macOS (Metal GPU) — не в Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama не найден. Установите:"
  echo "  brew install ollama"
  echo "  brew services start ollama"
  exit 1
fi

echo "==> Ollama: проверка http://localhost:11434"
if ! curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "Сервис не отвечает. Запустите:"
  echo "  brew services start ollama"
  echo "  или: ollama serve"
  exit 1
fi

echo "==> Pull моделей (имена как в config.py / LOCAL_*_MODEL)"
# Алиасы qwen2.5:* при необходимости — подтяните вручную; в проекте — coder-теги.
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b

echo ""
echo "Готово. OLLAMA_BASE_URL=http://localhost:11434"
ollama list
