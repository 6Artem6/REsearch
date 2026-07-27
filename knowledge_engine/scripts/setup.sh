#!/usr/bin/env bash
# Полная подготовка хоста: SearXNG (Docker) + Ollama (Metal) + Python .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Docker: только SearXNG"
docker compose up -d searxng

"${ROOT}/knowledge_engine/scripts/setup-host-ollama.sh"
"${ROOT}/knowledge_engine/scripts/setup-host-python.sh"

echo ""
echo "Запуск dev API:"
echo "  ./knowledge_engine/scripts/dev-native.sh"
