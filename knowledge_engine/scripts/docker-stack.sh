#!/usr/bin/env bash
# SearXNG в Docker; Ollama и Python — на хосте.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
docker compose up -d searxng
echo ""
echo "Далее:"
echo "  ./knowledge_engine/scripts/setup-host-ollama.sh"
echo "  ./knowledge_engine/scripts/setup-host-python.sh"
echo "  ./knowledge_engine/scripts/dev-native.sh"
echo ""
echo "Опционально API в Docker:"
echo "  ./knowledge_engine/scripts/sync-venv.sh"
echo "  docker compose --profile api up -d knowledge-api"
