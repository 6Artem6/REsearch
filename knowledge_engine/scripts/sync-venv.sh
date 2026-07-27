#!/usr/bin/env bash
# pip в Docker volume ke_python_venv (опциональный API в контейнере).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

docker compose --profile api build knowledge-api
docker compose --profile api run --rm knowledge-api python -c \
  "import sys; import fastapi, langgraph; print('venv:', sys.prefix, 'fastapi', fastapi.__version__)"

echo ""
echo "Python venv: Docker volume ke_python_venv"
echo "Нативный dev: ./knowledge_engine/scripts/dev-native.sh"
echo "Docker API: docker compose --profile api up -d knowledge-api"
