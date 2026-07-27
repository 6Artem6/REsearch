#!/usr/bin/env bash
# Legacy: API в Docker с watch. На Mac лучше dev-native.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Подсказка: на macOS используйте ./knowledge_engine/scripts/dev-native.sh"
echo ""

docker compose up -d searxng
./knowledge_engine/scripts/sync-venv.sh

exec docker compose -f docker-compose.yml -f docker-compose.dev.yml \
  --profile dev up --watch knowledge-api-dev
