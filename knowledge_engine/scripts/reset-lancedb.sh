#!/usr/bin/env bash
# Сброс LanceDB на хосте (несовместимость версий lance / смешение host vs Docker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LANCE="${ROOT}/knowledge_engine/.lancedb"
if [ ! -d "$LANCE" ]; then
  echo "Нет ${LANCE}"
  exit 0
fi
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP="${ROOT}/knowledge_engine/.lancedb.bak-${STAMP}"
mv "$LANCE" "$BACKUP"
mkdir -p "$LANCE"
echo "Backup: $BACKUP"
echo "Перезапустите analyze или: .venv/bin/python -c 'from knowledge_engine.services.vector_store import VectorStore; VectorStore()'"
