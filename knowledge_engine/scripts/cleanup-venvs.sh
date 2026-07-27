#!/usr/bin/env bash
# Удалить устаревшие venv (не трогаем REsearch/.venv — нативный dev).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

for path in ".venv-host" "knowledge_engine/.venv"; do
  if [ -e "${path}" ]; then
    echo "remove: ${path}"
    rm -rf "${path}"
  fi
done
echo "OK. Нативный venv: ./knowledge_engine/scripts/setup-host-python.sh"
echo "    Docker API venv: ./knowledge_engine/scripts/sync-venv.sh"
