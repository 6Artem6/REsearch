#!/usr/bin/env bash
# Сборка Skill Tree UI (локальный bundle, без CDN esm.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${ROOT}/knowledge_engine/web/static/skill-tree"
cd "$DIR"
if ! command -v npm >/dev/null 2>&1; then
  echo "Нужен npm (Node.js) для сборки Skill Tree UI"
  exit 1
fi
npm install --no-fund --no-audit
npm run build
echo "✓ ${DIR}/skill-tree.bundle.js"
