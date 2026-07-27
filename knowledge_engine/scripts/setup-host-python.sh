#!/usr/bin/env bash
# Нативный Python venv на хосте (для dev-native / CLI без Docker app).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv"
REQ="${ROOT}/knowledge_engine/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен python3"
  exit 1
fi

if [ ! -x "${VENV}/bin/python" ]; then
  echo "==> python3 -m venv .venv"
  python3 -m venv "${VENV}"
fi

"${VENV}/bin/pip" install -q -U pip wheel
"${VENV}/bin/pip" install -q -r "${REQ}"
"${VENV}/bin/pip" install -q -r "${ROOT}/knowledge_engine/requirements-dev.txt"

echo "==> playwright install chromium (для fallback fetch)"
export PLAYWRIGHT_BROWSERS_PATH=0
"${VENV}/bin/python" -m playwright install chromium

echo ""
echo "Активация:"
echo "  source .venv/bin/activate"
echo "  export PYTHONPATH=\"${ROOT}\""
echo "  python -m knowledge_engine.main analyze -c '...' 'задача'"
