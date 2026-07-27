#!/usr/bin/env bash
# Chromium для Playwright в нативном .venv (.local-browsers рядом с driver).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${ROOT}/.venv"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "Сначала: ./knowledge_engine/scripts/setup-host-python.sh"
  exit 1
fi
export PLAYWRIGHT_BROWSERS_PATH=0
"${VENV}/bin/python" -m playwright install chromium
echo ""
echo "Browsers:"
"${VENV}/bin/python" -c "import pathlib, playwright; p=pathlib.Path(playwright.__file__).parent/'driver/package/.local-browsers'; print(p)"
