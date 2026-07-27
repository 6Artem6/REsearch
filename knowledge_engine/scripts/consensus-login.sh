#!/usr/bin/env bash
# Вход в Consensus с тем же Chromium, что dev-native (.venv/.local-browsers).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

set -a
[ -f .env ] && source .env
set +a

export PYTHONPATH="${ROOT}"

if [ ! -x "${ROOT}/.venv/bin/python" ]; then
  echo "Нет .venv — ./knowledge_engine/scripts/setup-host-python.sh"
  exit 1
fi

if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
  export PLAYWRIGHT_BROWSERS_PATH="$("${ROOT}/.venv/bin/python" -c \
    "import pathlib, playwright; print(pathlib.Path(playwright.__file__).parent / 'driver/package/.local-browsers')")"
fi

if [ ! -d "${PLAYWRIGHT_BROWSERS_PATH}" ]; then
  echo "==> Chromium не найден — install-playwright.sh"
  "${ROOT}/knowledge_engine/scripts/install-playwright.sh"
fi

echo "PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH}"
exec "${ROOT}/.venv/bin/python" -m knowledge_engine.main consensus-login
