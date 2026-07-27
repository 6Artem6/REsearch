#!/bin/sh
# Python venv только в Docker volume: /app/.venv (ke_python_venv).
# На хосте не создаём .venv — избегаем дубля с knowledge_engine/.venv.
set -e

VENV="${VIRTUAL_ENV:-/app/.venv}"
REQ="/app/knowledge_engine/requirements.txt"
MARKER="${VENV}/.ke-deps-installed"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "==> knowledge-engine: create venv in volume ${VENV}"
  python3 -m venv "${VENV}"
fi

export PATH="${VENV}/bin:${PATH}"

"${VENV}/bin/pip" install -q -U pip wheel

if [ -f "${REQ}" ]; then
  if [ ! -f "${MARKER}" ] || [ "${REQ}" -nt "${MARKER}" ]; then
    echo "==> knowledge-engine: pip install -r requirements.txt"
    "${VENV}/bin/pip" install -r "${REQ}"
    touch "${MARKER}"
  fi
fi

if command -v playwright >/dev/null 2>&1; then
  if [ ! -d /root/.cache/ms-playwright ] || [ -z "$(ls -A /root/.cache/ms-playwright 2>/dev/null)" ]; then
    echo "==> knowledge-engine: playwright chromium"
    playwright install --with-deps chromium
  fi
fi

exec "$@"
