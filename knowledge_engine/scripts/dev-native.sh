#!/usr/bin/env bash
# Dev на хосте: SearXNG в Docker + API с uvicorn --reload (Metal Ollama).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
set -a
[ -f .env ] && source .env
set +a

export PYTHONPATH="${ROOT}"
export KE_API_RELOAD="${KE_API_RELOAD:-true}"
export KE_TRACE_STDOUT="${KE_TRACE_STDOUT:-true}"
export KE_LOG_PLAIN="${KE_LOG_PLAIN:-true}"
export KE_API_HOST="${KE_API_HOST:-127.0.0.1}"
export KE_API_PORT="${KE_API_PORT:-8765}"
export SEARXNG_BASE_URL="${SEARXNG_BASE_URL:-http://localhost:8080}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "==> SearXNG + Redis (Docker)"
docker compose up -d searxng redis

export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"

if [ ! -x "${ROOT}/.venv/bin/python" ]; then
  echo "==> Нет .venv — setup-host-python.sh"
  "${ROOT}/knowledge_engine/scripts/setup-host-python.sh"
fi

# Playwright 1.61 → chromium-1228 в .venv (не в ~/Library/Caches/ms-playwright)
if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
  export PLAYWRIGHT_BROWSERS_PATH="$("${ROOT}/.venv/bin/python" -c \
    "import pathlib, playwright; print(pathlib.Path(playwright.__file__).parent / 'driver/package/.local-browsers')")"
fi

if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "WARN: Ollama не отвечает на ${OLLAMA_BASE_URL} — запустите setup-host-ollama.sh"
fi

echo ""
echo "==> KE Worker (watch → auto-reload на .py, как API)"
if pgrep -f "[-m ]knowledge_engine\\.worker" >/dev/null 2>&1; then
  echo "    WARN: останавливаем старые процессы knowledge_engine.worker"
  pkill -f "[-m ]knowledge_engine\\.worker" 2>/dev/null || true
  sleep 0.6
fi
if pgrep -f "dev_worker_watch\\.py" >/dev/null 2>&1; then
  pkill -f "dev_worker_watch\\.py" 2>/dev/null || true
  sleep 0.3
fi
"${ROOT}/.venv/bin/python" "${ROOT}/knowledge_engine/scripts/dev_worker_watch.py" &
WORKER_WATCH_PID=$!
echo "    worker watch pid=$WORKER_WATCH_PID (reload при правках knowledge_engine/**/*.py)"
trap 'kill "$WORKER_WATCH_PID" 2>/dev/null || true; pkill -f "[-m ]knowledge_engine\\.worker" 2>/dev/null || true' EXIT INT TERM

echo ""
echo "==> Native API reload → http://${KE_API_HOST}:${KE_API_PORT}/docs"
echo "    trace: tail -f knowledge_engine/.runs/*.log"
echo ""

exec "${ROOT}/.venv/bin/python" -m knowledge_engine.api
