#!/usr/bin/env bash
# Просмотр job: Rich-матрица / unravel. Без аргументов — last-wait-response.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
LAST="${ROOT}/knowledge_engine/.runs/last-wait-response.json"

if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = "--no-interactive" ]; }; then
  exec "$PYTHON" -m knowledge_engine.cli.job_view -f "$LAST" "${@:1}"
fi

# Совместимость: ./view-job.sh JOB_ID [опции] → --id JOB_ID
if [[ "$1" != -* ]] && [[ "$1" != --* ]]; then
  set -- --id "$1" "${@:2}"
fi

exec "$PYTHON" -m knowledge_engine.cli.job_view "$@"
