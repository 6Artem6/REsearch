#!/usr/bin/env bash
# Создать job, long poll до матрицы, Rich-вывод + опциональный unravel.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
BASE="${KE_API_BASE:-http://127.0.0.1:${KE_API_PORT:-8765}}"
PROBLEM="${1:?usage: wait-analysis.sh \"вопрос\" [constraints] [timeout_sec]}"
CONSTRAINTS="${2:-}"
TIMEOUT="${3:-600}"
PYTHON="${ROOT}/.venv/bin/python"
VIEW="${PYTHON} -m knowledge_engine.cli.job_view"

INTERACTIVE="${KE_JOB_INTERACTIVE:-1}"
JSON_ONLY="${KE_JOB_JSON:-0}"

BODY=$(jq -n \
  --arg problem "$PROBLEM" \
  --arg constraints "$CONSTRAINTS" \
  '{problem: $problem, constraints: $constraints, async_mode: true}')

echo "POST /analyses"
CREATE=$(curl -sf -X POST "${BASE}/api/v1/analyses" -H 'Content-Type: application/json' -d "$BODY")
JOB=$(echo "$CREATE" | jq -r '.job.id')
echo "job=$JOB"
echo "Long poll GET /analyses/${JOB}/wait?timeout_sec=${TIMEOUT}&target=matrix"
echo ""

WAIT_JSON=$(curl -sf "${BASE}/api/v1/analyses/${JOB}/wait?timeout_sec=${TIMEOUT}&interval_sec=2&target=matrix")
LAST="${ROOT}/knowledge_engine/.runs/last-wait-response.json"
printf '%s' "$WAIT_JSON" > "$LAST"
echo "saved: $LAST"
echo ""

if [ ! -x "$PYTHON" ]; then
  echo "$WAIT_JSON" | jq .
  exit 0
fi

if [ "$JSON_ONLY" = "1" ]; then
  "$PYTHON" -m knowledge_engine.cli.job_view -f "$LAST" --json --no-interactive
  exit 0
fi

if [ "$INTERACTIVE" = "1" ]; then
  "$PYTHON" -m knowledge_engine.cli.job_view -f "$LAST" --interactive
else
  "$PYTHON" -m knowledge_engine.cli.job_view -f "$LAST" --no-interactive
fi
