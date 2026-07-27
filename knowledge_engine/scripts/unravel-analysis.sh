#!/usr/bin/env bash
# Unravel для уже готовой матрицы (без нового analyze).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
BASE="${KE_API_BASE:-http://127.0.0.1:${KE_API_PORT:-8765}}"
JOB_ID="${1:?usage: unravel-analysis.sh JOB_ID OPTION_ID [timeout_sec]}"
OPTION_ID="${2:?usage: unravel-analysis.sh JOB_ID OPTION_ID}"
TIMEOUT="${3:-600}"
FORCE="${UNRAVEL_FORCE:-0}"
PYTHON="${ROOT}/.venv/bin/python"
LAST="${ROOT}/knowledge_engine/.runs/last-wait-response.json"

_fr=false
if [ "$FORCE" = "1" ] || [ "$FORCE" = "true" ]; then
  _fr=true
fi

echo "POST /analyses/${JOB_ID}/unravel option_id=${OPTION_ID}"
POST_BODY=$(jq -n --argjson oid "$OPTION_ID" --argjson fr "$_fr" \
  '{option_id: $oid, async_mode: true, force_rerun: $fr}')

POST_RAW=$(curl -s -w "\n%{http_code}" -X POST "${BASE}/api/v1/analyses/${JOB_ID}/unravel" \
  -H 'Content-Type: application/json' \
  -d "$POST_BODY")
POST_CODE=$(echo "$POST_RAW" | tail -n1)
POST_JSON=$(echo "$POST_RAW" | sed '$d')

if [ "$POST_CODE" != "200" ]; then
  echo "HTTP ${POST_CODE}:" >&2
  echo "$POST_JSON" | jq . 2>/dev/null || echo "$POST_JSON" >&2
  echo "" >&2
  echo "Подсказка: job уже completed? Смотрите результат:" >&2
  echo "  ./knowledge_engine/scripts/view-job.sh --id ${JOB_ID} --no-interactive" >&2
  echo "  ./knowledge_engine/scripts/view-job.sh --no-interactive" >&2
  exit 1
fi

STATUS=$(echo "$POST_JSON" | jq -r '.status // empty')
if [ "$STATUS" = "completed" ]; then
  echo "Уже completed (тот же option_id) — без повторного Gemini."
  printf '%s' "$(echo "$POST_JSON" | jq '{job: ., done: true, timed_out: false, waited_sec: 0}')" > "$LAST"
else
  echo "Long poll completed (timeout=${TIMEOUT}s)"
  WAIT_JSON=$(curl -sf "${BASE}/api/v1/analyses/${JOB_ID}/wait?timeout_sec=${TIMEOUT}&interval_sec=2&target=completed")
  printf '%s' "$WAIT_JSON" > "$LAST"
fi
echo "saved: $LAST"
echo ""

if [ -x "$PYTHON" ]; then
  "$PYTHON" -m knowledge_engine.cli.job_view -f "$LAST" --no-interactive
else
  cat "$LAST" | jq .
fi
