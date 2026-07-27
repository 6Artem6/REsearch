#!/usr/bin/env bash
# Опрос статуса analysis job каждые N секунд (если long poll не подходит).
set -euo pipefail

JOB_ID="${1:?usage: poll-analysis.sh <job_id> [interval_sec] [base_url]}"
INTERVAL="${2:-5}"
BASE="${3:-http://127.0.0.1:${KE_API_PORT:-8765}}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"

echo "Polling ${BASE}/api/v1/analyses/${JOB_ID} every ${INTERVAL}s"
echo ""

while true; do
  RESP=$(curl -sf "${BASE}/api/v1/analyses/${JOB_ID}") || {
    echo "curl failed"
    sleep "$INTERVAL"
    continue
  }
  STATUS=$(echo "$RESP" | jq -r '.status')
  ERR=$(echo "$RESP" | jq -r '.error // empty')
  CLARIFY=$(echo "$RESP" | jq -r '.clarify_question // empty')
  LOG=$(echo "$RESP" | jq -r '.log_path // empty')

  TS=$(date '+%H:%M:%S')
  echo "[$TS] status=$STATUS"
  if [ -n "$CLARIFY" ] && [ "$CLARIFY" != "null" ]; then
    echo "  clarify: $CLARIFY"
  fi
  if [ -n "$ERR" ] && [ "$ERR" != "null" ]; then
    echo "  error: $ERR"
  fi
  if [ -n "$LOG" ] && [ "$LOG" != "null" ]; then
    echo "  log: $LOG"
  fi

  case "$STATUS" in
    matrix_ready|completed|failed)
      echo ""
      if [ -x "${ROOT}/.venv/bin/python" ]; then
        echo "$RESP" | "${ROOT}/.venv/bin/python" -m knowledge_engine.cli.job_view --no-interactive
      else
        echo "$RESP" | jq .
      fi
      exit 0
      ;;
  esac

  if [ -n "$CLARIFY" ] && [ "$CLARIFY" != "null" ]; then
    echo ""
    echo "Нужен clarify: POST ${BASE}/api/v1/analyses/${JOB_ID}/clarify"
    echo "$RESP" | jq '{status, clarify_question, log_path}'
    exit 0
  fi

  sleep "$INTERVAL"
done
