#!/usr/bin/env bash
# Полный прогон Knowledge Engine v0.7 (guardrails → search → dedup → Gemini matrix)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
# .env перекрывает устаревший export GRAPH_VERSION=0.7 в shell
if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export KE_TRACE_STDOUT="${KE_TRACE_STDOUT:-true}"
export KE_LOG_PLAIN="${KE_LOG_PLAIN:-true}"
QUERY="${1:?usage: run-v07-analysis.sh \"IT вопрос\"}"
exec "${ROOT}/.venv/bin/python" -m knowledge_engine.scripts.run_v07 "${QUERY}"
