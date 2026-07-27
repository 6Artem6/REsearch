#!/usr/bin/env bash
# Smoke: v0.7 LangGraph (guardrails only if SKIP_V07_FETCH=1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
cd "$ROOT"

QUERY="${1:-Как спроектировать invalidation кэша эмбеддингов в GraphRAG на LanceDB?}"
PROFILE="${ROOT}/knowledge_engine/user_profile.md"
THREAD="v07-smoke-$$"

if [ "${SKIP_V07_FETCH:-0}" = "1" ]; then
  echo "SKIP_V07_FETCH=1 — только Stage 0 personal context (Ollama)"
  exec "${ROOT}/.venv/bin/python" -c "
import asyncio
from knowledge_engine.src.guardrails import run_personal_context_stage
profile = open('${PROFILE}', encoding='utf-8').read()
ctx = asyncio.run(run_personal_context_stage('${QUERY}', profile))
print(ctx.model_dump())
"
fi

echo "Full v0.7 graph | thread=${THREAD}"
echo "query: ${QUERY}"
"${ROOT}/.venv/bin/python" -m knowledge_engine.scripts.smoke_v07 \
  --query "${QUERY}" \
  --profile "${PROFILE}" \
  --thread-id "${THREAD}"
