# Переменные окружения Knowledge Engine

**Единая точка чтения:** `knowledge_engine/config.py` (`_load_dotenv()` при импорте).  
В сервисах импортируйте константы из `config`, не `os.getenv`.

Шаблон: `.env.example`. Секреты: `.env`.

Полный машинный список (~200 ключей): скрипт в репо:

```bash
.venv/bin/python -c "
import re, pathlib
pat=re.compile(r\"os\\.getenv\\(\\s*['\\\"]([^'\\\"]+)['\\\"]|_env_bool\\(\\s*['\\\"]([^'\\\"]+)['\\\"]\")
keys=set()
for p in pathlib.Path('knowledge_engine').rglob('*.py'):
    for m in pat.finditer(p.read_text(encoding='utf-8', errors='ignore')):
        keys.add(m.group(1) or m.group(2))
print('\\n'.join(sorted(keys)))
"
```

---

## Core / API

| Variable | Default (if unset) |
|----------|-------------------|
| `GRAPH_VERSION` | `0.4` |
| `SEARXNG_BASE_URL` | `http://localhost:8080` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `KE_API_HOST` | `127.0.0.1` |
| `KE_API_PORT` | `8765` |
| `KE_API_BASE` | cli: `http://127.0.0.1:{port}` |
| `KE_API_RELOAD` | `false` |

## Redis / worker

| Variable | Default |
|----------|---------|
| `REDIS_URL` | empty |
| `REDIS_SOCKET_TIMEOUT_SEC` | `120` |
| `KE_USE_REDIS` | true if `REDIS_URL` |
| `KE_REDIS_LOGS` | = `KE_USE_REDIS` |
| `KE_TASKS_CHANNEL` | `ke:tasks` |
| `KE_REDIS_LOG_MAX_LINES` | `20000` |
| `KE_WORKER_POLL_SEC` | `0.4` |
| `KE_WORKER_HEARTBEAT_SEC` | `10` |
| `KE_WORKER_STALE_RUNNING_SEC` | `300` |
| `KE_WORKER_INLINE_FALLBACK` | `false` |
| `KE_WORKER_RELOAD_DEBOUNCE_SEC` | `1.0` |
| `KE_WORKER_STOP_TIMEOUT_SEC` | `30` |
| `KE_NODE_DIVE_TIMEOUT_SEC` | `900` |

## Logging

| Variable | Default |
|----------|---------|
| `KE_TRACE_STDOUT` | `false` |
| `KE_LOG_PLAIN` | `false` |
| `KE_LLM_FULL_TRACE` | `false` |

## Ollama

| Variable | Default |
|----------|---------|
| `LOCAL_ROUTER_MODEL` | `qwen2.5-coder:1.5b` |
| `LOCAL_HEAVY_MODEL` | `qwen2.5-coder:7b` |
| `LOCAL_L2_MODEL` | `qwen2.5-coder:7b` |
| `REACT_EVAL_MODEL` | router |
| `GUARDRAILS_OLLAMA_MODEL` | `qwen2.5-coder:7b` |
| `GUARDRAILS_MODEL` | legacy alias |
| `CONTEXT_EVAL_MODEL` | router (1.5b) |
| `CONTEXT_EVAL_NUM_PREDICT` | `2048` |
| `OLLAMA_ROUTER_NUM_CTX` | `2048` |
| `OLLAMA_HEAVY_NUM_CTX` | `4096` |
| `OLLAMA_NUM_CTX` | alias → heavy |
| `OLLAMA_ROUTER_KEEP_ALIVE` | `2m` |
| `OLLAMA_HEAVY_KEEP_ALIVE` | `2m` |
| `OLLAMA_NUM_PREDICT` | `1024` |
| `OLLAMA_GUARDRAILS_NUM_PREDICT` | `1536` |
| `OLLAMA_STRUCTURE_NUM_PREDICT` | `3072` |
| `SELECTION_PROMPTS_OLLAMA_MODEL` | `LOCAL_ROUTER_MODEL` |
| `SELECTION_PROMPTS_TIMEOUT_SEC` | `3` |
| `SELECTION_PROMPTS_NUM_PREDICT` | `256` |

## Gemini, CSE, SearXNG, SS, Exa, RAG, Curriculum, Consensus

См. таблицы в `.env.example` (комментарии) и блоки `CURRICULUM_*`, `CONSENSUS_*`, `GEMINI_*` в `config.py` (строки ~298–765).

Ключевые для tutor/RAG:

| Variable | Default |
|----------|---------|
| `RAG_CE_AUTO_UNLOAD` | `false` |
| `RAG_CE_AUTO_UNLOAD_IDLE_SEC` | `300` |
| `LECTURE_RAG_*` | см. `config.py` |
| `LIGHT_RAG_MIN_COSINE_SIM` | `0.42` |
| `KE_RAG_TIMEOUT_SEC` | `45` |
