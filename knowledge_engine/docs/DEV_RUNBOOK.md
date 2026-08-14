# Разработка: как запускать и смотреть логи

Все команды из **корня** `REsearch`.

**AI Skill Tree / Tutor (генерация графа, worker, Exa):** [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md) · UI: [SKILL_TREE_UI.md](SKILL_TREE_UI.md).

## 1. Один раз (окружение)

```bash
cp .env.example .env          # GRAPH_VERSION=0.8, GEMINI_API_KEY
./knowledge_engine/scripts/setup.sh
```

`setup.sh` поднимает SearXNG, настраивает **Ollama на macOS (Metal)** и создаёт **`.venv` на хосте**.

### Pre-commit (black / isort / flake8)

После `setup.sh` или `make dev-deps`:

```bash
make pre-commit-install   # git hook перед каждым commit
make pre-commit-run       # проверить всё дерево вручную
```

На `git commit` автоматически: `autoflake` → `isort` → `black` (правят файлы), затем `flake8` (блокирует коммит при ошибках). Если форматтеры изменили файлы — добавьте их в коммит и снова `git commit`.

Ручные аналоги: `make format`, `make lint`, `make check`.

## 2. Постоянная разработка (рекомендуется на Mac)

**Терминал A** — нативный API + uvicorn reload (без Docker VM для Python/Ollama):

```bash
chmod +x knowledge_engine/scripts/dev-native.sh
./knowledge_engine/scripts/dev-native.sh
```

Или через Makefile: `make dev`

Что происходит:

- `docker compose up -d searxng` — только метапоиск в Docker (~200 МБ RAM)
- Ollama на `http://localhost:11434` (GPU)
- uvicorn reload при изменении кода (`KE_API_RELOAD=true`)
- в **этом терминале** — `NODE ▶/✓`, `OLLAMA ▶/✓`, `PIPELINE ▶` (`KE_TRACE_STDOUT`, `KE_LOG_PLAIN`)

**Web UI (v0.8):** [http://127.0.0.1:8765/app](http://127.0.0.1:8765/app) — после прогона URL `/app?run=<id>`.

### Статус web-run (не SQLite)

| Что | Где |
|-----|-----|
| Персистентность | `knowledge_engine/.runs/v07_runs.json` (до 80 run, JSON) |
| В процессе API | `V07RunStore` в памяти (`knowledge_engine/services/v07_run_store.py`), при старте читает JSON |
| Поля | `status` (`pending` / `running` / `completed` / `failed`), `current_step`, `result` (partial state), `error`, `log_path` |
| Кто пишет | `v07_run_service.run_v07_job` — финал; `publish_web_run_progress` — шаги L2a…reasoner; `merge_result` — частичный `result` |
| API | `GET /api/v1/v07/runs/{id}` — poll; `GET …/view` — UI (`partial` пока `status != completed`) |

SQLite в проекте — **domain trust** и **source archive** (`domains.sqlite`, `links.sqlite`), а также ingestion схем источников (`.runs/article_diagrams.db`). Web-runs в SQLite не хранятся; подробнее о схемах: [ARTICLE_DIAGRAMS.md](ARTICLE_DIAGRAMS.md).

Ручной правка: править JSON или `python -c "from knowledge_engine.services.v07_run_store import …"` **после остановки API** (или перезапустить `dev-native.sh`), иначе в памяти процесса останется старый статус.

Тема Monokai Pro в сайдбаре; снимок версии: [V0_8_SNAPSHOT.md](V0_8_SNAPSHOT.md).
**Consensus (один раз):** остановите API → `./knowledge_engine/scripts/consensus-login.sh` (или `python -m knowledge_engine.main consensus-login` с тем же `PLAYWRIGHT_BROWSERS_PATH`, что в dev-native). Не `browser-login` (Gemini).

Альтернатива без reload:

```bash
export PYTHONPATH="$(pwd)"
source .venv/bin/activate
python -m knowledge_engine.api
# или: uvicorn knowledge_engine.api:app --reload --port 8765
```

**Legacy:** API в Docker + watch — `./knowledge_engine/scripts/dev-watch.sh` (Ollama всё равно на хосте через `host.docker.internal`).

## 3. Терминал B — запрос на FastAPI

```bash
export KE_API_PORT=8765
BASE="http://127.0.0.1:${KE_API_PORT}"

curl -s -X POST "${BASE}/api/v1/analyses" \
  -H 'Content-Type: application/json' \
  -d '{
    "problem": "Как спроектировать invalidation кэша эмбеддингов в GraphRAG на LanceDB?",
    "constraints": "Mac M1 unified memory, Ollama 7B локально, Gemini API для синтеза, tail latency",
    "matrix_only": false,
    "async_mode": true
  }' | tee /tmp/ke-job.json | jq .
```

```bash
JOB=$(jq -r '.job.id' /tmp/ke-job.json)
./knowledge_engine/scripts/wait-analysis.sh "ваш вопрос" "ограничения" 600
```

Файловый trace:

```bash
tail -f knowledge_engine/.runs/*.log
```

## 4. Быстрые проверки

```bash
curl -s http://127.0.0.1:8765/api/v1/health | jq .
curl -s -X POST http://127.0.0.1:8765/api/v1/search/test \
  -H 'Content-Type: application/json' \
  -d '{"query": "LanceDB hybrid search", "flat": false}' | jq '.hits[:5]'
```

## 5. CLI на хосте (без API)

```bash
export PYTHONPATH="$(pwd)"
source .venv/bin/activate
python -m knowledge_engine.main analyze -c "Mac M1, LanceDB" "Кэш эмбеддингов в RAG"
```

## 6. Prod-like API в Docker (CI / без нативного Python)

```bash
./knowledge_engine/scripts/sync-venv.sh
docker compose --profile api up -d knowledge-api
docker compose logs -f knowledge-api
```

Ollama в контейнере **не используется** — `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

## 7. Логи

| Что | Где |
|-----|-----|
| NODE / OLLAMA / PIPELINE | терминал `dev-native.sh` |
| v0.7 CLI (`run-v07-analysis.sh`) | тот же формат строк ▶/✓ + `TIMING` в терминале |
| полный trace | `knowledge_engine/.runs/*.log` |
| Docker API only | `docker compose logs -f knowledge-api` |

Переменные:

- `KE_TRACE_STDOUT=true`
- `KE_LOG_PLAIN=true`

### Lecture RAG (dense_material, CE + MMR)

См. [LECTURE_RAG_CONTEXT.md](LECTURE_RAG_CONTEXT.md). В trace ищите `LECTURE_RAG pool`, `ce_filter`, `mmr_pick`.

| Env | Default |
|-----|---------|
| `LECTURE_RAG_CANDIDATE_LIMIT` | 15 |
| `LECTURE_RAG_MMR_TOP_K` | 5 |
| `LECTURE_RAG_CE_MIN_SCORE` | 0.38 |
| `LECTURE_RAG_MMR_LAMBDA` | 0.62 |
| `LECTURE_RAG_RERANK_TIMEOUT_SEC` | 60 |

Требует Ollama (`EMBED_MODEL`) и опционально `sentence_transformers` + `RAG_CROSS_ENCODER_MODEL` для CE.
