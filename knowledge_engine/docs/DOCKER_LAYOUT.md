# Docker layout (Knowledge Engine)

## Что в Docker

| Сервис | Порт | RAM (типично) |
|--------|------|----------------|
| `postgres` (`pgvector/pgvector:pg18`) | 5432 | лимит 4G / резерв 1G (см. docker-compose.yml) |
| `migrator` (ephemeral, без profile) | — | 277 МБ образ, поднимается за секунды |
| `redis` | 6379 | ~50 МБ |
| `searxng` | 8080 | ~200 МБ |
| `knowledge-api` (profile `api`) | 8765 | только если нужен контейнерный API |

**Не в Docker:** Ollama (Metal GPU), основной Python dev (`REsearch/.venv`).

## Postgres/pgvector и автоматические миграции (Phase 0-3)

`docker compose up -d` (без `--profile api`) поднимает `postgres` + `redis` +
`searxng` + **`migrator`** — миграции накатываются автоматически на каждый
`up`, вручную `alembic upgrade head` вызывать не нужно. `migrator`:
1. Ждёт `postgres: service_healthy`.
2. `alembic upgrade head`.
3. Integrity-check: `vector` extension + HNSW-индексы на всех 9 vector-таблицах.
4. Падает с ненулевым exit code при любой проблеме — `knowledge-api` (если
   поднимается, `--profile api`) объявлен с
   `depends_on: migrator: condition: service_completed_successfully` и не
   стартует на сломанной/недомигрированной БД.

`migrator` — **отдельный лёгкий образ** (`knowledge_engine/docker/migrator/Dockerfile`,
`requirements-migrator.txt`: alembic/sqlalchemy/asyncpg/psycopg/pgvector/
pydantic-settings — БЕЗ PyTorch/CUDA/sentence-transformers/exa-py основного
`requirements.txt`), не общий `x-ke-python-image`/`entrypoint.sh` с
`knowledge-api` — иммутабельный, baked-at-build-time, без runtime venv-volume.

Переключатели бэкенда (см. `knowledge_engine/config.py`, `.env.example`):
`VECTOR_STORE_BACKEND` (`postgres` дефолт | `qdrant` fallback),
`GRAPH_CHECKPOINTER_BACKEND` (`postgres` дефолт | `memory` fallback,
RAM-only, теряется при рестарте). Оба легаси-пути (Qdrant-client/LanceDB,
in-memory checkpointer) остаются рабочими — переключение через `.env` не
требует правок кода и не ломает импорты.

Ручной прогон миграций (без docker, локальный `.venv`):
```bash
PYTHONPATH=. ./.venv/bin/alembic upgrade head
```

## Пути и данные

| Что | Где |
|-----|-----|
| Нативный pip | `REsearch/.venv` |
| Опциональный Docker pip | volume `ke_python_venv` → `/app/.venv` |
| Код | bind `./knowledge_engine` (в контейнере API) |
| `.env` | корень репо; для хоста: `localhost` URLs, для контейнеров — `postgres`/Docker DNS (см. `x-ke-python-env` в docker-compose.yml) |
| Trace / LanceDB | `knowledge_engine/.runs/`, `.lancedb/`, `.browser_state/` |
| Postgres data | named volume `ke_postgres_data` → `/var/lib/postgresql` (PG18+ конвенция — НЕ `.../data`) |

Очистка legacy venv: `./knowledge_engine/scripts/cleanup-venvs.sh` (не удаляет `REsearch/.venv`).

## Команды

```bash
# Инфра (postgres + миграции автоматически + redis + searxng)
docker compose up -d

# Только конкретный сервис
docker compose up -d postgres

# Хост
./knowledge_engine/scripts/setup-host-ollama.sh
./knowledge_engine/scripts/setup-host-python.sh
./knowledge_engine/scripts/dev-native.sh

# Опционально API в Docker (Ollama на хосте)
./knowledge_engine/scripts/sync-venv.sh
docker compose --profile api up -d knowledge-api
```

## Отладка

```bash
tail -f knowledge_engine/.runs/*.log
curl -s http://127.0.0.1:11434/api/tags | jq .
curl -s http://127.0.0.1:8080/ | head

# Postgres / миграции
docker logs ke-migrator                 # почему упал init-контейнер
docker exec ke-postgres psql -U knowledge_engine -d knowledge_engine -c "\dt"
PYTHONPATH=. ./.venv/bin/alembic upgrade head   # накатить вручную, если нужно
```

Watch overlay (`docker-compose.dev.yml`) — только для `knowledge-api-dev` (profile `dev`).
