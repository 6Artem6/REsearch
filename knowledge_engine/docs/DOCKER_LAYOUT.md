# Docker layout (Knowledge Engine)

## Что в Docker

| Сервис | Порт | RAM (типично) |
|--------|------|----------------|
| `searxng` | 8080 | ~200 МБ |
| `knowledge-api` (profile `api`) | 8765 | только если нужен контейнерный API |

**Не в Docker:** Ollama (Metal GPU), основной Python dev (`REsearch/.venv`).

## Пути и данные

| Что | Где |
|-----|-----|
| Нативный pip | `REsearch/.venv` |
| Опциональный Docker pip | volume `ke_python_venv` → `/app/.venv` |
| Код | bind `./knowledge_engine` (в контейнере API) |
| `.env` | корень репо; для хоста: `localhost` URLs |
| Trace / LanceDB | `knowledge_engine/.runs/`, `.lancedb/`, `.browser_state/` |

Очистка legacy venv: `./knowledge_engine/scripts/cleanup-venvs.sh` (не удаляет `REsearch/.venv`).

## Команды

```bash
# Инфра
docker compose up -d searxng

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
```

Watch overlay (`docker-compose.dev.yml`) — только для `knowledge-api-dev` (profile `dev`).
