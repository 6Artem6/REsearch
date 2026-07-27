# Knowledge Engine (Local MVP)

**Текущая dev-версия:** **v0.8 Consensus** (`GRAPH_VERSION=0.8`) — [docs/V0_8_CONSENSUS_AGENT.md](docs/V0_8_CONSENSUS_AGENT.md) · [docs/V0_8_SNAPSHOT.md](docs/V0_8_SNAPSHOT.md) · Web UI `http://127.0.0.1:8765/app`

Локальный CLI для архитектурного анализа: декомпозиция на CS-абстракции, поиск по **трём горизонтам** (SOTA / Infra / Prod), Trade-off матрица и точечная раскрутка выбранного варианта (графы 0.4–0.7).

Подробно: [docs/SEARCH_HORIZONS.md](docs/SEARCH_HORIZONS.md) · **legacy 0.4–0.6:** [docs/V0_6_CURRENT_SOLUTION.md](docs/V0_6_CURRENT_SOLUTION.md)

**История изменений и архитектурные решения:** [CHANGELOG.md](CHANGELOG.md)

## Требования

- Python 3.10+
- **Ollama** — отдельное системное приложение (не ставится в Python venv). На macOS: `brew install ollama`
- Модели в Ollama:
  - `qwen2.5-coder:1.5b` (роутер)
  - `qwen2.5-coder:7b` (основная)

## Docker (SearXNG only) + нативный Mac dev

На macOS **Ollama и Python не в Docker** — Metal GPU и без 16+ ГБ RAM на VM.

| Компонент | Где |
|-----------|-----|
| SearXNG | Docker, порт 8080 |
| Ollama | хост, `brew install ollama`, порт 11434 |
| API / CLI dev | хост, `REsearch/.venv` |

Схема: [docs/DOCKER_LAYOUT.md](docs/DOCKER_LAYOUT.md)

```bash
docker compose up -d searxng
./knowledge_engine/scripts/setup-host-ollama.sh
./knowledge_engine/scripts/setup-host-python.sh
./knowledge_engine/scripts/dev-native.sh   # API + reload
```

Или одной командой: `make setup` → `make dev`

Опционально API в контейнере (Ollama на хосте через `host.docker.internal`):

```bash
./knowledge_engine/scripts/sync-venv.sh
docker compose --profile api up -d knowledge-api
```

### Разработка

Пошагово: **[docs/DEV_RUNBOOK.md](docs/DEV_RUNBOOK.md)**

```bash
# Терминал 1
./knowledge_engine/scripts/dev-native.sh

# Терминал 2 — wait-analysis / curl POST /api/v1/analyses
```

| Сервис | Порт | Профиль |
|--------|------|---------|
| `searxng` | 8080 | — |
| `knowledge-api` | 8765 | `api` |
| `knowledge-api-dev` | 8765 | `dev` (+ overlay, legacy) |

## REST API (FastAPI)

Нативно (после `dev-native.sh` или `python -m knowledge_engine.api`):

```bash
open http://127.0.0.1:8765/docs
```

Или в Docker:

```bash
docker compose --profile api up -d knowledge-api
```

Эндпоинты: `GET /api/v1/health`, … `POST …/unravel`. После матрицы: `./knowledge_engine/scripts/unravel-analysis.sh JOB_ID 2`.

```bash
curl -s -X POST http://127.0.0.1:8765/api/v1/analyses \
  -H 'Content-Type: application/json' \
  -d '{"problem":"RAG cache invalidation","constraints":"Mac M1, LanceDB"}'
```

## Trace-лог прогона (`analyze`)

При `analyze` создаётся текстовый лог в `knowledge_engine/.runs/` (путь печатается в начале прогона).

Следить в реальном времени:

```bash
tail -f knowledge_engine/.runs/*.log
```

В логе: `NODE ▶/✓`, `OLLAMA ▶/✓`, `STATUS |`, `EMBED ▶`.

В **консоли** (Rich Live): в заголовке панели — `[MM:SS]` и текущая фаза; строки с `▶/✓ NODE` и `▶/✓ OLLAMA` (+ секунды). Между фазами Live (например, выбор варианта матрицы) важные строки печатаются в консоль отдельно и не теряются.

## SearXNG

Конфиг: `knowledge_engine/docker/searxng/settings.yml`. Метапоиск Bing + Google.

## Установка (один раз)

```bash
./knowledge_engine/scripts/setup.sh
```

`browser-login` (окно Google): нативный `.venv` на Mac с GUI; headless в Docker API без логина.

## Конфигурация

`config.py`:

- `OLLAMA_BASE_URL` — по умолчанию `http://localhost:11434`
- `ROUTER_MODEL` — `qwen2.5-coder:1.5b`
- `MAIN_MODEL` — `qwen2.5-coder:7b`
- `MAX_SEARCH_ITERATIONS` — максимум 3 итерации Re-Act

## Поток графа (0.4, `GRAPH_VERSION=0.4`)

1. **decomposition** (Gemini) — L0, CS-абстракции  
2. **query_expansion** (7B) + **query_expander** (site/minus операторы)  
3. **discovery** — SearXNG v0.6, Domain Trust, архив ссылок  
4. **document_fetch** → **structure_filter** (7B) → **deep_extractor** (Gemini)  
5. **research_evaluator** (Gemini) → **decision_router** (Re-Act, лимиты URL/depth)  
6. **matrix** (Gemini) → **lancedb_save**  
7. Interrupt — выбор варианта (CLI/API)  
8. **unraveling** (Gemini) — отдельный `POST /unravel` или `unravel-analysis.sh`

Детали: [docs/V0_6_CURRENT_SOLUTION.md](docs/V0_6_CURRENT_SOLUTION.md)

## Поток графа (0.2, legacy)

1. **decomposition** — CS-абстракции (7B)
2. **local_rag_check** — hybrid search в LanceDB
3. **ai_react_loop** + **multi_search** — если RAG не достаточен (Playwright, API, vision, summarizer)
4. **matrix** — Trade-off матрица (7B), с учётом `found_summaries`
5. Пауза (interrupt) — выбор ID в CLI
6. **unraveling** — детальный разбор (7B)

Перед Gemini в CLI: `python -m knowledge_engine.main browser-login`.  
**v0.8:** Consensus login — `./knowledge_engine/scripts/consensus-login.sh` ([V0_8_CONSENSUS_AGENT.md](docs/V0_8_CONSENSUS_AGENT.md)).

## Поток v0.8 (`GRAPH_VERSION=0.8`)

Light RAG → sanitize + grounding → Consensus → validate → L2a–L2c → Reasoner; Web `/app`, runs в `.runs/v07_runs.json`. Снимок: [docs/V0_8_SNAPSHOT.md](docs/V0_8_SNAPSHOT.md).

Состояние v0.2–0.4: `MemorySaver` для resume после выбора варианта.
