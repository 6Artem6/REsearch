# Knowledge Engine — текущее решение (v0.4–v0.6)

Документ описывает **актуальную** архитектуру при `GRAPH_VERSION=0.4` (рекомендуется в `.env`). Версии 0.2–0.3 остаются в коде для совместимости; новые прогоны — через v0.4 API/CLI.

Связанные материалы: [DEV_RUNBOOK.md](DEV_RUNBOOK.md), [DOCKER_LAYOUT.md](DOCKER_LAYOUT.md), [SEARCH_HORIZONS.md](SEARCH_HORIZONS.md), [FRUGAL_ROUTING.md](FRUGAL_ROUTING.md).

---

## Суть решения (одним абзацом)

**Knowledge Engine** — локальный исследовательский агент для инженерных задач: пользователь формулирует проблему и ограничения (Mac M-series, стек, tail latency); **LangGraph** раскладывает задачу на CS-абстракции (**Gemini** + **Ollama 7B**), собирает источники через **таргетированный SearXNG** и архив ссылок, отфильтровывает SEO/e-commerce (**Domain Trust**), извлекает L2-факты в **LanceDB**, оценивает достаточность (**Re-Act** с лимитами depth/URL), строит **Trade-off матрицу** (Классика / SOTA / Минимализм), сохраняет checkpoint и ждёт выбор варианта; **unravel** (Gemini) даёт глубокий разбор с failure modes без повторного полного analyze. Оркестрация: **FastAPI** + job store, **нативный Ollama (Metal)** и Python на Mac, **SearXNG** только в Docker.

---

## Целевая проблема

| Что не так с «обычным» поиском/чатом | Что делает Engine |
|--------------------------------------|-------------------|
| SEO, магазины, общий web-мусор | Domain Trust, minus-words, IT/science категории SearXNG, приоритет HN/GitHub/arXiv |
| Один проход без структуры | L0 → discovery → L2 в LanceDB → матрица → опциональный unravel |
| Нет явных trade-offs и failure modes | Профиль в `user_profile.md`, промпты evaluator/matrix/unravel |
| Тяжёлый Docker на Mac | Ollama + API на хосте; в Docker только SearXNG (~200 МБ) |

---

## Трёхуровневый LLM (v0.4 hybrid)

| Слой | Модель | Роль |
|------|--------|------|
| Router / structured | Ollama **1.5B** | decision router, короткие JSON-решения |
| Structure / expansion | Ollama **7B** | query expansion, junk filter по HTML |
| Reasoning / L0/L2/unravel | **Gemini API** (stateless) | декомпозиция, deep extractor, evaluator, matrix, unraveling |

Fallback моделей Gemini при 429: `GEMINI_FALLBACK_MODELS`. Лимит шагов графа: `GRAPH_RECURSION_LIMIT` (авто из `MAX_URLS` × `MAX_RESEARCH_DEPTH`).

---

## Граф v0.4 (основной поток)

```text
decomposition (Gemini L0)
  → query_expansion (7B + query_expander v0.6)
  → discovery (SearXNG + archive + domain trust)
  → document_fetch → structure_filter (7B) → deep_extractor (Gemini)
  → research_evaluator (Gemini) → decision_router (1.5B + Python caps)
  → [loop: fetch | query_expansion | exit]
  → pre_synthesis_clusterizer → matrix (Gemini) → lancedb_save
  → interrupt_before unraveling
  → unraveling (Gemini) → END
```

**Re-Act лимиты:** `MAX_URLS`, `MAX_RESEARCH_DEPTH`; при cap router идёт в pre_synthesis, не раздувает `pending_urls`.

**Checkpoint:** `MemorySaver` + `thread_id` на job; unravel = `graph.invoke` после interrupt с `selected_option_id`.

---

## Discovery: v0.5 Domain Trust + архив + v0.6 Smart Search

### v0.6 Smart Targeted Search

- `services/query_expander.py` — post-process запросов: `site:github.io|substack|dev.to|arxiv.org`, minus-слова (`-inurl:cart`, `-site:amazon.*`, …).
- `services/searxng_client.py` — JSON-поиск с `categories=it,science,general`, поле `engine` в результате.
- `docker/searxng/settings.yml` — движки github, hackernews, stackoverflow, arxiv, google scholar с повышенным weight.
- Env: `SMART_QUERY_SYNTAX_ENABLED`, `SEARXNG_DISCOVERY_CATEGORIES`.

### v0.5 Domain Trust Engine

- SQLite: `knowledge_engine/.domain_trust/`.
- Batch Gemini profiler доменов; статический deny (IKEA, Amazon, …).
- Отсев низкого trust; **приоритет** high-trust и движков SearXNG (HN/GitHub/…) в `pending_urls`, без жёсткого whitelist-only.

### Архив ссылок

- SQLite: `knowledge_engine/.source_archive/links.sqlite`.
- Все URL discovery/fetch с trust/status.
- `DISCOVERY_MODE=cache_first` или API `reuse_cached_sources` — сначала релевантные URL из архива, затем SearXNG.

### Горизонты SOTA/Infra/Prod

Полноценный `multi_search_horizons_sync` живёт в v0.2 `multi_search_node` и `test-search` CLI. В v0.4 discovery — плоский multi_search с v0.6 операторами (см. [SEARCH_HORIZONS.md](SEARCH_HORIZONS.md) для расширения).

---

## API и жизненный цикл job

| Статус | Смысл |
|--------|--------|
| `running` | analyze до матрицы или unravel |
| `matrix_ready` | матрица готова, interrupt перед unravel |
| `completed` | unravel сохранён (или idempotent повтор того же option_id) |
| `failed` | ошибка графа/Gemini/лимит |

Эндпоинты: `POST /api/v1/analyses`, `GET …/wait?target=matrix|completed`, `POST …/unravel` (`option_id`, `force_rerun`).

**Важно:** unravel **не** перезапускает analyze; нужен job в `job_store` (после reload API — 404, матрица в `last-wait-response.json`).

### CLI / скрипты

| Скрипт | Назначение |
|--------|------------|
| `wait-analysis.sh` | POST analyze + wait matrix + Rich-матрица + опциональный unravel |
| `unravel-analysis.sh` | POST unravel + wait completed |
| `view-job.sh` | Матрица/unravel из API, `--id`, `-f`, или last-wait по умолчанию |

Вывод: Rich-таблицы с `show_lines`; markdown-таблицы в unravel через `ui/markdown_terminal.py`.

---

## Данные и память

- **LanceDB** — `knowledge_engine/.lancedb`, иерархия L0/L1/L2.
- **Job store** — `knowledge_engine/.runs/job_store.json`.
- **Trace** — `knowledge_engine/.runs/*.log`, `last-wait-response.json`.
- **Профиль** — `user_profile.md` (критерии trade-off, Mac, failure modes).

---

## Типичный прогон (Mac)

```bash
docker compose up -d searxng
./knowledge_engine/scripts/dev-native.sh
./knowledge_engine/scripts/wait-analysis.sh "Задача" "Mac M4, Ollama 7B, Gemini"
./knowledge_engine/scripts/unravel-analysis.sh JOB_ID 2
./knowledge_engine/scripts/view-job.sh --no-interactive
```

После смены `searxng/settings.yml`: `docker compose up -d --force-recreate searxng`.

---

## Конфигурация (ключевые env)

См. `.env.example`: `GRAPH_VERSION=0.4`, `GEMINI_*`, `DOMAIN_TRUST_*`, `SOURCE_ARCHIVE_*`, `DISCOVERY_MODE`, `SMART_QUERY_SYNTAX_ENABLED`, `SEARXNG_DISCOVERY_CATEGORIES`, `GRAPH_RECURSION_LIMIT`, `MAX_URLS`, `MAX_RESEARCH_DEPTH`.

---

## Версионная шкала

| Версия | Фокус |
|--------|--------|
| 0.2 | multi_search, горизонты, LanceDB summaries |
| 0.3 | Stateless Gemini, extractor loop |
| **0.4** | 3-tier hybrid graph, structure filter, pre_synthesis |
| **0.5** | Domain Trust + source link archive |
| **0.6** | Smart Targeted Search (query syntax + SearXNG categories/engines) |

Дальнейшее: подключение horizon discovery в v0.4, расширение curated dork-провайдеров, persist material URLs в API response.
