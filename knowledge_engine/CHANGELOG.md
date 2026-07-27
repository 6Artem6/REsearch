# История проекта Knowledge Engine

Документ фиксирует значимые изменения в коде и архитектуре. Спецификация MVP: `../init.md` в корне `REsearch`.

## Статус (2026-07-26)

**Версия:** **0.8 Consensus** (снимок зафиксирован) — `knowledge_engine/src/` + Web UI `/app`. Legacy API graphs: **0.4–0.7**.

Документ снимка: [docs/V0_8_SNAPSHOT.md](docs/V0_8_SNAPSHOT.md).

### 2026-07-26 — v0.8 snapshot (Consensus + Web UI)

- **Пайплайн:** `run_consensus_pipeline` — L2a/L2b/L2c после chunking; Reasoner с локальным импортом (устойчивость к uvicorn reload); grounding + `preserved_terms` перед Consensus sanitize; anchor sanitize без Light RAG.
- **Consensus session:** reuse браузера, `begin_new_run`, auth recovery, shutdown в `api/app.py`; `consensus-login.sh`; `PLAYWRIGHT_BROWSERS_PATH` в dev-native.
- **Web:** `present.py` / `linkify.py` / `source_present.py`; KaTeX + repair LaTeX; Monokai Pro themes (`themes/themes.css`, `data-theme`); `?run=` permalink в `app.js`.
- **Runs:** `v07_run_store` — 80 прогонов, полный `result`; `GET …/view` для UI.
- **Фиксы:** `linkify` tuple/`re.sub`; form-feed в формулах; `local_orchestrator` `invoke_reasoner`.

### 2026-07-26 — v0.8 Stateful Consensus Agent (начало)

- `GRAPH_VERSION=0.8`: Light RAG, Playwright Consensus session, Gemini Lite validator (OK/REJECT/RETRY), Gemini Reasoner — финальный ответ; FastAPI `/app` и CLI через тот же `GRAPH_VERSION`.
- Док: [docs/V0_8_CONSENSUS_AGENT.md](docs/V0_8_CONSENSUS_AGENT.md).

### 2026-07-26 — Semantic Scholar academic track

- v0.7: Stage 0 Personal Context (7B); retrieval Semantic Scholar + arXiv; Gemini architect prompt; **без SearXNG** в графе v0.7.


- `/app` SPA: TOC, sections, DOI/arXiv linkify; `POST /api/v1/v07/runs`.


- `src/fetcher/`: Unpaywall, Sci-Hub mirrors, PyMuPDF clean, integration L2a/L2b full docs + LanceDB chunks.


- Stage 0/1: `fast_grounding` (raw SearXNG) → Ollama + WEB CONTEXT prompt; `term_guard` только verbatim acronyms; удалён `it_lexicon`.


- **profiler/schemas:** исследовательский синтез и сравнение источников (L2a/L2c); профиль/железо — вторичный context_flags; развернутые mechanics_detail, cross_source_contrasts, context_synthesis.
- **run_v07.py:** полный вывод L2a–L2c; интерактивный REPL по контексту (`--no-repl`).


- Узлы guardrails → fetch/dedup loop → chunking → profiling
- `knowledge_engine_v07_graph`, `run_knowledge_engine_v07`, `GRAPH_VERSION=0.7`

### 2026-07-26 — v0.7 guardrails (`src/guardrails/`)

- `generate_validated_query_spec` — Ollama structured JSON, `uma_resource_lock`
- `GUARDRAILS_OLLAMA_MODEL` в config

### 2026-07-26 — v0.7 analytics (`src/analytics/`)

- **chunker:** Gemini Lite → `StructuredChunk`
- **profiler:** L2a ConceptGraph, L2b ProfileGapMap, L2c matrix (без raw markdown)
- `GEMINI_LITE_MODEL`, `GEMINI_FLASH_MODEL` в config

### 2026-07-26 — v0.7 scaffold (`knowledge_engine/src/`)

- **locks:** global `uma_resource_lock`, `staged_uma_lock`, `run_under_uma_lock`
- **state:** Pydantic + `KnowledgeEngineState` TypedDict
- **fetcher:** ar5iv / DOM masks / trafilatura, no LLM
- **dedup:** `ChunkDedupStore`, cosine 0.88 ingest, `density_delta` termination
- Док: `docs/V0_7_ARCHITECTURE.md`

### 2026-07-26 — v0.6 Smart Targeted Search + ops (unravel, CLI)

- **Smart Search:** `query_expander.py`, `searxng_client.py` (categories it/science/general, engine → trust boost), SearXNG engines github/hn/arxiv в `settings.yml`.
- **Discovery:** `discovery_collect` + Domain Trust + source archive; `reuse_cached_sources` / `DISCOVERY_MODE=cache_first`.
- **Graph ops:** `GRAPH_RECURSION_LIMIT`, router caps при MAX_URLS/depth.
- **API/CLI:** unravel при `completed` (idempotent), `force_rerun`; `unravel-analysis.sh`, `view-job.sh` (`--id`, last-wait default); Rich tables `show_lines`, `markdown_terminal` для unravel.
- **Файлы:** `services/query_expander.py`, `searxng_client.py`, `discovery_*`, `v04_decision_router.py`, `cli/job_view.py`, `scripts/*`, `docs/V0_6_CURRENT_SOLUTION.md`.

### 2026-07-25 — v0.5 Domain Trust + source archive (v0.4 discovery)

- SQLite domain trust + source link archive; batch Gemini profiler; prioritize trusted URLs; интеграция в `discovery_collect`.

### 2026-07-25 — v0.4 hybrid graph

- `graph/v04.py`: Gemini + 7B + 1.5B pipeline, pre_synthesis, interrupt_before unraveling; FastAPI job store.

### 2026-07-25 — Горизонты поиска SOTA / Infra / Prod

- **Контекст:** в `init.md` — поиск по временным горизонтам; в коде был только один запрос на все провайдеры.
- **Изменение:** `services/search/horizons.py`, `multi_search_horizons_sync()`, поле `SearchResult.horizon`, state `search_horizon_queries`; документ `docs/SEARCH_HORIZONS.md`.
- **Затронутые файлы:** `services/search/*`, `nodes/multi_search.py`, `schemas.py`.

### 2026-07-25 — Gemini + гибкий SearchRegistry

- **Контекст:** замена Perplexity на `gemini.google.com/app`, единый реестр бесплатных провайдеров.
- **Изменение:**
  - `config.py`: `GEMINI_*` селекторы, `SEARXNG_BASE_URL`, Crossref/Habr API URL, `SEARCH_ACTIVE_PROVIDERS`.
  - `services/ai_dialogue/gemini_session.py`: async Playwright + `BrowserGeminiDialogueSession` для узлов.
  - `services/search/providers.py`: SearXNG, Semantic Scholar, Habr, Consensus, ArXiv, Crossref.
  - `services/search/registry.py`: `multi_search` / `multi_search_sync`.
  - `ai_react_loop` → Gemini; `multi_search` → `default_registry()`.
  - `browser-login` открывает Gemini; SearXNG через Docker (`localhost:8080`).
- **Затронутые файлы:** `config.py`, `services/search/*`, `services/ai_dialogue/*`, `nodes/ai_react_loop.py`, `nodes/multi_search.py`, `main.py`, `README.md`.

### 2026-07-25 — SearXNG: Bing/Google вместо DuckDuckGo (CAPTCHA)

- **Контекст:** при `keep_only` оставался только активный DuckDuckGo (Bing `disabled`, Google `inactive` в дефолтах) → CAPTCHA; botdetection без proxy headers.
- **Изменение:** `keep_only: [bing, google]` + явное включение engines; healthcheck/`test-search` на `bing`; клиенты шлют `X-Forwarded-For`; `limiter.toml` pass_ip для private nets.
- **Затронутые файлы:** `docker/searxng/*`, `config.py`, `providers.py`, `searxng_health.py`, `docker-compose.yml`.

### 2026-07-25 — SearXNG: keep_only engines + healthcheck headers

- **Контекст:** `disabled: true` для wikidata не останавливает INIT → 403 на `query.wikidata.org`; healthcheck без proxy headers → `X-Forwarded-For nor X-Real-IP`.
- **Изменение:** `settings.yml` — `use_default_settings.engines.keep_only: [duckduckgo, bing, google]`; healthcheck wget с `X-Forwarded-For` / `X-Real-IP`; `SEARXNG_SETTINGS__server__limiter=false`.
- **Затронутые файлы:** `knowledge_engine/docker/searxng/settings.yml`, `docker-compose.yml`.

### 2026-07-25 — Docker healthcheck + SearXNG engines

- **Ollama unhealthy:** в `ollama/ollama` нет `curl` → healthcheck `ollama list`; `ollama-init` монтирует `ollama_models`.
- **SearXNG:** healthcheck только `engines=duckduckgo`; отключены brave/startpage; limiter.toml; провайдер начинает с duckduckgo.

---

- **Контекст:** голый `docker run searxng` без `settings.yml` → нет JSON, ошибки engines (wikidata 403).
- **Изменение:**
  - `docker-compose.yml` (корень REsearch): `searxng`, `ollama`, `ollama-init`, `knowledge-engine`.
  - `knowledge_engine/docker/searxng/settings.yml` — `formats: [html, json]`, `limiter: false`, отключены проблемные engines.
  - `test-search` CLI, `check_searxng()`, fallback в `SearXNGProvider`.
  - `SEARXNG_BASE_URL` / `OLLAMA_BASE_URL` из env.
- **Затронутые файлы:** `docker-compose.yml`, `knowledge_engine/docker/**`, `config.py`, `main.py`, `providers.py`, `searxng_health.py`.

---

## [0.2.0] — Search & Knowledge Discovery Agent

### Контекст

Переход от симулированного Re-Act к реальному сбору знаний: локальный LanceDB, API-поиск, браузер (Playwright), диалог с внешним ИИ, vision pipeline, profile-guided summarization.

### Изменение

| Область | Что добавлено |
|---------|----------------|
| `user_profile.md` | Профиль разработчика для персонализации саммари |
| `services/vector_store.py` | LanceDB + Ollama embeddings, `save_summary`, `hybrid_search` |
| `services/search/` | `SearchRegistry`, ArXiv + Semantic Scholar, `browser_search` (persistent context) |
| `services/ai_dialogue/` | `BrowserAIDialogueSession` (Perplexity через Playwright) |
| `services/vision.py` | Поиск `<img>`, описание схем через LLM |
| `services/summarizer.py` | Сжатие статей → `DocumentSummary` |
| `nodes/` | `local_rag_check`, `ai_react_loop`, `multi_search` (замена `react_search` в графе) |
| `ui/logger.py` | `rich.live.Live`, потоковый вывод токенов |
| `graph.py` | decomposition → LanceDB → [dialogue → multi_search] → matrix → interrupt → unraveling |

### Граф 0.2

```text
decomposition → local_rag_check → (RAG ok?) → matrix
                              └→ ai_react_loop → multi_search → matrix → interrupt → unraveling
```

### Зависимости

- `lancedb`, `pyarrow`, `playwright`, `httpx`
- Ollama: дополнительно `ollama pull nomic-embed-text`
- `playwright install chromium`

### CLI

- `python -m knowledge_engine.main browser-login` — ручная авторизация в `.browser_state/`

### Затронутые файлы

`config.py`, `schemas.py`, `graph.py`, `main.py`, `llm.py`, `requirements.txt`, `nodes/*`, `services/**`, `ui/logger.py`, `user_profile.md`

---

## [0.1.0] — MVP и стабилизация

### Реализовано по ТЗ

| Компонент | Назначение |
|-----------|------------|
| `config.py` | URL Ollama, `ROUTER_MODEL` / `MAIN_MODEL`, лимиты Re-Act |
| `schemas.py` | Pydantic v2: `CSAbstraction`, `TradeOffOption`, `AnalysisReport`, `EngineState` |
| `nodes/` | `decomposition`, `react_search`, `matrix`, `unraveling` |
| `graph.py` | `StateGraph`, условный цикл Re-Act, `MemorySaver`, interrupt перед unraveling |
| `main.py` | Typer + Rich: таблица вариантов, ввод ID, resume графа |
| `scripts/setup.sh` | venv, pip, Homebrew Ollama, pull моделей |

### Архитектура графа

```text
decomposition (7B) → react_search (1.5B) ⇄ [до 3 итераций] → matrix (7B)
    → [INTERRUPT] → unraveling (7B) → END
```

Две модели: роутер для Re-Act, основная для декомпозиции, матрицы и раскрутки.

### Серьёзные изменения после первого запуска

#### 1. CLI (Typer)

- **Проблема:** при одной подкоманде Typer «поднимал» `analyze` на корень; вызов `main analyze -c … "задача"` воспринимал `analyze` как текст задачи.
- **Решение:** `@app.callback()` + подкоманда `analyze`; `_normalize_argv()` подставляет `analyze`, если пользователь вызвал `main -c … "задача"` без подкоманды.

#### 2. Structured output + Ollama

- **Проблема:** `with_structured_output()` по умолчанию использует `function_calling`; для `qwen2.5-coder` через Ollama возвращался `None` → падение на `result.items`.
- **Решение:** модуль `llm.py`, helper `structured_chat(..., method="json_schema")` для всех узлов с structured output.

#### 3. Состояние графа + MemorySaver

- **Проблема:** `StateGraph(dict)` с `MemorySaver` передавал в следующий узел только последний патч (например, только `abstractions`), без `user_problem` → ошибка Pydantic `Field required`.
- **Решение:** `EngineGraphState` (`TypedDict` в `schemas.py`), граф собран как `StateGraph(EngineGraphState)`.

#### 4. Окружение

- Ollama — **системный** сервис (`brew install ollama`), не пакет venv.
- `langchain-ollama` в venv — только HTTP-клиент к `http://localhost:11434`.

---

## Запуск (актуальный)

Из корня `REsearch`:

```bash
source knowledge_engine/.venv/bin/activate
export PYTHONPATH="$(pwd)"
python -m knowledge_engine.main -c "ограничения" "инженерная задача"
```

Эквивалент с явной подкомандой:

```bash
python -m knowledge_engine.main analyze "задача" -c "ограничения"
```

---

## Известные ограничения MVP

- Re-Act «поиск» симулируется LLM (нет реального RAG/веб-поиска).
- Локальный 7B: долгий inference на матрице и unraveling.
- Предупреждение LangGraph `allowed_objects` при старте — не блокирует работу.

---

## Шаблон для следующих записей

```markdown
### YYYY-MM-DD — краткий заголовок

- **Контекст:** …
- **Изменение:** …
- **Затронутые файлы:** …
```
