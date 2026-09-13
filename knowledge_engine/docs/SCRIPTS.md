# Каталог `knowledge_engine/scripts`

Канонический список setup-, diagnostic-, maintenance- и legacy-скриптов.
Актуальность CLI сверена с `argparse` и shell-кодом 2026-09-08.

Python-скрипты сгруппированы в поддиректориях по подсистеме, которую они
обслуживают (`dev/`, `curriculum_sources/`, `db/`, `benchmarks/`,
`pdf_diagrams/`, `search_diagnostics/`, `consensus/`, `legacy_research/`) —
mirror структуры `knowledge_engine/services|src`. Shell-скрипты (`.sh`)
остаются плоско в `knowledge_engine/scripts/`: почти все они — dev/deploy
bootstrapping (venv, Docker, Playwright), не привязанное к конкретной
подсистеме, и вложение их на уровень глубже сломало бы относительную
адресацию `dirname "$0")/../..` → корень репозитория.

Все команды выполняются из корня `REsearch`. Для Python-команд:

```bash
export PYTHONPATH="$(pwd)"
source .venv/bin/activate
```

У большинства Python CLI доступна встроенная справка:

```bash
python -m knowledge_engine.scripts.<subdir>.<module> --help
```

## Обозначения безопасности

| Метка | Значение |
|-------|----------|
| **read-only** | Читает локальное состояние или вызывает probe без изменения данных проекта |
| **writes** | Изменяет `.env`, `.runs`, session JSON, SQLite, LanceDB или generated bundle |
| **external/cost** | Делает сетевые запросы либо вызывает Gemini/Gemma/VLM |
| **destructive** | Удаляет/перемещает данные; сначала используйте dry-run или остановите API/worker |
| **legacy** | Поддержка старого research/analysis flow; не основной Skill Tree pipeline |

## 1. Setup и разработка (`scripts/`, `scripts/dev/`)

| Скрипт | Ключи / аргументы | Назначение и побочные эффекты |
|--------|-------------------|-------------------------------|
| `setup.sh` | нет | Полная первичная настройка: запускает SearXNG, затем `setup-host-ollama.sh` и `setup-host-python.sh`. **writes, external** |
| `setup-host-ollama.sh` | нет | Проверяет локальный Ollama и загружает `qwen2.5-coder:1.5b`, `qwen2.5-coder:7b`. **external** |
| `setup-host-python.sh` | нет | Создаёт корневой `.venv`, ставит runtime/dev requirements и Chromium Playwright. **writes, external** |
| `install-playwright.sh` | нет | Доустанавливает Chromium в `.venv` с `PLAYWRIGHT_BROWSERS_PATH=0`. **writes, external** |
| `dev-native.sh` | env: `KE_API_HOST`, `KE_API_PORT`, `SEARXNG_BASE_URL`, `OLLAMA_BASE_URL`, `REDIS_URL` | Основной dev entry point: SearXNG+Redis в Docker, worker-watch (`dev/dev_worker_watch.py`) и API с reload. Останавливает старые процессы worker/watch. |
| `dev-watch.sh` | нет | Legacy Docker API с Compose Watch; на macOS предпочтителен `dev-native.sh`. **legacy** |
| `sync-venv.sh` | нет | Собирает Docker API image/venv volume и проверяет импорты. **writes, external** |
| `cleanup-venvs.sh` | нет | Удаляет только `.venv-host` и `knowledge_engine/.venv`; корневой `.venv` не трогает. **destructive** |
| `docker-stack.sh` | нет | Поднимает SearXNG и печатает дальнейшие команды настройки; helper без собственного Python runtime. |
| `consensus-login.sh` | env из `.env`, `PLAYWRIGHT_BROWSERS_PATH` | Интерактивный вход в Consensus через тот же Chromium, что использует dev runtime. **writes** |
| `build-skill-tree-ui.sh` | нет | `npm install` + production build; обновляет `knowledge_engine/web/static/skill-tree/skill-tree.bundle.js`. **writes, external** |
| `dev/dev_worker_watch.py` | нет; запускается автоматически | Следит за Python-файлами и перезапускает `knowledge_engine.src.app.worker`. Вручную обычно не запускается. |
| `dev/sync_env_catalog.py` | `--write-example`, `--merge-env` | Без ключей печатает рассинхронизацию env-каталога. `--write-example` перезаписывает `.env.example`; `--merge-env` меняет локальный `.env`. **writes** |

Примеры:

```bash
./knowledge_engine/scripts/setup.sh
./knowledge_engine/scripts/dev-native.sh
python knowledge_engine/scripts/dev/sync_env_catalog.py
python knowledge_engine/scripts/dev/sync_env_catalog.py --write-example
```

## 2. Work jobs, ноды и curriculum sources (`scripts/curriculum_sources/`)

### `curriculum_sources/cancel_work_job.py`

Закрывает зависшие `pending/running` jobs и освобождает Redis claim/grounding locks
и `knowledge_engine/.runs/worker_dev_busy.json`.

```bash
python knowledge_engine/scripts/curriculum_sources/cancel_work_job.py \
  [--id JOB_ID] \
  [--curriculum CURRICULUM_ID --node NODE_ID] \
  [--all-active] [--complete] [--release-locks-only] [--reason TEXT]
```

- по умолчанию job переводится в `failed`;
- `--complete` восстанавливает result из session и завершает job успешно;
- `--release-locks-only` не меняет статус;
- `--id` допускает уникальный префикс от 8 символов;
- без selector выводит help и активные jobs.

**writes, destructive:** неверный `--complete` или cancel может зафиксировать
неконсистентное состояние.

### Source/session maintenance

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `curriculum_sources/clear_node_data.py` | `--curriculum-id ID --node-id ID [--apply] [--skip-cloud] [--skip-library-gc] [--no-clear-blocklist] [--no-scrub-registry] [--json]` | **Единая точка входа** для очистки данных одной ноды: последовательно вызывает `cleanup_cloud_resources` (Qdrant+Redis) → `clear_node_sources` (LanceDB/graph/session/blocklist) → `sync_curriculum_library_sources` (GC осиротевших registry-записей после очистки ноды). Порядок important — cloud обязан отработать ДО того, как local-этап очистит `mapped_source_ids`. Без `--apply` — dry-run (Stage 3 preview в dry-run занижен — граф ещё не пропатчен). **destructive** |
| `curriculum_sources/clear_node_sources.py` | `--curriculum ID --node ID [--apply] [--no-clear-blocklist] [--no-scrub-registry] [--json]` | Планирует или удаляет sources одной ноды из graph/session/LanceDB и, по умолчанию, из blocklist. Без `--apply` — dry-run. Вызывается из `clear_node_data.py`, но применим и отдельно. **destructive** |
| `curriculum_sources/cleanup_cloud_resources.py` | `--node-id ID --curriculum-id ID [--dry-run]` (apply по умолчанию, `--dry-run` для предпросмотра) | Удаляет из **Qdrant** (rag_chunks/document_summaries/knowledge_atoms по url/doc_id) и Redis (`ke:lock:node_ground:*`) записи ноды/куррикулума. URL резолвятся через `clear_node_sources._collect_node_source_urls` (registry + `resource_urls`/`source_ref` — не только registry, т.к. `mapped_source_ids` может ссылаться на несуществующие registry-записи). Gemini Cache — не применимо (нет node/curriculum linkage). LanceDB не трогает. **destructive, external** |
| `curriculum_sources/sync_node_session_sources.py` | `--curriculum ID --node ID [--apply] [--json]` | Сверяет `session.source_registry` с `mapped_source_ids`, retarget/scrub ссылок. Без `--apply` — dry-run. **writes** |
| `curriculum_sources/backfill_verified_source_registry.py` | `[--curriculum ID] [--node ID] [--apply] [--json]` | Регистрирует в `curriculum_sources_registry`/`mapped_source_ids` URL, у которых уже есть `resource_urls` + `document_summaries`, но не привязан registry (см. `persist_verified_external_sources_to_node` в `lecture_search_orchestrator.py` — без реестра `coerce_references_to_registry` отбрасывает все references, лекция цитирует `[n]` вместо `[Sn]`). Без `--curriculum` — сканирует все куррикулумы. URL без `document_summaries` пропускает. Без `--apply` — dry-run. **writes** |
| `curriculum_sources/sync_curriculum_library_sources.py` | `--curriculum ID [--apply] [--json]` | Удаляет orphan registry entries, синхронизирует sessions; из глобального LanceDB удаляет URL только если он не используется другими curriculum. Без `--apply` — dry-run. Вызывается из `clear_node_data.py` как финальный шаг, но применим и отдельно для GC всей библиотеки курса. **destructive** |
| `curriculum_sources/run_lazy_ground_node.py` | `[--curriculum-id ID] [--node-id ID] [--on-demand] [--full-academic] [--also-spatial-url URL] [--also-spatial-pdf PATH]` | Принудительный targeted search/ingest одной ноды. `--full-academic` отключает fast on-demand path. **writes, external/cost** |
| `curriculum_sources/inspect_node_source_collection.py` | `--curriculum ID --node ID [--source-id ID] [--url URL] [--probe-fetch] [--probe-smart] [--timeout 25] [--json]` | Сверяет graph/session/LanceDB source collection. Probe-ключи выполняют реальные HTTP/fetch запросы. |
| `curriculum_sources/sync_personal_rag_profile.py` | `[--force]` | Индексирует `user_profile.md` в `light_rag_facts`; `--force` сбрасывает SHA-предохранитель. **writes** |
| `curriculum_sources/sweep_orphaned_knowledge_atoms.py` | `[--curriculum ID] [--apply] [--json]` | One-off sweep: удаляет локальные LanceDB `knowledge_atoms` строки И живые Qdrant Cloud точки (document_summaries/rag_chunks/knowledge_atoms), чей URL больше не используется ни одним сохранённым curriculum. Regression-хвост миграции на Qdrant-only запись атомов — pre-migration строки не имели пути очистки. Без `--apply` — dry-run. **destructive, external** |

Рекомендуемый способ полностью очистить одну ноду (local + cloud + library GC
одной командой) — `clear_node_data.py`:

```bash
python knowledge_engine/scripts/curriculum_sources/clear_node_data.py \
  --curriculum-id agentic_systems_architecture \
  --node-id governed_agent_pipelines
# проверить план (Stage 3 preview занижен до --apply, см. описание скрипта)
python knowledge_engine/scripts/curriculum_sources/clear_node_data.py \
  --curriculum-id agentic_systems_architecture \
  --node-id governed_agent_pipelines --apply
```

`clear_node_sources.py`/`cleanup_cloud_resources.py`/`sync_curriculum_library_sources.py`
остаются рабочими самостоятельно (например, для GC всей библиотеки курса без
привязки к конкретной ноде).

## 3. LanceDB, Postgres/Qdrant и кэши (`scripts/db/`)

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `db/inspect_knowledge.py` | `[--db PATH] [--doc-id ID] [--title FRAGMENT]` | Rich-инспектор passports, knowledge atoms и rag chunks. Если selector не задан, берёт первый документ. **read-only** |
| `db/backfill_document_passports.py` | `[--dry-run] [--limit N] [--curriculum ID] [--doc-id ID] [--include-window-summary-only] [--force-gemma-cloud\|--no-force-gemma-cloud]` | Восстанавливает passports, atoms и `window_summary` через MAP→REDUCE. Cloud включён по умолчанию; сначала используйте `--dry-run`. **writes, external/cost** |
| `reset-lancedb.sh` | нет | Перемещает `knowledge_engine/.lancedb` в `knowledge_engine/.lancedb.bak-<timestamp>` и создаёт пустой каталог. Остановите API/worker; потребуется re-ingest. **destructive** |
| `db/invalidate_gemini_prompt_cache.py` | `[--local-only] [--dry-run]` | Без ключей удаляет remote Gemini cached content и локальный registry. `--local-only` не трогает remote cache. **destructive, external** |
| `db/run_phase3_integration_test.py` | `[--url URL] [--max-files N] [--all-files] [--out PATH]` | Smoke Phase 3A (GitHub Trees API → `github_trees`) + Phase 3B (`CLAIM_DEDUP_MODE=entity_consensus`, `primary_anchors` ≤ 3). Пишет `knowledge_engine/.runs/phase3_integration_report.md`. **Не** пишет LanceDB. **external/cost** |
| `db/check_qdrant_connection.py` | нет | Standalone проверка связи с Qdrant Cloud: ping, создаёт+удаляет одноразовую `test_ping` коллекцию, upsert/read-back одной точки через `QdrantVectorStore.fetch_by_url`. Не печатает `QDRANT_API_KEY`. **external** |
| `db/migrate_lancedb_to_qdrant.py` | нет | One-time перенос `rag_chunks`/`document_summaries` из LanceDB в Qdrant Cloud без ре-эмбеддинга. Пропускает строки с несовпадающим `embed_model`. Идемпотентен (детерминированные point id). **writes, external, destructive-adjacent** |
| `db/backfill_qdrant_to_postgres.py` | нет | Backfill Qdrant Cloud → Postgres/pgvector: переносит существующие точки (document_summaries/knowledge_atoms/rag_chunks) в Postgres-таблицы до переключения `vector_store.py` на `PostgresVectorRepository` как основной бэкенд. **writes, external** |
| `db/run_migrations_and_verify.py` | нет; env через `.env`/compose | Init-container entrypoint (`migrator` в `docker-compose.yml`, `CMD python -m knowledge_engine.scripts.db.run_migrations_and_verify`): накатывает Alembic до head и проверяет целостность (vector extension + HNSW-индексы) при старте стека. Ненулевой exit блокирует старт зависимых сервисов (`depends_on: service_completed_successfully`). **writes, destructive-adjacent** |

## 4. Benchmarks (`scripts/benchmarks/`)

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `benchmarks/benchmark_gemma_chunk_sizes.py` | `[--file PATH\|--url URL\|--pdf PATH] [--title TEXT] [--sizes CSV] [--max-chunks N] [--overlap N] [--model ID] [--api-base URL] [--api-key KEY] [--max-out N] [--timeout 180] [--sleep 1.25] [--config-sleep 5] [--dry-run] [--json-out PATH]` | Сравнивает MAP schema success для размеров окон. Без source-ключа использует synthetic corpus; `--dry-run` только делит текст, обычный запуск расходует quota. **external/cost** |
| `benchmarks/benchmark_pipeline_recall.py` | см. `--help` | Изолированный бенчмарк MAP+REDUCE pipeline: реальный production code path (`map_reduce_summarize_blog_outcome_async`) на одной тестовой статье — baseline-тайминг + recall knowledge_atoms. Без LanceDB/Qdrant persistence и HTML annotation/ingest-gate. Импортирует `benchmark_gemma_chunk_sizes` для общих helper'ов. **external/cost** |

## 5. Mermaid, VLM и PDF QA (`scripts/pdf_diagrams/`)

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `pdf_diagrams/sanitize_curriculum_diagrams.py` | `--curriculum-id ID [--node-id ID ...] [--diagram-ids ID/TITLE ...] [--with-gemma] [--force-gemma] [--concurrency N] [--dry-run]` | Массовая нормализация Mermaid в sessions. Gemma вызывается только с `--with-gemma`; `--force-gemma` требует его же. **writes, optional cost** |
| `pdf_diagrams/repair_session_diagrams.py` | `--curriculum-id ID --node-id ID --diagram-ids ID... [--no-gemma] [--force-gemma] [--dry-run]` | Точечный sanitize→validate→Gemma repair выбранных диаграмм. По умолчанию Gemma разрешена при необходимости. **writes, optional cost** |
| `pdf_diagrams/rerun_vlm_session_diagrams.py` | `--curriculum-id ID --node-id ID --article-id ID --diagram-ids ID... [--figures-dir PATH] [--dry-run]` | Повторный VLM по локальным изображениям; обновляет session и `article_diagrams`. Default figures dir: `knowledge_engine/.runs/acm_figures_qa`. **writes, external/cost** |
| `pdf_diagrams/export_pdf_figures.py` | `PDF [-o\|--out DIR]` | Экспортирует `FIG_*` production-путём `build_annotated_pdf`; не вызывает VLM. **writes** |
| `pdf_diagrams/export_pdf_annotated_text.py` | `PDF [-o\|--out DIR] [--url URL] [--title TITLE] [--with-registry]` | Экспортирует raw annotated text, MAP payloads и manifest; `--with-registry` добавляет anchors без VLM. **writes** |
| `pdf_diagrams/check_paper_structure_analyzer.py` | `[--url PDF_URL] [--pdf PATH] [--topic TEXT] [--min-relevance 4] [--fallback-only] [--json-out PATH] [--save-input-json PATH]` | Проверяет PDF structure/credibility analyzer и retention. Без `--fallback-only` вызывает Gemini Lite. **optional cost** |

## 6. Gemini и curriculum search diagnostics (`scripts/search_diagnostics/`)

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `search_diagnostics/check_gemini_grounding.py` | `[--models CSV] [--all-candidates] [--json] [--pause 3] [--query TEXT] [--compare-plain] [--metadata] [--save] [--list-models]` | Probe Gemini с Google Search grounding; `--save` пишет `knowledge_engine/.runs/gemini_grounding_probe.json`. **external/cost** |
| `search_diagnostics/check_gemini_quotas.py` | `[--models CSV] [--json] [--pause 2] [--save] [--summary-only]` | Проверяет модели на quota/429. `--summary-only` не вызывает API; `--save` обновляет local quota store. **external**, кроме summary |
| `search_diagnostics/check_curriculum_quotas.py` | `[--json] [--clear-ss-block]` | Показывает local CSE/Semantic Scholar counters; `--clear-ss-block` удаляет quota-state файл текущего дня. **writes** |
| `search_diagnostics/check_curriculum_search_providers.py` | `[--goal TEXT] [--json]` | Health/probe arXiv, Semantic Scholar, SearXNG и статусов optional CSE/DDGS. **external** |
| `search_diagnostics/smoke_curriculum_sources.py` | `[--goal TEXT] [--policy hybrid\|practical_only\|academic_only] [--with-collect] [--with-playwright] [--json]` | Smoke source collectors. `--with-collect` запускает реальный сбор; вместе с ним `--with-playwright` сохраняет Gemini web/grounding enabled. Без `--with-collect` этот ключ не действует. **external/cost** |
| `search_diagnostics/run_curriculum_search_first.py` | `[--goal TEXT] [--mode fast\|consensus] [--policy hybrid\|academic_only\|practical_only] [--depth TEXT]` | Полный Search-First curriculum trace. Default mode — `consensus`. **writes, external/cost** |
| `search_diagnostics/trafilatura_test.py` | `[--url URL] [--async]` | Изолированный quality-probe Trafilatura: fetch HTML → extract markdown → пишет в `output/`. **external, writes** |
| `search_diagnostics/test_trafilatura_async.py` | см. `--help` | Async-вариант проверки Trafilatura extraction поверх реального URL. **external** |

## 7. Consensus diagnostics (`scripts/consensus/`)

| Скрипт | Ключи | Назначение |
|--------|-------|------------|
| `consensus/check_consensus_playwright.py` | `[--query TEXT] [--send] [--headless] [--record-har] [--har-path PATH] [--log-json] [--json]` | Проверяет browser session; `--send` отправляет запрос, `--record-har` сохраняет network trace. **external** |
| `consensus/test_consensus_quick.py` | `[--query TEXT] [--headless] [--json]` | Минимальный Playwright smoke `/quick/`. **external** |
| `consensus/analyze_consensus_har.py` | `[--har PATH] [--out PATH] [--top 5] [--json]` | Ищет paper-list JSON endpoint в HAR и записывает descriptor. Файл может содержать auth headers — не коммитить без проверки. **writes** |
| `consensus/poc_consensus_api.py` | `[--endpoint PATH] [--query TEXT] [--storage-state PATH] [--skip-playwright] [--via auto\|curl\|playwright] [--headless] [--warmup-url URL] [--user-agent UA] [--impersonate chrome124] [--json]` | POC direct API через `curl_cffi` и/или Playwright cookies. Storage state содержит credentials. **external, writes sensitive local state** |
| `consensus/smoke_consensus_direct.py` | `[--query TEXT]` | Быстрый smoke production `ConsensusDirectClient`; печатает JSON. **external** |

`consensus-login.sh` (интерактивный вход) остаётся плоско в `scripts/` — см. Раздел 1.

## 8. LLM trace и legacy research pipeline (`scripts/legacy_research/`)

| Скрипт | Ключи / аргументы | Назначение |
|--------|-------------------|------------|
| `legacy_research/run_pipeline_llm_trace.py` | `[--query TEXT] [--mode consensus\|fast] [--thread-id ID] [--profile PATH]` | v0.8 research pipeline с `KE_LLM_FULL_TRACE=1`; пишет `knowledge_engine/.runs/*.log`. **external/cost** |
| `legacy_research/extract_llm_prompt_samples.py` | `[LOG_FILE] [-o\|--out PATH] [--max-per-function 2]` | Извлекает prompt/response samples из full trace. Выход может содержать пользовательские данные и model output. **writes** |
| `legacy_research/log_profiler.py` | `LOG_FILE [--top 15] [--since "HH:MM:SS"\|"YYYY-MM-DD HH:MM:SS"] [--llm-audit]` | Разбор произвольного лога прогона ноды/воркера (`trace()`, `logging`-формат, pytest caplog): тайминг по 6 стадиям пайплайна, топ самых долгих событий. `--since` обрезает лог с метки (для анализа конкретного прогона в общем `perf_debug.log`). `--llm-audit` — реестр реальных LLM-вызовов (Gemini/Gemma): токены in/out, латентность, конкурентность (пиковая одновременность вызовов, простои ≥5s без LLM-вызовов), RPM-spacing/quota-события, размер payload по стадиям (Triage/Bulk Gate/Code Dedup). Не привязан к v0.7/v0.8 research pipeline — общий инструмент для аудита RPM/TPM/параллелизма Skill Tree node-прогонов (см. `PERFORMANCE.md` §"Token & Rate Governor"). **read-only** |
| `legacy_research/run_v07.py` | `QUERY [--profile PATH] [--thread-id ID] [--json] [--no-repl]` | CLI research graph; фактическая версия выбирается через `GRAPH_VERSION`. **legacy, external/cost** |
| `legacy_research/smoke_v07.py` | `--query TEXT [--profile PATH] [--thread-id ID]` | Compile + полный smoke v0.7 graph. **legacy, external/cost** |
| `run-v07-analysis.sh` | `"QUERY"` | Shell-wrapper над `legacy_research/run_v07.py` (`python -m knowledge_engine.scripts.legacy_research.run_v07`), подхватывает `.env`. **legacy** |
| `smoke_v07.sh` | `[QUERY]`; env `SKIP_V07_FETCH=1` | Shell smoke; с `SKIP_V07_FETCH=1` запускает только Stage 0. **legacy** |

## 9. Legacy analysis-job helpers (`scripts/`, плоско)

Эти команды относятся к `/api/v1/analyses`, а не к основному Skill Tree flow.
Остаются плоско — четыре shell-скрипта без Python-эквивалента для переноса.

| Скрипт | Аргументы / env | Назначение |
|--------|-----------------|------------|
| `wait-analysis.sh` | `"PROBLEM" [CONSTRAINTS] [TIMEOUT=600]`; `KE_API_BASE`, `KE_JOB_INTERACTIVE`, `KE_JOB_JSON` | Создаёт analysis job, ждёт `matrix`, сохраняет `knowledge_engine/.runs/last-wait-response.json` и открывает Rich viewer. **legacy, writes** |
| `poll-analysis.sh` | `JOB_ID [INTERVAL=5] [BASE_URL]` | Polling до `matrix_ready/completed/failed` либо clarify. **legacy** |
| `unravel-analysis.sh` | `JOB_ID OPTION_ID [TIMEOUT=600]`; `UNRAVEL_FORCE=1` | Запускает/дожидается unravel для готовой матрицы. **legacy, external/cost** |
| `view-job.sh` | `[JOB_ID] [опции job_view]` | Wrapper над `knowledge_engine.src.app.cli`; без аргументов читает last response. **legacy, read-only** |

## 10. Полнота каталога

В каталоге перечислены все текущие файлы `knowledge_engine/scripts/` (флат `.sh`
+ 8 Python-поддиректорий):

- 18 shell-скриптов (плоско в `scripts/`);
- 47 Python-скриптов (в `dev/`, `curriculum_sources/`, `db/`, `benchmarks/`,
  `pdf_diagrams/`, `search_diagnostics/`, `consensus/`, `legacy_research/`);
- всего 65.

Новый файл в `scripts/` должен получить строку здесь одновременно с добавлением
CLI, в подходящей по теме поддиректории (для Python) или плоско в `scripts/`
(для `.sh`, если это не analysis-job helper — см. Раздел 9). Для
destructive-команд обязательны default dry-run либо явное предупреждение.
