# Переменные окружения Knowledge Engine

**Единая точка чтения:** `knowledge_engine/config.py` (`_load_dotenv()` при импорте).
В сервисах импортируйте константы из `config`, не `os.getenv`.

Шаблон: `.env.example`. Секреты: `.env`.

Полный машинный список (~272 ключей):

```bash
.venv/bin/python knowledge_engine/scripts/sync_env_catalog.py --write-example
.venv/bin/python knowledge_engine/scripts/sync_env_catalog.py --merge-env
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
| `DATABASE_URL` | `sqlite:///knowledge_engine/.runs/article_diagrams.db` |

## Postgres / pgvector / бэкенды (Phase 0-3)

Типизированная конфигурация — `knowledge_engine/db/pg_settings.py` (Pydantic
Settings), не голый `os.getenv`; `config.py` реэкспортирует готовые константы
(`POSTGRES_DSN` и т.д.), остальной код по-прежнему делает
`from knowledge_engine.config import POSTGRES_DSN`. См. `docker-compose.yml`
(`postgres` + ephemeral `migrator`, накатывает Alembic автоматически на
каждый `docker compose up`) и `docs/DOCKER_LAYOUT.md`.

| Variable | Default |
|----------|---------|
| `POSTGRES_DSN` | `postgresql://knowledge_engine:knowledge_engine@localhost:5432/knowledge_engine` (сырой DSN, без `+driver` — `+psycopg`/`+asyncpg` для SQLAlchemy выводятся из него программно) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_PORT` | `knowledge_engine` / `knowledge_engine` / `knowledge_engine` / `5432` (только для `docker-compose.yml`, сам контейнер) |
| `VECTOR_EMBED_DIM` | `1024` (BAAI/bge-m3 dense dim) |
| `VECTOR_HNSW_M` | `16` |
| `VECTOR_HNSW_EF_CONSTRUCTION` | `64` |
| `VECTOR_HNSW_EF_SEARCH` | `100` |
| `VECTOR_STORE_BACKEND` | `postgres` — переключатель `services/vector_store.py`. `qdrant` — старый путь (fallback, зависимости не убраны) |
| `GRAPH_CHECKPOINTER_BACKEND` | `postgres` — `AsyncPostgresSaver` через `TutorGraphService` (открывается/закрывается на каждый ход воркера, см. `services/tutor_graph_service.py`). `memory` — старый `MemorySaver` (RAM-only, теряется при рестарте) |

## Redis / worker

| Variable | Default |
|----------|---------|
| `REDIS_URL` | empty |
| `REDIS_SOCKET_TIMEOUT_SEC` | `10` |
| `KE_USE_REDIS` | true if `REDIS_URL` |
| `KE_REDIS_LOGS` | = `KE_USE_REDIS` |
| `KE_TASKS_CHANNEL` | `ke:tasks` |
| `KE_PROCESS_ROLE` | set by entrypoint (`api` / `worker`); do not put in `.env` |
| `KE_REDIS_LOG_MAX_LINES` | `20000` |
| `KE_WORKER_POLL_SEC` | `0.4` |
| `KE_WORKER_HEARTBEAT_SEC` | `10` |
| `KE_WORKER_STALE_RUNNING_SEC` | `300` |
| `KE_WORKER_INLINE_FALLBACK` | `false` (ignored for ML: API never loads BGE/CE) |
| `KE_WORKER_RELOAD_DEBOUNCE_SEC` | `1.0` |
| `KE_WORKER_STOP_TIMEOUT_SEC` | `30` |
| `KE_NODE_DIVE_TIMEOUT_SEC` | `900` |

## Logging

| Variable | Default |
|----------|---------|
| `KE_TRACE_STDOUT` | `false` |
| `KE_LOG_PLAIN` | `false` |
| `KE_LLM_FULL_TRACE` | `false` |
| `ENABLE_PROMPT_TRACE_LOGS` | `false` (если `KE_API_RELOAD=true`, по умолчанию `true`) |
| `PROMPT_TRACE_DIR` | `logs/session_traces` (от корня репо) |
| `PROMPT_TRACE_ALL_LLM` | `false` — при `true` логируются все Gemini in/out |
| `ENABLE_GEMINI_EXPLICIT_CACHE` | `true` |
| `GEMINI_CACHE_TTL_SECONDS` | `3600` |
| `GEMINI_CACHE_MIN_EST_TOKENS` | `32000` (ниже — без create, fallback на digest pinned) |

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
| `OLLAMA_NUM_PARALLEL` | `1` (сервер Ollama; для `BLOG_SPATIAL_MAP_CONCURRENCY>1` задайте `2`+) |

### Blog spatial Map-Reduce (ingest)

| Variable | Default |
|----------|---------|
| `BLOG_SPATIAL_SUMMARIZER_MODEL` | `MAIN_MODEL` |
| `BLOG_SPATIAL_NUM_CTX` | `16384` |
| `BLOG_SPATIAL_MAP_MAX_TOKENS` | `2800` |
| `BLOG_SPATIAL_OVERLAP_TOKENS` | `400` |
| `MAX_CONCURRENT_MAP_REQUESTS` | `4` (unified MAP in-flight for all models) |
| `BLOG_SPATIAL_MAP_CONCURRENCY` | `= MAX_CONCURRENT_MAP_REQUESTS` |
| `GEMMA_MAP_MAX_OUTPUT_TOKENS` | `4096` (fixed) |
| `GEMMA_REDUCE_MAX_OUTPUT_TOKENS` | `4096` |
| `REDUCE_STRATEGY` | `two_phase` (`legacy` = single FinalArticleSummary call) |
| `CODE_PARSER_MODE` | `linear` — structural line-batch splitter (`~40` lines/block), no AST awareness. `ast` activates `AstCodeChunker` (tree-sitter chunk boundaries by top-level function/class units; any parse failure falls back to `linear`). Real tree-sitter grammar coverage (`tree_sitter_<lang>`, `ast_code_chunker.py::EXTENSION_TO_LANGUAGE`): `c`, `cpp`, `python`, `javascript`, `typescript`, `tsx`, `go`, `rust`, `java`, `c_sharp`, `ruby`, `php`, `kotlin`, `swift` — not just Python/JS/TS. |
| `BLOG_SPATIAL_TRIAGE_ENABLED` | `true` — TOC-based section pruning (`DOC_TRIAGE` trace marker, `document_triage_engine.py`) before Map-Reduce; drops whole `[P_n]` ranges (nav/footer/off-topic) using `UniversalTOCExtractor` + `ArticleSectionPruner`. Independent of `CODE_PARSER_MODE`; applies to HTML/PDF/markdown/code alike (needs ≥4 paragraph blocks). |
| `BLOG_SPATIAL_TRIAGE_KEEP_FIGURES` | restores `FIG_*` refs dropped by text pruning, for VLM |
| `CHUNK_ANCHOR_INJECTION` | `false` (opt-in `[A1]`…`[An]` on MAP/REDUCE context) |
| `ANCHOR_REGEX_VALIDATE` | `false` (opt-in `[A99 (? unverified)]`; never touches `[S*]`/`[R*]`/`arr[0]`) |
| `CLAIM_DEDUP_MODE` | `none` (`exact` = identical SPO; `entity_consensus` = bge-m3 + reranker + cloud) |
| `CLAIM_MMR_LAMBDA` | `0.7` (does **not** change `LECTURE_RAG_MMR_LAMBDA`) |
| `SPO_CLUSTER_THRESHOLD` | `0.85` (cosine gate before reranker) |
| `SPO_RERANKER_DUPLICATE_THRESHOLD` | `0.88` (bge-reranker-v2-m3 duplicate merge) |
| `MAX_CONSENSUS_BATCH_TOKENS` | `3072` (system + batch; Gemma tokenizer; packer SSOT) |
| `MAX_CONSENSUS_NODES_PER_BATCH` | `10` (soft cap; token budget of 3072 can split smaller) |
| `MAX_PRIMARY_ANCHORS` | `3` (anti-bloat `primary_anchors`; full set in `all_anchors`) |
| `MIGRATION_USE_CONTEXT_CACHING` | `false` (Gemini ingest REDUCE cache; fallback Gemma) |
| `INGEST_CACHE_TTL_SECONDS` | `86400` (ingest cache only; tutor uses `GEMINI_CACHE_TTL_SECONDS`) |
| `USE_GITHUB_TREES_API` | `false` (opt-in Git Trees API; repo root → corpus; `/blob/` → target + depth-1 AST deps ≤ 5; 401/403/404/timeout → zip then HTML) |
| `GITHUB_TOKEN` | empty (optional; raises GitHub REST limit to 5000 req/h) |
| `MAX_GITHUB_FILE_SIZE_BYTES` | `102400` (skip blobs larger than 100 KB before download) |

Параллельный MAP на клиенте всегда `asyncio.Semaphore(4)` (Gemma cloud и Ollama). Для Ollama задайте `OLLAMA_NUM_PARALLEL >= 4`.

Токены окон: при установленном `transformers` используется HF tokenizer Qwen2.5; иначе `tiktoken` с запасом ×1.15.

**Три независимых pruning/chunking-механизма** (не путать друг с другом — ни один не управляется тем же флагом, что другой; подробнее: [CODE_TRIAGE_AND_PRUNING.md](CODE_TRIAGE_AND_PRUNING.md)):

| Механизм | Уровень | Управляется | Trace-маркер |
|----------|---------|--------------|--------------|
| `tiered_code_pruner.py` | тело функции (HIGH/MEDIUM/LOW → full/signature-only/dropped) | ничем — работает безусловно для code-языков, вызывается из `raw_source.py` до аннотации | `TieredCodePrune` |
| `AstCodeChunker` | границы MAP-чанков (top-level функции/классы) | `CODE_PARSER_MODE=ast` (default `linear`) | — (использует `[Pipeline Audit] Phase: Annotate`) |
| `DOC_TRIAGE` (`document_triage_engine.py`) | целые диапазоны `[P_n]` (TOC-секции: nav/footer/off-topic) | `BLOG_SPATIAL_TRIAGE_ENABLED` | `DOC_TRIAGE` |

### Article diagrams / VLM

Поток и назначение переменных: [ARTICLE_DIAGRAMS.md](ARTICLE_DIAGRAMS.md).

| Variable | Default |
|----------|---------|
| `ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL` | `MAIN_MODEL` |
| `ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC` | `45` |
| `ARTICLE_DIAGRAM_FILTER_NUM_PREDICT` | `256` |
| `ARTICLE_DIAGRAM_FILTER_NUM_CTX` | `4096` |
| `ARTICLE_MAX_DIAGRAMS_PER_ARTICLE` | `4` |
| `VLM_GEMINI_MODEL` | `gemini-3.5-flash-lite` |
| `VLM_GEMINI_MODELS` | empty → primary + Lite fallback chain |
| `GEMINI_FLASH_LITE_MAX_RPM` / `TPM` / `RPD` | `14` / `250000` / `490` | shared per-model caps for **all** Flash Lite uses (VLM, tutor, map overflow, curriculum…) |
| `VLM_GEMINI_MAX_RPM` / `MAX_TPM` / `MAX_RPD` | same as Flash Lite | VLM aliases (default = shared caps) |
| `VLM_GEMINI_CONCURRENCY` | `3` |
| `VLM_GEMINI_EST_INPUT_TOKENS` / `EST_OUTPUT_TOKENS` | `12000` / `1024` |
| `VLM_GEMINI_QUOTA_TRACK` | `true` |

### Token & Rate Governor

`services/token_rate_governor.py` — единый zero-wait-if-free пейсер RPM/TPM
для Gemini (через `_call_with_model_fallback` и `gemini_search_grounding.py`);
подробности: [PERFORMANCE.md](PERFORMANCE.md#token--rate-governor-gemini-pacing).

| Variable | Default | Кратко |
|----------|---------|--------|
| `GEMINI_GOVERNOR_TARGET_RPM` | `14` | Рабочий RPM-таргет Governor для Gemini 3.5 Flash Lite (лимит провайдера 15) |
| `GEMINI_GOVERNOR_TARGET_TPM` | `240000` | Рабочий TPM-таргет Governor для Gemini 3.5 Flash Lite (лимит провайдера 250K) |
| `GEMMA_GOVERNOR_TARGET_RPM` | `27` | Рабочий RPM-таргет для Gemma 4 (26B/31B); лимит провайдера 30. Применяется к `AsyncRateLimiter` через `GEMMA_BUDGET_MAX_RPM` (см. ниже), не через новый Governor-класс — у Gemma уже был корректный sliding-window limiter |
| `GEMMA_GOVERNOR_TARGET_TPM` | `15200` | Рабочий TPM-таргет для Gemma 4; лимит провайдера 16K. Применяется через `GEMMA_BUDGET_MAX_TPM` |

Прежний `_rpm_pause_for_model()` (безусловный `time.sleep(60/RPM)` перед
каждым Gemini-запросом) полностью удалён из `gemini_stateless.py` и
`gemini_search_grounding.py`; `GEMINI_RPM_PAUSE_SEC` / `GEMINI_RPM_JITTER_SEC`
теперь используются только внутри backoff-паузы после реальных 429/5xx-ошибок
(`_sleep_with_jitter`), не как безусловный пейсер.

## Gemini, CSE, SearXNG, SS, Exa, RAG, Curriculum, Consensus

См. таблицы в `.env.example` (комментарии) и блоки `CURRICULUM_*`, `CONSENSUS_*`, `GEMINI_*` в `config.py`.

**Exa Search (полный справочник `EXA_*`):** [EXA_SEARCH.md](EXA_SEARCH.md). Пул источников: [SOURCE_POOL.md](SOURCE_POOL.md).

### arXiv API

| Variable | Default | Кратко |
|----------|---------|--------|
| `ARXIV_MIN_INTERVAL_SEC` | `3.25` | Exclusive gate: ≥3.25s between successive arXiv API calls (official ≥3s); lock held for whole HTTP + retries |
| `ARXIV_MAX_RETRIES` | `3` | Retries on HTTP 503/429/403 with exponential backoff |
| `ARXIV_BACKOFF_BASE_SEC` | `3.0` | Base for `base * 2^n + jitter` before retry |
| `ARXIV_ID_LIST_CHUNK` | `50` | Max IDs per `id_list` hydrate request |
| `CURRICULUM_ACADEMIC_SEARXNG_ENGINES` | `arxiv,google scholar` | SearXNG engines for academic track (no bing/google fallback) |
| `CURRICULUM_ACADEMIC_SEARXNG_CATEGORIES` | `science` | SearXNG categories for academic (not `it`) |

### Academic rerank / relaxation

| Variable | Default | Кратко |
|----------|---------|--------|
| `ACADEMIC_RERANK_ENABLED` | `false` | Hybrid pre-sort before CE/MMR |
| `ACADEMIC_RERANK_WEIGHTS` | `0.45,0.25,0.20,0.10` | α,β,γ,δ (sim, trust, cites, recency) |
| `ACADEMIC_RERANK_C_SAT` | `40` | Citation log saturation |
| `ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS` | `6` | Recency decay half-life |
| `ACADEMIC_RELAXATION_ENABLED` | `true` | Cascade when hits &lt; min |
| `ACADEMIC_RELAXATION_MIN_HITS` | `3` | Trigger threshold |
| `ACADEMIC_RELAX_L0_MIN_TRUST` | `0.35` | Strict gate |
| `ACADEMIC_RELAX_L0_MIN_CITATIONS` | `5` | Strict gate |
| `ACADEMIC_RELAX_L1_MIN_TRUST` | `0.2` | Soft gate |
| `ACADEMIC_RELAX_L1_MIN_CITATIONS` | `0` | Soft gate |
| `ACADEMIC_RELAX_L1_YEAR_PAD` | `3` | Years padded before date drop |

| Variable | Default | Кратко |
|----------|---------|--------|
| `EXA_API_KEY` | empty | Ключ Exa API |
| `EXA_SEARCH_ENABLED` | `true` | Вкл/выкл поиск |
| `CURRICULUM_PRACTICAL_EXA_LIMIT` | `12` | Cap simple/bulk `num_results` |
| `EXA_FETCH_NUM_RESULTS` | `20` | DEEP recall budget |
| `EXA_MAX_CONCURRENT_SEARCH` | `3` | Параллель multi-vector |
| `EXA_RECALL_MAX_PER_DOMAIN` | `2` | Max/host в DEEP RR |
| `EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN` | `1` | Дефолт fair RR |
| `EXA_DOMAIN_CAP_PER_HOST` | `1` | В config; runtime → `EXA_RECALL_*` / `EXA_FAIR_*` |
| `EXA_RERANK_LITE_THRESHOLD` | `5` | Порог Lite rerank |
| `EXA_DUAL_QUERY_EN_RATIO` | `0.7` | Доля EN dual-merge |
| `EXA_EXCLUDE_TEXT` | api reference… | Exa excludeText ≤5 слов |
| `EXA_PRACTICAL_HIGHLIGHT_QUERY` | (см. `config.py`) | Fallback highlights |
| `EXCLUDED_SOURCES_BLACKLIST` | medium,dev.to,… | exclude_domains |
| `DOMAIN_REGISTRY_EMBED_MODEL` | `BAAI/bge-m3` | Bi-Encoder gist доменов |
| `DOMAIN_REGISTRY_COSINE_MIN` | `0.82` | Порог Pre-Discovery lookup |
| `DOMAIN_REGISTRY_SEARCH_LIMIT` | `8` | Max official hosts из LanceDB |

Ключевые для tutor/RAG:

| Variable | Default |
|----------|---------|
| `EMBED_MODEL` | `BAAI/bge-m3` |
| `EMBED_MODEL_REVISION` | `5617a9f…` | Pin Hub commit; load from `~/.cache/huggingface` |
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RAG_CROSS_ENCODER_REVISION` | `953dc6f…` | Pin reranker snapshot |
| `RAG_DEFAULT_MIN_RELEVANCE` | `0.55` |
| `RAG_CE_AUTO_UNLOAD` | `false` |
| `RAG_CE_AUTO_UNLOAD_IDLE_SEC` | `300` |
| `RAG_MPS_MEMORY_THRESHOLD_GB` | `3.5` | `services/ml_memory_guard.py` — экстренный порог OS-agnostic footprint-а (см. [PERFORMANCE.md](PERFORMANCE.md#ml-memory-guard-bge-m3--cross-encoder-mps-ram)); превышение выгружает не занятую прямо сейчас модель немедленно, не дожидаясь `RAG_MPS_REQUEST_COOLDOWN_SEC` |
| `RAG_MPS_REQUEST_COOLDOWN_SEC` | `300` | Idle-выгрузка ВСЕХ моделей по завершении ВСЕГО RAG-запроса (`rag_request_finished()`), не отдельного вызова embed/rerank; таймер сбрасывается каждым новым запросом (`rag_request_started()`) |
| `LECTURE_RAG_*` | см. `config.py` |
| `LIGHT_RAG_MIN_COSINE_SIM` | `0.42` |
| `KE_RAG_TIMEOUT_SEC` | `45` |
