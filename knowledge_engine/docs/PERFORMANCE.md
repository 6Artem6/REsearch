# Время прогона `analyze`

## Где уходили ~22 мин (ваш фоновый прогон)

| Этап | Почему долго |
|------|----------------|
| **Ollama 7B на CPU** (Docker OrbStack) | ~5–8 tok/s; один structured JSON ≈ 3–5 мин при длинном ответе |
| **summarizer × N URL** | каждый URL = **отдельный 7B** + embed; в логе было ≥3 (Microsoft, YouTube…) |
| **Playwright** | `fetch_page_html` на каждый URL |
| **Vision** | скриншоты/картинки (Microsoft, YouTube thumbnails) |
| **Gemini** | Playwright + ожидание ответа |
| **Горизонты поиска** | 3 × несколько API + Bing |
| **matrix** | ещё один 7B |
| **unraveling** | ещё один 7B (если не `--matrix-only`) |

До паузы на выбор варианта — почти всё выше **без** unraveling.

## Что уже сокращено в коде (defaults)

- `MAX_FETCH_URLS=3` (было до 10 URL с парсингом)
- `MULTI_SEARCH_SKIP_VISION=true` (vision по умолчанию выключен)
- Блоклист URL: YouTube, Microsoft support, Geeksforgeeks, Wikipedia
- Приоритет: arxiv, doi, Semantic Scholar, Habr
- `OLLAMA_NUM_PREDICT=1024` — не раздувать JSON
- `--matrix-only` — только матрица, без unraveling
- `SKIP_GEMINI=true` — пропуск Gemini (сразу multi_search)
- Сводка `print_timing_summary()` после матрицы

## Рекомендуемый быстрый прогон (Mac)

```bash
export OLLAMA_BASE_URL=http://localhost:11434   # host Ollama + Metal, не Docker CPU
export SKIP_GEMINI=true
export MAX_FETCH_URLS=2
export MULTI_SEARCH_SKIP_VISION=true

python -m knowledge_engine.main analyze -c "…" "…" --matrix-only
```

Ожидание: **~3–8 мин** до матрицы на Metal (зависит от сети), не 20+.

## Переменные

| Env | Default | Эффект |
|-----|---------|--------|
| `MAX_FETCH_URLS` | 3 | Лимит URL с парсингом + summarizer |
| `MULTI_SEARCH_SKIP_VISION` | true | Без vision pipeline |
| `SKIP_GEMINI` | false | true → без Playwright Gemini |
| `OLLAMA_NUM_PREDICT` | 1024 | Лимит токенов генерации (legacy alias → heavy ctx) |
| `OLLAMA_ROUTER_NUM_CTX` | 2048 | KV для 1.5B / router |
| `OLLAMA_HEAVY_NUM_CTX` | 4096 | KV для 7B / summarizer |
| `OLLAMA_ROUTER_KEEP_ALIVE` | 2m | Выгрузка router после паузы |
| `OLLAMA_GUARDRAILS_NUM_PREDICT` | 1536 | Guardrails JSON |
| `OLLAMA_STRUCTURE_NUM_PREDICT` | 3072 | JSON AnalysisReport после Gemini (без обрезки 3-го option) |
| `MAX_AI_DIALOGUE_TURNS` | 3 | Реплики Gemini (если не SKIP) |

Лог с `NODE` / `OLLAMA` таймингами: `knowledge_engine/.runs/*.log` и блок «Время по этапам» в консоли после матрицы.

## Token & Rate Governor (Gemini pacing)

`knowledge_engine/services/token_rate_governor.py::TokenRateGovernor` — единая
sync-точка контроля RPM/TPM для Gemini-вызовов через
`gemini_stateless.py::_call_with_model_fallback` (и через
`gemini_search_grounding.py`).

Заменяет прежний безусловный `_rpm_pause_for_model()`, который усыплял
**каждый** запрос на `max(60/RPM, 4.0)` + jitter вне зависимости от
фактического использования (аудит зафиксировал 109.4s чистого простоя при
факте 6/15 RPM за минуту).

- `governor.acquire(model, estimated_tokens)` — 60s sliding-window по
  RPM и TPM. Если в окне есть место — задержка **0** (без искусственных пауз
  и джиттера). Если бюджет упёрся — ждёт ровно до освобождения ближайшего
  слота в deque, затем сразу продолжает.
- `governor.confirm(model, actual_tokens)` — после ответа API корректирует
  последнюю TPM-резервацию реальным `usage_metadata.total_token_count` (было
  недоступно нигде в кодовой базе до этого — TPM-бюджет считался по
  фиксированной оценке в 800 токенов на любой запрос).
- Оценка токенов ДО вызова — через единый `fast_tokenizer.token_counter`
  (см. `article_ingestion/paragraph_token_splitter.py::_count_tokens`), а не
  через прежние рассинхронизированные Qwen HF / tiktoken cl100k_base /
  chars÷4 оценщики.
- Реестр `get_governor(model)` — по одному `TokenRateGovernor` на модель;
  таргет-лимиты (`GEMINI_GOVERNOR_TARGET_RPM/TPM`,
  `GEMMA_GOVERNOR_TARGET_RPM/TPM`, см. [ENV_VARIABLES.md](ENV_VARIABLES.md))
  берутся с небольшим безопасным отступом от жёсткой квоты провайдера.

Gemma (`services/llm/gemma_client.py` + `services/llm/rate_limiter.py`) на
этот модуль **не переведена** — там уже был корректный async sliding-window
`AsyncRateLimiter` с тем же zero-wait-if-free поведением, dual-basket wave
allocation (MAP+fallback параллельно) и реальным
`reconcile_batch_total(usage.total_tokens)`. Вместо замены целевые
RPM/TPM (27/15200) применены к нему напрямую через
`GEMMA_BUDGET_MAX_RPM`/`GEMMA_BUDGET_MAX_TPM` (config.py), которые теперь по
умолчанию равны `GEMMA_GOVERNOR_TARGET_RPM/TPM` — одна точка истины для
обоих провайдеров, без риска регресса wave/priority-логики Gemma.

### Feedback loop и chat-managed вызовы

`governor.confirm()`/`_record_actual_usage()` (`gemini_stateless.py`) логируют
`[GOVERNOR FEEDBACK] model=... est=... actual=... diff=...` — единая точка
проверки, что TPM-бюджет корректируется реальным usage, а не застрявшей
оценкой. Изначально этот мост был подключён только к
`_generate_once()` (обычные stateless-вызовы) — **chat-managed** вызовы через
`chat_session_manager.py::send_chat_message`/`send_chat_message_stream`
(используются, например, в two-pass inbound ingest gate —
`paper_structure_analyzer.py::_analyze_pass1_in_session`) шли мимо этого
моста: `governor.confirm()` получал протухшее `actual_tokens` от совсем
другого, более раннего вызова в том же потоке (наблюдалось в проде:
`est=8090 actual=975 diff=-7115` вместо реальных ~45K токенов). Исправлено —
`reset_actual_usage()` сбрасывает thread-local перед каждым chat-managed
вызовом, а `_record_actual_usage()` вызывается сразу после получения
ответа/последнего streaming-чанка (`response.usage_metadata`/
`last_chunk.usage_metadata`), как и в `_generate_once()`.

Те же `send_chat_message`/`send_chat_message_stream` теперь эмитят
`GEMINI HTTP ▶/✓`-трейсы в формате, который парсит `log_profiler.py --llm-audit`
(`_HTTP_START_RE`/`_HTTP_END_RE`) — раньше chat-managed вызовы (в т.ч. Ingest
Gate) были полностью невидимы в реестре `--llm-audit`, хотя `GEMINI IO`-трейс
токенов там уже был.

### Gemma: queue-aware overflow

`services/gemma_rate_limiter.py::GemmaTokenBudgetManager.acquire_budget()` —
отдельный от `AsyncRateLimiter` глобальный (один на процесс) 60s
sliding-window бюджет (`GEMMA_BUDGET_MAX_RPM/TPM`, по умолчанию =
`GEMMA_GOVERNOR_TARGET_RPM/TPM`), который решает overflow на Gemini
Flash-Lite ДО фактического HTTP-запроса к Gemma. Раньше overflow срабатывал
только если расчётное ожидание превышало `GEMMA_BUDGET_OVERFLOW_WAIT_SEC`
(10s) — при НЕСКОЛЬКИХ параллельных запросах к одному и тому же общему
бюджету каждый ждал СВОЮ ≤10s паузу *последовательно* (наблюдалось в
проде: поздняя MAP/consensus-волна из 8 Gemma-вызовов почти без overflow,
близко к строго серийному выполнению). Теперь `acquire_budget()`
дополнительно проверяет `_waiting` (сколько вызовов уже спят в ожидании
того же бюджета) — если бюджет уже занят другим ожидающим вызовом,
overflow срабатывает немедленно, не складывая ожидания друг на друга.

`services/gemini_quota_store.py::_GeminiMinuteGuard` (RPD/24h-счётчик и
`filter_models_for_quota` ранний soft-cap переход по chain) не заменён —
это отдельная забота (какую модель вообще пробовать в chain), не пейсинг
внутри одной модели; Governor теперь единственный источник задержки.

## ML Memory Guard (BGE-M3 / Cross-Encoder, MPS RAM)

`services/ml_memory_guard.py` — eviction для локальных ML-моделей RAG-пайплайна
(bge-m3 bi-encoder, RAG cross-encoder): async-прогрев в порядке вызова,
request-scoped 5-минутный cooldown и OS-agnostic экстренная выгрузка по
превышению RAM. Найдено исходным аудитом "8.5GB RAM": `ps`'s RSS на macOS
систематически занижает реальный footprint процесса — Metal/MPS GPU-shared
буферы учитываются как `IOAccelerator (graphics)` в `footprint`/Activity
Monitor, а НЕ в `ps RSS`. На живом воркере после обычного прогона
`footprint -s <pid>` показал phys_footprint=6279MB (пик 10GB) при
`ps RSS`≈80MB. `torch.mps.empty_cache()` для реально загруженных весов
модели память НЕ отдаёт (bge-m3 сама по себе стабильно держит
`driver_allocated_memory()`≈3GB, не растёт от вызова к вызову) —
единственный способ снизить footprint ниже порога — выгрузить модель
целиком через её зарегистрированный `unload_*()`
(`register_model(name, unload_fn)`, сейчас `"bge_m3"` /
`bge_m3_embed.py::unload_bge_m3_model` и `"cross_encoder"` /
`cross_encoder.py::unload_cross_encoder`).

### OS-agnostic footprint (REFACTOR: ASYNC PIPELINE WARMUP)

`current_phys_footprint_mb()` больше не жёстко привязан к macOS-бинарнику:
на Darwin — сначала `footprint -s <pid>` (единственный источник, эмпирически
подтверждённый совпадающим с Activity Monitor; на реальном прогоне под
нагрузкой `torch.mps.driver_allocated_memory()` ни разу не превысил порог
3.5GB, пока `footprint` показывал пик 6.9-7.1GB — та первая версия guard'а
молчала всю дорогу). На остальных ОС, и как fallback при недоступности
бинарника `footprint` — `psutil.Process().memory_full_info().uss`, с
graceful fallback на `memory_info().rss`, если USS недоступен на данной
платформе/правах. `psutil` — новая обязательная зависимость
(`requirements.txt`).

### Два независимых механизма выгрузки

- **Request-scoped 5-минутный cooldown** (`RAG_MPS_REQUEST_COOLDOWN_SEC`,
  300s по умолчанию) — идёт от момента ПОЛНОГО завершения всего
  RAG-запроса (`rag_request_finished()`, когда счётчик активных запросов
  доходит до нуля через `rag_request_scope()` вокруг
  `gateway.py::query_directional_rag`), НЕ от последнего локального вызова
  embed/rerank. Каждый новый запрос (`rag_request_started()`) отменяет и
  сбрасывает таймер. По истечении — выгружаются ВСЕ зарегистрированные
  модели безусловно (не только сверх порога RAM): раз простой на уровне
  всего пайплайна подтверждён, держать веса незачем. Это заменяет прежний
  independent idle-таймер `RAG_MPS_IDLE_CHECK_SEC` (60s от последнего
  вызова модели), который решал ту же задачу ("должно опускаться при
  простое" — на живом прогоне footprint зависал на плато ~4.5GB без него),
  но недостаточно точно отражал границы реального RAG-запроса.
- **Emergency threshold override** (`RAG_MPS_MEMORY_THRESHOLD_GB`,
  `guard_after_use(name)`, вызывается из `bge_m3_embed.py` и
  `cross_encoder.py` после каждого encode/predict, троттлится раз в 5с):
  если footprint ПРЯМО СЕЙЧАС выше порога — даже посреди активного
  запроса, не дожидаясь cooldown — немедленно выгружает другую
  зарегистрированную и реально использованную модель (не ту, что вызвали
  только что). Модель, которая была только прогрета (`register_model`), но
  ещё ни разу не использована реально, этой проверкой не трогается — иначе
  экстренная выгрузка сводила бы на нет пользу async-прогрева.

### Async pipeline warmup

`warmup_pipeline_async()` / `spawn_warmup_task()` — прогревает модели
ПОСЛЕДОВАТЕЛЬНО (не `asyncio.gather`, каждая загрузка — блокирующий вызов
в `asyncio.to_thread`) в том порядке, в котором пайплайн реально их
вызывает (Embedding `bge_m3` → Rerank `cross_encoder`), чтобы к моменту
реального обращения модель была уже прогрета и чтобы JIT/кэш интерпретатора
прогревался в реальном порядке использования. Запускается
`gateway.py::query_directional_rag`'ом параллельно `rag.vector_search` в
самом начале обработки запроса (`spawn_warmup_task()` сразу после входа в
`rag_request_scope()`), а не только на стадии Exa. Последовательность — не
жадный parallel-load — намеренная мера против пикового всплеска RAM
(двойная одновременная инициализация двух ~3GB+ MPS-моделей на живом
прогоне давала пик 6.9-7.3GB).

Независимо от `RAG_CE_AUTO_UNLOAD` — работает даже если тот выключен для
постоянной резидентности cross-encoder'а при непрерывной нагрузке (см.
[ENV_VARIABLES.md](ENV_VARIABLES.md)).

### Известное ограничение

Оба механизма выше не могут предотвратить транзиентный ПИК RAM в момент,
когда обеим моделям легитимно нужно быть загруженными одновременно (начало
запроса — прогрев/использование embedding сразу за которым следует
cross-encoder): реактивная проверка не выгружает модель, которая явно
понадобится следующим шагом того же запроса. На живом прогоне пик
footprint во время такой одновременной загрузки достигал 6.9-7.3GB.
