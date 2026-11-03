# Наблюдаемость: OpenTelemetry → локальный Phoenix

Трейсы связывают один пользовательский сценарий через процессы:
API → очередь Redis → worker → граф LangGraph → вызовы LLM. Хранилище и UI —
[Arize Phoenix](https://github.com/Arize-ai/phoenix), локально в Docker. Текстовый
`trace()` (`src/core/run_log.py`) и `log_profiler.py` остаются как были —
спаны их дополняют.

## Запуск

```bash
docker compose --profile observability up -d phoenix   # UI: http://localhost:6006
```

В `.env`:

```bash
KE_OTEL_ENABLED=true
# KE_OTEL_ENDPOINT=http://localhost:6006/v1/traces    (по умолчанию)
# KE_OTEL_CAPTURE_LLM_IO=true                         (промпты/ответы в спанах)
```

Перезапустить API и worker. В run log появится `OTEL ✓ init | role=api|worker`.
Без `KE_OTEL_ENABLED` трейсер no-op: ни сетевых вызовов, ни накладных расходов
на экспорт.

Phoenix хранит данные в SQLite в volume `ke_phoenix_data` (не в основном
Postgres), лимит памяти контейнера — 1 ГБ.

## Что трейсится

| Источник | Как | Спаны |
|----------|-----|-------|
| Постановка задачи | `WorkJobStore.create` | `work_job.enqueue` |
| Выполнение задачи | `work_handlers.run_work_job` | `work_job.<kind>` — продолжение трейса постановщика |
| Граф тьютора и «Штурвала» | OpenInference LangChain instrumentor | `LangGraph`, узлы графа |
| Gemini (google-genai SDK) | OpenInference Google GenAI instrumentor | `GenerateContent` — модель, токены (в т.ч. reasoning) |
| Gemma (httpx, OpenAI-совместимый API) | `llm_span` в `gemma_client.py` | `gemma.chat_completions` — по спану на каждую HTTP-попытку |
| Этапы сбора и RAG | декоратор `@traced` | `search.*`, `select.*`, `dedup.pre_map`, `ingest.*`, `map_reduce.*`, `vlm.figure_registry`, `rag.*`, `steering.*`, `node_gate.*`, `grounding.lazy_deep_node` |
| Оценщик тьютора | `graph/nodes/sub_concept_eval.py` | `eval.sub_concept` + оценка `sub_concept_eval` |
| Гейты «Штурвала» | `domains/steering/routes.py` | `steering.generate`, `steering.gate1`, `steering.gate2` + оценки `steering_gate1/2` |

Контекст между процессами передаётся полем `WorkJob.trace_context` (W3C
`traceparent`), а не через `payload`: payload валидируется доменными
моделями. Старые записи без поля читаются как пустой контекст — задача
начинает новый трейс.

## Оценки (span annotations)

Вердикты пишутся функцией `record_evaluation` на спан: атрибуты
`ke.eval.<name>.label/score` и событие `ke.evaluation` — всегда; при
`KE_OTEL_ANNOTATIONS_ENABLED` ещё и span annotation в Phoenix (колонка
оценок в UI, фильтр по label/score).

| Оценка | Annotator | label | score | metadata |
|--------|-----------|-------|-------|----------|
| `sub_concept_eval` | LLM | `verified` / `partial` / `gap` / `unchecked` | 1 / 0.5 / 0 / — | `concept_id`, `directive`, `accuracy_grade`, `failed_attempts`, `why/how/mechanic_passed`, `layer`, `interaction_axis` |
| `sub_concept_eval` (пропуск) | CODE | `skipped` | — | причина в `explanation` (`no pending …` — ход без зачёта) |
| `steering_gate1` | HUMAN | `all_approved` / `partial` / `none_approved` / `empty` | доля одобренных кандидатов | `target_goal`, `approved_urls`, `rejected[]` (url, title, source) |
| `steering_gate2` | HUMAN | то же | доля одобренных дайджестов | то же |

Низкий score на Gate 1 — шумный Discovery; `rejected[]` — это шум, который
не отсекли правила, материал для настройки triage и запросов.

Phoenix привязывает аннотацию только к уже записанному спану: в
sync-режиме иначе отвечает 404, в async — 200 без проверки, а применяет с
неизвестной задержкой. Мы используем sync с повторами — доставка
подтверждена. Поэтому отправка идёт фоновым
потоком: первая попытка через 5 с (интервал BatchSpanProcessor), затем
повторы на 404/5xx/сетевых ошибках; `shutdown_observability` сначала
сбрасывает спаны, потом очередь оценок. Не доставленная оценка остаётся в
атрибутах спана, в run log — `OTEL ⚠ annotation dropped`.

```bash
curl -s "http://localhost:6006/v1/projects/knowledge-engine/span_annotations?span_ids=<span_id>"
```

## Ошибки: GlitchTip (Sentry-совместимый)

Трейсы отвечают «где и сколько», GlitchTip — «какие ошибки, как часто,
с каким стеком»: группировка одинаковых исключений в issues, first/last
seen, счётчики. Локально, профиль `observability`:

```bash
docker compose --profile observability up -d glitchtip   # UI: http://localhost:8800
```

Один контейнер `glitchtip/glitchtip:6.2.6` (`SERVER_ROLE=all_in_one`, без
Valkey) + свой `postgres:18-alpine`; основной Postgres не задействован.
Порт опубликован только на `127.0.0.1`. Память в простое: около 165 МБ +
50 МБ Postgres. Первый вход: зарегистрировать пользователя в UI, создать
организацию и Python-проект, DSN — Project → Settings → Client Keys →
`KE_SENTRY_DSN` в `.env`, перезапустить API и worker
(`SENTRY ✓ init | role=…` в run log).

Источники событий (`src/core/error_tracking.py`):

| Источник | Уровень | Теги |
|----------|---------|------|
| `trace()` внутри `except` — fail-open: исключение проглочено, конвейер идёт по fallback | warning | `ke.failopen=true`, `ke.trace_label`; текст trace-строки в extra |
| Падение work-задачи (`redis_worker_dispatch._handle_work_job`) | error | `ke.job.kind`, `ke.job.id` |
| API 500 и `HTTPException` 5xx (`app.py`) — для 5xx отправляется исходная причина (`raise … from exc`) | error | `ke.api.status`, `ke.api.path` |
| `logger.exception` / `logger.error` (LoggingIntegration SDK) | error | — |

Хук в `trace()` покрывает все места вида `except Exception: trace(...)`
без правки каждого (через `sys.exc_info()` — вне `except` он ничего не
делает). Не события: `CancelledError`, `KeyboardInterrupt`, `SystemExit`,
любые `HTTPException` в `trace()` (4xx — ответ клиенту, 5xx уже отправлены
явно). Fail-open ограничен 5 событиями в минуту на ключ «тип исключения +
метка trace-строки» — сбой в цикле по 500 документам не даёт 500 событий.
Одно исключение из нескольких источников подряд отсекает DedupeIntegration
SDK; захват 5xx по статус-коду в Starlette/FastAPI-интеграции выключен,
иначе обёртка `HTTPException` шла бы вторым событием.

Связь с трейсами: тег `otel.trace_id` (текущий спан или
`WorkJob.trace_context` упавшей задачи) →
`log_profiler.py --phoenix --trace <id>`; в обратную сторону — атрибут
спана `ke.sentry.event_id`. Performance-режим Sentry выключен
(`traces_sample_rate=None`): трейсинг только через OpenTelemetry. Без PII и
без локальных переменных в кадрах (там тексты статей, промпты и ключи).

## Известные пробелы

- Гейты «Штурвала»: generate, approve-gate1/2 и Map-Reduce в worker теперь
  один трейс (контекст из `WorkJob.trace_context`), но он длится столько,
  сколько человек думает над гейтом — в Phoenix это один долгий трейс
  с паузами между ветками.
- Fail-open без `trace()` (`except …: continue` / `pass`, например в
  `pre_flight_triage.py`) и `trace()` вне `except` (ошибка передана строкой)
  в GlitchTip не попадают — для них нужен явный `capture_error`.
- Хук `trace()` видит и исключение внешнего `except`: строка-«успех»
  внутри fallback-кода, вызванного из обработчика, тоже станет warning
  (дубли того же исключения отсекает Dedupe).
- Оценки сессии тьютора не сгруппированы в Phoenix Sessions (`session.id`
  на спанах не ставится); фильтруйте по `ke.tutor.node_id`.
- Daemon-потоки (фоновый ингест схем, отложенный академический ингест) и
  ручные `ThreadPoolExecutor.submit` контекст не наследуют — их спаны станут
  отдельными трейсами. `asyncio.to_thread` и `asyncio.run` контекст переносят.
- LangChain instrumentor пишет служебные спаны `ChannelWrite<…>` — шум в
  дереве; фильтра пока нет.
- Автоинструментация Gemini проверена на `run_stateless_gemini`;
  chat-managed путь (`chat_session_manager.py`) идёт через тот же SDK, но
  отдельно не проверялся.

## Доступ для анализа (человек и модель)

Основной путь — `log_profiler.py --phoenix`: только GET-запросы, компактный
текстовый отчёт без сырых промптов (агенту не нужно разбирать JSON спанов):

```bash
P=knowledge_engine/scripts/legacy_research/log_profiler.py
python $P --phoenix                                  # 5 свежих трейсов за час: сводка + стадии
python $P --phoenix --since "14:00:00" --llm-audit   # токены/латентность/пик параллельности по label
python $P --phoenix --errors --evals --last-traces 0 # все ошибки и оценки окна
python $P --phoenix --trace <trace_id> --evals       # один трейс целиком
```

| Флаг | Отчёт |
|------|-------|
| (всегда) | трейсы: корень, длительность, спаны, LLM-вызовы, токены, ошибки; стадии по имени спана (служебные `ChannelWrite<…>` скрыты); топ-N долгих |
| `--llm-audit` | LLM по `ke.llm.label`/модели: prompt/completion/reasoning токены, avg/max латентность, пик одновременных вызовов |
| `--errors` | спаны ERROR с типом и текстом исключения, сводка и хронология |
| `--evals` | оценки по имени: распределение label, средний score, худшие вердикты с пояснением, отклонённые на гейтах статьи по источникам |

`--phoenix-url` (env `KE_PHOENIX_URL`), `--project` (env `KE_OTEL_SERVICE_NAME`).

Альтернатива для MCP-клиентов — официальный `@arizeai/phoenix-mcp`; в
проекте не подключён: кроме чтения он умеет писать (промпты, датасеты,
эксперименты), а сырые спаны дороже по токенам, чем агрегированный отчёт.

Сырой REST API Phoenix (проверено на `version-20.25.0`):

```bash
curl -s http://localhost:6006/v1/projects
curl -s "http://localhost:6006/v1/projects/knowledge-engine/spans?limit=50"
```

Спан содержит `context.trace_id`, `parent_id`, `span_kind` (`LLM` / `CHAIN`),
`status_code` и атрибуты (`llm.token_count.*`, `llm.model_name`,
`input.value`/`output.value`, `ke.job.*`, `ke.llm.label`).

## Конфигурация

| Variable | Default | Роль |
|----------|---------|------|
| `KE_OTEL_ENABLED` | `false` | Включить трейсинг |
| `KE_OTEL_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP/HTTP приёмник |
| `KE_OTEL_SERVICE_NAME` | `knowledge-engine` | Имя проекта в Phoenix; сервисы — `<name>-api`, `<name>-worker` |
| `KE_OTEL_CAPTURE_LLM_IO` | `true` | Писать промпты и ответы LLM в спаны; токены пишутся всегда |
| `KE_SENTRY_DSN` | пусто | DSN проекта GlitchTip; пусто — sentry-sdk не инициализируется |
| `KE_SENTRY_ENVIRONMENT` / `KE_SENTRY_RELEASE` | `local` / пусто | environment и release событий |
| `KE_SENTRY_CAPTURE_FAILOPEN` | `true` | `trace()` внутри `except` → warning-событие |
| `KE_OTEL_ANNOTATIONS_ENABLED` | `true` | Отправлять оценки как span annotations в Phoenix (только если endpoint — Phoenix `…/v1/traces`) |

## Код

| Компонент | Путь |
|-----------|------|
| Инициализация, `span`/`traced`/`llm_span`, `record_evaluation`, передача контекста | `src/core/observability.py` |
| Контекст в задаче | `src/shared/job_queue/work_job_store.py` (`WorkJob.trace_context`) |
| Спан задачи в worker | `src/shared/job_queue/work_handlers.py` (`run_work_job`) |
| Ошибки → GlitchTip, хук fail-open | `src/core/error_tracking.py`, `src/core/run_log.py` (`set_trace_exception_hook`) |
| Тесты | `tests/src/core/test_observability.py`, `tests/src/core/test_error_tracking.py` |
