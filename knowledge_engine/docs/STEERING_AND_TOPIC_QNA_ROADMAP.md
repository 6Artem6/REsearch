# Штурвал (Steering) и Topic Q&A — мастер-план внедрения

Add-only реализация двух независимых осей выбора режима работы поверх
действующего Autopilot/Lecture-контура. Актуальную архитектуру самого
Autopilot/Lecture & Self-Check см. в [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md),
[RAG_PIPELNES.md](RAG_PIPELNES.md), [PROMPT_SYSTEM_REFERENCE.md](PROMPT_SYSTEM_REFERENCE.md)
— этот документ их не заменяет и не переопределяет.

**Источник:** зафиксировано по постановке задачи в чате.
Код должен ссылаться сюда.

**Правило реализации (действует для всех этапов):** не переписываем и не
модифицируем действующий рабочий код — Autopilot-пайплайны, текущие DDL,
существующие эндпоинты. Всё новое внедряется строго параллельно (add-only) —
новыми модулями, Pydantic-схемами, независимыми промптами, изолированными
эндпоинтами. Autopilot и текущие ветки должны работать без единого изменения.

## Две независимые оси

1. **Control Axis:** `Autopilot` (существующий, без изменений) vs
   `Steering`/Штурвал (новый) — пошаговая воронка сбора источников с двумя
   точками одобрения пользователем.
2. **Interaction Axis:** `Lecture & Self-Check` (существующий, без изменений)
   vs `Socratic`/Topic Q&A (новый) — формат подачи готового материала.

Штурвал, шаг за шагом:

1. Flash Lite генерирует теги, хабы компаний, ключевики под цель (`TaxonomyService`).
2. Экспресс-поиск по интро/заголовкам (`Light Discovery`) → **Gate 1** (approve пользователем).
3. Поверхностные дайджесты одобренных статей без тяжёлого Map-Reduce
   (`Batch Digest Generator`) → **Gate 2** (approve пользователем).
4. Глубокий Map-Reduce строго по утверждённым на Gate 2 статьям (переиспользует
   существующий DEEP-пайплайн, без изменений в нём — только сузить входной
   список URL).

Topic Q&A, коротко: вместо `Lecture & Self-Check` — сжатая выжимка (готовый
REDUCE Phase 2) + `socratic_focus`-фаза тьютора, роутинг через
`coverage_router.py`/`prompt_factory.py` — без изменений в самом графе.

## Этапы

| № | Этап | Что делает | Статус |
|---|------|------------|--------|
| 1 | Контракты и схемы данных (Add-Only) | Pydantic-контракты для обеих осей, Gate 1/Gate 2, расширенные (дочерние) запросы генерации/сессий с backward-compatible дефолтами | ✅ реализован |
| 2 | Сервисы фильтрации и поверхностного анализа | `TaxonomyService`, `Light Discovery Service`, `Batch Digest Generator` — чистые async-функции/stateless-классы, без завязки на графы | ✅ реализован |
| 3 | Изолированный микро-граф «Штурвала» на LangGraph | Независимый граф с `AsyncPostgresSaver`-чекпоинтером, намеренные паузы `interrupt()` на Gate 1/2, новые внешние статусы `WorkJobStore` (`AWAITING_GATE_1`/`AWAITING_GATE_2`) без излома текущей статус-машины | ✅ реализован |
| 4 | Topic Q&A в Prompt Factory | `topic_qna_prompt.py`, роутинг в `coverage_router.py`: при выборе `topic_qna` — сжатая выжимка (готовый REDUCE Phase 2) → сразу `socratic_focus` | ✅ scaffolding (см. журнал — сшивка в live coverage_router_node/_invoke_tutor сознательно отложена) |
| 5 | REST API & Gateway | Изолированные эндпоинты `/api/v1/curriculum/steering/*` — создание задач, approve Gate 1/2 | ✅ реализован |

## Журнал реализации

| Этап | Файлы | Комментарий |
|---|---|---|
| 1 | `src/curriculum/steering_contracts.py` | `ControlAxis`/`InteractionAxis`, `CandidateArticleMeta`, `TaxonomyDiscoveryResponse`, `Gate1ApprovePayload`, `SurfaceDigestItem`, `SurfaceDigestResponse`, `Gate2ApprovePayload`, `SteeringCurriculumGenerateInput`, `TopicQnaNodeDeepDiveRequest` |
| 2 | `src/curriculum/services/taxonomy_service.py` | Flash Lite → `TaxonomySeed` (tags/company_hubs/keywords); `tags`/`keywords` — русские (видны пользователю на Gate 1, `RUSSIAN_OUTPUT_RULE`), `company_hubs` — технические идентификаторы (bare slug/hostname), не переводятся (см. прецедент `services/search/exa_source_expand.py::_EXPAND_SYSTEM` для самой идеи разделения полей) |
| 2 | `src/curriculum/services/light_discovery_service.py` | Exa (per-hub search) + `fetch_html`/`_extract_paragraphs` (Trafilatura, переиспользованы как есть) → `TaxonomyDiscoveryResponse` для Gate 1; RSS из заголовка задачи не реализован — в детальной спецификации функционала описан только Exa+Trafilatura путь |
| 2 | `src/curriculum/services/batch_digest_service.py` | Один батч-вызов Flash Lite на все approved-статьи → `SurfaceDigestResponse` для Gate 2, `RUSSIAN_OUTPUT_RULE` (пользователь читает дайджест), строгий запрет вступительных историй в системном промпте |
| 3 | `services/work_job_store.py` | Add-only расширение (не новый файл, разрешено формулировкой задачи): `WorkJobStatus.AWAITING_GATE_1`/`AWAITING_GATE_2` (новые значения enum) + `WorkJobStore.set_status()` (новый метод — минимально необходимая обвязка, чтобы граф вообще мог выставлять эти статусы; `complete()`/`fail()`/остальные методы не тронуты, диф чисто аддитивный) |
| 3 | `src/curriculum/steering_graph/state.py` | `SteeringState` (TypedDict): `target_goal`, `work_job_id`, `taxonomy`, `approved_gate1_urls`, `surface_digests`, `final_approved_urls` |
| 3 | `src/curriculum/steering_graph/builder.py` | `SteeringGraphService` (тот же AsyncPostgresSaver-lifecycle паттерн, что `TutorGraphService`, но БЕЗ MemorySaver dev-фолбэка) + `build_steering_graph()`/`compile_steering_graph()` (явный `ValueError` при не-Postgres чекпоинтере для `control_axis == "steering"`) + 5 нод (`node_taxonomy_and_discovery` → `node_gate1_interrupt` → `node_generate_batch_digests` → `node_gate2_interrupt` → `node_finalize_steering`) + `steering_graph_session()`. Реальный HITL через `interrupt()`/`Command(resume=...)` — тот же примитив, что уже используется в `nodes/intent_and_clarify.py`/`services/analysis_service.py` (не изобретался заново). `node_finalize_steering` только пишет `final_approved_urls` в `WorkJob.result` (`complete()`, существующий метод) — реальный запуск Map-Reduce по этим URL передан на Этап 5 (Gateway), чтобы не импортировать `generator.py` в изолированный граф. Проверено end-to-end (`MemorySaver`, замоканные сервисы): оба interrupt срабатывают, `WorkJobStatus` последовательно AWAITING_GATE_1 → AWAITING_GATE_2 → COMPLETED |
| 4 | `src/curriculum/prompts/topic_qna_prompt.py` (+ `prompts/__init__.py`, новый пакет) | `TOPIC_QNA_SYSTEM_PROMPT` (сырой English, без RUSSIAN_OUTPUT_RULE — та же конвенция, что `SOCRATIC_MODE_PROMPT`/etc.) + `build_topic_qna_context_block(FinalArticleSummaryResponse)` — форматирует executive_summary/key_takeaways вместо полного текста статьи |
| 4 | `src/node_deep_dive/prompt_factory.py` | Add-only: новая функция `select_interaction_axis_system_prompt()` (тот же контракт, что существующая `select_isolated_prompt_for_mode`) — для `topic_qna` возвращает `TOPIC_QNA_SYSTEM_PROMPT`, для `lecture_self_check` — `None`. Нигде не вызывается из `_invoke_tutor` — диф чисто аддитивный (+24/-0), 12/12 существующих тестов `test_prompt_factory.py` проходят без изменений |
| 4 | `src/node_deep_dive/graph/nodes/coverage_router.py` | Add-only: новая функция `resolve_topic_qna_entry()` — чистое решение (True/False), НЕ вызывается из `coverage_router_node`. Ключевая находка: `socratic_focus` уже существует как значение `LearningPhase` и уже полностью маршрутизируется (`learning_loop.set_learning_mode(memory, "socratic_point")` → `tutor_behavior_state.resolve_tutor_mode` → `SOCRATIC_MODE_PROMPT`) — Topic Q&A нужен только свой промпт поверх уже готовой фазы, а не новая ветка графа. Диф чисто аддитивный (+23/-0), `coverage_router_node` не тронут |
| 4 | **Сознательно отложено** | Реальная сшивка (`coverage_router_node` читает `interaction_axis` из `request`/`memory` и вызывает `resolve_topic_qna_entry`/`select_interaction_axis_system_prompt`) НЕ сделана в этом этапе — это горячий путь всех текущих Autopilot/Lecture-сессий; пользователь подтвердил вариант "только scaffolding" при уточнении по Этапу 4. Live-сшивка — отдельный, точечно проверяемый шаг на будущее |
| 5 | `services/work_job_store.py` | Add-only: `WorkJobKind.STEERING_FUNNEL`/`STEERING_MAP_REDUCE` (новые значения enum) + `create(..., publish: bool = True)` (новый keyword-параметр с дефолтом = 100% прежнее поведение для всех текущих вызовов) — `publish=False` нужен, чтобы STEERING_FUNNEL-джобу (её ведёт напрямую `api/routes/steering.py`, не обычный worker) не подхватил Redis pub/sub worker |
| 5 | `services/work_handlers.py` | Add-only: один новый `if` в `run_work_job()` (диспетчеризация по kind) + новая функция `_run_steering_map_reduce()` — `_run_curriculum_generate()` и остальные обработчики не тронуты |
| 5 | `src/curriculum/steering_generator_bridge.py` | `generate_curriculum_from_steering_digests()` — Gate 2 `final_approved_urls` + их `SurfaceDigestItem` → `CurriculumGraph`, СТРОГО минуя Discovery Phase: `CurriculumSearchHit` строится прямо из готовых дайджестов (не из `collect_curriculum_source_hits`/`summarize_whitelist_blog_hits`/`enrich_search_hits_with_extracts`), дальше — те же существующие `assign_source_ids`/`search_hits_as_prompt_json`/`generate_curriculum_search_first`, что и Search-First путь `generator.py` (сам `generator.py` не изменён) |
| 5 | `api/routes/steering.py` (+ `api/app.py`, 2 строки) | `POST /generate` (ждёт Gate 1 внутри запроса, TaxonomyService+Discovery — секунды), `POST /approve-gate1` (→ Batch Digest → Gate 2), `POST /approve-gate2` (→ `node_finalize_steering`, COMPLETED, затем ставит `STEERING_MAP_REDUCE` в очередь обычному worker'у и сразу возвращает `generation_job_id` для `GET /work-jobs/{id}/wait` — тяжёлый Map-Reduce НЕ ждём инлайн). 404/409 на невалидные переходы (`_require_job`/`_require_status`) |
| 5 | **Проверено** | End-to-end через сам роутер (`post_steering_generate` → `approve_gate1` → `approve_gate2`, замоканные Taxonomy/Discovery/BatchDigest, реальный `build_steering_graph()`): полный цикл, оба гейта, оба guard'а (404 unknown job, 409 invalid state), `STEERING_MAP_REDUCE`-джоба создаётся с payload'ом из готовых дайджестов и остаётся `PENDING` (не гоняется инлайн). `steering_generator_bridge` — отдельно юнит-тестами (порядок hits по `final_approved_urls`, `skip_ollama_summary=True`, пропуск URL без дайджеста, `GeminiUnavailableError`/`ValueError` guard'ы). Живой сквозной прогон с реальным `AsyncPostgresSaver` временно не завершался — Postgres/Redis стали недоступны ИЗ ЭТОЙ ПЕСОЧНИЦЫ посреди этапа (диагностика, пришедшая позже, показала, что `ke-redis` весь этот час был healthy в Docker — сбой был в сетевом доступе песочницы к localhost, а не в самих сервисах); роутер-тест на тот момент использовал `MemorySaver` с одним общим инстансом (эмулирует персистентность Postgres между вызовами). ДОБАВЛЕНО ПОСЛЕ: как только доступ восстановился, тот же сценарий (generate → 409 guard → approve-gate1 → approve-gate2) прогнан ещё раз через настоящий `AsyncPostgresSaver` + настоящий Redis-backed `WorkJobStore` — прошёл полностью, тестовые джобы подчищены (`work_job_store.fail(...)`) |

## Node Grounding Gate (per-node, независимо от Штурвала)

Отдельный, более лёгкий HITL-гейт для ОДНОЙ DEEP-ноды при её открытии/
перегрунде (перехватывает существующий per-node lazy-grounding в
`engine.py`/`targeted_node_search.py`, у которого есть SOTA-override,
дающий неконтролируемый Consensus/arXiv/S2 харвест даже в
`practical_only`). Решение пользователя: новый изолированный сервисный
слой, `targeted_node_search.py`/`engine.py` НЕ модифицируются вообще
(только читаются, чтобы понять точку интеграции для будущего Этапа 3).

| Этап | Файлы | Комментарий |
|---|---|---|
| 0 | `src/curriculum/node_gate_contracts.py` | `NodeSearchProfile`, `NodeCandidatePassport`, `NodeGate1DiscoveryResponse`, `NodeGate1ApprovePayload` (Pydantic `max_length` кап на `NODE_GATE1_MAX_APPROVED=4`, без кастомного валидатора) |
| 0 | `src/curriculum/services/node_search_profile_service.py` | Единый Flash Lite вызов → Habr hubs/tags/keywords + Exa keywords. Academic-часть НЕ генерируется здесь — после док-аудита (`docs/SOURCE_POOL.md`/`ACADEMIC_AND_CONSENSUS.md`) выяснилось, что это дублировало уже существующий Academic Query Architect; `build_node_academic_plan()` — тонкая обёртка над `lite_search_pipeline.build_academic_search_plan` (реюз как есть, даёт и query, и структурированные `arxiv_params`) |
| 1 | `src/curriculum/services/node_candidate_collection_service.py` | Сбор (Habr/Exa/academic, параллельно `asyncio.gather(..., return_exceptions=True)` — fail-open по каналам) → дедуп по URL + по эмбеддингам (BGE-M3, Union-Find) → сбалансированный пул 5-10 → батч Flash Lite паспортов. Academic-канал переиспользует `academic_source_fetch._primary_academic_hits` (SS→SearXNG-science→arXiv) и `academic_consensus.harvest_consensus_for_node(..., defer_ingest=True)` — существующий флаг, останавливающийся на метаданных БЕЗ полного харвеста/Gemma-ingest (ключевая находка док-аудита: закрывает и дублирование, и риск повторного 43-минутного харвеста одним и тем же механизмом) |
| 1 | Habr-канал (`_collect_habr` в том же файле) | ИСПРАВЛЕНО: изначально был Exa-поиск с доменным ограничением на Habr — по итогам не соответствовал заданию ("для Хабра — API/прямые ссылки для Trafilatura и RSS для XML"). Переписано на genuine RSS-обнаружение: `_fetch_habr_company_rss_items()` тянет `https://habr.com/ru/rss/companies/{slug}/articles/?fl=ru` (URL-паттерн подтверждён живым `curl` по `<link type="application/rss+xml">` реальных страниц Habr, не угадан), парсит `xml.etree.ElementTree`, предпочитает `<guid>` (чистый URL) над `<link>` (с `utm_*`, обрезается по `?`). `_rss_item_matches_keywords()` — фильтр по `habr_tags`/`habr_keywords` на title+description ДО дорогого полнотекстового fetch (RSS даёт только хронологию, не релевантность). Дальше — тот же существующий `fetch_html`+`_extract_paragraphs` (Trafilatura), что и раньше — "остальные шаги такие же" |
| Gate 1 | `api/routes/node_skill.py` (`POST /node/grounding-discover`) | Add-only (+114, затем +157 с Gate 2 — 0 удалений). Instant empty response для BASE-нод и уже прогруженных DEEP (без сетевых/LLM вызовов). SOTA-override пересчитывается так же, как в `targeted_node_search.py` (`is_sota_rd_node`+`consensus_allowed_for_policy` из `academic_consensus.py`, НЕ `source_policy.py`) — чтобы решение "нужен ли academic" совпадало с боевым путём |
| 2 | `src/curriculum/services/node_digest_service.py` | По ≤4 Gate-1-одобренным URL: полный текст (тот же `fetch_html`+`_extract_paragraphs`) → `_top_k_chunks()` — BGE-M3 (`embed_texts_bge_m3`, косинус, векторы уже L2-нормированы) грубо отбирает `NODE_GATE2_COARSE_K=16` абзацев по теме ноды, затем локальный Cross-Encoder (`src/rag_gateway/cross_encoder.py::score_relevance_pairs` — тот же примитив, что Directional RAG Gateway, `docs/RAG_GATEWAY_MODULE_3.md`, детерминированный, без LLM) уточняет до `NODE_GATE2_TOP_K_CHUNKS=6` → ОДИН батч-вызов `GemmaCloudClient.complete_structured` (не Flash Lite — Gate 2 по заданию требует именно Gemma cloud; `services/llm/gemma_client.py`, тот же клиент, что MAP/REDUCE в `blog_spatial_summarizer.py`, но здесь без Map-Reduce и без записи в LanceDB/registry — это Этап 3) → дайджест на статью (Архитектура/Практический кейс/Ограничения). Fail-open на каждом уровне (embed/rerank/Gemma) — см. docstring файла. Защита от URL-галлюцинаций Gemma: ответ фильтруется по фактически переданным в промпт `url` |
| Gate 2 | `api/routes/node_skill.py` (`POST /node/grounding-digest`) | Add-only. Тело запроса переиспользует `NodeGate1ApprovePayload` (та же форма: `curriculum_id`/`node_id`/`approved_urls`, уже капнута на 4) — не заведён отдельный payload-класс, раз форма идентична. 404 на неизвестный curriculum/node |
| — | **Проверено** | `knowledge_engine/tests/test_node_grounding_gate.py` — 23 теста (контракты, Этап 0 профиль+академия, Habr RSS парсинг/фильтр-до-fetch, Этап 1 дедуп/выбор/оркестрация fail-open, Этап 2 coarse-filter→rerank порядок и fail-open на обеих ML-моделях, батч-дайджест + защита от галлюцинаций + fail-open на пустой Gemma-ответ, оба эндпоинта). `_collect_habr`/`_fetch_habr_company_rss_items` также провалидированы вручную против ЖИВОГО Habr RSS (не только моками) |
| 3 | `src/curriculum/services/node_grounding_finalize_service.py` | ``finalize_node_grounding(graph, node, approved_urls, target_goal="")`` — Gate 2 финальное согласие → ``summarize_whitelist_blog_hits_async`` (ТОТ ЖЕ унифицированный Map-Reduce ingest, что уже используется ``search_sources_for_deep_node_async`` — ничего нового не изобреталось, как и указал пользователь) → тот же persist-хвост, что ``targeted_node_grounding.lazy_ground_deep_node_on_demand`` (``hit_to_registry_entry``/``_attach_hits_to_node``/``cap_curriculum_sources_registry``/``sync_route_sources_from_registry``/``refresh_node_session_diagrams_from_articles``, импортированы как есть). Единственный новый кусок — ``_infer_source_tier(url)``: у Gate 1/Gate 2 человекочитаемый тир ("habr"), а ``summarize_whitelist_blog_hits_async`` нужен канонический (``whitelist_blog``/``arxiv``/``semantic_scholar``/``consensus``/``exa``) — иначе хит молча пропускался бы без ingest. ``save_curriculum_record`` НЕ вызывается внутри сервиса — как и в ``lazy_ground_deep_node_on_demand``, это ответственность вызывающего эндпоинта |
| 3 | `api/routes/node_skill.py` (`POST /node/grounding-finalize`) | Add-only. **Ключевое: ``targeted_node_search.py``/``engine.py`` по-прежнему НЕ изменены.** Корректность цепочки обеспечена по конструкции: ``_apply_lazy_grounding_for_init`` (engine.py) запускает поиск ТОЛЬКО когда ``grounding_status == "pending_grounding"``; раз ``finalize_node_grounding`` выставляет ``grounded``+``source_ref`` ДО того, как фронтенд вообще позовёт ``POST /node/init``, — штатная проверка `if status == "grounded" and node.source_ref: skip search` срабатывает сама, без единой правки в hot-path |
| Frontend | `web/static/skill-tree/api.js` | Add-only: ``nodeGroundingDiscover``/``nodeGroundingDigest``/``nodeGroundingFinalize`` — тот же fetch-паттерн, что ``steeringApproveGate1``/``steeringApproveGate2`` |
| Frontend | `web/static/skill-tree/SteeringGatePanel.js` | Расширен (не переписан): новый опциональный проп ``kind: "node"`` — Gate 1/2 подписи и кнопка меняют текст под контекст одной ноды (без ``kind`` — прежнее поведение Штурвала, 0 изменений для существующих вызовов). ``Gate2Digest`` теперь рендерит ЛИБО ``NodeDigestItem`` (architecture/practical_case/limitations), ЛИБО прежний ``SurfaceDigestItem`` Штурвала — определяется по наличию поля ``architecture`` |
| Frontend | `web/static/skill-tree/NodeDrawer.js` | Add-only: второй, независимый блок рендера ``SteeringGatePanel`` по ``session?.nodeGateStatus`` (Штурвал остаётся на ``session?.steeringStatus``, оба блока не пересекаются для одной сессии) + проброс нового пропа ``onNodeGateApprove`` |
| Frontend | `web/static/skill-tree/RoadmapDashboard.js` | Живая сшивка (в отличие от Topic Q&A, где сшивка сознательно отложена — здесь пользователь явно попросил "соединить пайплайн"): ``openNode`` перед запуском ``nodeInitStream`` проверяет ``node.node_risk_kind === "DEEP" && node.grounding_status !== "grounded"`` — если да, сперва ``nodeGroundingDiscover``; пустой список кандидатов (BASE-нода/уже готово/фейл-опен бэкенда) — молча идёт прежним путём (0 лишних кликов, 0 изменений поведения для большинства нод); непустой — рисует Gate 1 и НЕ вызывает ``nodeInitStream``, пока пользователь не пройдёт Gate 1→Gate 2 (``handleNodeGateApprove``). После ``nodeGroundingFinalize`` — сброс gate-состояния сессии и рекурсивный вызов ``openNode`` для того же (уже ``grounded``) узла — штатный ``nodeInitStream`` идёт по быстрому пути. Повторный клик по ноде с незавершённым гейтом (``awaiting_gate_1``/``awaiting_gate_2``) НЕ перезапускает discovery — не роняет прогресс с Gate 2 обратно на Gate 1 |
| — | **Проверено** | Backend: 30/30 тестов (`test_node_grounding_gate.py`), включая fail-open по всем сервисам Этапа 3 и защиту от повторного persist уже обработанных тиров. Frontend: `npm run build` (esbuild) проходит чисто. **НЕ проверено** живым браузером — в этой фоновой сессии нет поднятого dev-сервера/Postgres/Redis; логика Gate 1→2→finalize→re-open прослежена вручную построчно, но не прокликана глазами — нужен ручной прогон перед тем, как полагаться на неё в проде |

## Перевод введений на Gate 1 (общий для обоих гейтов)

Пользователь прогнал реальный Gate 1 Штурвала и увидел непереведённые
английские тексты — перевод изначально был подключен только для Node
Grounding Gate (задание в prompt.txt использовало терминологию "Паспорта",
специфичную для Node Gate). По подтверждению пользователя расширено на оба
гейта одним общим модулем.

| Файл | Комментарий |
|---|---|
| `src/utils/translation_service.py` | `translate_batch_to_russian(texts) -> list[str]` — бесплатный неофициальный Google REST-эндпоинт (`translate.googleapis.com/translate_a/t`, `client=gtx`), ОДИН HTTP-запрос на весь батч через несколько `q=`, порядок сохраняется. Нулевая нагрузка VRAM/RAM — без локальных LLM/SLM, без новых тяжёлых зависимостей. Fail-open на каждом шаге (сеть/парсинг/несовпадение длины) — оригинальные тексты без исключения наружу; проверено вживую — этот sandbox блокируется Google по абьюз-детекту (HTML "automated queries" вместо JSON), фоллбэк не гипотетический. `needs_ru_translation(text) -> bool` — общая эвристика (латиница vs кириллица, без langdetect) для решения "нужен перевод или уже по-русски" |
| `src/curriculum/services/node_candidate_collection_service.py` | Использует `translate_batch_to_russian`/`needs_ru_translation` в `_translate_latin_snippets`, вызывается между `_select_balanced_pool` и `_generate_passports` (Этап 1, Node Grounding Gate) |
| `src/curriculum/services/light_discovery_service.py` | ДОБАВЛЕНО: `_translate_latin_leads`, вызывается в конце `discover_candidates` (после policy-фильтра, до возврата `TaxonomyDiscoveryResponse`) — переводит `lead_paragraph` кандидатов Штурвала. В отличие от Node Gate, на Gate 1 Штурвала нет прохода через Flash Lite (тот идёт позже, при Batch Digest) — перевод подставляется в лид напрямую |
| — | **Проверено** | 46/46 тестов (`test_node_grounding_gate.py` — 41, включая перенесённый тест эвристики; новый `test_light_discovery_translation.py` — 5: перевод только нерусских лидов, no-op на всём русском, fail-open на сетевой ошибке и на несовпадении длины, полный `discover_candidates` end-to-end с переводом). `black`/`isort`/`flake8` чисто |

## "Маршрут не найден" при обновлении страницы во время Штурвала

Обнаружено пользователем вживую: обновление страницы на Gate 1/2 Штурвала
показывало 404 "Маршрут не найден". Причина — виртуальный
`curriculum_id = steering-{work_job_id}` существует только в
LangGraph-чекпоинтере (Postgres, `AsyncPostgresSaver`) и в React-состоянии
браузера; он никогда не попадает в обычный `skill_tree_store`, а
`loadWorkspace()` при перезагрузке пытается получить его именно оттуда
(`GET /skill-tree/curricula/{id}/workspace`).

Данные Gate 1/2 физически не терялись — они уже жили в чекпоинтере;
не хватало read-only способа их оттуда перечитать без продвижения графа
вперёд. Пользователь подтвердил: нужно полное восстановление состояния.

| Файл | Комментарий |
|---|---|
| `api/routes/steering.py` (`GET /{work_job_id}/status`, новый) | Read-only: `job = _require_job(...)` + `graph.aget_state(config)` — та же операция, что approve-эндпоинты уже делают ПОСЛЕ `run_or_resume` для формирования ответа, но без самого `run_or_resume` (не двигает граф). Отдаёт `status`/`target_goal`/`taxonomy_discovery`/`surface_digests`/`result`/`error` — тот же shape полей, что и POST-эндпоинты, чтобы фронтенд мог переиспользовать существующую логику сборки виртуальной ноды-гейта |
| `api/routes/steering.py` (`approve-gate2`) | Add-only фикс: после создания `WorkJobKind.STEERING_MAP_REDUCE` джобы `generation_job_id` дополнительно дописывается в `result` уже `COMPLETED` funnel-джобы (`work_job_store.complete(job.id, {...})`) — раньше он уходил только в HTTP-ответ и терялся при перезагрузке страницы после Gate 2, `GET .../status` не мог сказать, какую Map-Reduce джобу ждать |
| `web/static/skill-tree/api.js` | `steeringGetStatus(workJobId)` — тот же fetch-паттерн, что остальные steering-обёртки |
| `web/static/skill-tree/RoadmapDashboard.js` | Новая `resumeSteeringSession(jobId)`, вызывается из mount-эффекта вместо `loadWorkspace`, когда `?curriculum=` начинается с `"steering-"` (единственный источник такого ID — сам `handleGenerate`, localStorage/backend "активный курс" никогда его не хранит, т.к. `rememberActiveCurriculumId`/`setActiveCurriculum` для виртуальной Штурвал-сессии не вызываются). Восстанавливает ТУ ЖЕ виртуальную ноду-гейт/session, что `handleGenerate`/`handleGateApprove` строят "вживую" — с точки зрения остального UI неотличимо. Для `status="completed"` — дожидается `generation_job_id` через `waitWorkJob` и переключается на реальный `loadWorkspace(curriculum_id)`, который сам заменяет `steering-*` в URL на настоящий ID (дальше обычные reload работают штатно) |
| — | **Проверено** | Новый `test_steering_status_endpoint.py` — 5 тестов (восстановление awaiting_gate_1/2, completed с generation_job_id, 404 на неизвестной джобе, регрессия на сам фикс approve-gate2). Джобы создаются реальным `work_job_store` и подчищаются через `fail()` — без замусоривания `.runs/work_jobs.json`. `npm run build` — чисто. Живым браузером не проверено (нет dev-сервера/Postgres в этой фоновой сессии) |

## Дубли кандидатов на Gate 1 → "пропавшие" дайджесты на Gate 2

Диагностировано на реальной живой джобе (`steering-cf3df66b9cbd`, проверено
напрямую через `work_job_store.get`/`graph.aget_state` — не гипотеза).
Находка: `approved_gate1_urls` содержал 9 записей, из них только 5
уникальных (одна и та же статья Postgres Pro встретилась 4 раза, статья
ВТБ — дважды) — сам чекпоинтер хранил именно то, что было утверждено;
восстановление после reload (`GET .../status`) лишь честно это показало.

Причины (три независимых, не связаны с фиксом reload-восстановления):

| # | Причина | Файл | Фикс |
|---|---|---|---|
| 1 | `light_discovery_service.py` не дедуплицировал кандидатов по URL — одна и та же статья находится через разные `company_hubs`, попадает в Gate 1 отдельными карточками, обе можно отметить | `src/curriculum/services/light_discovery_service.py` | `_dedupe_candidates_by_url()` (нормализация регистра/слэша, как `node_candidate_collection_service._dedupe_by_url`, без BGE-M3 — разные статьи по смежным темам курса это желаемое разнообразие) — вызывается сразу после сбора, до policy-фильтра/перевода |
| 2 | `generate_surface_digests` (уже существующий, не менялся) молча роняет URL, которые не удалось скачать (fail-open by design) — без единого сигнала пользователю, какие из утверждённых статей не попали в Gate 2. Один из "пропавших" URL в живой джобе — `netflixtechblog.com`, живой `curl` подтвердил HTTP 403 (Medium блокирует datacenter-IP), т.е. это НЕ баг конкретно в Trafilatura/фетчере, а блокировка конкретного сайта | — (без изменений в самом сервисе) | `SteeringGatePanel.js` — новая подсказка на Gate 2: если `approvedCount > digests.length`, показывает "Из N утверждённых дайджест удалось построить для M — остальные не открылись". `approvedCount` пробрасывается и из живого `handleGateApprove`/`handleNodeGateApprove` (`selectedUrls.length`), и из `resumeSteeringSession` (новое поле `approved_gate1_urls` в `GET .../status`) |
| 3 | `Gate2Digest` никогда не рендерил ссылку на исходную статью (в отличие от `Gate1Candidate`) — на Gate 2 нельзя было ни проверить источник дайджеста, ни открыть оригинал | `web/static/skill-tree/SteeringGatePanel.js` | Добавлен `<a href={item.url} target="_blank">` — тот же элемент, что уже был у `Gate1Candidate` |
| — | **Важно** | Фикс НЕ ретроактивен — джоба `cf3df66b9cbd` уже прошла Gate 1 с дублями, её `approved_gate1_urls`/`surface_digests` в чекпоинтере не переписываются; для чистого результата нужен новый прогон Штурвала с нуля (Gate 1 теперь дедуплицирован) |
| — | **Проверено** | Новый `test_light_discovery_dedup.py` (3 теста: точные дубли, регистр/слэш, различные URL остаются). `test_steering_status_endpoint.py` дополнен проверкой `approved_gate1_urls` в ответе. Итого 54/54 теста. `npm run build` — чисто |

## Штурвал никогда не делал Map-Reduce (несмотря на название) → ленивый ingest при открытии ноды

Пользователь заметил отсутствие Map/Reduce для утверждённых на Gate 2
статей. Диагностировано через `scripts/log_profiler.py` на реальном
завершённом прогоне (`steering_funnel=da5efaaa1be0` →
`steering_map_reduce=c5a9c34f60df`): за весь job — **ровно один**
`GEMINI HTTP` вызов (`curriculum_generator / search_first`, 6.1s), ни
одной Map/Reduce-стадии. Лог хранился в Redis (`redis_logs_enabled()`),
не в файле `.runs/*.log` — вытащен через `redis_run_log.read_lines`.

Причина — не баг исполнения, а изначальный дизайн:
`steering_generator_bridge.py::generate_curriculum_from_steering_digests`
строит `CurriculumSearchHit` прямо из лёгких Gate-2 дайджестов
(`SurfaceDigestItem`, 2-3 предложения Flash Lite) и зовёт
`generate_curriculum_search_first` — тот же "Search-First" путь, что и
быстрый Autopilot: один LLM-вызов на весь курс, без чанкинга/MAP/REDUCE/
записи в pgvector. Название `WorkJobKind.STEERING_MAP_REDUCE` и докстринг
файла ("Steering → тяжёлый Map-Reduce") при этом обещают то, чего в коде
нет. Подтверждено на реальных данных: у всех нод сгенерированного курса
`grounding_status="model_only"`, у записей реестра `source_tier=
"steering_approved"` — ни одного реального ингест-тира.

**Решение (по прямому указанию пользователя)**: не эйджерно на генерации
курса (замедлило бы Gate 2 → курс с ~6s до многих секунд), а **лениво,
при первом открытии ноды** — с дедупом между нодами (одна и та же
approved-статья часто примаплена на несколько нод; подтверждено живой
находкой: у всех 5 нод тестового курса совпадали `primary_source_id`/
`mapped_source_ids`).

| Файл | Комментарий |
|---|---|
| `src/curriculum/services/steering_lazy_ingest_service.py` (новый) | `ensure_node_steering_sources_ingested(graph, node, target_goal="")` — для `mapped_source_ids` ноды с `source_tier=="steering_approved"` в реестре реально доингещивает их через ТОТ ЖЕ `summarize_whitelist_blog_hits_async` (переиспользован из `source_material_pipeline.py`, ничего нового не изобретено), обновляет и реестр (extracts/snippet/tier), и `node.source_ref`/`grounding_status` (лекция реально читает `node.source_ref`, не реестр напрямую — см. `lecture_search_orchestrator.py:840`). Дедуп бесплатный: как только тир меняется с `steering_approved` на канонический, следующие открытия других нод с тем же `source_id` это видят и ничего не переделывают. Fail-open на каждом шаге. `_infer_source_tier` переиспользован (импортирован, не скопирован) из `node_grounding_finalize_service.py` |
| `api/routes/node_skill.py` (`POST /node/ensure-steering-sources`) | Add-only. Instant no-op для любой ноды без `steering_approved` источников — то есть для подавляющего большинства (Autopilot, уже доингещенные Штурвал-ноды) |
| `web/static/skill-tree/api.js` | `ensureSteeringSourcesIngested(curriculumId, nodeData)` |
| `web/static/skill-tree/RoadmapDashboard.js` | `openNode` вызывает новый эндпоинт ПЕРЕД `nodeInitStream` (для любой ноды, не только DEEP — Штурвал-ноды генерируются как BASE) с fail-open try/catch; `nodeInitStream` затем получает уже свежий узел с бэкенда (`_apply_lazy_grounding_for_init`/`_merge_node_data_from_graph` в engine.py — НЕ менялись — сами перечитывают граф по `curriculum_id`, клиентский `node_data` не авторитетен) |
| — | **Проверено** | `test_steering_lazy_ingest.py` — 9 тестов (no-op на каноническом тире, обновление реестра+source_ref, дедуп между двумя нодами с общим source_id, fail-open на ошибке/пустом ответе ingest, идентичность `_infer_source_tier` с оригиналом, 3 эндпоинт-теста). Итого по всем Steering/Node-Gate тестам — 63/63. `npm run build`, `black`/`isort`/`flake8` — чисто |
| — | **Отдельная находка, не в этом фиксе** | У сгенерированного тестового курса ВСЕ 5 нод получили `primary_source_id="src_1"`/`mapped_source_ids=["src_1"]`, хотя по смыслу `relevant_extracts` разных нод явно относятся к разным approved-статьям — похоже на баг атрибуции источников в `generate_curriculum_search_first`/`assign_source_ids` (общий код, используется и Autopilot). Не чинил — не было в задании этого хода, но worth a separate look |

## Баг атрибуции источников в generate_curriculum_search_first (общий с Autopilot)

По указанию пользователя разобрался с "отдельной находкой" из предыдущего
фикса: все ноды сгенерированного курса получали `primary_source_id`
первой статьи в реестре, независимо от того, о какой статье они реально
писали.

**Корень (`src/curriculum/search_first_flash.py::_resolve_source_ref`)**:
приоритет проверки был перепутан. Функция сначала пыталась резолвить
`source_id`, который LLM возвращает в формате "src_N"/"SN" — но часто
путает (пустая строка, голое число, другой формат). `_norm_src_id(raw_sid,
1)` при нераспознанном формате молча возвращала ЗАХАРДКОЖЕННЫЙ `"src_1"`
(второй аргумент — индекс фоллбэка — был жёстко `1`, а не позиция ноды).
`"src_1"` — это РЕАЛЬНАЯ запись реестра (не отсутствующая), поэтому
`reg_by_id.get(sid)` её находила, и весь url-based фоллбэк ниже (гейтился
на `if not entry`) никогда не успевал сработать: любая нода с "неудобным"
для LLM `source_id` молча получала источник первой статьи реестра.

**Фикс**: поменял приоритет — сначала пытаемся резолвить по `url` (самый
надёжный сигнал, LLM почти всегда копирует его дословно из входных
материалов), и только если url не резолвился — по `source_id`, причём
ТОЛЬКО если он реально похож на `src_N`/`SN` (без угадывания дефолта).
Последний резерв (первая запись реестра) — как и раньше, для случая, когда
не резолвилось вообще ничего.

Код общий — используется и Autopilot Search-First путём, и Штурвалом
(`steering_generator_bridge.py`). Других существующих тестов на этот
модуль не было (`grep` по `tests/` — 0 совпадений до этого фикса).

| Файл | Комментарий |
|---|---|
| `src/curriculum/search_first_flash.py` (`_resolve_source_ref`) | Реордер приоритета: url → (только well-formed) source_id → registry[0]. Убрана и мёртвая строка `if not entry and sid in reg_by_id` (недостижима — `reg_by_id.get(sid)` уже покрывал этот случай) |
| — | **Проверено** | Новый `test_search_first_source_attribution.py` — 6 тестов, включая прямую регрессию на сам баг (4 ноды с одинаково "плохим" source_id, но разными url → 4 разных source_id вместо одного и того же), приоритет url над ошибочным-но-well-formed source_id, обычный случай (well-formed id без url — не сломан), fail-through на registry[0]. 15/15 (вместе со Steering lazy-ingest тестами). `black`/`isort`/`flake8` — чисто |

## "Проблема"/"Практический кейс" на Gate 2 — не всегда на русском

Причина: `RUSSIAN_OUTPUT_RULE` (`llm_locale.py`) явно перечисляет
"title, description, pros, cons, takeaways, failure_modes" как поля,
на которые распространяется правило — `problem_solved`/
`two_sentence_summary` (Штурвал) и `architecture`/`practical_case`/
`limitations` (Node Gate) в этот список не входят, и их СОБСТВЕННОЕ
англоязычное описание в промпте ("the engineering problem...", "the
architecture/approach...") иногда перевешивает общее правило для конкретно
этих полей. Тот же паттерн уже был решён для `_PASSPORT_SYSTEM`
(`node_candidate_collection_service.py`) — там у каждого текстового поля
явная пометка `(Russian)`.

| Файл | Комментарий |
|---|---|
| `src/curriculum/services/batch_digest_service.py` (`_DIGEST_SYSTEM`) | `problem_solved`/`two_sentence_summary` — добавлена явная `(Russian)`. `main_tech_stack` — добавлена явная пометка НЕ переводить названия технологий (тот же принцип, что `_PASSPORT_SYSTEM`) |
| `src/curriculum/services/node_digest_service.py` (`_DIGEST_SYSTEM`) | Та же находка в собственном промпте Node Gate — `architecture`/`practical_case`/`limitations` — добавлена явная `(Russian)` |
| — | Изменение только текста промпта, логика не тронута — существующие тесты (6/6 на digest) остаются зелёными как есть |

## Разделение на Mode 1/Mode 2

**Запрос**: до этой правки `POST /curriculum/steering/generate` имел только
один сценарий — TaxonomyService + Light Discovery → Gate 1 → Batch Digest →
Gate 2 → `generate_curriculum_from_steering_digests` (`steering_generator_
bridge.py`) → `generate_curriculum_search_first`. Этот путь строил
`CurriculumGraph` напрямую из Gate-2-approved дайджестов, где каждый
дайджест детерминированно превращался ровно в один `CurriculumSearchHit`
(`_hit_from_digest`) — глобальный поиск/выбор статей (по своей природе
задача "обзор темы", Mode 2) принудительно применялся к построению графа
курса, из-за чего одна approved-статья становилась ровно одной нодой,
независимо от реального количества нужных курсу нод.

**Решение** — два независимых, не пересекающихся сценария под одним
`POST /generate`, разводимые новым полем `steering_mode` (`steering_
contracts.py`):

- **Mode 1 (`per_node`, дефолт)** — курс строится РОВНО как у Autopilot по
  умолчанию: `generate_curriculum_graph` → Model-First DAG (Flash Lite, без
  единого источника заранее) → классификация BASE/DEEP → DEEP-ноды получают
  `pending_grounding`. Никакого Gate 1/2 на уровне курса нет вообще —
  `api/routes/steering.py::_post_steering_generate_per_node` вообще не
  открывает `steering_graph_session`, это тонкий алиас поверх
  `enqueue_curriculum_generate` (тот же `WorkJobKind.CURRICULUM_GENERATE`,
  что и обычный `/curriculum/generate`). Заземление остаётся точечным и
  ленивым — при открытии DEEP-ноды её перехватывает уже существующий Node
  Grounding Gate (`node/grounding-discover/-digest/-finalize`), который сам
  даёт до `NODE_GATE1_MAX_APPROVED=4` источников на ноду через настоящий
  Map-Reduce (`finalize_node_grounding`) — ровно то "3-4 источника через
  Map-Reduce на ноду", что требовалось, безо всякого нового кода: этот
  механизм уже существовал и уже работает одинаково для Autopilot и
  Штурвала (он не завязан на `control_axis`).
- **Mode 2 (`standalone_digest`)** — прежний путь TaxonomyService + Light
  Discovery → Gate 1 → Batch Digest → Gate 2 не тронут ни на строчку. Но
  ИТОГ теперь — не `CurriculumGraph`, а один объединённый
  `TopicDigestDocument` (`steering_contracts.py`): `executive_summary`
  (3-5 предложений, один синтез-вызов Flash Lite,
  `services/steering_topic_digest_service.py`) + `sections` (по одной на
  approved-статью, `key_points` — сухая выжимка её же Gate-2-дайджеста).
  Fail-open на каждом уровне: сбой синтез-вызова или пустой
  `sections` в ответе LLM — откат на `_fallback_document` (секции строятся
  напрямую из уже готовых `SurfaceDigestItem`, без `executive_summary`, а
  не пустой документ). После Gate 2 `approve-gate2` ставит
  `WorkJobKind.STEERING_TOPIC_DIGEST` (обычный worker,
  `_run_steering_topic_digest`) — `save_curriculum_record` НЕ вызывается,
  это не курс. Фронтенд рендерит документ тем же виртуальным
  `steering-gate-node`, что и Gate 1/2, компонентом
  `SteeringTopicDigestView.js`.

`WorkJobKind.STEERING_MAP_REDUCE`/`steering_generator_bridge.py` оставлены
как LEGACY (обработчик `_run_steering_map_reduce` не удалён) — новый код их
больше не создаёт, но десериализация уже существующих в Redis/`.runs`
записей этого типа не должна падать.

Выбор режима — новый селектор в `CurriculumInputBar.js` (виден только при
`controlAxis === "steering"`); `RoadmapDashboard.js` разводит генерацию на
`runSteeringCreatePerNode`/`runSteeringCreateStandaloneDigest`.

| Файл | Комментарий |
|---|---|
| `src/curriculum/steering_contracts.py` | `SteeringMode`, `steering_mode` на `SteeringCurriculumGenerateInput` (дефолт `per_node`), `TopicDigestSection`/`TopicDigestDocument` |
| `api/routes/steering.py` | `/generate` разведён на `_post_steering_generate_per_node`/`_post_steering_generate_standalone_digest`; `/approve-gate2` ставит `STEERING_TOPIC_DIGEST` с урезанным payload'ом (без user_level/depth_level/generation_mode/source_policy) |
| `services/work_job_store.py` | Новый `WorkJobKind.STEERING_TOPIC_DIGEST`; `STEERING_MAP_REDUCE` помечен legacy в докстринге, не удалён |
| `services/work_handlers.py` | Новый `_run_steering_topic_digest`; `_run_steering_map_reduce` оставлен как legacy-обработчик |
| `src/curriculum/services/steering_topic_digest_service.py` (новый) | `generate_standalone_topic_digest` — один Flash Lite синтез-вызов + fail-open `_fallback_document` |
| `web/static/skill-tree/CurriculumInputBar.js`, `RoadmapDashboard.js`, `NodeDrawer.js`, `SteeringTopicDigestView.js` (новый) | UI-селектор режима + рендер итогового документа |
| `tests/test_steering_mode_split.py` (новый, 9 тестов) | per_node не трогает steering_graph, payload без control_axis/steering_mode, standalone_digest не тронут, approve-gate2 создаёт STEERING_TOPIC_DIGEST с урезанным payload, generate_standalone_topic_digest fail-open (success/exception/None/empty sections/no-match ValueError) |
| — | Суммарно по всем Steering/Node-Gate тестам: 78 passed |

## Gate 2: статья с медленного/CDN-защищённого сайта тихо выпадает из дайджеста

**Запрос**: `curriculum=steering-76a3fcaf2765&node=steering-gate-node` — на Gate 1
пользователь утвердил 7 URL (6 Habr + `netflixtechblog.com`), на Gate 2 дайджест
построился только для 6 — Netflix пропал без ошибки в UI. Сайт при этом
открывается у пользователя в браузере без проблем.

**Диагноз** (воспроизведено напрямую вызовом `generate_surface_digests` с теми
же 7 URL): никакой блокировки по 403/боту на уровне "сайт закрыт" нет — причина
в двух независимых факторах на пути `batch_digest_service.py::_fetch_article`/
`node_digest_service.py::_fetch_article` → `lecture_passage_fetch.py::fetch_html`:

1. **Слишком короткий таймаут.** Оба Gate-2-сервиса переиспользовали
   `LECTURE_PASSAGE_FETCH_TIMEOUT_SEC=1.8с` — константу, рассчитанную на
   необязательный лекционный добор абзацев (там уместно жертвовать медленным
   источником). `netflixtechblog.com` (Medium, ~267КБ HTML) грузится 1.4–2.5с
   в зависимости от нагрузки — регулярно не укладывался.
2. **Нестабильный 403 от CDN под конкурентной нагрузкой** (обнаружено уже
   после фикса #1, прямыми параллельными `httpx`-запросами к тому же URL):
   Fastly/Medium иногда отдаёт 403 с телом ~6КБ (страница анти-бот-блока)
   именно когда запросы идут параллельно — при последовательных запросах
   или через браузер пользователя всегда 200. `fetch_html` на `status_code
   >= 400` возвращает `""`, что дальше по цепочке неотличимо от "сайт
   недоступен" — статья тихо выпадает из `articles`, LLM про неё даже не
   узнаёт (batch-промпт получает только успешно скачанные статьи).

**Решение** (оба фактора, `src/config/settings.py`/`batch_digest_service.py`/
`node_digest_service.py`):
- Новая отдельная константа `STEERING_DIGEST_FETCH_TIMEOUT_SEC=8.0` (env
  override) — используется ТОЛЬКО в Gate-2 fetch обоих сервисов,
  `LECTURE_PASSAGE_FETCH_TIMEOUT_SEC` (и лекционный сценарий) не тронуты.
- Один retry с паузой 1.5с в `_fetch_article` при пустом `fetch_html` —
  статья уже approved пользователем (≤10 на Штурвале / ≤`NODE_GATE1_MAX_
  APPROVED` на Node Gate), одна лишняя попытка оправдана. **Не панацея**:
  вероятностный 403 у CDN снижает шанс потери с ~50% до ~25% (два
  независимых испытания), но не гарантирует 100% — это внешнее,
  недетерминированное поведение чужого CDN, полностью убрать его нельзя,
  не меняя стратегию fetch (реальный браузер/прокси и т.п.).

Проверено: `generate_surface_digests` с теми же 7 URL — 2 из 3 повторных
прогонов дают 7/7, один всё ещё 6/7 (ожидаемо, см. выше). `test_node_grounding_
gate.py` — 41 passed (регрессий нет, retry новых прямых тестов не имеет —
сетевой сценарий, не мокается).

## Mode 2 (standalone_digest): финал переделан на реальный Map-Reduce

**Запрос**: после первой версии Mode 2 (лёгкий Flash Lite синтез поверх уже
тонких Gate-2 дайджестов → бесструктурный `TopicDigestDocument`, без графа)
пользователь дважды уточнил, что ожидал ровно того же, что Autopilot/Node
Grounding Gate: **реальный тяжёлый Map-Reduce** по approved-статьям (полный
текст, MAP-окна, REDUCE-синтез — не пересборка уже тонких дайджестов), и
итог — **загружаемая нода** (со своим чатом/источниками), а не отдельный тип
"документ" без графа.

**Ограничение схемы, из-за которого решение не тривиально**:
`CurriculumGraph.nodes` (`schemas.py:289`) требует `min_length=3` — общий
инвариант с Autopilot, пользователь явно попросил его НЕ трогать
("это не должно касаться контрактов для Автопилота"). Буквально "одна нода"
как `CurriculumGraph` из 1 элемента невалидна.

**Решение** (`services/steering_topic_node_service.py`, новый, заменяет
удалённый `steering_topic_digest_service.py`):
1. Approved URL → `stub_hits` → `summarize_whitelist_blog_hits_async` +
   `enrich_search_hits_with_extracts_async` — ТОТ ЖЕ ingest, что
   `node_grounding_finalize_service.finalize_node_grounding`/
   `targeted_node_grounding.py`, никакого нового Map-Reduce не изобретено.
2. Хиты **детерминированно** (без LLM — во избежание точно того
   атрибуционного бага 1-статья-на-ноду, который уже чинили в
   `search_first_flash.py`) режутся на `max(3, ceil(N/
   CURRICULUM_DEEP_NODE_MAX_HITS))` бакетов по ≤`CURRICULUM_DEEP_NODE_MAX_
   HITS` источников — это и обходит ограничение схемы (≥3 ноды), и держит
   системный инвариант "≤4 источника на ноду" (`_chunk_evenly`).
3. Один батч-LLM-вызов (Flash Lite) только ПОДПИСЫВАЕТ уже фиксированные
   бакеты (title/category/brief_summary/core_concepts) — состав бакета
   LLM не выбирает. Fail-open: сбой вызова → `_fallback_label` (генерическая
   подпись из target_goal, без выдумывания фактов).
4. Собранный `CurriculumGraph` проходит через `cap_curriculum_sources_
   registry`/`sync_route_sources_from_registry` (тот же хвост, что
   `finalize_node_grounding`) и `save_curriculum_record` — ровно как обычный
   курс. `WorkJobKind.STEERING_TOPIC_DIGEST`/имя функции `_run_steering_
   topic_digest` сохранены (не курс-специфичное имя, но минимальные
   изменения истории джоб) — по сути строит граф, не документ.
5. Фронтенд (`RoadmapDashboard.js`/`NodeDrawer.js`) откачен к простой логике
   "результат = `curriculum_id` → `loadWorkspace`", как и у Mode 1 —
   `TopicDigestDocument`/`SteeringTopicDigestView.js`/бесструктурный
   документ полностью удалены (были добавлены и убраны в рамках одной и той
   же правки).

Проверено: `test_steering_topic_node_service.py` (7 тестов, первая версия) —
≥3 ноды при малом числе статей, ≤4 источника на ноду при 10 approved,
ValueError на пустой/несуществующий approved-набор, fallback-подписи при
сбое LLM, `_chunk_evenly` unit-тесты. `test_steering_mode_split.py` обновлён
(payload теперь несёт `funnel_job_id`). Суммарно по Steering/Node-Gate:
66 passed. `black`/`isort`/`flake8` чисты, `npm run build` чист.

## Mode 2, итерация 2: одна нода вместо ≥3 бакетов (prompt.log)

**Запрос** (сразу после предыдущей правки — файл переименован в
`prompt.log`, тот же смысл): деление на ≥3 ноды-бакета — искусственное,
сделано ТОЛЬКО ради `CurriculumGraph.nodes >= 3`. Смысл Gate 2 — строгий
отбор источников; сам сгенерированный контент должен подаваться через
СУЩЕСТВУЮЩИЕ механизмы (Лекция/Topic Q&A/одна сфокусированная нода), а не
оформляться как искусственный граф. Требования:
1. Убрать нарезку на 3+ бакета — 1 сфокусированная нода с глубоким разбором
   (4-12 подтем), заземлённая через Map-Reduce по ВСЕМ Gate-2-статьям.
2. Материалы сразу становятся базой для Лекции/Topic Q&A (существующий
   `node_deep_dive`).
3. Если используется `CurriculumGraph` — ослабить `min_items` под тип
   `steering_node`/`standalone`, не создавать фейковые ноды-заглушки.

**Уточнено вопросами** (архитектурная развилка была реальной): (1) кап
`mapped_source_ids<=CURRICULUM_DEEP_NODE_MAX_HITS=4` тоже блокировал "одна
нода — все источники" — решено отдельным дискриминированным типом с
условной валидацией (не трогая Autopilot); (2) "4-12 подтем" — генерируется
ЯВНО при Map-Reduce (не просто ожидаемое качество обычной Лекции).

**Решение**:
- `schemas.py`: новый `NodeKind = Literal["standard", "steering_standalone"]`;
  `CurriculumNode.node_kind`/`CurriculumGraph.graph_kind` (default
  `"standard"` — 100% обратная совместимость). Жёсткие `Field(max_length=
  CURRICULUM_DEEP_NODE_MAX_HITS)` на `mapped_source_ids` и `Field(min_length
  =3)` на `CurriculumGraph.nodes` УБРАНЫ из декларативных constraints (Pydantic
  не умеет их делать условными на уровне `Field`) и заменены
  `@model_validator(mode="after")`, который применяет прежний жёсткий кап
  ТОЛЬКО когда `node_kind`/`graph_kind == "standard"` — для Autopilot и Node
  Grounding Gate поведение побитово то же (default остаётся "standard").
  `NodeCurriculumBreakdown` получил новое поле `subtopics: list[NodeSubtopic]`
  (add-only, пусто для обычных узлов).
- `steering_topic_node_service.py` переписан: убран `_chunk_evenly`/
  bucket-labelling, вместо него — один content-generation LLM-вызов
  (`_StandaloneNodeContent`: title/category/brief_summary/core_concepts/
  subtopics 1-12 — `min_length=1`, не 4, чтобы не заставлять LLM выдумывать
  подтемы при малом числе approved-статей) поверх ВСЕХ ingest-хитов сразу.
  Собирается РОВНО ОДНА `CurriculumNode` (`node_kind="steering_standalone"`,
  `mapped_source_ids` = все sid без обрезки, `source_ref.relevant_extracts`
  агрегирует выдержки со ВСЕХ статей, не только первой, до 12 — предел
  самого `NodeSourceRef`). `CurriculumGraph.graph_kind="steering_standalone"`,
  `total_nodes=1`.
- `node_deep_dive` НЕ изменялся — заземлённая (`grounding_status="grounded"`,
  `source_ref` заполнен) нода открывается штатным путём, Лекция/Topic Q&A
  строится существующим RAG поверх реального Map-Reduce ingest (LanceDB) —
  ровно как у любой другой grounded DEEP-ноды.

Проверено: `test_steering_topic_node_service.py` переписан (7 тестов) —
ровно 1 нода, все approved-источники замаплены без обрезки, явная структура
подтем в ответе, `source_ref.relevant_extracts` агрегирует выдержки из ВСЕХ
статей (не только первой), ValueError на пустом/несуществующем наборе,
fallback на сбое/`None` от LLM. Плюс `test_graph_integrity.py`/
`test_curriculum_topology.py`/`test_node_grounding_gate.py` (регрессия схемы
для Autopilot/Node Gate) — без изменений в поведении. Суммарно по Steering/
Node-Gate/graph-схеме: **86 passed**. `black`/`isort`/`flake8` чисты,
`npm run build` чист.

## Gate 2: дефолт выбора — все статьи отмечены

**Запрос**: на Gate 2 все статьи в дайджесте должны оставаться отмеченными
по умолчанию (не пустой выбор с нуля).

**Причина старого поведения**: Gate 1 и Gate 2 в `SteeringGatePanel.js`
делили один и тот же дефолт — пустой `Set()` — сознательно (ранее в этой же
сессии убрали автовыбор всех кандидатов на Gate 1: "пользователь должен
явно решить, какие статьи брать"). Но Gate 2 показывает НЕ новый пул
кандидатов, а уже те статьи, что пользователь САМ утвердил на Gate 1 (тут
только дайджест по ним) — заставлять отмечать их заново было лишним
трением, Gate 2 по смыслу — ревью ("убрать то, что по дайджесту оказалось
нерелевантным"), а не отбор с нуля.

**Решение** (`SteeringGatePanel.js`): дефолт выбора теперь зависит от
`isGate2` — `new Set(itemUrls)` (все отмечены) на Gate 2, пустой `Set()`
на Gate 1 (без изменений). Общий компонент для Штурвала (курс) и Node
Grounding Gate — правило действует одинаково для обоих (cap `maxApproved`
на Node Gate 2 не конфликтует: список дайджестов там уже ≤4, кап был
применён раньше, на Gate 1).

Проверено: `npm run build` чист (синтаксис). Логика простая, без нового
Python-кода — только React-хук; отдельных unit-тестов на JS-компонент в
проекте нет (текущий паттерн — только сборка + ручная проверка в браузере,
не покрыта в этой сессии).

## Забытый mirror-фикс: NodeDataInput тоже капал источники в 4

**Запрос**: реальная ошибка с фронтенда (`prompt.log`) — `422 too_long,
mapped_source_ids: List should have at most 4 items after validation, not 8`
при открытии/чате Штурвал-ноды (steering_standalone, 8 источников).

**Причина**: `toNodeDataInput()` (api.js) сериализует ПОЛНЫЙ объект ноды в
каждый запрос `/node/init|chat|verify|restart|grounding-*` — устоявшийся,
широко используемый архитектурный паттерн (в `engine.py` ~15 мест напрямую
читают `req.node_data`, не перечитывая граф с сервера; переделывать это в
lookup по `curriculum_id`/`node_id` — большой рискованный рефакторинг, от
которого по явному запросу отказались в пользу точечного фикса). Условную
валидацию (`node_kind` снимает кап `CURRICULUM_DEEP_NODE_MAX_HITS`) я добавил
только в `CurriculumNode` (src/curriculum/schemas.py) — но `NodeDataInput`
(node_deep_dive/schemas.py) — ОТДЕЛЬНАЯ, параллельная схема с тем же старым
безусловным `max_length=CURRICULUM_DEEP_NODE_MAX_HITS`, её забыли.

**Решение**: тот же паттерн, mirror в `NodeDataInput` — новое поле
`node_kind: NodeKind` (переиспользован тип из `curriculum/schemas.py`,
default `"standard"`), `max_length` убран из декларативного `Field(...)`,
заменён `@model_validator(mode="after")` — кап применяется только когда
`node_kind != "steering_standalone"`. `toNodeDataInput()` (api.js) теперь
прокидывает `node.node_kind || "standard"` в каждый запрос.

Проверено: новый `test_node_data_input_steering_cap.py` (4 теста) —
`standard` по-прежнему падает на >4 (регрессия воспроизведена и подтверждена
фиксом прямым скриптом, ровно как в prompt.log), `steering_standalone`
проходит с 8+, дефолт `node_kind="standard"`. Плюс
`test_node_grounding_gate.py`/`test_steering_lazy_ingest.py` (50 passed) —
без регрессий. `black`/`isort`/`flake8` чисты, `npm run build` чист.

## Interaction Axis подключён + [Rn] RAG-инжест включён при Map-Reduce

**Запрос**: три диагностических вопроса про ноду `curriculum=...&node=topic`
— (1) как задавать вопросы в Topic Q&A, если формат ввода тот же, что после
обзора; (2) почему видно только `[Sn]`, а не `[Rn]` из уже сделанного
Map-Reduce; (3) какие промпты и где используются. По итогам разбора
пользователь запросил оба технических фикса из трёх находок.

**Диагноз (см. также ответ в чате)**: `interactionAxis` в session state
никогда не попадал в тело запроса (`toNodeDataInput`/`NodeDeepDiveRequest`
его не несли), а `select_interaction_axis_system_prompt`
(prompt_factory.py) существовал, но не вызывался ниоткуда — выбор
"Topic Q&A" был чисто визуальным. `[Rn]` (chunk-level RAG, LanceDB
`RAG_CHUNKS_TABLE`, см. `retrieve_lecture_rag_context`/
`fetch_rag_chunks_by_doc_id`) не заполнялся для контента, заземлённого через
общий Map-Reduce (`persist_approved_curriculum_hits_to_lancedb_async`),
потому что тот вызывал `save_summary(..., skip_rag_ingest=True)` — этот флаг
пишет только `[Sn]`-каталог (Qdrant, `document_summaries`), пропуская
`ingest_document_summary` (LanceDB chunks). Перепроверено по прямому запросу
пользователя — подтверждено: `[Sn]`=Qdrant, `[Rn]`=LanceDB, оба живут в
одном классе `VectorStore` (services/vector_store.py).

**Решение 1 — Interaction Axis (живой)**:
- `NodeDeepDiveRequest` (node_deep_dive/schemas.py) получил
  `interaction_axis: Literal["lecture_self_check","topic_qna"]` напрямую
  (убран отдельный `TopicQnaNodeDeepDiveRequest` в steering_contracts.py —
  стал избыточен).
- `NodeSessionBody` (node_skill.py, база для init/chat/restart) получил то
  же поле, прокинуто во все payload `enqueue_node_deep_dive`
  (`/init`, `/init-stream`, `/chat-stream`, через общий
  `_enqueue_and_maybe_inline`).
- `work_handlers.py` (`_run_node_deep_dive`/`_run_node_deep_dive_stream`)
  читает `payload["interaction_axis"]` в `NodeDeepDiveRequest`.
- `generate_dense_material` (services/node_content_generator.py) — новый
  параметр `interaction_axis`; при `"topic_qna"` добавляет
  `TOPIC_QNA_SYSTEM_PROMPT` ПОВЕРХ обычного dense-system (не заменяет —
  цитатная политика/RAG-контекст остаются теми же). `engine.py::
  run_dense_lecture_turn` прокидывает `req.interaction_axis`.
- Фронтенд: `nodeInitStream`/`nodeChatStream`/`nodeChat` (api.js) — новый
  параметр `interactionAxis`; `RoadmapDashboard.js` передаёт
  `sessions[nid]?.interactionAxis`; `TopicFocusView.js` передаёт жёстко
  `"topic_qna"` (это его единственное назначение). Заодно нашёл и поправил
  мелкую несостыковку в `NodeMasteryPanel.js` — value дропдауна был
  `"lecture"`, а не канонический `"lecture_self_check"` (раньше было
  неважно — значение было полностью инертным).

**Решение 2 — `[Rn]` RAG-инжест**:
`curriculum_lancedb_persist.py::persist_approved_curriculum_hits_to_lancedb_async`
теперь вызывает `save_summary(..., skip_rag_ingest=False)`. Чанкуется тот
же текст, что уже идёт в `[Sn]`-паспорт (title/executive_summary/takeaways
— `_summary_document` в vector_store.py), НЕ полный исходный текст статьи
(та в эту функцию не приходит, только сжатые `key_extracts`). Область
действия — только этот общий Map-Reduce choke-point (Node Grounding Gate,
Штурвал); `academic_gemma_ingest.py`/`lecture_search_orchestrator.py` (свои,
отдельные `skip_rag_ingest=True`) осознанно не тронуты — вне запрошенной
области "при Map-Reduce".

Проверено: новый `test_interaction_axis_topic_qna_wiring.py` (5 тестов) —
диспетчеризация `select_interaction_axis_system_prompt`, дефолт
`NodeDeepDiveRequest.interaction_axis`, и главное — `generate_dense_material`
реально добавляет `TOPIC_QNA_SYSTEM_PROMPT` в system только при
`interaction_axis="topic_qna"` (перехват на границе LLM-вызова, без
реального сетевого вызова). Новый
`test_curriculum_lancedb_persist_rag_ingest.py` — `save_summary` вызывается
с `skip_rag_ingest=False`. Плюс `test_node_grounding_gate.py`/
`test_steering_lazy_ingest.py`/`test_lazy_ground_map_summarize.py`/
`test_graph_integrity.py`/`test_curriculum_topology.py` — без регрессий.
Суммарно: **94 passed**. `black`/`isort`/`flake8` чисты, `npm run build`
чист.

## Topic Q&A: вопрос физически невозможен вместо "запрещён текстом"

**Запрос**: серия live-тестов ноды в Topic Q&A подряд показывала один и тот
же симптом — модель заканчивает ответ техническим вопросом, хотя режим
"эксперт-консультант" явно запрещает квизовать. Каждый раз фикс на уровне
текста промпта переставал работать на следующем прогоне на новой ноде/сессии.

**Диагноз (пять независимых каналов, найдены по очереди через live-тесты,
не одним аудитом)**:
1. Шесть отдельных "MANDATORY checkpoint"-фрагментов в общем dense-промпте
   (`LECTURE_MODE_STRUCTURE_RULES`, `STRUCTURED_LECTURE_FIELD_RULES` правило
   8, `NO_CLOSING_QUESTIONNAIRES`, `DENSE_FUNDAMENTALS_BLOCK`,
   `LECTURE_GAP_STEERING_RULES`, `DENSE_LECTURE_INTERACTION_MODE`) — одна
   вставка-override в конце промпта не может надёжно перекрыть несколько
   независимо сформулированных MANDATORY-инструкций в начале/середине.
2. Даже после того как все шесть текстовых фрагментов стали
   axis-aware, `checkpoint_prompt`/`follow_up_question` содержали
   безусловное `Field(description=...)` ("the ONLY field for...", "MUST be
   named here") — Gemini structured output учитывает описание поля схемы
   как отдельный, не менее весомый канал инструкции, независимый от
   `system_instruction`.
3. `ChatSessionManager.resolve_for_model` сравнивал только `model_name` —
   переключение `interaction_axis` для того же chat label (например
   `node_deep_dive/dense_material`) переиспользовало старую сессию и
   реплеило старые `api_turns` (сырые JSON-ответы с заполненным
   `checkpoint_prompt` из lecture_self_check) как историю в новый Gemini
   chat — модель имитировала свой же прошлый паттерн вопреки текущей
   инструкции.
4. `engine.py::_invoke_tutor` для свободного диалога (не dense_material)
   независимо маршрутизировал на `DeepDiveExplainContract`/
   `DeepDiveTutorContract` — те же безусловные Field-описания, ещё один
   необследованный канал той же природы.
5. Даже когда `checkpoint_prompt`/`follow_up_question` были обнулены
   текстом для всех topic_qna-ходов без исключения, это сломало
   self-check-цикл: после незачтённого ответа на самопроверку тьютору
   было физически некуда положить повторный/уточняющий вопрос.

**Решение — кодирование инварианта в схеме, а не в тексте инструкции**:
там, где для данной оси/состояния вопрос заведомо не нужен, поле вопроса
просто отсутствует в Pydantic-схеме, которую видит Gemini structured
output — модель не может туда написать, а не "должна не писать".

- `tutor.py`: общие поля вынесены в базовые классы
  (`_DenseLectureFieldsBase`, `_ExplainFieldsBase`, `_TutorFieldsBase`);
  `checkpoint_prompt`/`follow_up_question` объявлены ТОЛЬКО в
  lecture_self_check-вариантах (`StructuredLectureResponse`,
  `DeepDiveExplainContract`, `DeepDiveTutorContract`). Параллельные
  `TopicQnaLectureResponse`, `TopicQnaExplainContract`,
  `TopicQnaTutorContract` — те же базовые поля, без поля вопроса вовсе.
- `generate_dense_material`/`engine.py::_invoke_tutor` выбирают
  `response_schema` по `interaction_axis`, а не только компонуют другой
  system prompt — `structured_lecture_to_dense` принимает оба типа через
  `getattr(out, "checkpoint_prompt", "")`.
- Три случая, где вопрос ВСЁ ЖЕ нужен в topic_qna (`engine.py::
  _resolve_tutor_response_schema`, вынесена отдельной тестируемой
  функцией): (а) сам ход `[mode:self_check]` (SELF_CHECK_MODE_PROMPT
  формулирует контрольный вопрос); (б) любой ход, пока self-check ещё не
  отвечен (`has_pending_self_check` — `stored_pending_evaluation_id`,
  покрывает и «Переформулируй вопрос», и любой другой control-chip,
  случившийся в открытой самопроверке, без хардкода по конкретному
  интенту); (в) self-check отвечен, но НЕ зачтён
  (`last_eval_directive` не терминальный, `PROBE_NEXT_LAYER:*`) — цикл
  самопроверки продолжает спрашивать до зачёта, как в lecture_self_check.
  Зачтено (терминальная директива) — `TopicQnaTutorContract` без вопроса,
  автопереход к следующему модулю остаётся на Host-owned
  `ready_for_transition`/`suggested_next_step` (не зависят от оси).
- `sub_concept_eval.py`: убран отдельный безусловный skip для
  `topic_qna` — теперь работает та же общая логика "нет pending → skip",
  что и для lecture_self_check; она достаточна сама по себе именно
  потому, что обычный Q&A в topic_qna больше не может выставить pending
  (нет поля вопроса).
- `ChatSessionManager.resolve_for_model`/`create_new_session`: новое поле
  `StoredChatSession.interaction_axis`; смена оси для того же label
  теперь начинает новую сессию (Summary handoff), как и смена модели —
  старые api_turns другой оси не реплеятся как history.
- Стриминг: `TopicQnaTutorContract`/`TopicQnaExplainContract` не совпадали
  ни с одной веткой `schema_name`-диспетчера в
  `ChatSessionManager.send_chat_message_stream` (тот же regex-based
  `partial_json_string_field_state`/`JsonFieldStreamFilter`, что и
  everywhere else) — в чат летел сырой растущий JSON вместо текста по
  словам. Добавлены `TOPIC_QNA_TUTOR_STREAM_FIELDS`/
  `TOPIC_QNA_EXPLAIN_STREAM_FIELDS` (`gemini_json_stream.py`) и
  соответствующие ветки диспетчера.

Проверено: `test_interaction_axis_topic_qna_wiring.py` вырос до **18
тестов** — прямые unit-тесты `_resolve_tutor_response_schema` по всем
комбинациям (зачтено / не зачтено / self_check-триггер / pending
control-chip / прочие skip-причины / lecture_self_check-контроль),
`resolve_for_model` на смену оси, схемы без поля вопроса напрямую через
`model_fields`. Плюс `test_tutor_dialogue_stream.py` (стрим-фильтры) и
весь домен `grounding` — **355+ passed** без регрессий (кроме 3
предсуществующих flaky-тестов на живую модель, не связанных с этой
правкой). `black`/`isort`/`flake8` чисты.
