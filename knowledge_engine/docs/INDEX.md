# Каталог документации Knowledge Engine

Индекс `knowledge_engine/docs/` + аудит **код ↔ docs** (2026-08-18).
Корневой обзор продукта: [README.md](../../README.md).

**Когда обновлять этот файл:** новый подсистемный модуль без своего `.md`, смена канонического DAG тьютора, новый LLM-контракт.

---

## Что задокументировано

| Документ | Тема | Соответствие коду |
|----------|------|-------------------|
| [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md) | Карта create / expand / тьютор | Актуальна (§8 = LangGraph) |
| [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md) | Модуль 2: DAG, MemorySaver, single-writer | Актуальна для оркестрации |
| [TUTOR_PROMPT_AND_UI_TEXT.md](TUTOR_PROMPT_AND_UI_TEXT.md) | Контракт тьютора, режимы, BLOCK 1–3 | Актуальна для dialogue/lecture |
| [TUTOR_LANGGRAPH_MIGRATION.md](TUTOR_LANGGRAPH_MIGRATION.md) | ADR миграции | **Устарела как «Phase 0/1»** — граф уже полный |
| [LLM_CONTRACTS.md](LLM_CONTRACTS.md) | Реестр Pydantic Gemini + регламент prompt/contract | Частично: нет `EvaluatorCritiqueContract`, `ArxivQueryParamsContract` |
| [PROMPT_SYSTEM_REFERENCE.md](PROMPT_SYSTEM_REFERENCE.md) | Матрица моделей/контуров, режимы `prompt_factory.py`, реестр prompt-переменных, DEEP MAP/REDUCE/Dedup контракты, Evaluator Bypass guardrails, чек-лист расширения | Актуальна (2026-09-01) |
| [EXA_SEARCH.md](EXA_SEARCH.md) | Exa: plan, rank, domains, `EXA_*` | Актуальна |
| [RAG_PIPELNES.md](RAG_PIPELNES.md) | Lecture vs DEEP: сквозное сравнение обоих RAG-контуров, Mermaid, backfill_margin dedup | Актуальна (2026-08-31) |
| [ACADEMIC_AND_CONSENSUS.md](ACADEMIC_AND_CONSENSUS.md) | Papers: query sanitize, SS/arXiv, Consensus harvest | Актуальна |
| [SOURCE_POOL.md](SOURCE_POOL.md) | Провайдеры discovery | Актуальна |
| [ENV_VARIABLES.md](ENV_VARIABLES.md) | Env-каталог | Краткий; детали Exa → EXA_SEARCH |
| [SCRIPTS.md](SCRIPTS.md) | Все 52 setup / diagnostic / maintenance / legacy CLI из `scripts/` | Актуальна; ключи сверены с кодом |
| [CURRICULUM_MODULE_1.md](CURRICULUM_MODULE_1.md) | Модуль 1 curriculum | Актуальна |
| [RAG_GATEWAY_MODULE_3.md](RAG_GATEWAY_MODULE_3.md) | Directional RAG | Актуальна |
| [LECTURE_RAG_CONTEXT.md](LECTURE_RAG_CONTEXT.md) | Dense: LanceDB → CE → MMR | Актуальна |
| [SKILL_TREE_UI.md](SKILL_TREE_UI.md) | UI / worker / explain SSE | Частично: нет ActionChips / overlay / Gloss |
| [ARTICLE_DIAGRAMS.md](ARTICLE_DIAGRAMS.md) | Mermaid / VLM у ноды | Актуальна |
| [ARTICLE_ETL_AND_FIGURE_EXTRACTION.md](ARTICLE_ETL_AND_FIGURE_EXTRACTION.md) | ETL фигур | Актуальна |
| [CODE_TRIAGE_AND_PRUNING.md](CODE_TRIAGE_AND_PRUNING.md) | Tiered Code Pruner & Code Ingestion: 3 независимых механизма (tiered prune / AST chunker / DOC_TRIAGE) | Актуальна |
| [SEARCH_HORIZONS.md](SEARCH_HORIZONS.md) | SOTA / Infra / Prod | Research-граф |
| [V0_8_CONSENSUS_AGENT.md](V0_8_CONSENSUS_AGENT.md) / [V0_8_SNAPSHOT.md](V0_8_SNAPSHOT.md) | Consensus **research-агент** `/app` | Актуальна; curriculum harvest → ACADEMIC_AND_CONSENSUS |
| [CONSENSUS_API_DIRECT.md](CONSENSUS_API_DIRECT.md) | Direct `paper_search` API | Актуальна (транспорт, не семантика запросов) |
| [ARCHITECTURE_DEDUP.md](ARCHITECTURE_DEDUP.md) | Единые точки сбора + Pre-MAP Dedup (source-level, code_deduplicator.py) | Актуальна |
| [DEV_RUNBOOK.md](DEV_RUNBOOK.md) / [DOCKER_LAYOUT.md](DOCKER_LAYOUT.md) | Запуск | Актуальна |
| [PERFORMANCE.md](PERFORMANCE.md) | Perf | Узкая |
| [FRUGAL_ROUTING.md](FRUGAL_ROUTING.md) | SLM + Gemini research | **Legacy research**, не Skill Tree |
| [V0_3](V0_3_ARCHITECTURE.md) / [V0_6](V0_6_CURRENT_SOLUTION.md) / [V0_7](V0_7_ARCHITECTURE.md) | Исторические графы | Legacy |

---

## Пробелы: реализовано в коде, нет канонического docs

Host-слой тьютора (чипы, overlay, векторы) **не** описан в Module 2 / Prompt docs.

| Подсистема | Код | Что документировать |
|------------|-----|---------------------|
| Control intents / chips | `intent_definitions.py`, `control_intent.py` | Gloss / HOW / MECH / lecture / skip; `[mode:]` / `[action:]` |
| Vector intent router | `vector_intent_router.py`, `db/intent_vectors_schema.py` | LanceDB `intent_vectors`, cosine, `VECTOR_INTENT_*` |
| Prompt factory | `prompt_factory.py` + `*_prompt.py` (gloss, how, mech, deep_*) | Изолированные system prompts поверх compositor |
| Star Task FSM | `star_task_fsm.py` | Overlay L4 `advanced_analysis` / L5–L6 `deep_design` |
| Socratic poles | `socratic_poles.py`, `db/socratic_poles_schema.py` | repulsion / attraction, FACT_* в payload |
| Edge-case lexicon | `edge_case_lexicon.py`, `db/edge_case_vectors_schema.py` | Векторная классификация тезисов |
| Weakness ledger | `context_drift_manager.py` | Cross-node drift, `HostPrep.ledger_block` |
| Host parallel | `host_parallel.py` | Chip + factory + ledger до LLM |
| Evaluator critique | `schemas/llm_contracts/evaluator_critique.py` | Контракт критики gap-eval |
| Resilience | `src/resilience_manager.py` | Деградация intent / LLM 429 / FSM hops |
| Telemetry | `src/telemetry_auditor.py` | `HostTurnTelemetry` (exact / vector / fallback) |
| Graph integrity | `src/graph_validator.py` | Циклы DAG, orphan ids, overlay refs |
| LanceDB pool | `db/lancedb_pool.py` | Пул соединений (не только document_summaries) |
| UI overlay | `ActionChips.js`, `DepthSection.js`, `SubConceptsList.js`, `nodeProgressTypes.js` | Чипы и прогресс в drawer |

Не писать отдельные энциклопедии: достаточно одного хост-дока тьютора + строки в [LLM_CONTRACTS.md](LLM_CONTRACTS.md) и [SKILL_TREE_UI.md](SKILL_TREE_UI.md).

---

## Канонические точки входа по задаче

| Задача | Документ |
|--------|----------|
| Поднять dev | [DEV_RUNBOOK.md](DEV_RUNBOOK.md) |
| Найти CLI / maintenance-скрипт | [SCRIPTS.md](SCRIPTS.md) |
| Понять продукт Skill Tree | [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md) |
| Curriculum DAG | [CURRICULUM_MODULE_1.md](CURRICULUM_MODULE_1.md) |
| Чат тьютора (граф) | [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md) |
| Промпты / JSON тьютора | [TUTOR_PROMPT_AND_UI_TEXT.md](TUTOR_PROMPT_AND_UI_TEXT.md) |
| Модели/контракты/переменные по всем LLM-контурам целиком | [PROMPT_SYSTEM_REFERENCE.md](PROMPT_SYSTEM_REFERENCE.md) |
| Поиск практики | [EXA_SEARCH.md](EXA_SEARCH.md), [SOURCE_POOL.md](SOURCE_POOL.md) |
| Lecture vs DEEP пайплайны целиком (сравнение, Mermaid) | [RAG_PIPELNES.md](RAG_PIPELNES.md) |
| Papers / Consensus / sanitize | [ACADEMIC_AND_CONSENSUS.md](ACADEMIC_AND_CONSENSUS.md) |
| Лекция RAG | [LECTURE_RAG_CONTEXT.md](LECTURE_RAG_CONTEXT.md) |
| Триаж/prune кода перед MAP (tree-sitter, tiered pruner) | [CODE_TRIAGE_AND_PRUNING.md](CODE_TRIAGE_AND_PRUNING.md) |
| Env | [ENV_VARIABLES.md](ENV_VARIABLES.md), `.env.example` |
| Research v0.8 (не курс) | [V0_8_CONSENSUS_AGENT.md](V0_8_CONSENSUS_AGENT.md) |
