# Knowledge Engine: быстрый вход в проект

**Stack:** Python 3.10+ / FastAPI / LangGraph / Pydantic v2 / LanceDB / Redis / Ollama & Gemini / React 18 / XYFlow / esbuild

## Quick Commands

- **Test:** `PYTHONPATH=. ./.venv/bin/python -m pytest knowledge_engine/tests`
- **Lint:** `make check`
- **Config source of truth:** `knowledge_engine/config.py`

Перед изменениями сначала прочитай документацию в указанном порядке. Не пытайся
загружать все документы сразу: начни с обзорного маршрута, затем открой только
файлы, относящиеся к текущей задаче.

## 1. Главные точки входа (Core)

1. [README.md](README.md) — назначение продукта и основные точки входа.
2. [Индекс документации](knowledge_engine/docs/INDEX.md) — актуальность
   документов, известные пробелы и навигация по подсистемам.
3. [Dev Runbook](knowledge_engine/docs/DEV_RUNBOOK.md) — локальный запуск,
   worker, Redis и диагностика.

Для реализации конкретных фич сразу переходи к Разделу 4 (Маршруты).

## 2. Документы по области задачи

### Tutor, prompt и UI

- [Карта pipeline](knowledge_engine/docs/TUTOR_PIPELINES.md)
- [Node Deep Dive](knowledge_engine/docs/NODE_DEEP_DIVE_MODULE_2.md)
- [LLM-контракты](knowledge_engine/docs/LLM_CONTRACTS.md)
- [Tutor prompt и UI text](knowledge_engine/docs/TUTOR_PROMPT_AND_UI_TEXT.md)
- [Skill Tree UI](knowledge_engine/docs/SKILL_TREE_UI.md)
- [Lecture RAG context](knowledge_engine/docs/LECTURE_RAG_CONTEXT.md)
- [Directional RAG Gateway](knowledge_engine/docs/RAG_GATEWAY_MODULE_3.md)

### Search, sources и ingest

- [Curriculum DAG](knowledge_engine/docs/CURRICULUM_MODULE_1.md)
- [Exa search](knowledge_engine/docs/EXA_SEARCH.md)
- [Source pool](knowledge_engine/docs/SOURCE_POOL.md)
- [Academic и Consensus](knowledge_engine/docs/ACADEMIC_AND_CONSENSUS.md)
- [Consensus Direct API](knowledge_engine/docs/CONSENSUS_API_DIRECT.md)
- [Article ETL и figures](knowledge_engine/docs/ARTICLE_ETL_AND_FIGURE_EXTRACTION.md)
- [Article diagrams](knowledge_engine/docs/ARTICLE_DIAGRAMS.md)

### Runtime и эксплуатация

- [Каталог скриптов](knowledge_engine/docs/SCRIPTS.md)
- [Environment variables](knowledge_engine/docs/ENV_VARIABLES.md)
- [Docker layout](knowledge_engine/docs/DOCKER_LAYOUT.md)
- [Performance](knowledge_engine/docs/PERFORMANCE.md)
- [Architecture deduplication](knowledge_engine/docs/ARCHITECTURE_DEDUP.md)

### Research pipeline

- [v0.8 Consensus Agent](knowledge_engine/docs/V0_8_CONSENSUS_AGENT.md)
- [v0.8 snapshot](knowledge_engine/docs/V0_8_SNAPSHOT.md)
- [Search horizons](knowledge_engine/docs/SEARCH_HORIZONS.md)

## 3. Актуальность и ограничения

- `knowledge_engine/docs/INDEX.md` — источник истины о том, какой документ
  актуален, частичен или устарел.
- `V0_3_ARCHITECTURE.md`, `V0_6_CURRENT_SOLUTION.md`,
  `V0_7_ARCHITECTURE.md`, `FRUGAL_ROUTING.md` и ранние фазы
  `TUTOR_LANGGRAPH_MIGRATION.md` являются legacy/историческими материалами.
- Документация ускоряет навигацию, но перед изменением поведения проверяй
  фактические call sites, `knowledge_engine/config.py`, Pydantic schemas и тесты.
- Не копируй старые defaults из docs без сверки с `config.py` и `.env.example`.
- Destructive-скрипты запускай только после чтения предупреждений в
  [SCRIPTS.md](knowledge_engine/docs/SCRIPTS.md); сначала используй dry-run.

## 4. Минимальный маршрут по типу задачи

- Tutor/LLM: `NODE_DEEP_DIVE_MODULE_2` → `TUTOR_PROMPT_AND_UI_TEXT` →
  `LLM_CONTRACTS`.
- RAG/embeddings: `LECTURE_RAG_CONTEXT` → `RAG_GATEWAY_MODULE_3` →
  `ARCHITECTURE_DEDUP`.
- Search/ingest: `CURRICULUM_MODULE_1` → `SOURCE_POOL` → `EXA_SEARCH` →
  `ACADEMIC_AND_CONSENSUS`.
- UI: `TUTOR_PIPELINES` → `SKILL_TREE_UI`.
- Dev/operations: `DEV_RUNBOOK` → `DOCKER_LAYOUT` → `SCRIPTS`.

## Rules & Guidelines
Перед внесением изменений ОБЯЗАТЕЛЬНО ознакомься с правилами проекта:
- **Общие правила и гайдлайны:** прочитай файлы в `.cursor/rules/*.mdc` (или `.md`).
- **Стилистика кода:** соблюдай запреты из `.cursor/rules/`.
