# Модуль 1 — Curriculum Generator

Генератор направленного учебного графа (DAG) под инженерную цель. **Не** использует Light RAG, историю чатов и Consensus.

**Карта всех пайплайнов Tutor (create / expand / нода):** [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md).

## API

```http
POST /api/v1/curriculum/generate
Content-Type: application/json

{
  "target_goal": "Проектирование отказоустойчивых распределённых хранилищ на Go",
  "user_level": "Intermediate/Advanced",
  "depth_level": "Deep Mechanics"
}
```

`depth_level`: `Overview` | `Standard` | `Deep Mechanics`

## Поток Targeted Node Grounding (по умолчанию)

1. **Model-First** (`model_first_flash.py`): Flash строит **ветвящийся** DAG (не линейная цепочка), ≥8–12 нод, пустой реестр. Валидатор: `validate_dag_branching`.
2. **Risk** (`node_risk_classification.py`): Lite → `BASE` (без поиска) | `DEEP` (нужен RAG).
3. **Targeted Search** (`targeted_node_search.py`): **Exa** (whitelist blogs + highlights) → SearXNG fallback / SS / arXiv / (опц.) Consensus **только для DEEP**; запросы из терминологии ноды; Lite batch **strict** (без fallback approve).
4. **Grounding** (`targeted_node_grounding.py`): источники к DEEP-нодам; при пустом поиске — `unverified_deep`, граф не сжимается.

`CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED=false` + `CURRICULUM_SEARCH_FIRST_ENABLED=true` — legacy Search-First.

## Поток Search-First (legacy)

1. **Источники** (`collect_sources_by_policy` — тот же код, что smoke `--with-collect`):
   - **practical_only** — архив → **SearXNG** (primary) → (опц.) Gemini web / API grounding → Lite site-suggest
   - **academic_only** — **Semantic Scholar** → arXiv → (опц.) Consensus Playwright если `CURRICULUM_USE_V08_CONSENSUS=true` и API пуст
   - **hybrid** — практика + академика (последовательно)

**source_policy** в UI (create / expand): `practical_only` | `hybrid` | `academic_only` → `POST /curriculum/create` / `expand` → worker → `generate_curriculum_graph` / `expand_curriculum`.

2. **Выдержки** (`source_material_pipeline.py`): открытый сбор → Lite-валидация → архив → Summarizer → LanceDB.
3. **Flash** (`search_first_flash.py`): маршрут **вокруг материалов** — ноды с `source_ref` (url + `relevant_extracts`) и `node_curriculum_breakdown` (key_concepts, architectural_focus). Слои: foundation | advanced | sota.
4. **Лекция** (`[mode:lecture]`): Lite — план урока + выдержки (`format_node_lesson_plan_for_lecture`).

Отключить Search-First: `CURRICULUM_SEARCH_FIRST_ENABLED=false` → legacy Reasoner + Lite `whitelist_sources`.

`CURRICULUM_GEMINI_GROUNDING_ENABLED=false` — без API Search tool (рекомендуется при 429/404).

`CURRICULUM_GEMINI_WEB_HARVEST_ENABLED=false` — не открывать gemini.google.com через Playwright (тогда только архив + SearXNG fallback).

Перед первым web-harvest: `python -m knowledge_engine.main browser-login` (persistent `.browser_state`).

Модели Search grounding (env): `GEMINI_GROUNDING_MODEL`, `CURRICULUM_GEMINI_GROUNDING_MODEL`, `CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS`. Не использовать Gemini 3.x для Google Search tool (Search grounding 0/0 в free tier).

**Диагностика tooling:** `make check-gemini-grounding` или `python -m knowledge_engine.scripts.check_gemini_grounding` — один запрос с `google_search` на каждую модель из chain; `--list-models` — что видит ключ в API; `--all-candidates --compare-plain` — расширенный прогон. JSON: `knowledge_engine/.runs/gemini_grounding_probe.json`.

## Расширение графа (expand)

`POST /api/v1/curriculum/expand` → `expand_curriculum`:

1. **Lite** — `expansion_vector` (текст направления, без нод).
2. **Сбор** — `collect_sources_for_expand(vector, source_policy)` → SearXNG / SS / arXiv → Summarizer → LanceDB.
3. **Flash** — `new_nodes`, `new_edges` на объединённом пуле выдержек.
4. **Merge** — существующие ноды и прогресс в `session_store` не сбрасываются.

Legacy: `knowledge_engine.services.curriculum_service.expand_curriculum(...)`.

## Ответ (legacy enrich)

**После DAG (legacy):** этап `whitelist_sources` (Lite):

1. `curriculum_sources_registry` — библиотека курса (8–15 источников Whitelist, `source_id`: `src_1`, …).
2. Для каждой ноды: `mapped_source_ids` (1–3 ID из реестра, не пусто), `learning_goal`, `learning_materials`, `learning_resources`, `primary_source_id`.
3. `route_sources` — зеркало реестра с URL для UI [src_1] / legacy [S1].

Валидация: каждый `mapped_source_id` должен существовать в `curriculum_sources_registry` (`source_registry.validate_curriculum_source_links`).

## Код

| Компонент | Путь |
|-----------|------|
| Схемы | `knowledge_engine/src/curriculum/schemas.py` |
| Валидатор DAG | `knowledge_engine/src/curriculum/dag_validator.py` |
| Reasoner | `knowledge_engine/src/curriculum/generator.py` |
| Реестр + Lite enrich | `knowledge_engine/src/curriculum/source_enrichment.py`, `source_registry.py` |
| Пул источников (без дублей) | `knowledge_engine/docs/SOURCE_POOL.md`, `ARCHITECTURE_DEDUP.md` |
| Search-First | `search_prestep.py`, `search_first_flash.py` |
| HTTP | `knowledge_engine/api/routes/curriculum.py` |

После генерации выполняется проверка: существующие `prerequisites`, отсутствие циклов, наличие слоёв `foundation` и `sota`. При ошибке — один повторный вызов Reasoner с подсказкой валидатора.

## Пример curl

```bash
curl -s -X POST "http://127.0.0.1:8765/api/v1/curriculum/generate" \
  -H "Content-Type: application/json" \
  -d '{"target_goal":"Консенсус в распределённых системах для backend на Go"}' | jq .
```
