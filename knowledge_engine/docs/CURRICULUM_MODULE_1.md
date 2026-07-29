# Модуль 1 — Curriculum Generator

Генератор направленного учебного графа (DAG) под инженерную цель. **Не** использует Light RAG, историю чатов и Consensus.

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

## Поток Search-First (по умолчанию)

## Поток Search-First (по умолчанию)

1. **Источники:** **Consensus** (Playwright + Lite validate + Summarizer + chunks → LanceDB) — приоритет. Дополнение: **SearchRegistry** только URL из **whitelist** (инженерные блоги), Summarizer для страниц.
2. **Выдержки** (`source_material_pipeline.py`): LanceDB Consensus-конспекты → `key_extracts` в JSON для Flash.
3. **Flash** (`search_first_flash.py`): маршрут **вокруг материалов** — ноды с `source_ref` (url + `relevant_extracts`) и `node_curriculum_breakdown` (key_concepts, architectural_focus). Слои: foundation | advanced | sota.
4. **Лекция** (`[mode:lecture]`): Lite — план урока + выдержки (`format_node_lesson_plan_for_lecture`).

Отключить Search-First: `CURRICULUM_SEARCH_FIRST_ENABLED=false` → legacy Reasoner + Lite `whitelist_sources`.

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
| Search-First | `search_prestep.py`, `search_first_flash.py` |
| HTTP | `knowledge_engine/api/routes/curriculum.py` |

После генерации выполняется проверка: существующие `prerequisites`, отсутствие циклов, наличие слоёв `foundation` и `sota`. При ошибке — один повторный вызов Reasoner с подсказкой валидатора.

## Пример curl

```bash
curl -s -X POST "http://127.0.0.1:8765/api/v1/curriculum/generate" \
  -H "Content-Type: application/json" \
  -d '{"target_goal":"Консенсус в распределённых системах для backend на Go"}' | jq .
```
