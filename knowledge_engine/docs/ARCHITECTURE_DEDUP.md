# Верхнеуровневые точки входа (без дублирования)

Краткий реестр: **один модуль — одна ответственность**. Изолированные треки (v0.4 graph nodes, node deep-dive) не сливать без необходимости.

## Gemini API (stateless)

| Что | Модуль |
|-----|--------|
| Клиент, RPM, retry, model chain | `services/gemini_stateless.py` |
| Lite / Flash с пином модели | `src/analytics/gemini_v07.py` (`run_gemini_lite_structured`, `run_gemini_flash_*`) |
| Reasoner / curriculum Flash | `run_gemini_structured_with_chain` из stateless (не второй клиент) |
| Legacy re-export v0.3 | `services/gemini_session.py` → только stateless |

**Не добавлять:** прямые `google.genai` вызовы вне stateless / search grounding.

## Playwright (браузер)

| Что | Модуль |
|-----|--------|
| Persistent profile launch | `services/search/playwright_launch.py` |
| Gemini web чат | `services/ai_dialogue/gemini_session.py` |
| Consensus.app | `src/retrieval/consensus_session.py` (+ shared acquire/release) |
| URL fetch в discovery | `services/search/browser_search.py` |

Один `user_data_dir` на процесс — **не** параллельный Consensus + Gemini web (см. `collect_sources_by_policy` sequential hybrid).

## Consensus query prep

| Что | Модуль |
|-----|--------|
| preserved terms, SearXNG grounding payload | `src/processors/consensus_query_prep.py` |
| Lite sanitize → academic EN | `validator.sanitize_query_for_consensus` |
| Anchor «только user query» | `consensus_query_prep.consensus_sanitize_anchor` |
| v0.8 orchestrator | `local_orchestrator.py` |
| Curriculum academic harvest | `curriculum_v08_harvest.harvest_curriculum_sources_v08` |

**Не копировать** `_consensus_sanitize_anchor` в других файлах.

## Curriculum: сбор источников

| Что | Модуль |
|-----|--------|
| Политики hybrid / practical / academic | `source_policy.py` + `collect_sources_by_policy` |
| Практические блоги (архив → web → expand → fallback) | `source_material_pipeline.collect_practical_blog_hits` |
| Академика (Consensus) | `collect_academic_source_hits` → v08 harvest |
| Единая точка для generator | `search_prestep.collect_curriculum_source_hits` |
| Приоритетные engineering `site:` | `curriculum_search_sites.CURRICULUM_PRIORITY_ENGINEERING_SITES` |
| Пул источников + Lite + архив | см. [SOURCE_POOL.md](SOURCE_POOL.md) |

Удалён legacy: `_collect_fast_source_hits`, `_collect_consensus_source_hits`, `run_hybrid_source_collection` (дублировали policy pipeline).

## Поиск (SearXNG / registry)

| Что | Модуль |
|-----|--------|
| multi_search | `services/search/registry.py` `default_registry()` |
| Discovery graph | `discovery_collect.py` + `discovery_trust.py` |
| Curriculum SearXNG fallback | `_collect_whitelist_blog_hits` / `source_discovery_expand` |

## Summarizer + LanceDB (статьи)

| Что | Модуль |
|-----|--------|
| 7B summarizer | `services/summarizer.summarize_article` |
| Curriculum blogs ingest | `source_material_pipeline._ingest_blog_url` |
| v0.8 papers | `curriculum_v08_harvest` (отдельный трек papers) |

При добавлении tier для summarizer — расширять `_BLOG_SOURCE_TIERS` в `source_material_pipeline.py`.

## UI text repair

| Что | Модуль |
|-----|--------|
| Backend (escapes, mermaid, trade-off JSON→markdown) | `web/llm_text_repair.py` (`repair_structured_analysis_json`) |
| Skill Tree bundle | `web/static/skill-tree/llmTextRepair.js` (параллель, не импорт Python) |
| Документация | [TUTOR_PROMPT_AND_UI_TEXT.md](TUTOR_PROMPT_AND_UI_TEXT.md) |

## Node Deep-Dive: lecture context

| Что | Модуль |
|-----|--------|
| Сбор + CE/MMR | `services/lecture_rag_context.py`, `services/lecture_context_rerank.py` |
| Cross-Encoder (shared) | `src/rag_gateway/cross_encoder.py` |
| Документация | [LECTURE_RAG_CONTEXT.md](LECTURE_RAG_CONTEXT.md) |

## Directional RAG (init ноды)

| Что | Модуль |
|-----|--------|
| CE + dedup фактов | `src/rag_gateway/gateway.py` |
| Процесс | только **worker** (`WorkJobKind.RAG_GATEWAY` / `NODE_DEEP_DIVE`); API — очередь |
| Документация | [RAG_GATEWAY_MODULE_3.md](RAG_GATEWAY_MODULE_3.md) |

## Local ML weights (BGE-M3 / Cross-Encoder)

| Что | Модуль |
|-----|--------|
| Role guard | `services/ml_runtime.py` (`KE_PROCESS_ROLE=api\|worker`) |
| Bi-encoder load | `services/search/bge_m3_embed.py` |
| CE load | `src/rag_gateway/cross_encoder.py` |
| SSE worker→API | `services/job_stream.py` |

## Pre-MAP Dedup: дедупликация источников до MAP+REDUCE

Точка входа: `src/deduplication/pre_map_deduplicator.py::deduplicate_before_map_reduce()`.
Отличается от `services/deduplication/entity_consensus_engine.py` (тот дедупит
уже извлечённые ФАКТЫ внутри одного REDUCE-батча) — этот модуль дедупит целые
ИСТОЧНИКИ (URL/файл), до того как MAP вообще запустился, чтобы не тратить
дорогой MAP+REDUCE на источник, который уже является дублем другого.

Работает **строго внутри одного батча кандидатов**, переданного в один вызов —
кросс-ранового сравнения с уже сохранёнными узлами (LanceDB / graph index) на
сегодня **нет** (см. «Известное ограничение» ниже).

| Что | Модуль |
|-----|--------|
| Оркестрация 4 шагов, Union-Find, TPM-guard батчинг | `src/deduplication/pre_map_deduplicator.py` |
| Изолированная дедупликация кода (README/дерево/AST) | `src/deduplication/code_deduplicator.py` |
| Схема Bulk Gate contract (Array of Pairs, не dict) | `schemas/llm_contracts/pre_map_dedup.py` |
| Интеграция в инджест блогов | `src/curriculum/source_material_pipeline.py` (`_pre_map_dedup_batch_items`) |
| `alias_of` на исходящем хите | `src/curriculum/schemas.py::CurriculumSearchHit` |

### Пайплайн (per batch)

1. **Context Extraction** — по каждому кандидату: TEXT → абзацы
   (`_extract_paragraphs`); CODE → AST-сигнатуры (`_ast_semantic_extracts`).
   Затем **Triage одним Group Batching вызовом** на весь батч
   (`_flash_lite_triage_core_units_batch` — все кандидаты сворачиваются в ОДИН
   `InputPaperJson`, один Flash Lite вызов вместо N; откат на поштучные вызовы
   при переразмеренном payload/ошибке) классифицирует юниты CORE/CONTEXT/DROP
   через существующий `paper_structure_analyzer.PaperStructureAnalyzer`
   (не локальная эвристика). TEXT CORE-юниты идут через MMR
   (`_mmr_top_by_centroid` → `greedy_mmr_select`); CODE CORE-юниты — через
   Head-3/Tail-3 (см. ниже).
2. **BGE-кластеризация** (только TEXT, 0 LLM-вызовов) — Union-Find по
   косинусу пуловых отпечатков ≥ `PRE_MAP_DEDUP_COSINE_THRESHOLD` →
   suspect groups. Код никогда не кластеризуется по BGE — косинус по
   AST-сигнатурам не надёжный сигнал «тот же алгоритм, другой язык».
3. **Step 3a — Bulk Gate (TEXT)**: suspect-группы, TPM-guard
   (`PRE_MAP_DEDUP_BULK_GATE_MAX_TPM`) с жадным разбиением на суб-батчи
   (`_pack_bulk_gate_sub_batches`), общий системный промпт-компаратор.
   **Step 3b — изолированная дедупликация кода**: все код-кандидаты идут
   ОТДЕЛЬНЫМ, выделенным вызовом через `code_deduplicator.py` вместо Bulk
   Gate — см. ниже, почему это отдельный модуль.
4. **Canonical Pooling** — `_sanitize_canonical_map` чистит галлюцинированные
   id/self-alias/циклы; кандидат либо CANONICAL, либо `ALIAS -> canonical_id`.

Fail-open на каждой границе (BGE/Flash Lite/GitHub): ошибка не роняет батч —
худший случай, все кандидаты остаются CANONICAL, как если бы модуля не было.

### code_deduplicator.py: почему код обособлен от Bulk Gate

Плоский AST-only payload (только сигнатуры) не даёт Flash Lite достаточно
сигнала, чтобы уверенно распознать межъязыковой дубль (например, Union-Find/
DSU на C++ и Python) — а обогащение контекстом проекта помогает. Обогащение
per candidate:

- **Two-level README Extraction**: module README (соседняя папка файла) +
  root README репозитория. Источник — GitHub Git Trees API (один рекурсивный
  вызов на репозиторий, `GitHubTreeLoader` из
  `services/article_ingestion/github_tree_loader.py`, токен
  `GITHUB_TOKEN` подставляется автоматически) либо локальный диск
  (`file://`/существующий путь) — что подходит candidate.url. README режется
  на блоки по markdown-заголовкам (`_split_markdown_by_headers`), предварительно
  очищенные `_sanitize_readme_md` (HTML-шапки/бейджи shields.io срезаются до
  нарезки — bs4 как аналог markdownify/html2text/selectolax, ни одна из
  которых не установлена в проекте).
- **In-RAM BGE Concept Anchors**: `ANCHOR_PURPOSE` / `ANCHOR_ALGORITHMS` —
  захардкоженные текстовые якоря (не конфиг), Top-K блоков README на каждый
  якорь отдельно по косинусу (`_top_readme_chunks_by_anchor`), без сети.
- **Сбалансированный tree snippet**: родительская папка файла (файлы + имена
  соседних каталогов первого уровня, без рекурсии внутрь них), width ≤
  `CODE_DEDUP_TREE_WIDTH`, фильтрация `.git/node_modules/build/dist/
  __pycache__/.venv`.
- **Tree-Sitter AST Head-3/Tail-3**: самодостаточный экстрактор
  (`_head_tail_core_with_calltree`, не зависит от Triage-прохода выше) —
  находит ВСЕ функции/методы файла (рекурсивно внутрь классов — CORE-класс
  разворачивается в отдельные методы, а не остаётся одной свёрнутой
  сигнатурой), если их > 6 — берёт первые 3 + последние 3 по порядку в
  файле, с ПОЛНЫМИ телами (все комментарии/docstrings/аннотации типов
  сохранены). Плюс call-tree строкой на каждую CORE-функцию/метод (в т.ч. не
  попавшие в отобранные 6) — компенсирует пропуск средних утилит.
- Всё собирается в один промпт (`=== CODE MODULE {id} ===` блоки), который
  просит Flash Lite сравнивать по **Behavioral Intent & Data Structure
  State**, а не по синтаксису.
- Fail-open на КАЖДОМ шаге отдельно: GitHub недоступен/403/timeout → пустой
  README/дерево, но AST Head-3/Tail-3 всё равно строится и уходит в
  промпт — деградация до сравнения по коду, а не отказ целиком.

### Известное ограничение: Downstream Merge & Re-linking не реализован

`alias_map`/`alias_of` действуют **только внутри одного вызова**
`deduplicate_before_map_reduce()` — `_pre_map_dedup_batch_items` строит
кандидатов исключительно из текущего батча, ничего не сравнивает с уже
persisted узлами. `CurriculumSearchHit.alias_of` — поле только на этой
Pydantic-модели, в LanceDB (`services/vector_store.py`) **не персистится** и
никак не участвует в pre-insert проверках (там только identity-guard по URL,
`doc_id_for_url`). Промоушен/демоушен canonical-статуса и re-link рёбер
графа при повторном инджесте узла из другого батча — **не спроектировано и
не реализовано** ни частично, ни заглушкой; это отдельная будущая задача, а
не текущее поведение.

---

При новой фиче: сначала найти строку в этой таблице; если нет — расширить **один** модуль и обновить этот файл.
