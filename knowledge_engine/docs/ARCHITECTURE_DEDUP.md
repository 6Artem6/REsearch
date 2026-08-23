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

---

При новой фиче: сначала найти строку в этой таблице; если нет — расширить **один** модуль и обновить этот файл.
