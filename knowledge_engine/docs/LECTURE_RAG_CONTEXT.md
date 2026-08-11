# Lecture RAG — контекст для плотной лекции (dense_material)

Сбор локального материала перед `generate_dense_material()` / блоком `=== НАЧАЛО МАТЕРИАЛА ===` в промпте тьютора.

Связанные документы: [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md), [RAG_GATEWAY_MODULE_3.md](RAG_GATEWAY_MODULE_3.md).

## Когда вызывается

- `POST /node/chat` с `[mode:lecture]` или явным запросом плотной лекции.
- `engine.run_node_deep_dive` → `needs_dense` → `retrieve_lecture_rag_context()` → (условно) `fetch_verified_external_sources()` → `generate_dense_material()`.

Перед **LECTURE_SEARCH** (Exa / Semantic Scholar / Consensus): если после RAG достаточно локальных фрагментов (`local_sources_count >= LECTURE_MIN_LOCAL_SOURCES`, по умолчанию 3) или есть pinned whitelist — первичный внешний поиск пропускается (`[LECTURE_PIPELINE] External search bypassed…`). Жёсткое отключение: `LECTURE_EXTERNAL_SEARCH_ENABLED=false`. Запрос модели `search_external_materials` после лекции по-прежнему вызывает внешний поиск без этого guardrail.

**Не вызывается** на каждый turn диалога (`dialogue_feedback`) — там только `memory.rag_profile_compressed` (из init) + sliding window + fact manifest.

## Пайплайн (с 2026-07)

```text
whitelist foundation (pinned, без MMR)
  +
пул кандидатов (до LECTURE_RAG_CANDIDATE_LIMIT):
  route URL → LanceDB summaries
  registry stubs
  LanceDB hybrid_search (document_summaries)
  knowledge_nodes hybrid
  LightRAG vector_search (profile + fact)
        ↓
Cross-Encoder rerank (запрос ↔ plain текст чанка)
        ↓
отсечка LECTURE_RAG_CE_MIN_SCORE
        ↓
MMR (λ · relevance − (1−λ) · max sim к уже выбранным)
        ↓
Top LECTURE_RAG_MMR_TOP_K → склейка с pinned
```

### Step 1 — Candidate retrieval

Fine `rag_chunks` ranking uses `final_score = vector_similarity × trust_score`
(`metadata.trust_score`, default `1.0` for legacy rows). Trust scores come from
OpenAlex at ingest (`OPENALEX_TRUST_*`) for **any** work with a DOI or arXiv id
(Consensus / Semantic Scholar / arXiv). Vendor docs stay at `1.0`. URLs without
DOI/arXiv get a soft fallback (`0.3`), not a free pass.

### Hybrid academic rerank (optional)

When `ACADEMIC_RERANK_ENABLED=true`, candidates are **pre-sorted** before CE/MMR with:

```text
score = α·relevance_sim + β·trust_score + γ·log1p(cites)/log1p(C_sat) + δ·recency
```

Defaults: `ACADEMIC_RERANK_WEIGHTS=0.45,0.25,0.20,0.10`, `C_sat=40`,
recency half-life `ACADEMIC_RERANK_RECENCY_HALF_LIFE_YEARS=6`. Flag defaults to
**false** so production behavior stays `sim × trust` until explicitly enabled.

### Academic relaxation cascade (curriculum search)

If academic hits &lt; `ACADEMIC_RELAXATION_MIN_HITS` (default 3), arXiv precision
params are softened in levels:

| Level | Name | Effect |
|-------|------|--------|
| 0 | Strict | Year window + min trust/citations gates |
| 1 | Soft date & cites | Drop/widen dates; lower citation floor |
| 2 | Broad relevance | Drop `cat:` / excludes; prefer semantic relevance |

Implemented in `src/retrieval/academic_rerank.py`, wired from
`academic_source_fetch` and lecture RAG pre-sort.

Hard cutoff (`RAG_TRUST_HARD_CUTOFF`): drop chunks with `trust < 0.2` **and**
`vector_similarity < 0.85`. Applied as **early exit** immediately after LanceDB
vector hits are scored (`search_rag_chunk_rows`) and again on the candidate pool
**before** CE / cross-attention / MMR — never after Map-Lite or prompt stitch.
Surviving chunks are ordered by trust (desc) before `[R#]` assignment.
Dialog `chat_history` is untouched — trust markers appear only in RAG context blocks.

Note: blog/academic **Map-Reduce ingest** summarizes article windows (not vector
hits). Hard cutoff gates **retrieval → lecture context**, not window splitting.

| Источник | Модуль | Лимит |
|----------|--------|--------|
| Hybrid конспекты | `VectorStore.hybrid_search` | `LECTURE_RAG_CANDIDATE_LIMIT` (15) |
| Конспекты по URL маршрута | `fetch_summaries_by_urls` | min(urls, limit) |
| Knowledge nodes | `hybrid_search_nodes` | `LECTURE_RAG_KNODE_CANDIDATE_LIMIT` (4) |
| LightRAG | `LightRAG.vector_search` | same pool limit |
| Registry / archive stubs | без LanceDB | по маршруту |

Первичный поиск — **вектор + hybrid FTS** (LanceDB), не отдельный BM25 pipeline.

### Step 2 — Cross-Encoder rerank

- Критерий: **фокус пользователя** (`user_query`), если пусто — search query ноды.
- Вызов: `score_relevance_pairs()` из `src/rag_gateway/cross_encoder.py` (тот же стек, что Directional RAG Gateway).
- Модель: `RAG_CROSS_ENCODER_MODEL` (`BAAI/bge-reranker-v2-m3`) или cosine fallback на Ollama `EMBED_MODEL`.
- Отсечка шума: `LECTURE_RAG_CE_MIN_SCORE`.

### Step 3 — MMR

- Реализация: greedy MMR в `services/lecture_context_rerank.py` (не библиотека).
- Relevance: scores CE (0…1).
- Similarity между чанками: косинус эмбеддингов Ollama (не cross-encoder между чанками).
- λ: `LECTURE_RAG_MMR_LAMBDA` (default 0.62) — выше → ближе к чистой релевантности, ниже → больше разнообразия.

### Step 4 — Склейка

- Pinned whitelist foundation **не** проходит CE/MMR.
- Итог: `"\n\n---\n\n".join(chunks)` → `build_lecture_generation_payload()`.

## Отказоустойчивость

| Ситуация | Поведение |
|----------|-----------|
| Таймаут collect (`LECTURE_RAG_COLLECT_TIMEOUT_SEC`) | minimal fallback: whitelist foundation + route URLs, без LanceDB hybrid |
| Таймаут LightRAG (`LECTURE_RAG_LIGHT_TIMEOUT_SEC`) | пул без vector hits |
| Таймаут CE/MMR (`LECTURE_RAG_RERANK_TIMEOUT_SEC`) | `fallback_dedupe_candidates` — URL + exact-text, лимит как legacy |
| Ошибка всего блока rerank | полный fallback: сбор пула + legacy dedupe |
| CE недоступен | уже внутри `cross_encoder.py` → Ollama cosine |

Тяжёлые операции: `run_blocking_timed` + пулы `blocking_pools` (`pool_rag_io`, `pool_rag_ce`) для collect/CE (без глобального UMA-lock на весь collect — иначе таймаут оставляет «зомби»-поток).

## Логи (trace)

| Префикс | Смысл |
|---------|--------|
| `LECTURE_RAG ▶ collect` / `collect ✓` | старт / конец сбора кандидатов |
| `LECTURE_RAG collect timeout` | превышен `LECTURE_RAG_COLLECT_TIMEOUT_SEC` |
| `LECTURE_RAG light_rag timeout` | превышен `LECTURE_RAG_LIGHT_TIMEOUT_SEC` |
| `LECTURE_RAG pool ▶` | размер пула до rerank |
| `LECTURE_RAG rerank ▶` | старт CE |
| `LECTURE_RAG ce_filter` / `ce_drop` | прошли / отсечены по score |
| `LECTURE_RAG mmr ✓` / `mmr_pick #N` | финальный набор |
| `LECTURE_RAG rerank/mmr fallback` | откат |
| `LECTURE_RAG full fallback` | откат всего retrieve |

## Конфиг (.env)

| Переменная | Default | Описание |
|------------|---------|----------|
| `LECTURE_RAG_CANDIDATE_LIMIT` | 15 | Первичный пул |
| `LECTURE_RAG_MMR_TOP_K` | 5 | Чанков после MMR |
| `LECTURE_RAG_CE_MIN_SCORE` | 0.38 | Мин. CE score |
| `LECTURE_RAG_MMR_LAMBDA` | 0.62 | Баланс rel / diversity |
| `LECTURE_RAG_RERANK_TIMEOUT_SEC` | 60 | Таймаут rerank+MMR |
| `LECTURE_RAG_COLLECT_TIMEOUT_SEC` | 90 | Таймаут LanceDB/Ollama collect (в thread) |
| `LECTURE_RAG_LIGHT_TIMEOUT_SEC` | 45 | Таймаут LightRAG vector_search |
| `LECTURE_RAG_KNODE_CANDIDATE_LIMIT` | 4 | Knowledge nodes в пуле |
| `LECTURE_RAG_TOP_K` | 3 | Legacy лимит при full fallback |
| `RAG_CROSS_ENCODER_MODEL` | bge-reranker-v2-m3 | CE (общий с Gateway) |

## Код

| Файл | Роль |
|------|------|
| `services/lecture_rag_context.py` | Сбор пула, async orchestration |
| `services/lecture_context_rerank.py` | CE gate + MMR + fallback dedupe |
| `src/rag_gateway/cross_encoder.py` | CE / embedding fallback |
| `services/llm_markdown_service.py` | HTML для UI (не lecture pool) |

## Отличие от Directional RAG (init ноды)

| | Init `rag_profile` | Lecture `retrieve_lecture_rag_context` |
|--|-------------------|----------------------------------------|
| Когда | `user_action=init` | dense_material |
| Источник | LightRAG facts/profile | LanceDB summaries + route + LightRAG |
| CE | 3 search directions | один focus query |
| MMR | нет (text overlap dedup) | yes |
| В промпте | `layer_1_compressed_rag_profile` | `=== НАЧАЛО МАТЕРИАЛА ===` |
