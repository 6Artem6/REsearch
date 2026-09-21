# Lecture RAG — контекст для плотной лекции (dense_material)

Сбор локального материала перед `generate_dense_material()` / блоком `=== НАЧАЛО МАТЕРИАЛА ===` в промпте тьютора.

Связанные документы: [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md), [RAG_GATEWAY_MODULE_3.md](RAG_GATEWAY_MODULE_3.md).

## Когда вызывается

- `POST /node/chat` с `[mode:lecture]` или явным запросом плотной лекции.
- `engine.run_node_deep_dive` (worker: `NODE_DEEP_DIVE`) → `needs_dense` → `retrieve_lecture_rag_context()` → (условно) `fetch_verified_external_sources()` → `generate_dense_material()`.

Векторный поиск, LightRAG и Cross-Encoder **не вызываются из процесса API**. `POST /node/chat-stream` проксирует SSE с worker (`ke:jobstream:{job_id}` / JSONL).

Перед **LECTURE_SEARCH** (Exa / Semantic Scholar / Consensus): если после RAG достаточно локальных фрагментов (`local_sources_count >= LECTURE_MIN_LOCAL_SOURCES`, по умолчанию 3) или есть pinned whitelist — первичный внешний поиск пропускается (`[LECTURE_PIPELINE] External search bypassed…`). Жёсткое отключение: `LECTURE_EXTERNAL_SEARCH_ENABLED=false`. Запрос модели `search_external_materials` после лекции по-прежнему вызывает внешний поиск без этого guardrail.

**Не вызывается** на каждый turn диалога (`dialogue_feedback`) — там только `memory.rag_profile_compressed` (из init) + sliding window + fact manifest.

## Пайплайн

Хранилище — `VECTOR_STORE_BACKEND` (по умолчанию `postgres`: pgvector с HNSW-индексом; `qdrant` — запасной бэкенд). В логах и docstrings по-прежнему встречается слово «LanceDB» (историческое имя стадий), реально запросы идут в активное хранилище. Эмбеддинги — `BAAI/bge-m3`.

```text
whitelist foundation (pinned, без отбора)
  +
пул кандидатов (до LECTURE_RAG_CANDIDATE_LIMIT):
  route URL → конспекты документов
  registry stubs
  векторный поиск document_summaries
  knowledge_nodes (вектор)
  LightRAG vector_search (profile + fact)
  тонкие чанки rag_chunks (primary / secondary scope)
        ↓
presort (sim × trust × cites × recency, при ACADEMIC_RERANK_ENABLED)
        ↓
ОСНОВНОЙ ПУТЬ (LECTURE_CHUNK_CA_ENABLED=1): chunk cross-attention
  оценка = α·cos(чанк, тема) + β·cos(паспорт документа, тема)   [BGE-M3]
  → knee-cutoff + пол RAG_SCORE_MIN_FLOOR
  → якорь (top ≥ RAG_ANCHOR_THRESHOLD) или greedy MMR по источникам
  → semantic dedup (RAG_CHUNK_SEMANTIC_DEDUP)
  → positional reorder (лучшие в начало и конец)
        ↓ при ошибке / пустом выборе
ЗАПАСНАЯ ЦЕПОЧКА: Cross-Encoder rerank → отсечка LECTURE_RAG_CE_MIN_SCORE → MMR
        ↓ при ошибке
fallback_dedupe_candidates (URL + exact-text)
        ↓
склейка с pinned → [R*] чанки в промпт
```

Cross-Encoder (`BAAI/bge-reranker-v2-m3`) в основном пути не участвует: он нужен запасной цепочке и RAG Gateway ([RAG_GATEWAY_MODULE_3.md](RAG_GATEWAY_MODULE_3.md)).

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
`vector_similarity < 0.85`. Applied as **early exit** immediately after vector-store
vector hits are scored (`search_rag_chunk_rows`) and again on the candidate pool
**before** CE / cross-attention / MMR — never after Map-Lite or prompt stitch.
Surviving chunks are ordered by trust (desc) before `[R#]` assignment.
Dialog `chat_history` is untouched — trust markers appear only in RAG context blocks.

Note: blog/academic **Map-Reduce ingest** summarizes article windows (not vector
hits). Hard cutoff gates **retrieval → lecture context**, not window splitting.

| Источник | Модуль | Лимит |
|----------|--------|--------|
| Конспекты документов | `VectorStore.hybrid_search` (только вектор) | `LECTURE_RAG_CANDIDATE_LIMIT` (8) |
| Конспекты по URL маршрута | `fetch_summaries_by_urls` | min(urls, limit) |
| Knowledge nodes | `hybrid_search_nodes` | `LECTURE_RAG_KNODE_CANDIDATE_LIMIT` (4) |
| LightRAG | `LightRAG.vector_search` | same pool limit |
| Registry / archive stubs | без векторного хранилища | по маршруту |

Первичный поиск — только векторный (`BAAI/bge-m3`). Лексического канала (BM25 / `tsvector`) нет: точные термины и идентификаторы векторный поиск теряет. Планируется `tsvector` + `pg_trgm` + RRF (см. Roadmap в [README](../../README.md)).

### Step 2 — Выбор чанков (основной путь, без Cross-Encoder)

- Реализация: `select_diverse_chunks_with_cross_attention` в `src/shared/retrieval/chunk_cross_attention_mmr.py`, вход — `cross_attention_select_lecture_candidates_sync` (`lecture_context_rerank.py`).
- Тема — эмбеддинг запроса; для каждого чанка считаются вектор чанка и вектор паспорта документа (`BAAI/bge-m3`).
- Оценка: `LECTURE_CHUNK_CA_ALPHA · cos(чанк, тема) + LECTURE_CHUNK_CA_BETA · cos(паспорт, тема)` (0.7 / 0.3).
- **Knee-cutoff:** срез хвоста в первой точке, где перепад между соседними оценками ≥ `RAG_KNEE_DROP_RATIO` (0.12) от лучшей. Если лучшая оценка ниже `RAG_SCORE_MIN_FLOOR` (0.30), возвращается **пустой** результат (а не «лучший из плохих»).
- **Выбор:** если лучшая ≥ `RAG_ANCHOR_THRESHOLD` (0.70) — документ-якорь плюс до `RAG_ANCHOR_SUPPLEMENT_MAX` добавочных чанков; иначе greedy MMR (`LECTURE_CHUNK_CA_GAMMA`, не больше `LECTURE_CHUNK_CA_MAX_PER_SOURCE` чанков на источник), всего до `LECTURE_CHUNK_CA_TOP_K` (10).
- **Semantic dedup** (`RAG_CHUNK_SEMANTIC_DEDUP`, 0.85) и **positional reorder** (Lost in the Middle: два лучших чанка в начало и в конец).

### Step 3 — Запасная цепочка: Cross-Encoder → MMR

Включается, если основной путь выключен (`LECTURE_CHUNK_CA_ENABLED=0`), упал или ничего не выбрал.

- Критерий: **фокус пользователя** (`user_query`), если пусто — search query ноды.
- `score_relevance_pairs()` из `src/rag_gateway/cross_encoder.py` (тот же стек, что Directional RAG Gateway): `BAAI/bge-reranker-v2-m3` (`RAG_CROSS_ENCODER_MODEL`), сырые логиты → σ(x) ∈ [0, 1]. Если CE недоступен — cosine `BAAI/bge-m3`.
- Отсечка шума: `LECTURE_RAG_CE_MIN_SCORE` (0.50); если все ниже порога, результат пуст.
- MMR (`services/lecture_context_rerank.py`, не библиотека): relevance — оценки CE, сходство между чанками — cosine эмбеддингов BGE-M3; λ = `LECTURE_RAG_MMR_LAMBDA` (0.62), top-k = `LECTURE_RAG_MMR_TOP_K` (3).

### Step 4 — Склейка

- Pinned whitelist foundation не проходит отбор.
- Итог: `"\n\n---\n\n".join(chunks)` → `build_lecture_generation_payload()`.

## Отказоустойчивость

| Ситуация | Поведение |
|----------|-----------|
| Таймаут collect (`LECTURE_RAG_COLLECT_TIMEOUT_SEC`) | minimal fallback: whitelist foundation + route URLs, без векторного поиска |
| Таймаут LightRAG (`LECTURE_RAG_LIGHT_TIMEOUT_SEC`) | пул без vector hits |
| Таймаут CE/MMR (`LECTURE_RAG_RERANK_TIMEOUT_SEC`) | `fallback_dedupe_candidates` — URL + exact-text, лимит как legacy |
| Ошибка всего блока rerank | полный fallback: сбор пула + legacy dedupe |
| CE недоступен | уже внутри `cross_encoder.py` → BGE-M3 cosine |

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

Источник истины — `src/config/settings.py`; сводка всех переменных — [ENV_VARIABLES.md](ENV_VARIABLES.md).

| Переменная | Default | Описание |
|------------|---------|----------|
| `LECTURE_CHUNK_CA_ENABLED` | 1 | Основной путь (BGE-M3, knee, якорь/MMR); 0 → сразу CE → MMR |
| `LECTURE_CHUNK_CA_TOP_K` | 10 | Чанков после основного пути |
| `LECTURE_CHUNK_CA_ALPHA` / `_BETA` / `_GAMMA` | 0.7 / 0.3 / 0.55 | Вес чанка / паспорта / MMR |
| `LECTURE_CHUNK_CA_MAX_PER_SOURCE` | 2 | Чанков на источник |
| `RAG_SCORE_MIN_FLOOR` | 0.30 | Пол релевантности (ниже → пусто) |
| `RAG_KNEE_DROP_RATIO` | 0.12 | Перепад для knee-cutoff |
| `RAG_ANCHOR_THRESHOLD` | 0.70 | Порог режима «якорь» |
| `RAG_CHUNK_SEMANTIC_DEDUP` | 0.85 | Порог смыслового дубля |
| `LECTURE_RAG_CANDIDATE_LIMIT` | 8 | Первичный пул |
| `LECTURE_RAG_MMR_TOP_K` | 3 | Чанков после CE → MMR (запасная цепочка) |
| `LECTURE_RAG_CE_MIN_SCORE` | 0.50 | Мин. CE score после σ(logit) |
| `LECTURE_RAG_MMR_LAMBDA` | 0.62 | Баланс rel / diversity в запасной цепочке |
| `LECTURE_RAG_CONTEXT_MAX_CHARS` | 9000 | Лимит склейки контекста |
| `LECTURE_RAG_RERANK_TIMEOUT_SEC` | 60 | Таймаут rerank / MMR |
| `LECTURE_RAG_COLLECT_TIMEOUT_SEC` | 90 | Таймаут сбора кандидатов (в thread) |
| `LECTURE_RAG_LIGHT_TIMEOUT_SEC` | 45 | Таймаут LightRAG vector_search |
| `LECTURE_RAG_KNODE_CANDIDATE_LIMIT` | 4 | Knowledge nodes в пуле |
| `LECTURE_RAG_TOP_K` | 3 | Legacy лимит при full fallback |
| `LECTURE_MIN_LOCAL_SOURCES` | 3 | Минимум локальных источников, чтобы пропустить внешний поиск |
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | CE (общий с Gateway) |
| `EMBED_MODEL` | `BAAI/bge-m3` | Bi-Encoder |

## Код

| Файл | Роль |
|------|------|
| `src/shared/retrieval/lecture_rag_context.py` | Сбор пула, async orchestration |
| `src/shared/retrieval/lecture_context_rerank.py` | Вход основного пути, CE gate + MMR (запасной), fallback dedupe |
| `src/shared/retrieval/chunk_cross_attention_mmr.py` | Оценка, knee-cutoff, якорь/MMR, dedup, positional reorder |
| `src/rag_gateway/cross_encoder.py` | CE / cosine fallback |
| `services/llm_markdown_service.py` | HTML для UI (не lecture pool) |

## Отличие от Directional RAG (init ноды)

| | Init `rag_profile` | Lecture `retrieve_lecture_rag_context` |
|--|-------------------|----------------------------------------|
| Когда | `user_action=init` | dense_material |
| Источник | LightRAG facts/profile | Конспекты документов + route + LightRAG + тонкие чанки |
| CE | 3 search directions | один focus query |
| MMR | нет (text overlap dedup) | yes |
| В промпте | `layer_1_compressed_rag_profile` | `=== НАЧАЛО МАТЕРИАЛА ===` |
