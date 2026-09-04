# Модуль 3 — Directional RAG Gateway (Background Memory Engine)

Детерминированный брокер памяти: LanceDB + BGE-M3 embeddings + локальный Cross-Encoder. **Без LLM.** Выполняется **только в KE worker**; HTTP API ставит `WorkJobKind.RAG_GATEWAY` и ждёт результат (API не загружает веса моделей).

## Пайплайн

1. Векторный поиск (топ-5 на каждое `vector_query`)
2. Cross-Encoder: `relevance_criteria` ↔ текст чанка
3. Отсечка `score * weight < min_relevance_threshold`
4. Дедупликация (перекрытие текста > 90%)
5. Top `max_facts`

## API

Эндпоинты остаются синхронными для клиента; исполнение — worker.

```http
POST /api/v1/rag-gateway/query
```

```json
{
  "target_node": "raft_leader_election",
  "search_directions": [
    {
      "direction_label": "Опыт с горутинами",
      "vector_query": "синхронизация и горутины в Go",
      "weight": 1.0
    }
  ],
  "relevance_criteria": "Факты про консенсус и кворум в распределённых системах",
  "max_facts": 4,
  "min_relevance_threshold": 0.75
}
```

```http
POST /api/v1/rag-gateway/facts
```

```json
{
  "fact_text": "Пользователь путает кворум при split-brain",
  "category": "learning_gap",
  "node_id": "raft_leader_election"
}
```

## Конфиг (.env)

| Переменная | По умолчанию |
|------------|----------------|
| `EMBED_MODEL` | `BAAI/bge-m3` |
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RAG_CE_TORCH_DTYPE` | `auto` → fp16 на MPS/CUDA |
| `RAG_CE_AUTO_UNLOAD` | `false` — выгрузка CE после idle |
| `RAG_CE_AUTO_UNLOAD_IDLE_SEC` | `300` |
| `RAG_DEFAULT_MIN_RELEVANCE` | `0.55` |
| `RAG_RETRIEVAL_PER_DIRECTION` | `5` |
| `RAG_LATENCY_WARN_MS` | `100` |

Память (Apple Silicon): `RAG_CE_TORCH_DTYPE=auto` загружает CE в fp16 на MPS; после predict — `inference_mode`, `gc.collect()`, `mps.empty_cache()`. `unload_cross_encoder()` + `RAG_CE_AUTO_UNLOAD=true` снимает веса после idle.

## Код

- `knowledge_engine/src/rag_gateway/gateway.py`
- `knowledge_engine/src/rag_gateway/cross_encoder.py`
- `knowledge_engine/src/memory/light_rag.py` — `vector_search`, `save_user_fact`

Модуль 2 вызывает `query_directional_rag` на `init` и `save_user_fact` при пробелах (**в процессе worker**, не API).

Отладка через HTTP: `POST /rag-gateway/query` и `/facts` → очередь `rag_gateway`. `GET /memory-status` считает строки LanceDB без эмбеддера.

Установка: `pip install -r knowledge_engine/requirements.txt` (добавлен `sentence-transformers`).
