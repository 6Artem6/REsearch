# Модуль 3 — Directional RAG Gateway (Background Memory Engine)

Детерминированный брокер памяти: LanceDB + Ollama embeddings + локальный Cross-Encoder. **Без LLM.**

## Пайплайн

1. Векторный поиск (топ-5 на каждое `vector_query`)
2. Cross-Encoder: `relevance_criteria` ↔ текст чанка
3. Отсечка `score * weight < min_relevance_threshold`
4. Дедупликация (перекрытие текста > 90%)
5. Top `max_facts`

## API

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
| `RAG_CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RAG_DEFAULT_MIN_RELEVANCE` | `0.75` |
| `RAG_RETRIEVAL_PER_DIRECTION` | `5` |
| `RAG_LATENCY_WARN_MS` | `100` |

## Код

- `knowledge_engine/src/rag_gateway/gateway.py`
- `knowledge_engine/src/rag_gateway/cross_encoder.py`
- `knowledge_engine/src/memory/light_rag.py` — `vector_search`, `save_user_fact`

Модуль 2 вызывает `query_directional_rag` на `init` и `save_user_fact` при пробелах.

Установка: `pip install -r knowledge_engine/requirements.txt` (добавлен `sentence-transformers`).
