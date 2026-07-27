# Knowledge Engine v0.3

Иерархический Deep Research: **Stateless Gemini** (`google-genai`) + LanceDB **L0/L1/L2**.

## Запуск

```bash
export GEMINI_API_KEY=...   # или GOOGLE_API_KEY
export GRAPH_VERSION=0.3      # default
export PYTHONPATH="$(pwd)"
python -m knowledge_engine.main analyze -c "..." "Задача"
```

- `MAX_RESEARCH_DEPTH` (default 2) — Re-Act раунды discovery после `research_evaluator`

## Граф v0.3

`decomposition` → `discovery` → `extractor` (httpx→Playwright) → `research_evaluator` (Re-Act) → `matrix` → `lancedb_save` → interrupt → `unraveling`.

State: `graph/state.py` — `original_query`, `l0_summary`, `pending_urls`, `explored_urls`, `depth` (без сырого HTML).

LanceDB table `knowledge_nodes`: `get_hierarchical_context(node_id)` поднимает L2→L1→L0.

## Stateless Gemini

`services/gemini_stateless.py` — `run_stateless_gemini(system, payload, global_anchor, response_schema?)`.

Нет persistent chat; каждый вызов = anchor + instruction + payload.
