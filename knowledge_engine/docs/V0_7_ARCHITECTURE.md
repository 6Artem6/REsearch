# Knowledge Engine v0.7 — Architecture

Спецификация: staged UMA locking, Semantic Scholar retrieval, Gemini analytics, LangGraph.

## Реализовано (`knowledge_engine/src/`)

| Модуль | Назначение |
|--------|------------|
| `locks.py` | `uma_resource_lock`, `staged_uma_lock`, `run_under_uma_lock` |
| `state.py` | `PersonalContext`, `ScrapedDocument`, `StructuredChunk`, `KnowledgeEngineState` |
| `fetcher/` | Academic cascade, PyMuPDF clean, HTTP fallback |
| `retrieval/` | **Semantic Scholar** + arXiv fallback → documents |
| `dedup.py` | LanceDB cosine dedup, `density_delta` |
| `analytics/*` | Lite chunking, Flash L2a–L2c, `prompts.py` |
| `guardrails/personal_context.py` | Stage 0: Ollama 7B personal context |
| **`graph.py`** | `context_inject` → `scholar_fetch` → chunking → profiling |

## Граф v0.7

```text
context_inject (Ollama 7B PersonalContext)
  → scholar_fetch (Semantic Scholar / arXiv → fetch → LanceDB dedup)
  → chunking (Gemini Lite)
  → profiling (Gemini Flash L2a–L2c)
  → END
```

**SearXNG не используется** в этом треке v0.7 (остаётся для legacy API v0.4–0.6).

## Guardrails / Stage 0

1. **`personal_context.py`** — Ollama 7B → `PersonalContext` (архитектура, latency, стек, ресурсы). Без search_queries.
2. **`manager.run_stage_0`** — personal context под UMA lock.

## Retrieval (academic track)

1. **`retrieval/semantic_scholar.py`** — опционально ( `SEMANTIC_SCHOLAR_ENABLED` ); по умолчанию **arXiv only**; v0.8 — карточки Consensus.
2. **`retrieval/paper_documents.py`** — Open Access PDF / abstract → `ScrapedDocument`.
3. Env: `SEMANTIC_SCHOLAR_LIMIT`, `SEMANTIC_SCHOLAR_API_KEY` (опционально).

## Gemini

- **`analytics/prompts.py`** — «Главный Архитектор»: local context + papers + user query → L2a–L2c system prompts.

## Web UI

- FastAPI `/app` — `POST /api/v1/v07/runs`, TOC, linkified sources.

## Smoke

```bash
export PYTHONPATH="$(pwd)"
SKIP_V07_FETCH=1 ./knowledge_engine/scripts/smoke_v07.sh "ваш вопрос"
./knowledge_engine/scripts/run-v07-analysis.sh "ваш вопрос"
```

См. [V0_6_CURRENT_SOLUTION.md](V0_6_CURRENT_SOLUTION.md) (legacy API).
