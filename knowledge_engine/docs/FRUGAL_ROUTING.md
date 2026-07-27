# FrugalGPT / Hybrid routing (SLM + Gemini)

## Роли

| Слой | Модель | Узлы / сервисы |
|------|--------|----------------|
| SLM router | Qwen 2.5 **1.5B** (Ollama) | `intent_and_clarify`, `horizons` (запросы), `rolling_summarize_dialogue`, `local_rag_check` |
| Discovery | API + LanceDB | `multi_search` (без сырого HTML в Gemini) |
| Context | `context_blocks` + `assemble_gemini_payload` | Профиль по `##` в MD, источники, задача — стабильные блоки |
| Quality gate | Qwen **7B** (`CONTEXT_EVAL_MODEL`) | `evaluate_and_refine_context` — галочки по block_id + hints |
| Heavy reasoner | **Gemini** (Playwright) | `gemini_heavy_reasoning` → `ask_gemini(payload)` |
| Structure (cheap) | Qwen **7B** | JSON `AnalysisReport` из текста Gemini |
| Index | LanceDB | `lancedb_save_node` |

## Граф

```
intent_and_clarify → (interrupt: уточнение?) → local_rag_check
  → [достаточно RAG? → context_preparation → evaluate_and_refine_context → gemini_heavy_reasoning]
  → [нужен поиск → gemini_find_sources … | SKIP_GEMINI → multi_search → matrix]
  → lancedb_save → (interrupt) unraveling → END
```

## Deep Researcher loop (Gemini + 1.5B validator)

```
intent (1.5B: нужны уточнения? → по умолчанию **Gemini**, не пользователь)
  → local_rag
  → [SKIP_GEMINI] multi_search → matrix
  → [RAG ok] context_preparation → evaluate_and_refine_context → gemini_heavy_reasoning
  → gemini_find_sources (A) → gemini_extract (B) → profile_validator (1.5B)
     → VALID: save LanceDB + signal → next source / re-find
     → INVALID: signal only → next source / re-find
     → MIN_VALIDATED reached → gemini_final_matrix (C)
→ lancedb_save → unraveling
```

Env: `MIN_VALIDATED_SOURCES`, `MAX_RESEARCH_SOURCES`, `MAX_RESEARCH_FIND_ROUNDS`, `CLARIFY_VIA_GEMINI` (default true; false = вопрос в терминале).

Одна Playwright-сессия на прогон: `services/gemini_research_session.py`.

## Язык

Все LLM-вызовы используют `llm_locale.py`: **вывод на русском**, даже если источники (arxiv, Habr) на другом языке. Поисковые строки SOTA могут содержать английские термины для лучшего recall.


- `SKIP_GEMINI=true` — heavy шаг заменяется локальным `matrix_node` (7B).
- `SEARXNG_ENABLED=false` — без Bing/Google (часто с `--gemini-research`).
- `GEMINI_PAYLOAD_MAX_CHARS` — лимит Sandwich payload.
- `GEMINI_RESPONSE_MAX_SEC` (default 300) — макс. ожидание стриминга.
- `GEMINI_STREAM_STABLE_ROUNDS` — сколько пауз без роста текста считать «ответ готов».

**Два окна Firefox:** раньше `multi_search` открывал Playwright для URL и отдельно Gemini. При `SKIP_GEMINI=false` URL-парсинг через браузер отключён — только один браузер на heavy шаг.
