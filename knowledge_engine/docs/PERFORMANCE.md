# Время прогона `analyze`

## Где уходили ~22 мин (ваш фоновый прогон)

| Этап | Почему долго |
|------|----------------|
| **Ollama 7B на CPU** (Docker OrbStack) | ~5–8 tok/s; один structured JSON ≈ 3–5 мин при длинном ответе |
| **summarizer × N URL** | каждый URL = **отдельный 7B** + embed; в логе было ≥3 (Microsoft, YouTube…) |
| **Playwright** | `fetch_page_html` на каждый URL |
| **Vision** | скриншоты/картинки (Microsoft, YouTube thumbnails) |
| **Gemini** | Playwright + ожидание ответа |
| **Горизонты поиска** | 3 × несколько API + Bing |
| **matrix** | ещё один 7B |
| **unraveling** | ещё один 7B (если не `--matrix-only`) |

До паузы на выбор варианта — почти всё выше **без** unraveling.

## Что уже сокращено в коде (defaults)

- `MAX_FETCH_URLS=3` (было до 10 URL с парсингом)
- `MULTI_SEARCH_SKIP_VISION=true` (vision по умолчанию выключен)
- Блоклист URL: YouTube, Microsoft support, Geeksforgeeks, Wikipedia
- Приоритет: arxiv, doi, Semantic Scholar, Habr
- `OLLAMA_NUM_PREDICT=1024` — не раздувать JSON
- `--matrix-only` — только матрица, без unraveling
- `SKIP_GEMINI=true` — пропуск Gemini (сразу multi_search)
- Сводка `print_timing_summary()` после матрицы

## Рекомендуемый быстрый прогон (Mac)

```bash
export OLLAMA_BASE_URL=http://localhost:11434   # host Ollama + Metal, не Docker CPU
export SKIP_GEMINI=true
export MAX_FETCH_URLS=2
export MULTI_SEARCH_SKIP_VISION=true

python -m knowledge_engine.main analyze -c "…" "…" --matrix-only
```

Ожидание: **~3–8 мин** до матрицы на Metal (зависит от сети), не 20+.

## Переменные

| Env | Default | Эффект |
|-----|---------|--------|
| `MAX_FETCH_URLS` | 3 | Лимит URL с парсингом + summarizer |
| `MULTI_SEARCH_SKIP_VISION` | true | Без vision pipeline |
| `SKIP_GEMINI` | false | true → без Playwright Gemini |
| `OLLAMA_NUM_PREDICT` | 1024 | Лимит токенов генерации |
| `OLLAMA_STRUCTURE_NUM_PREDICT` | 3072 | JSON AnalysisReport после Gemini (без обрезки 3-го option) |
| `MAX_AI_DIALOGUE_TURNS` | 3 | Реплики Gemini (если не SKIP) |

Лог с `NODE` / `OLLAMA` таймингами: `knowledge_engine/.runs/*.log` и блок «Время по этапам» в консоли после матрицы.
