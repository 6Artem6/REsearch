# Knowledge Engine v0.8 — Consensus Agent

Stateful Consensus (Playwright) + Light RAG + Gemini Lite/Flash + Reasoner.

**Снимок версии:** [V0_8_SNAPSHOT.md](V0_8_SNAPSHOT.md) · **разработка:** [DEV_RUNBOOK.md](DEV_RUNBOOK.md)

## Сессия Consensus

Playwright **persistent profile** в `knowledge_engine/.browser_state/<chromium|firefox>/`.

1. **Один раз войти:** `./knowledge_engine/scripts/consensus-login.sh` (Google/email).
   CLI без `PLAYWRIGHT_BROWSERS_PATH` ищет Chrome в `~/Library/Caches/ms-playwright` — используйте скрипт или `dev-native.sh`.
2. Между прогонами браузер можно **не закрывать** (`CONSENSUS_REUSE_BROWSER_SESSION=true`); каждый анализ — **новый чат** (`CONSENSUS_NEW_THREAD_EACH_RUN=true`).
3. При **uvicorn reload** API закрывает браузер и сохраняет profile; после reload cookies с диска. Для стабильного окна без reload: `KE_API_RELOAD=false`.
4. Первый вход — **headed** (`CONSENSUS_BROWSER_HEADLESS=false`); headless без warm profile часто даёт login wall.

При **Sign in** / login wall: **soft** `goto` + new thread → **hard** restart browser (тот же profile) (до `CONSENSUS_AUTH_RECOVERY_CYCLES`, default **2**), затем ошибка.

`browser-login` в CLI — **только Gemini**, не Consensus.

## Direct API (без DOM)

HAR reverse-engineering: ручка **`POST /api/paper_search/`**. Отчёт и POC:

- [CONSENSUS_API_DIRECT.md](CONSENSUS_API_DIRECT.md)
- `python -m knowledge_engine.scripts.check_consensus_playwright --send --record-har`
- `python -m knowledge_engine.scripts.analyze_consensus_har`
- `python -m knowledge_engine.scripts.poc_consensus_api --via curl`

Auth: Cloudflare `cf_clearance` + Clerk `__session` (Bearer для `curl_cffi`, TTL ~60s) — нужен периодический Playwright prefetch.

`CONSENSUS_USE_DIRECT_API=true` (default) включает гибридный клиент в продуктовом поиске.

## Контекст и изоляция

| Компонент | Что получает |
|-----------|----------------|
| **Consensus.app** | `academic_query_en`: Lite sanitize + **SearXNG grounding** (`fast_grounding`) + **preserved_terms** (RPG, LLM, lore…). Anchor sanitize = **только user query** (без Light RAG в anchor). |
| **Light RAG** | Индексирует `user_profile.md`; в рантайме `get_relevant_profile_context(query)` — селективный фрагмент или `""`. |
| **Gemini Lite / Flash** | validate, chunking, L2a–L2c: `user_query` + selective profile + anchor. |
| **Gemini Reasoner** | `valid_docs`, papers block, raw consensus fallback, `partial_data_note`. |
| **Локальный агент** | Транслирует `user_final_answer` без перегенерации. |

**Hardcode профиля в коде и промптах запрещён** — только динамический Light RAG.

## Поток

```text
Light RAG index profile → grounding + preserved terms → sanitize query (Lite)
  → Consensus (EN academic only) → Lite validate (OK/REJECT/RETRY)
  → enrich metadata → fetch paper bodies → dedup ingest → chunking
  → L2a ConceptGraph → L2b ProfileGapMap → L2c TradeoffMatrix
  → Reasoner → ingest fact_nuggets → completed
```

## Web UI (`/app`)

- Запуск: форма на `/app`; API `POST /api/v1/v07/runs`.
- Прогоны: `knowledge_engine/.runs/v07_runs.json` (до 80 записей, полный `result`).
- Permalink: `/app?run=<id>` — повторное открытие после долгого прогона.
- Секции: запрос в Consensus (EN), валидация, источники (DOI/arXiv, без дубля URL), L2a–L2c, ответ Reasoner.
- **Темы:** `web/static/themes/themes.css` (`data-theme=monokai-pro|classic`), компоненты на `--ke-*`.
- **KaTeX:** `$...$` / `$$...$$`; ремонт битого TeX (form-feed, `\f\frac`, `\text`) в `linkify.py` и `app.js`.

## Env

| Переменная | Назначение |
|------------|------------|
| `GRAPH_VERSION=0.8` | включает consensus pipeline |
| `CONSENSUS_REUSE_BROWSER_SESSION` | держать браузер между прогонами (default true) |
| `CONSENSUS_NEW_THREAD_EACH_RUN` | новый чат на каждый анализ (default true) |
| `CONSENSUS_BROWSER_HEADLESS` | false для первого логина |
| `CONSENSUS_AUTH_RECOVERY_CYCLES` | циклы soft/hard recovery |
| `CONSENSUS_MAX_RETRIES` | RETRY в validator |
| `LIGHT_RAG_MIN_COSINE_SIM` | порог сегментов профиля (default 0.42) |
| `LIGHT_RAG_PROFILE_LIMIT` | max сегментов в контексте (default 5) |
| `GEMINI_REASONER_MODEL` | финальный ответ (см. config) |
| `SEMANTIC_SCHOLAR_ENABLED=false` | enrich без SS API |
| `KE_API_RELOAD` | uvicorn reload (закрывает Consensus при reload) |
| `PLAYWRIGHT_BROWSERS_PATH` | dev-native → `.venv` Playwright browsers |

См. [V0_7_ARCHITECTURE.md](V0_7_ARCHITECTURE.md) для `GRAPH_VERSION=0.7`.
