# v0.8 — зафиксированный снимок (2026-07-26)

Актуальный режим разработки: **`GRAPH_VERSION=0.8`** в `.env`. Спека агента: [V0_8_CONSENSUS_AGENT.md](V0_8_CONSENSUS_AGENT.md).

## Что входит в снимок

| Область | Содержание |
|---------|------------|
| **Пайплайн** | Consensus (EN academic query) → Lite validate (OK/REJECT/RETRY) → enrich papers → fetch bodies → dedup/chunk → **L2a → L2b → L2c** → **Gemini Reasoner** → Light RAG fact ingest |
| **Контекст** | Light RAG: селективный `user_profile.md`; Consensus query: sanitize + **SearXNG grounding** + **preserved_terms** (без подмены темы через полный anchor) |
| **Playwright** | Persistent profile `knowledge_engine/.browser_state/chromium/`; `consensus-login`; reuse сессии; новый чат на прогон; soft/hard auth recovery |
| **Web UI** | `/app` — TOC, L2, источники, KaTeX; **пошаговый poll** `/view` при `running`; темы Monokai Pro |
| **Персистентность** | `knowledge_engine/.runs/v07_runs.json` (до **80** прогонов с полным `result`) |
| **API** | `POST/GET /api/v1/v07/runs`, shutdown hook сохраняет Consensus profile |

## Быстрый старт

```bash
cp .env.example .env   # GRAPH_VERSION=0.8, GEMINI_API_KEY
make setup             # или setup.sh + host ollama/python
# Один раз, API остановлен:
./knowledge_engine/scripts/consensus-login.sh
make dev               # dev-native.sh
open http://127.0.0.1:8765/app
```

## Ключевые env (см. `.env.example`)

- `CONSENSUS_REUSE_BROWSER_SESSION`, `CONSENSUS_NEW_THREAD_EACH_RUN`, `CONSENSUS_BROWSER_HEADLESS`
- `CONSENSUS_AUTH_RECOVERY_CYCLES`, `CONSENSUS_MAX_RETRIES`
- `LIGHT_RAG_MIN_COSINE_SIM`, `LIGHT_RAG_PROFILE_LIMIT`
- `SEMANTIC_SCHOLAR_ENABLED=false` (v0.8 по умолчанию без SS API)
- `KE_API_RELOAD=true` — при reload браузер Consensus закрывается (profile на диске)
- `PLAYWRIGHT_BROWSERS_PATH` — dev-native выставляет путь в `.venv` (не `~/Library/Caches/ms-playwright`)

## Ключевые пути в коде

```text
knowledge_engine/src/agent/local_orchestrator.py   # v0.8 orchestration
knowledge_engine/src/retrieval/consensus_session.py
knowledge_engine/src/processors/validator.py
knowledge_engine/src/processors/consensus_query_prep.py
knowledge_engine/src/processors/reasoner.py
knowledge_engine/src/guardrails/fast_grounding.py
knowledge_engine/services/v07_run_store.py
knowledge_engine/web/present.py, linkify.py, source_present.py
knowledge_engine/web/static/ (app.css, app.js, themes/themes.css)
knowledge_engine/api/app.py                      # static /app, shutdown
knowledge_engine/scripts/consensus-login.sh
knowledge_engine/scripts/dev-native.sh
```

## Web: темы и LaTeX

- Темы: `web/static/themes/themes.css` — токены `--ke-*`, переключение `data-theme` (Monokai Pro / Classic); picker в сайдбаре, `localStorage` `ke-theme`.
- LaTeX: `linkify.repair_broken_latex` (сервер) + `sanitizeMathDelimited` в `app.js` (form-feed, `\f\frac`, `\t\text`, дубликаты в `$...$`).

## Проверка после изменений

```bash
export PYTHONPATH="$(pwd)"
.venv/bin/python -c "from knowledge_engine.web.linkify import repair_broken_latex; print(repair_broken_latex('\$\\text{x} \\f\\frac{a}{b}\$'))"
.venv/bin/python -c "from knowledge_engine.src.agent.local_orchestrator import run_knowledge_engine_v08"
curl -s http://127.0.0.1:8765/api/v1/health | jq .graph_version
```

## Отличие от v0.7

| | v0.7 | v0.8 |
|---|------|------|
| Retrieval | Semantic Scholar / arXiv track | **Consensus.app** |
| Финальный текст | Architect / graph nodes | **Reasoner** на valid_docs |
| SearXNG в графе | нет (guardrails grounding) | grounding **только** для Consensus sanitize |
| Web | `/app` + v07 runs API | то же + v0.8 pipeline + UI polish |
