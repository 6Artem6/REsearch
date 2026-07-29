# Source pool (curriculum + evaluator)

См. также [ARCHITECTURE_DEDUP.md](ARCHITECTURE_DEDUP.md) — Gemini, Consensus, curriculum search.

Единая логика источников — не дублировать в pipeline.

| Задача | Модуль |
|--------|--------|
| Статический whitelist | `whitelist.py` (`APPROVED_SOURCES_WHITELIST`, `format_whitelist_*`) |
| URL gate (blocklist, homepage) | `curriculum_source_pool.is_collectible_article_url` |
| Lite APPROVED / REJECTED | `evaluator.evaluate_source` |
| Curriculum hits (Search-First) | `lite_search_pipeline.batch_lite_eval_hits` / `batch_evaluate_sources` (один Lite) |
| Reasoner Re-Act citations | `batch_evaluate_sources` (не per-URL) |
| Пополнение SQLite архива | `curriculum_source_pool.register_curriculum_source` (только из `evaluate_source`) |
| Discovery graph (domain trust) | `discovery_trust.archive_urls_from_discovery` — отдельный трек, статусы `discovered` / `rejected_low_trust` |

**Не добавлять:** второй `run_gemini_lite_structured` для источников, отдельный `archive.upsert` после курации, локальные копии whitelist-форматтеров.

---

## Ключи и лимиты (curriculum сбор)

| Сервис | Переменные `.env` | Где взять | Лимиты (ориентир) | Локальный счётчик |
|--------|-------------------|-----------|-------------------|-------------------|
| **Google Custom Search** (опц.) | `GOOGLE_CSE_*`, `CURRICULUM_GOOGLE_CSE_ENABLED=true` | GCP + billing card | ~100 req/day | **по умолчанию выключен** — практика через **SearXNG** |
| **SearXNG** (практика, default) | `SEARXNG_BASE_URL`, `SEARXNG_ENABLED` | `docker compose up -d searxng` | engines bing/google | primary path после архива |
| **DuckDuckGo** (практика) | `CURRICULUM_PRACTICAL_DDGS_ENABLED=true` | пакет `duckduckgo-search` | ratelimit | по умолчанию **выключен**; только если CSE+SearXNG пусты |
| **Semantic Scholar** (академика) | `SEMANTIC_SCHOLAR_API_KEY` (опц.) | [Semantic Scholar API](https://www.semanticscholar.org/product/api) | **1 req/s** на все endpoints (глобальный throttle в коде, `SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC=1.05`) | `SEMANTIC_SCHOLAR_DAILY_LIMIT`, блок на 429/503 |
| **arXiv** | нет | публичный `export.arxiv.org` | разумный polling (1 запрос на сбор) | нет |
| **SearXNG** (legacy fallback) | `SEARXNG_BASE_URL` | локальный Docker `docker compose up -d searxng` | лимиты engines (Google/Bing) у провайдера | нет |
| **Gemini** (tutor, Flash, опц. grounding / web) | `GEMINI_API_KEY` или `GEMINI_API_KEYS` | [Google AI Studio](https://aistudio.google.com/apikey) | RPM/RPD по модели (free tier) | `GEMINI_QUOTA_TRACK` → `.runs/gemini_quota_state.json`; `python -m knowledge_engine.scripts.check_gemini_quotas` |
| **Consensus** (опц. академика) | сессия браузера | `consensus-login.sh`, не API key | ручной логин Playwright | нет |

### Проверка локальных лимитов curriculum

```bash
python -m knowledge_engine.scripts.check_curriculum_quotas
python -m knowledge_engine.scripts.check_curriculum_quotas --json
```

Сброс счётчиков: новый UTC-день автоматически, или удалить `knowledge_engine/.runs/curriculum_api_quota_state.json`.

### Минимальный `.env` для API-first curriculum

```env
# GOOGLE_CSE_API_KEY=   # опционально; нужен billing в GCP
# GOOGLE_CSE_ID=
# CURRICULUM_GOOGLE_CSE_ENABLED=false
SEARXNG_BASE_URL=http://localhost:8080
SEARXNG_ENABLED=true
# опционально
SEMANTIC_SCHOLAR_API_KEY=...
GEMINI_API_KEY=...
CURRICULUM_API_QUOTA_TRACK=true
GOOGLE_CSE_DAILY_LIMIT=100
```

Без `GEMINI_API_KEY` Search-First curriculum не стартует (генератор графа).

### Smoke всего API-сбора (без SS ключа)

```bash
# Сначала — что живо среди поисковиков
python -m knowledge_engine.scripts.check_curriculum_search_providers --goal "kafka replication"

python -m knowledge_engine.scripts.smoke_curriculum_sources
python -m knowledge_engine.scripts.smoke_curriculum_sources --with-collect --policy hybrid
python -m knowledge_engine.scripts.smoke_curriculum_sources --json
```

По умолчанию: SS search + paper probe, arXiv, CSE/DDGS, academic/practical fetch (без Playwright). `--with-collect` — полный `collect_sources_by_policy` (web/grounding выключены).
