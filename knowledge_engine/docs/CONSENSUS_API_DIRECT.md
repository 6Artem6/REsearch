# Consensus Direct API — reverse-engineering report

Дата: 2026-08-08. Цель: уйти от DOM/кликов Playwright к прямому JSON API.

## Найденная ручка

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `https://consensus.app/api/paper_search/` |
| **Content-Type** | `application/json` |

### Request body (quick search)

```json
{
  "query": "<academic query>",
  "product_feature": "quick_search",
  "filters": { "open_access": "true" }
}
```

### Response (фрагмент)

```json
{
  "search_id": "…",
  "papers": [
    {
      "title": "…",
      "authors": ["…"],
      "doi": "…",
      "paper_id": "…",
      "abstract": "…",
      "citation_count": 0
    }
  ]
}
```

Пагинация / повторная выдача того же поиска:

- `GET https://consensus.app/api/paper_search/{search_id}/?page=1&size=20`

## Авторизация и защита

| Артефакт | Роль |
|----------|------|
| `cf_clearance` (+ `__cf_bm`) | Cloudflare bot management |
| `__session` / `__session_*` | Clerk session JWT (~**60s** TTL) |
| `__client*` / `__refresh_*` | Clerk client / refresh |
| `Authorization: Bearer <__session JWT>` | **обязателен для curl_cffi** (браузер обходится одними Cookie) |

Дополнительно:

- Auth provider: **Clerk** (`clerk.consensus.app`).
- В HAR от браузера **нет** `Authorization` — только Cookie. Для внебраузерного TLS fingerprint Clerk на POST отвечает `307` с `session-token-expired-refresh-non-eligible-non-get`, если Bearer не передан.
- Persistent Chromium profile раздувает Cookie jar (Google/ads) → nginx `400 Request Header Or Cookie Too Large`. Нужна фильтрация по `*.consensus.app`.

## Устойчивость (оценка)

| Режим | Результат |
|-------|-----------|
| Чистый `curl` / httpx без CF cookies | Cloudflare challenge / 403 |
| `curl_cffi` + stale cookies | 307/401 Clerk |
| `curl_cffi` + Playwright prefetch cookies **и** `Authorization: Bearer __session` | **OK** (проверено: 19–20 papers) |
| In-page `fetch()` в Playwright (без DOM) | **OK**, самый стабильный fallback |

**Вывод:** полностью без браузера нельзя — нужен периодический pre-fetch сессии (Playwright persistent profile → `cf_clearance` + свежий `__session`). После prefetch прямой `curl_cffi` с `impersonate="chrome124"` работает в окне ~60s JWT.

Рекомендуемый production path:

1. Короткий Playwright warmup (или reuse session).
2. Снять consensus-only cookies + `__session` → Bearer.
3. `POST /api/paper_search/` через `curl_cffi`.
4. Fallback: in-page `fetch` при 307/401/403.

## Product integration

`CONSENSUS_USE_DIRECT_API=true` (default) → `ConsensusSessionManager` (token cache) +
`ConsensusDirectClient`:

1. **Singleton cache** (`services/search/consensus_session_manager.py`): TTL 45s → cache hit ~0 ms
2. **Fast warmup**: block CSS/fonts/images/trackers, `goto(..., wait_until="commit")`, сразу cookies, без `networkidle`
   - Блоклист хостов пополнен из `consensus_network_trace.har`: Datadog RUM, GTM/GA, Intercom, Statsig (`prodregistryv2.org` / `featureassets.org`), LinkedIn/Facebook pixels, `track.consensus.app`, Cloudflare Insights.
3. `curl_cffi` + `Authorization: Bearer <__session>`
4. Fallback: in-page `fetch()` при 307/401/403

Замер (smoke, headed + block static):

| Режим | Цель | Факт |
|-------|------|------|
| Cache hit `get_active_session` | ~0 ms | **0.1 ms** |
| Fast warmup (profile JWT fresh) | ≤2–3 s | **~1.1–1.3 s** |
| Fast warmup (commit+Clerk refresh) | ≤2–3 s | **~3.1 s** |
| Warm search (JWT в кэше, только curl) | ~1.0–1.5 s | **~2.0–2.1 s** (сеть Consensus) |
| Legacy DOM quick search | — | ~14 s+ |

Заметки:
- Cookie jar только `*.consensus.app` (`cookie_header_len≈5.5KB`) — без nginx 400.
- Headless **не** ротирует Clerk `__session` на этом profile → warmup использует `CONSENSUS_BROWSER_HEADLESS` (обычно headed).
- `ensure_warmup_async()` — фоновый prefetch до `search_papers()`.

Legacy DOM: `CONSENSUS_USE_DIRECT_API=false`.

Smoke: `PYTHONPATH=. python -m knowledge_engine.scripts.smoke_consensus_direct`

## Артефакты / команды

```bash
# 1) HAR + JSON traffic
PLAYWRIGHT_BROWSERS_PATH="$(.venv/bin/python -c "import pathlib,playwright;print(pathlib.Path(playwright.__file__).parent/'driver/package/.local-browsers')")" \
PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_consensus_playwright \
  --send --record-har --har-path consensus_network_trace.har

# 2) Найти ручку + cURL
PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.analyze_consensus_har \
  --har consensus_network_trace.har --out consensus_api_endpoint.json

# 3) POC без DOM
PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.poc_consensus_api \
  --endpoint consensus_api_endpoint.json --via curl --query "your query"
```

Env: `CONSENSUS_RECORD_HAR`, `CONSENSUS_HAR_PATH`, `CONSENSUS_LOG_JSON_TRAFFIC` (см. `.env.example`).

Файлы (в `.gitignore`): `consensus_network_trace.har`, `consensus_api_endpoint.json`, `consensus_storage_state.json`.
