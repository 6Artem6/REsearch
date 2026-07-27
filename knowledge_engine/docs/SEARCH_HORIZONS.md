# Три горизонты поиска (SOTA / Infra / Prod)

Knowledge Engine разделяет **сбор источников** на три горизонта. Это ответ на идею из `init.md` («поиск по нескольким временным горизонтам»).

**Не путать** с категориями **Trade-off матрицы** в узле `matrix_node`:

| Контекст | Три варианта |
|----------|----------------|
| **Горизонты поиска** (discovery) | SOTA, Infra, Prod |
| **Категории матрицы** (решения) | Классика, SOTA (современное), Минимализм |

---

## Горизонты

### SOTA — исследования и передний край

- **Цель:** papers, benchmarks, surveys, новые алгоритмы и оценки.
- **Провайдеры:** `arxiv`, `semantic_scholar`, `crossref`, `consensus` (dork через SearXNG).
- **Фокус запроса:** state of the art, survey, benchmark, recent paper, evaluation.

### Infra — стек, деплой, observability

- **Цель:** как это ставится и сопровождается (K8s, Docker, scaling, cost).
- **Провайдеры:** только `google_meta` (SearXNG / Bing) — отдельный web-канал от papers.
- **Фокус запроса:** architecture, deployment, observability, runbook (без дублирования SOTA-каналов).

### Prod — продакшен и failure modes

- **Цель:** инциденты, postmortem, деградация, реальные trade-offs в бою.
- **Провайдеры:** `habr` (dork `site:habr.com`), `google_meta`.
- **Фокус запроса:** production, failure mode, incident, SRE.

---

## Как это работает в коде

1. **Декомпозиция** (`decomposition_node`, 7B) → CS-абстракции.
2. **LanceDB** (`local_rag_check_node`) — если контекста достаточно, внешний поиск пропускается.
3. **Gemini** (`ai_react_loop_node`, 1.5B) — горизонт-агностичный диалог: ссылки и факты «поверх» всех горизонтов.
4. **Multi-search** (`multi_search_node`):
   - `build_horizon_queries()` в `services/search/horizons.py` строит **три запросы** из `user_problem`, `context_constraints` и `cs_concept` абстракций.
   - `SearchRegistry.multi_search_horizons_sync()` для каждого горизонта вызывает свой набор провайдеров.
   - URL дедуплицируются; каждый `SearchResult` помечен полем `horizon` (`sota` | `infra` | `prod`).
   - Запросы сохраняются в state: `search_horizon_queries`.
5. Парсинг URL → vision → **summarizer** (`user_profile.md`) → LanceDB.
6. **Matrix** (7B) — три **архитектурные** варианта (Классика / SOTA / Минимализм), уже на обогащённых фактах.

---

## Стек поиска (v0.2.1)

| Компонент | Роль |
|-----------|------|
| SearXNG (`google_meta`) | Bing (+ Google в конфиге); DuckDuckGo отключён (CAPTCHA) |
| Semantic Scholar / arXiv / Crossref | SOTA и часть Prod |
| Habr / Consensus dorks | Infra и Prod, русскоязычный контекст |
| Gemini (Playwright) | Углубление и ссылки, не заменяет горизонты |
| LanceDB | Кэш summary между прогонами |

---

## Конфигурация

- Провайдеры по горизонту: `HORIZON_PROVIDERS` в `services/search/horizons.py`.
- Глобальный список активных провайдеров: `SEARCH_ACTIVE_PROVIDERS` в `config.py` (для `test-search` без разбиения по горизонтам).
- Шаблоны фокуса запроса: `_QUERY_FOCUS` в том же модуле (можно расширить или позже заменить генерацией 1.5B).

---

## CLI

```bash
export PYTHONPATH="/path/to/REsearch"
python -m knowledge_engine.main test-search "RAG cache invalidation" -c "Mac M1 Python"
```

По умолчанию **три панели** (запрос + провайдеры) и **три таблицы** URL по горизонтам. Старый режим одного запроса: `--flat`.

`analyze`: горизонты только если LanceDB **не** покрыл задачу (`is_rag_sufficient=false`) и граф дошёл до `multi_search_node`. В Live-строке: `URL: sota=… infra=… prod=…`.

---

## Gemini-primary (целевая архитектура discovery)

| Роль | Модель / сервис |
|------|------------------|
| Основная исследовательская работа | **Gemini** (Playwright) + API горизонтов (SOTA/Infra/Prod) |
| Сверка и уточнение | **1.5B** Re-Act в `ai_react_loop_node` |
| Сжатие для LanceDB | **один** вызов 7B (`summarize_gemini_bundle`), не 7B×каждый URL |
| Матрица / unraveling | 7B как раньше |

Запуск теста с этим путём:

```bash
export PYTHONPATH="$(pwd)"
python -m knowledge_engine.main browser-login   # один раз
export GEMINI_BROWSER_HEADLESS=false
python -m knowledge_engine.main analyze \
  --gemini-research \
  -c "стек, железо" \
  "ваша задача" \
  --matrix-only
```

В логе ожидайте: `[Dialogue: Gemini]`, `ai_react_loop_node`, `Gemini-primary: горизонты API`, `summarizer / Gemini bundle` — **не** десятки `summarizer / DocumentSummary (youtube…)`.

`SKIP_GEMINI=true` обходит диалог и оставляет тяжёлый `multi_search` (для быстрых прогонов без браузера).

### Playwright: Firefox вместо Chromium

Playwright **не** подключается к вашему системному Firefox — нужен свой бинарник:

```bash
cd knowledge_engine && .venv/bin/playwright install firefox
export PLAYWRIGHT_BROWSER=firefox
export GEMINI_BROWSER_HEADLESS=false
python -m knowledge_engine.main browser-login
```

Профиль сессии: `knowledge_engine/.browser_state/firefox/` (отдельно от `chromium/`).

### Нужен ли логин в Google?

**Не всегда.** Knowledge Engine не «логинится» сам — он держит **одну persistent сессию** Playwright (`launch_persistent_context`): куки и локальное состояние в `.browser_state/<engine>/`.

| Ситуация | Действие |
|----------|----------|
| Gemini открывает чат без аккаунта (гость / Try Gemini) | `browser-login` один раз → дойти до поля ввода → Enter. Google-аккаунт не нужен. |
| Редирект на `accounts.google.com` | Один раз войти в Google в этом же окне Playwright — сессия сохранится в профиле. |
| Уже прошли `browser-login` | `analyze` подхватывает профиль; повторный логин не нужен. |

`browser-login` — не «обязательный Google OAuth», а **ручной первый проход** до рабочего чата (CAPTCHA, terms, гость или вход). Headless после warm profile обычно ок (`GEMINI_BROWSER_HEADLESS=true`).

**browser-login:** Enter — в **терминале** (не в Firefox). Если строка «После входа…» повторялась — это баг старого `typer.prompt`; обновите код и снова `browser-login`.

---

## Ограничения (честный статус)

### Логи SearXNG (если контейнер запущен, но web не нужен)

| Сообщение | Смысл |
|-----------|--------|
| `TRACKER_PATTERNS … clearurls` | Внешний список не скачался — **не критично**, поиск может работать. |
| `X-Forwarded-For nor X-Real-IP` | Запрос **без** заголовков (браузер на `localhost:8080`). Клиент Knowledge Engine шлёт заголовки; для UI можно игнорировать или открыть с теми headers. |
| `bing … ConnectError: All connection attempts failed` | Из **Docker** нет исходящего доступа к `www.bing.com` (сеть OrbStack/VPN/firewall). Infra/Prod web-горизонты пустые; SOTA (arxiv, scholar) с хоста работают без Bing. |

Для прогона **только Gemini + API** без SearXNG:

```bash
export SEARXNG_ENABLED=false
# или CLI --gemini-research (по умолчанию отключает SearXNG, если переменная не задана)
```

Контейнер можно остановить: `docker compose stop searxng`.

- Запросы по горизонтам **шаблонные** (не отдельный LLM-вызов на каждый горизонт); роутер дополняет картину в `ai_react_loop`.
- Узел `react_search.py` из v0.1 **не в графе** v0.2.
- Google через SearXNG может возвращать 0 результатов; основной web-канал — **Bing**.
- Vision pipeline и Playwright-парсинг статей — best-effort; PDF/arXiv — упрощённый fetch.
