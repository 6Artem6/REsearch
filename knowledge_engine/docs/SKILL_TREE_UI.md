# AI Skill Tree & Tutor Dashboard (UI)

Страница: **http://127.0.0.1:8765/app/skill-tree**

## Сборка (один раз после клонирования)

CDN `esm.sh` часто блокируется (CORS/MIME). UI собирается **локально**:

```bash
make skill-tree-ui
# или: ./knowledge_engine/scripts/build-skill-tree-ui.sh
```

Нужны Node.js и npm. Результат: `web/static/skill-tree/skill-tree.bundle.js` (тот же origin, без import map).

## Worker (Gemini / тяжёлый бэкенд)

Генерация маршрута, тьютор ноды, v0.7 runs и async analysis выполняются в **отдельном процессе**:

```bash
make dev          # API + worker
make worker       # только worker (если API уже запущен)
```

`GET /api/v1/health` → `worker_ok`. Без worker POST `/curriculum/generate` и `/node/*` → 503.

### Redis (очередь + логи)

При `REDIS_URL` (по умолчанию в `make dev`: `redis://127.0.0.1:6379/0`):

- Задачи worker: **pub/sub** канал `ke:tasks` (work jobs, analysis, v07).
- Статусы jobs/runs: ключи в Redis (не `work_jobs.json`).
- Trace прогонов: списки `ke:runlog:{id}` (не `.runs/*.log`).
- Heartbeat: `ke:worker:heartbeat` (TTL).

`GET /api/v1/health` → `redis_ok`, `worker_ok`.

Без Redis — прежний режим: JSON в `.runs/` и poll worker.

Опционально: `KE_WORKER_INLINE_FALLBACK=true` — выполнять в API, если worker не отвечает (не для prod).

## Локальное сохранение (продолжить позже)

| Файл | Содержимое |
|------|------------|
| `knowledge_engine/.runs/skill_tree_curricula.json` | Все учебные графы (Модуль 1) |
| `knowledge_engine/.runs/node_deep_dive_sessions.json` | Контент нод, ссылки, история чата тьютора |

API:

- `GET /api/v1/skill-tree/curricula` — список маршрутов
- `GET /api/v1/skill-tree/curricula/{id}/workspace` — граф + статусы + сессии
- `POST /api/v1/skill-tree/curricula/active` — активный маршрут

В браузере дополнительно: `localStorage` (`ke_skill_tree_active_curriculum`) и URL `?curriculum=...`.


| Файл | Аналог TSX |
|------|------------|
| `RoadmapDashboard.js` | `RoadmapDashboard.tsx` |
| `RoadmapCanvas.js` | `RoadmapCanvas.tsx` (React Flow) |
| `NodeDrawer.js` | `NodeDrawer.tsx` |
| `NodeTutorChat.js` | `NodeTutorChat.tsx` |
| `SkillNode.js` | кастомная нода графа |

## API

- `POST /api/v1/curriculum/generate`
- `POST /api/v1/node/init` | `/chat` | `/verify`
- `POST /api/v1/node/suggest-questions` | `/explain-selection` — выделение в материале (как v08)
- `GET /api/v1/rag-gateway/memory-status`
- `GET /api/v1/node/statuses/{curriculum_id}`

## Режимы генерации

- **Fast** → `depth_level: Standard`
- **Consensus** → `depth_level: Deep Mechanics`
