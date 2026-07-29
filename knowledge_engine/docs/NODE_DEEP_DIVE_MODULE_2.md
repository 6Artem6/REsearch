# Модуль 2 — Node Deep-Dive Engine

Интерактивное погружение в ноду учебного графа (Модуль 1): tiered memory, step pipeline, RAG Gateway → **ритмичный учебный цикл** (не бесконечный Сократовский допрос).

## Ритмичный учебный цикл (Learning Loop)

| Фаза | Что происходит |
|------|----------------|
| `intro_assessment` | Init: один экспресс-вопрос (Flash). |
| `dense_material` | Heavy: плотный материал, Mermaid, Rich Resource карточки, code snippets. |
| `checkpoint` | Короткая самопроверка в чате (Flash). |
| `pathway_decision` | Выбор траектории / финализация. |
| `socratic_focus` | Точечный Сократ только по запросу `[mode:socratic]`. |

**Модели:** `GEMINI_TUTOR_MODEL` (Lite chain) — intro, чат, dense в панели; `GEMINI_LITE_MODEL` — step_analysis / rolling_compress; `GEMINI_REASONER_MODEL` — только генерация маршрута (curriculum).

**Лекция (`dense_material`):** перед вызовом Gemini — `retrieve_lecture_rag_context()` (`lecture_rag_context.py`): hybrid LanceDB `document_summaries` (Consensus/скачанные) + `light_rag_facts` (TOP-K, `LECTURE_RAG_TOP_K`). Промпт с блоком `=== НАЧАЛО МАТЕРИАЛА ===` / fallback Best Practices.

**Probe:** перед основным запросом — `GEMINI probe ▶/✓/✗` (короткий ping, `GEMINI_PROBE_TIMEOUT_SEC`); quota store; первая рабочая модель в chain; при смене модели — `ChatSessionManager` Summary + handoff.

**UI:** панель mastery в drawer (`NodeMasteryPanel`), карточки ресурсов (`ResourceCard`), режимы Лекция / Блиц / Сократ. Лекция (`[mode:lecture]`) — полный текст в **чат** (`lecture_body` / `tutor_message`); панель (summary, Mermaid, refs) дополняет.

## Tiered Memory (4 слоя в промпте тьютора)

1. **Compressed RAG Profile** — сжатый срез фактов Модуля 3 (стек, опыт, пробелы по ноде).
2. **Core Concepts Matrix** — `core_concepts` с `status`, `evidence`, `mastery_score` (0–100).
3. **Rolling Dialogue Summary** — структурированный дайджест (`CURRENT_STATE`, `COVERED_POINTS`, `PENDING_ACTION`, `NEXT_ACTION_FOR_TUTOR`); не путает запрос лекции с выданным материалом.
4. **Active Dialogue Window** — последние 3 цикла (6 сообщений tutor/user); в API Gemini передаётся **один раз** при первом turn chat-сессии, далее только **delta** (`current_user_message`).

Персистентность: поле `memory` в `node_deep_dive_sessions.json` (+ `memory.chat_sessions` для Gemini); UI-история `history` сохраняется для чата.

## ChatSessionManager (изоляция Gemini)

- Каждая логическая сессия: `session_id` + `model_name` + `label` (intro / step_analysis / tutor / dense_material).
- **Первый turn** — полный static контекст (anchor + tiered без повторного окна в static; окно — один раз для tutor).
- **Следующие turns** — только `### current_user_message` (без дублирования истории в payload).
- **Fallback на другую модель** → новый `session_id`, `Context Type: Summary`, в API только `rolling_summary` + матрица (не сырая история прошлой модели).
- **Probe** (`GEMINI_PROBE_BEFORE_USE`) проверяет модели в chain до полного payload; `session_registry` синхронизирует `model_name` после probe.
- **Init ноды** → `clear_all` chat-сессий (Fresh).
- **Dense material** → всегда новая Fresh-сессия на Heavy.
- Лог: `[Session Created] ID: … | Model: … | Context Type: Fresh/Summary`.

Код: `knowledge_engine/services/chat_session_manager.py`, интеграция в `gemini_stateless.py` и `engine.py`.

## Step pipeline (chat / verify)

1. **Intent** — `ANSWER` | `INTENT_EXPLAIN` | `INTENT_SHIFT_FOCUS` | `INTENT_FINALIZE`.
2. **Mastery update** — сопоставление `user_message` + окна с матрицей концептов.
3. **Window rotation** — при >6 сообщений вытеснение в rolling summary (LLM-сжатие).
4. **Node status** — `in_progress` (0–39%), `deep_understanding` (40–99%), `gap`, `mastered` (100%, все verified).

## API

```http
POST /api/v1/node-deep-dive/interact
```

`user_action`: `init` | `chat` | `verify`.

## Ответ

`node_status`, `content`, `tutor_message`, `history`, `topic_mastery_score`, `concepts_matrix`, `mastery_dashboard`, `learning_phase`, `learning_mode`, …

## Надёжность (503 / Redis)

- `step_analysis` и `rolling_compress` — **GEMINI_LITE_MODEL** (меньше конкуренции с Flash).
- При ошибке step_analysis — **эвристический intent** (диалог не падает).
- При ошибке rolling_compress — текстовое слияние без LLM.
- Gemini 503/429: probe → retry + fallback по chain (логи `GEMINI probe`, `GEMINI wait`, `fallback ▶`).
- Redis: `REDIS_SOCKET_TIMEOUT_SEC` (default 30), retry на чтение job.

## Код

| Компонент | Путь |
|-----------|------|
| Engine | `engine.py` |
| Learning loop | `learning_loop.py` |
| Dense material (tutor Lite) | `services/node_content_generator.py` |
| Tiered memory | `tiered_memory.py` |
| Step pipeline | `step_pipeline.py` |
| Схемы памяти | `memory_schemas.py` |
| Store | `session_store.py` |

Сессии: `knowledge_engine/.runs/node_deep_dive_sessions.json`.
