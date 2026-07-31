# Тьютор: промпты, контекст диалога, отображение LLM-текста

Дополнение к [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md).

## Prompt compositor

Единая сборка system prompts — `src/node_deep_dive/tutor_prompt_builder.py`:

| Функция | Режим |
|---------|--------|
| `build_intro_system()` | init, один вопрос |
| `build_dialogue_system()` | dialogue_feedback |
| `build_lecture_chat_system()` | lecture_dense (Flash-маршрутизатор) |
| `build_dense_system()` | Heavy `generate_dense_material` |

Правила форматирования, grounded architecture, whitelist — в compositor, не дублировать в payload.

## Слои контекста чата (не лекция)

`src/node_deep_dive/dialog_context.py`:

1. **Static prefix** — RAG profile, матрица, phase/mode, node curriculum block.
2. **Anchor + fact manifest** — якорь intro + JSON manifest (не prose rolling summary на hot path).
3. **Dynamic suffix** — sliding window (6) + `current_user_message`.

State vector тьютора: `tutor_behavior_state.py` → JSON в payload (`step_intent`, `current_mode`, …).

## Step pipeline

`step_pipeline.py`:

- Lite **step_analysis** — intent, concept_updates, critical_gap.
- При вытеснении из окна — **fact manifest** (`fact_manifest.py`, Lite extract), не `rolling_compress` на каждый chat turn.
- `rolling_dialogue_summary` — legacy поле; handoff при смене модели.

## Отображение trade-off JSON в UI

LLM иногда вставляет в `summary` / `tutor_message` сырой JSON (`title`, `pros`, `cons`, `takeaways`, `failure_modes`).

| Слой | Модуль |
|------|--------|
| Backend markdown→HTML | `repair_structured_analysis_json()` в `web/llm_text_repair.py` → `llm_markdown_service.llm_markdown_to_html` |
| Skill Tree fallback | `structuredAnalysisToHtml()` в `web/static/skill-tree/llmTextRepair.js` |

Панель «Суть механики» и чат используют `summary_html` / `contentHtml`; при их отсутствии — клиентский fallback.

## Skill Tree bundle

После правок в `web/static/skill-tree/*.js`:

```bash
cd knowledge_engine/web/static/skill-tree && npm run build
```

Импорты из `api.js` (например `sortDialogMessages`) должны быть явными в каждом компоненте.
