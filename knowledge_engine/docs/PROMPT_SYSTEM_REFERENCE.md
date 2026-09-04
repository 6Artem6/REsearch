# Prompt System Reference — модели, контракты, переменные, роутинг

Единый справочник по всем LLM-ответственным контурам: Lecture Pipeline,
DEEP Map-Reduce Pipeline, Pre-Map Dedup, Tutor State Machine, Intent Router
Gate. Не дублирует [LLM_CONTRACTS.md](LLM_CONTRACTS.md) (полный реестр
Pydantic-контрактов Node Deep-Dive) — ссылается на него и добавляет то, чего
там нет: матрицу моделей по контурам, таблицу режимов `prompt_factory.py`,
инвентарь переменных промптов, контракты DEEP MAP/REDUCE/Dedup, guardrails
Evaluator Bypass и чек-лист расширения.

**См. также:** [LLM_CONTRACTS.md](LLM_CONTRACTS.md), [RAG_PIPELNES.md](RAG_PIPELNES.md),
[ARCHITECTURE_DEDUP.md](ARCHITECTURE_DEDUP.md), [EXA_SEARCH.md](EXA_SEARCH.md).

**Важная поправка к постановке задачи:** часть имён/классов в исходном ТЗ
(`IntentResult`, `EvaluatorOutputSchema`, `user_evaded_question`,
`lecture_prompt.py`, «Gemma Local SLM») — иллюстративные, не совпадают с
реальным кодом. Ниже — только проверенные по коду имена; расхождения отмечены
явно в сносках, а не молча скопированы.

---

## 1. Матрица ответственности моделей и провайдеров

| Контур / Модуль | Исполнитель | Режим | Задача | Вход | Тип выхода |
|---|---|---|---|---|---|
| `src/node_deep_dive/vector_intent_router.py` | **BGE-M3** (`BAAI/bge-m3`, `EMBED_MODEL`) | Sync / In-Memory Matrix (LanceDB `intent_vectors` кэшируется в RAM при старте) | Косинусный поиск интента, `sim ≥ VECTOR_INTENT_THRESHOLD` (0.82) | `user_message` (после regex-фильтра тегов) | `tuple[intent: str, score: float]` — **не** Pydantic-модель, см. §4.1 |
| `src/node_deep_dive/step_pipeline.py::run_step_analysis` | **Gemini Flash Lite** (`gemini-3.5-flash-lite` → fallback `gemini-3.1-flash-lite`) | Sync | Классификация intent (`INTENT_EXPLAIN`/`ANSWER`/…) + concept_updates | `user_msg`, `learning_phase`, `learning_mode`, matrix, fact manifest | Structured (`StepAnalysisContract`) |
| `sub_concept_evaluator.py` / `graph/nodes/sub_concept_eval.py` | **Gemini Flash Lite** | Sync | Педагогическая оценка ответа (`TechnicalConceptAudit`) — **только** если не control chip / не lecture / есть pending target | `user_msg`, sub-concept criteria, RAG-контекст | Structured (`audit` внутри `DeepDiveTutorContract`) |
| `prompt_factory.py::select_system_prompt_and_mode` + генерация чата | **Gemini Flash Lite** (`node_deep_dive/tutor` / `node_deep_dive/chat`) | Sync (stream) | Реплика тьютора в активном режиме (`default`, `lecture`, `blitz`, `socratic`, `self_check`, `next_module`, `gloss`, `deep_dive_how/mech`, `deep_analysis`/`advanced_analysis`/`deep_design`) | Isolated/default system prompt + user payload (RAG + sub-concepts + история) | `DeepDiveTutorContract` (eval) / `DeepDiveExplainContract` (skip) — Structured JSON, `technical_explanation`/`follow_up_question` стримятся в UI |
| `competency_extraction.py` | **Gemma Cloud** (`gemma-4-31b-it`, `COMPETENCY_EXTRACT_MODEL`-роль) | Async, fire-and-forget (`asyncio.Task`, UMA-lock non-blocking backoff) | Дельта компетенций пользователя из хода диалога | `user_message`, превью первых чанков ответа тьютора | Structured (`_ExtractJson` → `CompetencyDelta`) |
| `src/deduplication/pre_map_deduplicator.py` | **BGE-M3** (кластеризация) + **Gemini Flash Lite** (Bulk Gate, только suspect-группы) | Sync/Async, per batch | Union-Find дедуп источников до MAP, с backfill из резерва (`DEEP_INGEST_BACKFILL_MARGIN`/`LECTURE_PASSAGE_BACKFILL_MARGIN`) | Пуловые векторы кандидатов | `canonical_map` / `alias_of_url` — не Pydantic, `dict[str, str]` |
| `services/article_ingestion/blog_spatial_summarizer.py` (MAP) | **Gemma Cloud** (`GEMMA_PRIMARY_MODEL`, `BLOG_SPATIAL_MAP_PROVIDER=gemma_cloud`) | Async Worker, 16 000 TPM/модель sliding-window | Оконный (~2800 ток) извлекающий анализ, knowledge_atoms | `TokenWindowChunk` (AST/linear границы) | Structured (`MapWindowResponse`) |
| REDUCE Phase 1 (dedup atoms) | **BGE-M3** (`entity_consensus_engine`, приоритетный путь) **или** Gemma Cloud (fallback при отключении/ошибке) | Async Worker | Слияние `knowledge_atoms` без потери уникальных фактов | Пул `KnowledgeAtom` из всех MAP-окон | Structured (`DeduplicatedAtomsResponse`) — только при Gemma-фолбэке |
| REDUCE Phase 2 (synthesis) | **Gemma Cloud** (`GEMMA_PRIMARY_MODEL`, prompt caching) | Async Worker | Executive summary + takeaways из дедуплицированных atoms | Дедуп-atoms + summaries block | Structured (`FinalArticleSummaryResponse`) |

**Поправка к исходной таблице ТЗ:** Gemma в этой кодовой базе — **только Cloud**
(`GemmaCloudClient`, `llm.py`), локальных инстансов/Ollama нет ни в одном
контуре (см. очистку легаси-именования этой же сессией, `config.py`/`llm.py`
docstring «Gemma Cloud SSOT»). «REDUCE — Gemma / Flash Lite» из шаблона тоже
неточно: обе REDUCE-фазы бьют в Gemma Cloud, Flash Lite там не участвует;
Flash Lite занят в MAP-этапе только косвенно — через Domain Discovery/Content
Gate лекционного добора (см. [RAG_PIPELNES.md §2.5](RAG_PIPELNES.md)), не в
самом Map-Reduce.

---

## 2. Реестр сценариев диалога и Prompt Factory (`prompt_factory.py`)

### 2.1 Два механизма роутинга (Step 1 / Step 2)

```
user_message
    │
    ▼
Step 1 — Regex System Tag Parser (0 мс, без эмбеддинга)
    control_intent.py::_classify_explicit_and_exact
    паттерн: r"^\[(?:mode|action|intent|begin)(?::[^\]]+)?\]"
    │  найден тег → intent сразу (100% confidence)
    ▼ не найден
Step 2 — Dense Vector Match (BGE-M3 + LanceDB intent_vectors)
    control_intent.py::_classify_vector → VectorIntentRouter.classify()
    sim ≥ VECTOR_INTENT_THRESHOLD (0.82) → intent; иначе "" (свободный ответ)
```

Оба пути сходятся в `classify_control_chip()` → единая точка, откуда читают
и `sub_concept_eval_node` (bypass эвалюатора), и `coverage_router_node`
(переключение `learning_mode`), и `prompt_factory.select_system_prompt_and_mode`
(выбор изолированного промпта; для новых режимов — через
`_promote_vector_chip_to_mode`, т.к. векторный путь исторически не питал
Prompt Factory напрямую, только тег).

### 2.2 Сводка по режимам

| Active Mode | System Tag | `factory_modes` intent | Промпт-константа / файл | Evaluator Action | Persistent State? |
|---|---|---|---|---|---|
| `default` | — | — | нет изолированного; используется `default_system_prompt` (полный тьюторский compose) | Run Normal Eval | — |
| `gloss` | `[mode:gloss]` | `gloss` | `GLOSS_SUMMARY_PROMPT` / `gloss_summary_prompt.py` | Skip (`EVALUATOR_SKIP_INTENTS`) | Нет |
| `deep_dive_how` | `[mode:deep_dive_how]` | `how` | `DEEP_DIVE_HOW_PROMPT` / `deep_dive_how_prompt.py` | Run Normal Eval (`how`/`mech` в `EVALUATOR_SKIP_INTENTS` **не** входят) | Нет |
| `deep_dive_mech` | `[mode:deep_dive_mech]` | `mech` | `DEEP_DIVE_MECH_PROMPT` / `deep_dive_mech_prompt.py` | Run Normal Eval | Нет |
| `advanced_analysis`/`deep_design`/`deep_analysis` | `[mode:…]` | одноимённые | `ADVANCED_ANALYSIS_PROMPT`/`DEEP_DESIGN_PROMPT`/`DEEP_ANALYSIS_PROMPT` | Skip | Нет (overlay-стейт, не `learning_mode`) |
| `lecture` | `[mode:lecture]` | `lecture` | **нет отдельного изолированного файла** — `LECTURE_MODE_STRUCTURE_RULES` (`lecture_prompt_en.py`) **дописывается** к `default_system_prompt`, не заменяет его | Skip | Да (`memory.learning_mode = "lecture"`) |
| `blitz` | `[mode:blitz]` | `blitz` | `BLITZ_MODE_PROMPT` / `blitz_mode_prompt.py` | Skip | Да (`learning_mode = "express_blitz"`, `coverage_router.py::_apply_learning_mode_prefixes` + chip-веточка) |
| `socratic` | `[mode:socratic]` | `socratic` | `SOCRATIC_MODE_PROMPT` / `socratic_mode_prompt.py` | Skip | Да (`learning_mode = "socratic_point"`) |
| `self_check` | `[mode:self_check]` | `check` (переиспользует intent "check" — тот же, что у intro-чипа «проверка») | `SELF_CHECK_MODE_PROMPT` / `self_check_mode_prompt.py` | Skip | Нет (one-shot; персистентный `express_blitz` эффект intro-чипа «проверка» **не** триггерится mid-dialogue тегом — намеренно, см. `ARCHITECTURE_DEDUP`-уровня решение этой сессии) |
| `next_module` | `[mode:next_module]` | `next` (переиспользует intent "next" — тот же, что у чипа «Идем дальше») | `NEXT_MODULE_PROMPT` / `next_module_prompt.py` | Skip | Нет (transition, ноду переключает host/UI) |

**Поправка к исходной таблице ТЗ:** `lecture_prompt.py` как отдельный
изолированный файл не существует — лекционный режим устроен иначе, чем
blitz/socratic/self_check/next_module: `select_system_prompt_and_mode`
**дозаписывает** `LECTURE_MODE_STRUCTURE_RULES` поверх обычного тьюторского
system prompt (см. `prompt_factory.py`, ветка `mode == "lecture"`), а не
подменяет его целиком изолированным промптом.

`_FACTORY_CONTROL_MODES` (форсирует `route="tutor"`, никогда dense-лекцию):
`deep_dive_mech`, `deep_dive_how`, `deep_analysis`, `advanced_analysis`,
`deep_design`, `gloss`, `blitz`, `socratic`, `self_check`, `next_module`.
`lecture` в этот набор **не входит** — у него свой `route="dense"`.

---

## 3. Реестр динамических переменных (Prompt Variables Directory)

**Оговорка по формату:** в кодовой базе это **не** буквальные Python
`.format()`/f-string плейсхолдеры `{var}` внутри одного шаблона — system/user
промпты собираются конкатенацией текстовых блоков с `=== SECTION ===`
маркерами (`tutor_prompt_builder.py::compose_system_prompt`,
`dialog_context.py`). Ниже `{var}` — условное обозначение логической единицы
контекста для навигации по таблице, не буквальный синтаксис кода.

### А. Контекст знаний и RAG-слоя

| Переменная | Тип | Источник (модуль/таблица) | Потребитель | Описание |
|---|---|---|---|---|
| `{lecture_context}` | `str` (Markdown, склейка чанков) | `services/lecture_rag_context.py::retrieve_lecture_rag_context()` — LanceDB `document_summaries` + `knowledge_nodes` hybrid + LightRAG → CE → MMR | `generate_dense_material()`, блок `=== НАЧАЛО МАТЕРИАЛА ===` промпта тьютора | Локальный RAG-контекст перед лекцией; см. [LECTURE_RAG_CONTEXT.md](LECTURE_RAG_CONTEXT.md) |
| `{ex_search_results}` | `list[CurriculumSearchHit]`/passages | `src/node_deep_dive/lecture_search_orchestrator.py::_exa_sources_multi_vector` (Exa, только если `local_sources_count < LECTURE_MIN_LOCAL_SOURCES`) | Дозаполнение `{lecture_context}` при нехватке локального материала | Внешний добор через Exa, BGE-M3 MMR passages; см. [RAG_PIPELNES.md §2](RAG_PIPELNES.md) |
| `{current_node_title}` / `{topic_name}` | `str` | `NodeDataInput.title` (payload запроса) | Все system/user промпты тьютора | Заголовок текущей ноды curriculum |
| `{node_sub_concepts}` | `list[SubConceptRecord]` → сериализация в текст | `memory.sub_concepts` (`concept_map.py::ensure_sub_concept_map`) | `step_pipeline.py`, `sub_concept_evaluator.py`, генерация | Подконцепты + критерии (`why_passed`/`how_passed`/`mechanic_passed`) |

### Б. Диалоговое состояние

| Переменная | Тип | Источник | Потребитель | Описание |
|---|---|---|---|---|
| `{learning_mode}` | `Literal["lecture","express_blitz","socratic_point"] \| ""` | `memory.learning_mode` (`memory_schemas.py::LearningMode`) | `tutor_behavior_state.py::resolve_tutor_mode`, `learning_loop.py` | Персистентный активный режим; отдельная ось от `TutorFactoryMode` (§2) |
| `{evaluator_skipped}` | `bool` | `memory.evaluator_skipped` (`sub_concept_evaluator.py::mark_evaluator_skipped`/`mark_evaluator_ran`) | `select_system_prompt_and_mode` (переключает JSON-контракт на `DeepDiveExplainContract` + `EVALUATOR_SKIPPED_TUTOR_RULES`) | **Реальный** аналог иллюстративного `evaluator_skipped` из ТЗ — существует буквально под этим именем |
| `{evaluator_reasoning}` | — | **не существует как отдельное поле** | — | В ТЗ упомянуто иллюстративно; реально — `trace(...)`-лог (`mark_evaluator_skipped(memory, reason)`), не структурное поле payload'а промпта |
| `{dialogue_history}` | `list[dict]` → sliding window text | `memory.active_window` (`tiered_memory.py`), `ChatSessionManager` (stateful Gemini chat session, если `context != Fresh`) | User payload всех тьюторских вызовов | Скользящее окно предыдущих реплик; ротация — `step_pipeline.py::rotate_window_after_message` |

### В. Переменные асинхронного DEEP Pipeline (Gemma & Dedup)

| Переменная | Тип | Источник | Потребитель | Описание |
|---|---|---|---|---|
| `{chunk_text}` | `str`, ≤`BLOG_SPATIAL_MAP_MAX_TOKENS` (2800 ток, `config.py`, не env-переопределяемо) | `paragraph_token_splitter.py::TokenWindowChunk` (AST-границы при `CODE_PARSER_MODE=ast`, иначе linear) | MAP-промпт (`_prompt_for_window`, `blog_spatial_summarizer.py`) | Оконный фрагмент документа/кода перед MAP-вызовом |
| `{map_summaries_batch}` | `list[MapWindowResponse]` → `_format_reduce_summaries_block` | Результаты всех MAP-вызовов одного `MapReduceArticleJob` | REDUCE Phase 1/2 промпты | Сводки окон, объединяемые в один REDUCE-батч |
| `{backfill_margin}` | `int` | `DEEP_INGEST_BACKFILL_MARGIN` (2, DEEP) / `LECTURE_PASSAGE_BACKFILL_MARGIN` (3, Lecture) — `config.py` | `replenish_valid_hits_until_cap` / `postprocess_exa_hits_for_external_recall` | Резерв кандидатов сверх целевого `cap`, из которого добирается замена при обнаружении near-duplicate (BGE-M3 Union-Find, cosine ≥ 0.80 на обеих сторонах); подробности — [RAG_PIPELNES.md §4.1](RAG_PIPELNES.md) |

---

## 4. Контракты Pydantic-схем и JSON Structured Output

**Оговорка:** классы `IntentResult` и `EvaluatorOutputSchema` из ТЗ **не
существуют** в коде — ниже документированы реальные структуры возврата.

### 4.1 Intent Router Gate — реальная форма возврата

Нет единого Pydantic-класса; возврат — обычный кортеж/строка:

```python
# src/node_deep_dive/vector_intent_router.py
VectorIntentRouter.classify(user_text, *, allowed_intents=None, threshold=None) \
    -> tuple[str, float]          # (intent, cosine_score); ("", 0.0) при отсутствии матча

# src/node_deep_dive/control_intent.py
classify_control_chip(user_text, *, memory=None) -> str            # intent | ""
classify_control_chip_detailed(user_text, *, memory=None) -> tuple[str, str]  # (intent, source="exact"|"vector"|"fallback")
is_control_chip_message(user_text, *, memory=None) -> bool          # intent in EVALUATOR_SKIP_INTENTS
```

Если нужен структурный объект по образу иллюстративного `IntentResult` —
ближайший эквивалент: `(intent, source)` из `classify_control_chip_detailed`
+ `is_control_action = intent in EVALUATOR_SKIP_INTENTS` (вычисляется на
вызывающей стороне, не поле схемы). `canonical_tag` — реконструируется из
`IntentRule.system_mode` (`intent_definitions.py`), не возвращается роутером
напрямую.

### 4.2 Evaluator — `TechnicalConceptAudit` (реальное имя, `schemas/drill_schemas.py`)

```python
class TechnicalConceptAudit(BaseModel):
    feedback_kind: Literal["EXACT", "NEEDS_CORRECTION"]
    accuracy_grade: AnswerAccuracyGrade  # EXACT_AND_CORRECT | PARTIAL | NEEDS_CORRECTION | MISUNDERSTANDING
    user_claims_analysis: list[str]                 # min_length=1, max_length=16
    detected_errors_or_misconceptions: list[str] = []
    confirmation: str = ""            # EXACT-ветка; пусто на NEEDS_CORRECTION
    praise_points: list[str] = []     # обязательно на PARTIAL; пусто на EXACT
    correction_breakdown: str = ""    # NEEDS_CORRECTION-ветка; пусто на EXACT
```

Поле `user_evaded_question` из ТЗ **отсутствует** — вместо булева флага
внутри схемы оценки, «уклонение» решается **до** вызова эвалюатора: если
`is_control_chip_message()` истинно (или lecture-запрос, или нет
`pending_evaluation_concept_id`), `sub_concept_eval_node` вообще **не
вызывает** `process_sub_concept_user_answer` — `TechnicalConceptAudit` не
генерируется в этом ходу вовсе (см. §5).

`TechnicalConceptAudit` — вложенное поле `audit` внутри
`DeepDiveTutorContract` (см. [LLM_CONTRACTS.md](LLM_CONTRACTS.md)), не
самостоятельный top-level контракт.

### 4.3 DEEP MAP-фаза — `MapWindowResponse` (реальное имя, `blog_spatial_schemas.py`)

```python
class MapWindowResponse(BaseModel):
    window_role: str = ""                       # короткий тег, 2–6 слов
    window_summary: str                          # обязателен
    knowledge_atoms: list[KnowledgeAtom] = []     # max_length=24
    required_diagrams: list[WindowDiagramCheck] = []

class KnowledgeAtom(BaseModel):
    scope: ScopeType                 # PRINCIPLE | MECHANIC | INSTANCE
    statement: str                   # min_length=8, max_length=2000
    context_quote: str | None = None
    source_chunk_ids: list[str] = []
```

Иллюстративный `DeepMapChunkAnalysis{key_concepts, technical_facts,
code_snippets_meta}` из ТЗ — не совпадает с реальными именами полей;
смысловой эквивалент есть (`knowledge_atoms` со `scope=INSTANCE` покрывает
и код-факты, и числа), но конкретные ключи другие.

### 4.4 DEEP REDUCE-фаза — два отдельных контракта, не один

```python
class DeduplicatedAtomsResponse(BaseModel):   # Phase 1 (Gemma-фолбэк; приоритет — BGE-M3 entity_consensus, без LLM)
    knowledge_atoms: list[KnowledgeAtom] = []  # max_length=32

class FinalArticleSummaryResponse(BaseModel):  # Phase 2 — синтез
    executive_summary: str
    key_takeaways: list[str]        # 3–7 строк, префикс [SCOPE: ...]
    knowledge_atoms: list[KnowledgeAtom] = []
    target_diagrams_for_vlm: list[WindowDiagramCheck] = []
```

Иллюстративный `KnowledgeNodeSchema{node_id, prerequisites, sub_concepts}`
из ТЗ — это структура **графового узла curriculum** (`src/curriculum/schemas.py`,
отдельная подсистема генерации курса), а не выход Gemma REDUCE над статьёй.
REDUCE Map-Reduce пишет `FinalArticleSummaryResponse` в LanceDB
`document_summaries` (см. [RAG_PIPELNES.md §3.5](RAG_PIPELNES.md)) — это
источники/конспекты, не узлы графа курса.

---

## 5. Правила защиты Evaluator Node (Bypass Guardrails)

1. **`EVALUATOR_SKIP_INTENTS`** (`intent_definitions.py`) — 14 интентов:
   `gloss, how, mech, deep_analysis, advanced_analysis, deep_design, next,
   practice, check, skip, begin, lecture, blitz, socratic`. Любой intent из
   этого набора помечает ход как control-action, не content-answer.
2. **Точка отсечения** — `graph/nodes/sub_concept_eval.py::sub_concept_eval_node`:
   до вызова `process_sub_concept_user_answer` (который производит
   `TechnicalConceptAudit`) проверяются, в порядке:
   `empty user_message` → `no pending_evaluation_concept_id` →
   `is_lecture_request_message()` → `is_quick_reply_control_message()`
   (делегирует в `classify_control_chip`/`EVALUATOR_SKIP_INTENTS`). При любом
   срабатывании — `mark_evaluator_skipped(memory, reason)` и **немедленный
   выход**, без обращения к LLM-эвалюатору вообще.
3. **`memory.evaluator_skipped: bool`** — устанавливается
   `mark_evaluator_skipped`/`mark_evaluator_ran`; читается в
   `prompt_factory.py::select_system_prompt_and_mode`.
4. **Замена JSON-контракта** (не булев флаг внутри схемы, а смена самой
   схемы): при `evaluator_skipped=True` — `system` промпт дописывается
   `EVALUATOR_SKIPPED_TUTOR_RULES`:

   ```
   === EVALUATOR SKIPPED (HARD — Host did not score this turn) ===
   Do NOT emit `audit` / TechnicalConceptAudit / confirmation /
   correction_breakdown. There is no learner-answer verdict this turn
   (control chip, lecture request, empty/short text, or no pending
   evaluation target).
   Output DeepDiveExplainContract: technical_explanation + optional
   follow_up_question only. Teach or clarify; do not grade.
   ```

   Модель физически не может вернуть `audit`/обвинение в уклонении — схема
   `DeepDiveExplainContract` не содержит таких полей вовсе (см. §4.2).
5. **Итог vs формулировка ТЗ:** вместо `user_evaded_question = False`
   (иллюстративное поле) реальная защита двухслойная — (а) LLM не вызывается
   с эвалюационной схемой вообще (host-side gate, шаг 2), (б) если бы и была
   вызвана — контракт `DeepDiveExplainContract` структурно не допускает
   вердикта. Обвинение в уклонении невозможно на уровне схемы, а не
   опровергается постфактум флагом.

---

## 6. Чек-лист разработчика: добавление нового режима/промпта

1. **Модель и режим вызова.** Диалоговая реплика тьютора → Gemini Flash
   Lite, Sync. Асинхронная обработка документа/чанка (MAP/REDUCE/dedup) →
   Gemma Cloud, Async Worker (учесть 16 000 TPM/модель,
   `GEMMA_MAP_FORCE_PER_MODEL_LIMITS`). Лёгкая batch-классификация
   (Domain Discovery, Content/Bulk Gate, competency extraction) → Flash Lite
   или Gemma в зависимости от контура — см. §1.
2. **Файл промпта** — новый `*_mode_prompt.py` в `src/node_deep_dive/`
   (или `*_prompt.py` для DEEP/dedup контуров), константа EN-текстом +
   RU-докстринг-комментарий под ней (project convention, см.
   `blitz_mode_prompt.py` как образец). Явно указать в тексте промпта:
   держать `summary`/`references` пустыми, если режим one-shot и не должен
   тратить генерацию на панель Materials: в `DeepDiveExplainContract`/
   `DeepDiveTutorContract` поля `summary`/`references` объявлены **раньше**
   `technical_explanation`, а в UI стримятся только
   `technical_explanation`/`follow_up_question` (`TUTOR_EXPLAIN_STREAM_FIELDS`)
   — модель, не ограниченная явно, может потратить заметную часть генерации
   на невидимые для пользователя поля до того, как начнёт стримиться видимый
   текст (это добавлено в blitz/socratic/self_check/next_module prompt
   этой же сессией — см. п. 4 «summary=""/references=[]» в каждом файле).
3. **`intent_definitions.py`** — добавить `IntentRule` (новый intent) или
   расширить существующий (`reference_phrases`, `system_mode="[mode:...]"`,
   `factory_modes=("...",)`); если режим должен блокировать педагогическую
   оценку — добавить имя intent'а в `EVALUATOR_SKIP_INTENTS`. LanceDB
   `intent_vectors` пересоздаётся автоматически при рассинхроне с SSOT
   (`VectorIntentRouter.sync_and_validate_intents`, сверка по
   `COL_EMBED_MODEL`/hash id) — вручную ничего пересоздавать не нужно.
4. **`prompt_factory.py`** — добавить токен в `TutorFactoryMode` Literal и
   в **оба** места, где перечислен whitelist токенов: regex `_MODE_PREFIX_RE`
   **и** внутренний `if mode in (...)` в `parse_tutor_mode_prefix` (это два
   независимых списка, синхронизировать руками — учтено на своём опыте этой
   сессии: забыть второй список — тег будет матчиться регэкспом, но
   тихо схлопываться обратно в `"default"`). Добавить в
   `_FACTORY_CONTROL_MODES`, если режим не должен уходить в dense-лекцию.
   Подключить промпт-константу веткой `elif mode == "...":` в
   `select_system_prompt_and_mode` + запись в `select_isolated_prompt_for_mode`.
   Если нужна поддержка свободного текста без тега — добавить имя factory
   mode в `_VECTOR_PROMOTABLE_FACTORY_MODES`.
5. **Persistent state (опционально).** Если режим должен переключать
   `memory.learning_mode` на весь диалог (как blitz/socratic), а не быть
   one-shot (как self_check/next_module) — добавить ветку в
   `coverage_router.py::coverage_router_node` (chip-based, рядом с
   существующей `elif chip == "blitz": set_learning_mode(...)`), а не только
   в `_apply_learning_mode_prefixes` (та покрывает только буквальный тег,
   не векторный матч свободного текста).
6. **Pydantic-схема** — нужна только если режим возвращает НЕ
   `DeepDiveExplainContract`/`DeepDiveTutorContract` (эти два уже покрывают
   и skip-, и normal-eval сценарии тьютора). Новая схема — в
   `schemas/llm_contracts/tutor.py` + регистрация в
   `GEMINI_STRUCTURED_CONTRACTS` (см. [LLM_CONTRACTS.md](LLM_CONTRACTS.md)).
7. **Тесты** — `test_intent_definitions.py` (валидность каталога,
   `EVALUATOR_SKIP_INTENTS`, exact/vector резолюция тега без эмбеддинга),
   `test_prompt_factory.py` (parse + select_system_prompt_and_mode +
   `is_factory_control_mode` + explain-tail при `evaluator_skipped=True`),
   `test_control_intent_guardrails.py` (регрессия существующих чипов).
   Векторные тесты — через offline `lexical_probe_embed`
   (`tests/intent_embed_probe.py`), не настоящий BGE-M3/LanceDB.
