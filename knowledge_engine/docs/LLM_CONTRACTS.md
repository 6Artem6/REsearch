# LLM Contracts — реестр Pydantic-схем

Единый каталог structured JSON для Gemini. Код: `knowledge_engine/schemas/llm_contracts/`.  
Runtime-реестр label → тип: `GEMINI_STRUCTURED_CONTRACTS` в `__init__.py`.

**Инвариант:** system prompt + `response_schema` задают контракт; UI/API не парсят произвольный prose как источник истины (кроме display-склейки).

---

## Node Deep-Dive / Tutor

Файл: [`tutor.py`](../schemas/llm_contracts/tutor.py). Подробности полей тьютора: [TUTOR_PROMPT_AND_UI_TEXT.md](TUTOR_PROMPT_AND_UI_TEXT.md).

| Контракт | Назначение | Ключевые поля | Labels (`GEMINI_STRUCTURED_CONTRACTS`) |
|----------|------------|---------------|----------------------------------------|
| `DeepDiveTutorContract` | Chat / verify dialogue + lecture_chat | `feedback_on_answer`, `technical_explanation`, `follow_up_question`, `question_sub_concept_id`, `referenced_diagram_id` (catalog id only; **нет** raw Mermaid / **нет** `tutor_message`) | `node_deep_dive/tutor` |
| `IntroAssessmentContract` | Intro / lazy intro | `tutor_message`, `node_status` | `node_deep_dive / intro_assessment` |
| `StructuredLectureResponse` | Dense lecture | `lecture_body`, `extracted_concepts`, `used_sources`, `referenced_diagram_id` (catalog id only), … | `node_deep_dive / dense_material` |
| `StepAnalysisContract` | Intent + mastery patch | `intent`, `concept_updates`, `critical_gap` | `node_deep_dive / step_analysis` |
| `SubConceptGapEvalContract` | Gap eval одного pending id | `updates[]` (0–1 × `SubConceptStatusUpdate`) | `node_deep_dive / sub_concept_gap` |
| `DialogueFactManifestContract` | Extract при ротации окна | `agreed_concepts`, `open_bottlenecks`, … | `node_deep_dive / fact_manifest` |
| `NodeExplainContract` | Пояснение выделения | `explanation`, `cited_source_ids` (`R*` / `S*`) | `node_selection_explain`, `node_deep_dive / node_explain`, `contextual_explainer` |

Вложенные (не в реестре labels отдельно): `VerifiedSourceReference`, `ConceptUpdateContract`, `SubConceptStatusUpdate`.

Helper: `structured_lecture_to_dense`, константа `STRUCTURED_LECTURE_FIELD_RULES`.

**Diagrams:** LLM contracts expose `referenced_diagram_id` (catalog asset id or null). Raw Mermaid is never accepted from the tutor/lecture model; the server resolves Mermaid from `article_diagrams` / `content.diagrams` into `NodeContentBlock.diagram` for the UI.

---

## Curriculum

Файл: [`curriculum.py`](../schemas/llm_contracts/curriculum.py).

| Контракт | Назначение |
|----------|------------|
| `CurriculumReasonerContract` | Legacy Reasoner DAG |
| `FlashCurriculumPayloadContract` | Search-First Flash DAG |
| `ModelFirstPayloadContract` | Model-First DAG (без URL) |
| `GeminiSourcesEnrichmentContract` | Lite enrichment whitelist |
| `ExpansionVectorContract` | Expand: вектор расширения |
| `FlashExpansionPatchContract` | Expand: patch нод/рёбер |

---

## Lite curriculum / search curation

Файл: [`lite_curriculum.py`](../schemas/llm_contracts/lite_curriculum.py).

| Контракт | Назначение |
|----------|------------|
| `LiteQueryPlanContract` | План поисковых запросов |
| `LiteAcademicQueryContract` | Academic Query Architect: `academic_query_en` |
| `ArxivQueryParamsContract` | `ti:` / `abs:` / categories / годы для Atom |
| `LiteBatchEvalContract` | Batch eval hits |
| `LiteSourceBatchContract` | Batch source eval items |
| `LiteSiteSuggestionsContract` | Site suggestions |

---

## Consensus / academic

Файл: [`consensus.py`](../schemas/llm_contracts/consensus.py). Поток: [ACADEMIC_AND_CONSENSUS.md](ACADEMIC_AND_CONSENSUS.md).

| Контракт | Назначение |
|----------|------------|
| `AcademicQueryContract` | Sanitize → `academic_query_en` для Consensus (preserved_terms + grounding) |
| `ValidationResultContract` | OK / RETRY / REJECT papers vs вопрос |
| `ProfileApplicabilityContract` | Нужен ли personal profile в validator |
| `RefinementSanitizeContract` | RETRY follow-up на английский academic |

Academic Architect (не Consensus.app): `LiteAcademicQueryContract` + `ArxivQueryParamsContract` в [`lite_curriculum.py`](../schemas/llm_contracts/lite_curriculum.py) — SS/arXiv plan.

---

## Source eval / domain / reasoner

| Файл | Контракт | Назначение |
|------|----------|------------|
| [`source_eval.py`](../schemas/llm_contracts/source_eval.py) | `SourceEvaluatorLiteContract` | Lite source evaluator |
| [`domain.py`](../schemas/llm_contracts/domain.py) | `DomainProfilerBatchContract` | Domain profiler batch |
| [`reasoner.py`](../schemas/llm_contracts/reasoner.py) | `FinalResponseContract` | Research reasoner final |

---

## Research v0.4 Gemini

Файл: [`v04_gemini.py`](../schemas/llm_contracts/v04_gemini.py).

| Контракт | Назначение |
|----------|------------|
| `GeminiL0DecompositionContract` | L0 decomposition |
| `L2EvidenceExtractionContract` | L2 evidence extract |
| `ResearchEvaluationContract` | Research eval |
| `AnalysisReportContract` | Matrix / analysis report |

---

## Analytics v0.7

Файл: [`analytics_v07.py`](../schemas/llm_contracts/analytics_v07.py).

| Контракт | Назначение |
|----------|------------|
| `ChunkExtractionContract` | Chunk extract |
| `ConceptGraphContract` | Concept graph |
| `ProfileGapMapContract` | Profile gaps |
| `TradeoffMatrixContract` | Trade-off matrix |

---

## VLM (article diagrams)

Файл: [`vlm.py`](../schemas/llm_contracts/vlm.py).

| Контракт | Назначение |
|----------|------------|
| `VlmBatchResponseContract` | Batch VLM diagram items (`VlmDiagramItemContract`) |

---

## Ingest gate (Flash Lite, before Gemma Map-Reduce)

Файл: [`paper_structure_schema.py`](../src/parsers/paper_structure_schema.py). Двухпроходный inbound gate: [`paper_structure_analyzer.py`](../src/parsers/paper_structure_analyzer.py).

| Контракт | Назначение | Labels |
|----------|------------|--------|
| `PaperStructureAnalysis` | Проход 1: `priority` CORE/CONTEXT/DROP, `topic_relevance` 0–10 | `ingest_gate / paper_structure` |
| `PaperCredibilityAnalysis` / `ParagraphCredibility` | Проход 2: `semantic_level`, `technical_correctness`, `information_density` | `ingest_gate / paper_credibility` |

Хост считает `P_i` из трёх осей (`CONTRADICTION` → 0) и `Q_article` как взвешенную сумму по `topic_relevance`. Блоги с `Q < 0.65` отклоняются целиком (`Failed parametric credibility score`); `CONTRADICTION` вырезается даже если порог пройден.

---

## Streaming ↔ контракты

Реализация: `services/gemini_json_stream.py` → `ChatSessionManager.send_chat_message_stream` / `gemini_stateless.run_gemini_structured_with_chain`.

| Schema | Stream filter | UI видит |
|--------|---------------|----------|
| `DeepDiveTutorContract` | `TutorDialogueFieldsStreamFilter` | склейка `feedback` → `technical` → `follow_up` |
| `NodeExplainContract` | `JsonFieldStreamFilter("explanation")` | дельты `explanation` |
| `StructuredLectureResponse` | `JsonFieldStreamFilter("lecture_body")` | дельты `lecture_body` |
| `IntroAssessmentContract` | `tutor_message` (via `structured_stream_text_field`) | дельты intro |

Резолв поля: `structured_stream_text_field(schema)` — для `DeepDiveTutorContract` возвращает `None` (специальный dialogue filter).

Explain SSE: [SKILL_TREE_UI.md](SKILL_TREE_UI.md) § Explain selection.

---

## См. также

| Документ | Тема |
|----------|------|
| [TUTOR_PROMPT_AND_UI_TEXT.md](TUTOR_PROMPT_AND_UI_TEXT.md) | Dialogue fields, modes, BLOCK 1–3 |
| [NODE_DEEP_DIVE_MODULE_2.md](NODE_DEEP_DIVE_MODULE_2.md) | LangGraph + single-writer coverage |
| [TUTOR_PIPELINES.md](TUTOR_PIPELINES.md) | Curriculum + tutor product map |
| [ARTICLE_DIAGRAMS.md](ARTICLE_DIAGRAMS.md) | VLM / Mermaid |
