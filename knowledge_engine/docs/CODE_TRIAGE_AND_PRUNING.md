# Триаж и prune исходного кода перед MAP

Как система сжимает файлы кода (`.c`, `.py`, `.js`, …) до отправки в Gemma MAP.
Три независимых механизма — ни один не включается/выключается тем же флагом,
что другой, и каждый работает на своём уровне гранулярности.

## 1. Три механизма

| # | Механизм | Файл | Уровень гранулярности | Управляется | Trace-маркер |
|---|----------|------|------------------------|-------------|--------------|
| 1 | **Tiered code pruner** | `ingest/tiered_code_pruner.py` | тело функции | ничем — безусловно для code-языков | `TieredCodePrune` |
| 2 | **AST code chunker** | `services/article_ingestion/ast_code_chunker.py` | границы MAP-чанков (top-level функции/классы) | `CODE_PARSER_MODE=ast` (default `linear`) | — (`[Pipeline Audit] Phase: Annotate`) |
| 3 | **Document triage** | `services/article_ingestion/document_triage_engine.py` | целые диапазоны `[P_n]` (TOC-секции) | `BLOG_SPATIAL_TRIAGE_ENABLED` (default `true`) | `DOC_TRIAGE` |

Порядок вызова при ingest кода (`raw_source.py::wrap_raw_source_as_annotated`):
**(1) tiered prune → (2) chunk boundaries (ast/linear) → (3) DOC_TRIAGE** (только
в полном пайплайне `ingest_blog_with_spatial_mapping`, не в изолированных
MAP+REDUCE вызовах вроде `map_reduce_summarize_blog_outcome_async`).

## 2. Tiered code pruner (`tiered_code_pruner.py`)

Работает **безусловно** для языков `python`, `c`, `cpp`, `javascript`,
`typescript`, `tsx` (`_PRUNE_LANGS`) — не зависит от `CODE_PARSER_MODE`.
Вызывается из `raw_source.py::wrap_raw_source_as_annotated` до любой
аннотации, значит применяется независимо от того, `ast` или `linear` потом
режет файл на MAP-окна.

Шаги:
1. tree-sitter (или встроенный `ast` для Python) извлекает каждую функцию:
   `AstFunctionSpan{name, signature, leading_comment, docstring, body, calls}`.
2. Каталог имён + **весь исходный файл** (включая тела) уходит в Gemini
   Flash-Lite (`classify_code_tiers_flash_lite`, label `tiered_code_prune`) на
   классификацию.
3. Модель относит каждое имя к одному из трёх тиров:
   - **HIGH** (= `FULL`) — архитектура, алгоритмы, entry points, состояние
     рантайма/потоков. Полное тело в MAP-контексте.
   - **MEDIUM** (= `HEADER_ONLY`) — хелперы/обёртки. Только сигнатура +
     docstring + комментарий `Calls HIGH: …`, тело вырезано.
   - **LOW** (= `SKIP`) — геттеры/сеттеры, логирование, тривиальный
     boilerplate. Функция выброшена из MAP-контекста целиком.
   Промпт явно просит "if unsure → MEDIUM, never LOW for unknown architectural
   role" (`_TIER_SYSTEM`) — консервативный биас в сторону сохранения.
4. `assemble_tiered_context` пересобирает файл по этим правилам, сохраняя
   исходный порядок и межфункциональные промежутки (импорты, макросы и т.п. —
   не относящиеся к функциям — не трогаются).

**Известное ограничение (не устранено):** шаг 2 передаёт в Flash-Lite полный
`raw_code` с телами всех функций — сам классификационный вызов не экономит
входные токены (тела нужны только на выходе HIGH-функций, но модель их видит
и для MEDIUM/LOW тоже). Для `pystate.c` (raw ≈108K символов) это означает
~27K+ токенов на один только классификационный запрос. Эффективность prune
также зависит от состава файла: для архитектурно-плотного кода (много HIGH)
итоговое сжатие скромное — на живом прогоне `pystate.c` дало 108431→80697
символов (**25.6%**), несмотря на то что 104 из 161 функции (65%) ушли в
MEDIUM/LOW — потому что 43 HIGH-функции по суммарному объёму доминируют над
файлом.

## 3. AST code chunker (`ast_code_chunker.py`)

Активен только при `CODE_PARSER_MODE=ast` (default — `linear`, то есть этот
чанкер **выключен** по умолчанию). Работает **после** tiered pruner — режет
уже prune'нутый текст.

- Реальное покрытие tree-sitter-грамматик (`EXTENSION_TO_LANGUAGE` →
  `tree_sitter_<lang>`): `c`, `cpp`, `python`, `javascript`, `typescript`,
  `tsx`, `go`, `rust`, `java`, `c_sharp`, `ruby`, `php`, `kotlin`, `swift`.
  Шире, чем можно было бы предположить по названию — не только Python/JS/TS.
- Границы чанков — top-level функции/классы (`_top_level_units`), крупные
  классы режутся по методам (`_split_large_class`), затем упаковываются в
  окна `TARGET_MIN/MAX_TOKENS` (300–500 токенов) через `_pack_units`.
- Любая ошибка парсинга (нет грамматики, tree-sitter `ERROR`-узел, функция
  длиннее `MAX_FUNCTION_LINES=150`) → откат на `linear_chunk_code`
  (тот же `wrap_raw_source_linear`, что и default-режим).

**`CODE_PARSER_MODE=linear`** (default, всегда активен если `ast` не
включён явно) — `wrap_raw_source_linear`: механическая нарезка по 40 строк на
блок `[P_n]`, без какого-либо синтаксического анализа. Не различает тело
функции, комментарий или `#include` — просто считает строки.

## 4. Document triage (`document_triage_engine.py`)

`BLOG_SPATIAL_TRIAGE_ENABLED=true` по умолчанию. Работает на уровне уже
аннотированных `[P_n]`-параграфов (после шагов 1–2 выше), независимо от того,
код это или HTML-статья. Требует ≥4 параграфов, иначе no-op.

1. `UniversalTOCExtractor` строит структуру документа (TOC-узлы).
2. `ArticleSectionPruner` решает, какие диапазоны параграфов оставить.
3. Если решение реально сокращает набор — `prune_annotated_article` вырезает
   остальное; фигуры (`FIG_*`) опционально восстанавливаются
   (`BLOG_SPATIAL_TRIAGE_KEEP_FIGURES`), даже если их параграф был вырезан.

Применяется только в полном пайплайне ingest (`ingest_blog_with_spatial_mapping`
/ `prepare_spatial_diagram_job`) — изолированные вызовы MAP+REDUCE напрямую
(`map_reduce_summarize_blog_outcome_async`, как в
`scripts/benchmark_pipeline_recall.py`) этот шаг пропускают.

## 5. Диагностика (trace-маркеры)

| Маркер | Смысл |
|--------|-------|
| `[Pipeline Audit] Phase: TieredCodePrune` | High/Medium/Low counts + итоговый размер после tiered prune |
| `DOC_TRIAGE ▶` / `DOC_TRIAGE ✓` | TOC-triage начат/завершён; `P {before}→{after}` при реальном сокращении |
| `[Pipeline Audit] Phase: Annotate \| raw_linear P=N` | `wrap_raw_source_linear` отработал (значит `ast`-чанкер не использовался или упал в fallback) |
| `AstCodeChunker fallback to linear` (warning) | tree-sitter не смог разобрать файл — проверить грамматику/синтаксис |
