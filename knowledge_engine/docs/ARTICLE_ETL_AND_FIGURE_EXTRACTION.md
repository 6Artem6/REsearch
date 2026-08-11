# ETL статей: суммаризация, извлечение FIG и VLM для схем

Документ для handoff: как система превращает URL/PDF/HTML в текст для RAG, `FIG_*` для VLM и Mermaid в `article_diagrams`. Акцент на **разрыве между «идеальной» моделью FIG и реальными PDF** (ACM, векторные схемы, full-page raster).

## 1. Цели продукта

| Цель | Артефакт | Потребитель |
|------|----------|-------------|
| Суммаризация / ключевые тезисы | `DocumentSummary` → LanceDB | curriculum, tutor context |
| Технические схемы как Mermaid | `article_diagrams` (SQLite) | `content.diagrams` на ноде, pinned lecture |
| Размеченный текст | `AnnotatedArticle` (`[P_n]`, `[FIG_m]`) | Map-Reduce LLM, triage, VLM контекст |

Схемы **не** хранятся как оригинальные PNG в БД: VLM классифицирует картинку и генерирует **Mermaid** + caption/summary. Оригинальные байты живут только transiently в `AnnotatedArticle.fig_bytes` и в пайплайне VLM.

## 2. Единая модель разметки: `AnnotatedArticle`

Файл: `services/parsers/html_annotator.py` (`AnnotatedArticle`).

- `annotated_markdown` — линейный текст с маркерами `[P_1] …` и `[FIG_1: alt="…" | page="…"]`.
- `paragraph_map` / `paragraph_page` — id → текст / номер страницы PDF.
- `fig_map` — `FIG_n` → URL или `embedded:FIG_n`.
- `fig_bytes` — `FIG_n` → `(bytes, mime)` для embedded/VLM.
- `source_pdf_bytes` — сырой PDF для повторного resolve FIG (vector crop).

**Критично:** id `FIG_n` в разметке **не гарантирует** качественный bitmap. Это **логический** идентификатор места в документе; байты могут быть xref, caption-crop или «полоса колонки».

## 3. Входные точки (кто запускает ETL)

| Точка | Код | Что делает |
|-------|-----|------------|
| Harvest practical/blog | `src/curriculum/source_material_pipeline.py` | `_try_blog_spatial_diagrams`, batch `prepare_spatial_diagram_job` |
| Академический harvest | `maybe_ingest_article_diagrams` | spatial ingest + fallback |
| Ручной / CLI | `auto_ingest.py`, `run_blog_spatial_diagram_ingest` | то же |
| QA только картинки | `scripts/export_pdf_figures.py` | **только** `extract_figures_pymupdf`, **без** Map/VLM |

Полный spatial ingest: `blog_spatial_pipeline.py` — registry → VLM → Map-Reduce → LanceDB.

## 4. Fetch и discovery (до парсинга)

Цепочка для «универсального» URL (ACM DOI, блог, PDF):

1. `article_resource_discoverer` — manifest: canonical URL, DOI, ссылки PDF/eReader, кэш.
2. `smart_fetcher` — httpx + Playwright; приоритет `dl.acm.org/doi/pdf/` над epdf; `is_parseable_pdf` (PyMuPDF `page_count ≥ 1`, мин. размер).
3. Fallback: Sci-Hub PDF для academic (`_academic_pdf_fallback_bytes` в `blog_spatial_pipeline`).
4. **Плохой PDF** (~467 KB epdf intercept, `page_count=0`) → annotate находит **0 FIG**; нужен parseable `/doi/pdf/`.

Manifest кэшируется по URL+DOI; повторный harvest не должен закреплять битые байты без ручной очистки кэша.

## 5. Annotate: HTML vs PDF

### HTML / Markdown

- `build_annotated_article` / `build_annotated_markdown` — `<img>` → `FIG_n` + URL в `fig_map`; байты часто **не** загружаются до VLM (`load_image_bytes`).
- Схемы как отдельные DOM-узлы — **наиболее предсказуемый** путь.

### PDF

`build_annotated_pdf` (`pdf_annotator.py`):

1. По страницам: `get_text("dict")` — блоки type 0 (текст) → `P_n`; type 1 (image block) → `FIG_n` если не full-page.
2. `get_images()` + `extract_image(xref)` — embedded raster, **skip** если bbox > ~42% страницы.
3. `_merge_vector_crops` — `VectorPDFCropper` по подписям Fig. N.
4. Если `fig_bytes` пусто — `extract_figures_pymupdf` (pymupdf pass).

Legacy parser `PdfArticleParser` (`pdf_parser.py`) — список `ExtractedImage` **без** `[P_n]`/`FIG_n` разметки; используется в **fallback** `ArticleIngestionPipeline.ingest`, не в spatial Map-Reduce.

## 6. Извлечение FIG из PDF — три стратегии и почему они конфликтуют

Это **ядро текущей проблемы** (включая «текстовый» native PDF, где export всё равно «режет по тексту»).

### 6.1. Caption-driven crop (`VectorPDFCropper`)

Файл: `services/parsers/vector_pdf_cropper.py`.

- Находит якоря: regex + `search_for("Fig. N")` в **текстовом слое**.
- Колонка (левая/правая) по X подписи.
- Вертикальный clip: от предыдущей подписи на странице (или верха) до текущей; лимит высоты (~300 pt).
- Опционально сужение по `get_drawings()`; если drawings пусто — **рендер clip зоны** (`page.get_pixmap(clip=…)` 300 DPI).

**Плюсы:** работает когда есть текстовые подписи и нативный PDF (не один bitmap на страницу).

**Минусы:**

- Не извлекает «image object» — **скриншот прямоугольника страницы**.
- В clip попадают абзацы, несколько схем в одной колонке, подпись.
- Векторные ACM-схемы (сетки, формулы как текст) часто **без** отдельного xref и **без** полного покрытия в `get_drawings()`.

### 6.2. Embedded image xref / type=1 block

Файлы: `pdf_annotator._page_events`, `pymupdf_figure_extract`.

- Берёт **сырые** байты JPEG/PNG из PDF stream.
- Фильтр: bbox **не больше ~42%** площости страницы (защита от full-page raster).

**Плюсы:** чистая картинка без текста страницы, когда схема реально отдельный raster.

**Минусы:**

- Многие paper figures **не** отдельный xref (вектор + текст).
- Full-page scan → xref отбрасывается → остаётся только caption/layout путь.
- Dedup по hash в `extract_figures_pymupdf` может скрыть второй источник для того же визуала.

### 6.3. Raster layout bands (`raster_pdf_layout`)

Файл: `services/parsers/raster_pdf_layout.py`.

- Только если на странице **мало** extractable text (< ~120 символов) и страница выглядит как одна большая картинка.
- Делит колонки на вертикальные полосы по «чернилам» в pixmap (numpy).

**Плюсы:** хоть что-то для scan-PDF без текстового слоя.

**Минусы:** полосы ≠ Fig. N; смесь текста и схемы; **не активен** на native text PDF (триаж идёт через caption-crop).

### 6.4. Порядок в `extract_figures_pymupdf`

1. `VectorPDFCropper` (caption-crop)  
2. `discover_raster_column_figures` (если почти нет текста)  
3. type=1 blocks и xref (маленькие)  
4. Синтетические `FIG_SEQ_*` / alias `FIG_i`

**Скрипт `export_pdf_figures.py` использует только этот модуль** — он **не** отражает полный ingest (нет triage, Map, VLM). Если caption-crop доминирует, QA-папка будет «по тексту», даже при наличии image blocks в PDF.

## 7. Document triage (перед Map-Reduce)

`document_triage_engine.py` + `ArticleSectionPruner`:

- TOC из PDF/HTML (`toc_extractor`).
- Удаление «шумных» секций (references, boilerplate) → `prune_annotated_article`.
- `BLOG_SPATIAL_TRIAGE_KEEP_FIGURES`: после prune **восстанавливает** FIG из оригинала, чтобы VLM не потерять схемы из отрезанного текста.

Triage **не улучшает** геометрию FIG — только объём текста для LLM.

## 8. Spatial pipeline (registry → VLM → Map → Reduce)

Оркестрация: `blog_spatial_pipeline.py` — **registry → VLM → Map → Reduce → LanceDB**.

`blog_spatial_summarizer.py` (Gemma cloud / Ollama fallback):

- Текст режется на token windows; в MAP user-промпт вставляется `[ATTACHED_DIAGRAMS]` из `FigureRegistry` (`map_diagram_attach.py`).
- **MAP:** локальная выжимка окна с фактами из VLM-описаний схем (без выбора id для VLM).
- **REDUCE:** единый `FinalArticleSummaryResponse` / `DocumentSummary`.

Реестр: `persist_figure_registry` + `run_vlm_on_registry` до Map-фазы (`figure_registry_service.py`).

Legacy: `ingest_spatial_maps_batch_vlm`, `target_diagrams_for_vlm` — старые пути.

**LLM не чинит плохие crops.** Плохой bitmap → плохой Mermaid.


## 9. Загрузка байтов для VLM

`spatial_diagram_dispatch._load_figure_bytes`:

1. `annotated.fig_bytes[FIG_n]`
2. URL из `fig_map` → `load_image_bytes`
3. `VectorPDFCropper.resolve_figure(source_pdf_bytes, FIG_n)` — повторный caption-crop
4. Иначе `⊘ not loadable`

VLM batch: `ArticleIngestionPipeline.ingest_vlm_targets` → sanitizer (pHash, размер) → optional Ollama smart filter → Gemini VLM pool → validate/repair Mermaid → `save_diagram`.

Fallback ingest (`pipeline.ingest` без spatial): `PdfArticleParser` / Html — **все** картинки из парсера, smart filter, без Map-Reduce выбора.

## 10. Хранение и нода

- SQLite: `knowledge_engine/.runs/article_diagrams.db` (`ARTICLE_DIAGRAMS.md`).
- `canonical_article_id(source_id, normalized_url)` — связь с registry; DOI → `https://doi.org/{doi}`.
- `article_diagram_context` + `content_assets.hydrate_content_diagrams_from_articles` + `diagram_session` — Mermaid в UI ноды.

Пустой `article_diagrams` для ACM часто означает: **0 FIG после annotate**, **0 target после MAP**, или **VLM отсечил** / not loadable.

## 11. Диагностика (trace маркеры)

| Маркер | Смысл |
|--------|--------|
| `BLOG_SPATIAL ingest ⊘ | corrupt pdf` | epdf intercept, не parseable |
| `BLOG_SPATIAL ingest ⊘ | no figures after annotate` | нет fig_map/fig_bytes |
| `BLOG_SPATIAL ingest ▶ | pymupdf figure extract` | последний шанс pymupdf |
| `VECTOR_PDF` / `PYMUPDF_FIG` | caption-crop / xref / layout |
| `DOC_TRIAGE` | prune P/FIG |
| `BLOG_SPATIAL pool` | Map-Reduce |
| `BLOG_SPATIAL vlm ⊘ | FIG_X not loadable` | нет байтов |
| `ARTICLE_INGEST` / `VLM pool` | VLM и сохранение |

Для QA PDF: `export_pdf_figures.py` печатает `extractable_text_chars` — если 0, файл raster-only; если тысячи — caption-path активен.

## 12. Типы PDF и ожидаемое поведение

| Тип | Текст | Images | Типичный результат сейчас |
|-----|-------|--------|---------------------------|
| Блог HTML | DOM | `<img>` | Хорошо (URL → load) |
| ArXiv Figure 1 raster | Да | Один xref на фигуру | xref или caption-crop, часто ок |
| ACM native | Да | Вектор/текст-сетки, редкий xref | **Caption-crop**, часто с текстом в crop |
| ACM epdf / preview | Нет/мало | Full-page xref | layout bands или 0 FIG |
| Scan 10× full-page | Нет | 1 xref/page | skip xref + layout полосы |

## 13. Целевое направление решений (не реализовано целиком)

1. **Image-first:** для каждого малого xref/block — match ближайшей подписи Fig. N снизу; caption-crop только fallback.
2. **Две Fig. в одной колонке:** жёсткие вертикальные границы между caption anchors; не merge всей колонки.
3. **Вектор:** кластеризация drawings **между** caption_y(prev) и caption_y(curr) без merge страницы.
4. **Full-page raster:** OCR подписей (Tesseract / `get_textpage_ocr`) или отказ с явным «need native pdf».
5. **Единый FIG registry:** один canonical bytes per `FIG_n`, метаданные `source=xref|caption_crop|layout`.
6. **QA:** export script должен дублировать annotate path (`build_annotated_pdf` + список fig_bytes), не только pymupdf extract.

## 14. Карта файлов

| Задача | Путь |
|--------|------|
| Spatial ingest orchestration | `services/article_ingestion/blog_spatial_pipeline.py` |
| Figure registry + VLM-first | `figure_registry_service.py`, `figure_anchor_mapper.py`, `map_diagram_attach.py`, `models/figure_registry.py` |
| Map-Reduce | `services/article_ingestion/blog_spatial_summarizer.py` |
| VLM targets | `services/article_ingestion/spatial_diagram_dispatch.py` |
| Fallback ingest | `services/article_ingestion/auto_ingest.py`, `pipeline.py` |
| PDF annotate | `services/parsers/pdf_annotator.py` |
| Caption crop | `services/parsers/vector_pdf_cropper.py` |
| Pymupdf extract / export | `services/parsers/pymupdf_figure_extract.py`, `scripts/export_pdf_figures.py` |
| Raster bands | `services/parsers/raster_pdf_layout.py` |
| Fetch | `services/parsers/smart_fetcher.py`, `article_resource_discoverer.py` |
| Triage | `services/article_ingestion/document_triage_engine.py` |
| Диаграммы в ноде | `services/article_diagram_context.py`, `src/node_deep_dive/content_assets.py` |
| Обзор VLM/конфиг | `docs/ARTICLE_DIAGRAMS.md` |

---

**Короткий тезис для другой модели:** production ETL **зависит от списка `FIG_*` и байтов в `AnnotatedArticle`**, которые для PDF сейчас в основном строятся **геометрией подписей**, а не надёжным извлечением image objects. Map-Reduce и VLM **умны в выборе id и семантике**, но **не чинят** плохие crops. Export script тестирует только pymupdf-ветку — он воспроизводит ту же слабость, даже на «текстовом» PDF.
