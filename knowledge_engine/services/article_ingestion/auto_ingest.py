"""Авто-ingest схем при fetch источников (harvest / summarizer)."""

from __future__ import annotations

from knowledge_engine.services.article_diagram_context import canonical_article_id
from knowledge_engine.services.article_diagram_store import list_diagrams_for_article
from knowledge_engine.services.article_ingestion.pipeline import (
    ArticleFormat,
    ArticleIngestionPipeline,
)
from knowledge_engine.services.parsers.article_resource_discoverer import (
    discover_article_resources,
    get_cached_manifest,
)
from knowledge_engine.services.parsers.smart_fetcher import fetch_pdf_from_manifest
from knowledge_engine.ui.run_log import trace


def maybe_ingest_article_diagrams(
    source_id: str,
    url: str,
    *,
    data: bytes | None = None,
    content_type: ArticleFormat | str | None = None,
) -> int:
    """
    Сохранить схемы статьи в article_diagrams.
    Дубликаты отсекаются по phash; повторный запуск дополняет новые картинки.
    Возвращает число сохранённых диаграмм в этом прогоне.
    """
    article_id = canonical_article_id(source_id, url)
    if not article_id:
        return 0
    existing = len(list_diagrams_for_article(article_id))
    if existing:
        trace(
            f"ARTICLE_AUTO_INGEST incremental | {article_id} | "
            f"already={existing} diagrams in db"
        )

    pipeline = ArticleIngestionPipeline()
    page_url = (url or "").strip()
    fmt: ArticleFormat
    raw = data
    if raw is None:
        manifest = get_cached_manifest(page_url) or discover_article_resources(
            page_url, source_id
        )
        raw = manifest.fetched_pdf_bytes
        if not raw or (isinstance(raw, bytes) and raw[:5] != b"%PDF-"):
            raw = fetch_pdf_from_manifest(manifest)
        if raw and isinstance(raw, bytes) and raw[:5] == b"%PDF-":
            page_url = manifest.selected_pdf_url or manifest.canonical_url or page_url
            fmt = ArticleFormat.PDF
            trace(
                f"ARTICLE_AUTO_INGEST fetch | kind=pdf via=resource_manifest "
                f"| {page_url[:80]}"
            )
        elif manifest.html_snapshot:
            raw = manifest.html_snapshot.encode("utf-8", errors="replace")
            page_url = manifest.canonical_url or page_url
            fmt = ArticleFormat.HTML
            trace(
                f"ARTICLE_AUTO_INGEST fetch | kind=html via=resource_manifest "
                f"| {page_url[:80]}"
            )
        else:
            return 0
    else:
        fmt = (
            pipeline._normalize_format(content_type)
            if content_type is not None
            else _detect_format(page_url, raw)
        )

    if not raw or len(raw) < 40:
        return 0

    trace(f"ARTICLE_AUTO_INGEST ▶ | {article_id} | {page_url[:80]}")
    if fmt == ArticleFormat.HTML or fmt == ArticleFormat.PDF:
        try:
            from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
                run_blog_spatial_diagram_ingest,
            )

            kwargs: dict[str, object] = {}
            if fmt == ArticleFormat.PDF:
                kwargs["raw_bytes"] = raw
            else:
                kwargs["raw_html"] = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else str(raw)
                )
            n = run_blog_spatial_diagram_ingest(source_id, page_url, **kwargs)
            if n > 0:
                trace(f"ARTICLE_AUTO_INGEST spatial ✓ | saved={n}")
                return n
        except Exception as exc:
            trace(f"ARTICLE_AUTO_INGEST spatial fallback | {exc}")

    saved = pipeline.ingest(article_id, raw, fmt, page_url=page_url)
    trace(f"ARTICLE_AUTO_INGEST ✓ | {article_id} | saved={len(saved)}")
    return len(saved)


def maybe_ingest_article_diagrams_verbose(
    source_id: str,
    url: str,
    **kwargs: object,
) -> int:
    """CLI: печатает результат в stdout."""
    n = maybe_ingest_article_diagrams(source_id, url, **kwargs)
    aid = canonical_article_id(source_id, url)
    total = len(list_diagrams_for_article(aid)) if aid else 0
    print(f"saved_this_run={n} total_in_db={total} article_id={aid}")
    return n


def _detect_format(url: str, data: bytes) -> ArticleFormat:
    low = (url or "").lower().split("?", 1)[0]
    if low.endswith(".pdf") or data[:5] == b"%PDF-":
        return ArticleFormat.PDF
    head = data[:800].decode("utf-8", errors="ignore").lower()
    if "<html" in head or "<body" in head:
        return ArticleFormat.HTML
    return ArticleFormat.MD
