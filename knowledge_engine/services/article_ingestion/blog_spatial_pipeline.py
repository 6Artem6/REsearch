"""Blog / PDF ingest: annotate → Map-Reduce → LanceDB + targeted VLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from knowledge_engine.config import BLOG_SPATIAL_VLM_FALLBACK_ALL_FIGURES
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.article_diagram_context import canonical_article_id
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    BlogArticleSummaryResponse,
    FinalArticleSummaryResponse,
    WindowDiagramCheck,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    map_reduce_jobs_pooled_async,
    map_reduce_summarize_blog,
    summarize_blog_article_spatial,
)
from knowledge_engine.services.article_ingestion.document_triage_engine import (
    detect_source_format,
    triage_annotated_article,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
    split_annotated_text_by_tokens,
)
from knowledge_engine.services.article_ingestion.section_context import (
    infer_article_title,
)
from knowledge_engine.services.article_ingestion.triage_schemas import (
    TOCNode,
    TriageOutcome,
)
from knowledge_engine.services.parsers.article_manifest import ArticleResourceManifest
from knowledge_engine.services.parsers.article_resource_discoverer import (
    discover_article_resources,
    get_cached_manifest,
    store_manifest,
)
from knowledge_engine.services.parsers.html_annotator import (
    AnnotatedArticle,
    build_annotated_article,
)
from knowledge_engine.services.parsers.md_annotator import build_annotated_markdown
from knowledge_engine.services.parsers.pdf_annotator import build_annotated_pdf
from knowledge_engine.services.parsers.pdf_bytes import (
    is_acm_doi_pdf_url,
    is_parseable_pdf,
)
from knowledge_engine.services.parsers.smart_fetcher import (
    boost_manifest_pdf_candidates,
    fetch_pdf_from_manifest,
)
from knowledge_engine.services.parsers.toc_extractor import UniversalTOCExtractor
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.services.web_extract import smart_fetch_page_html
from knowledge_engine.ui.run_log import trace


def _is_pdf_bytes(data: bytes, url: str = "") -> bool:
    if (data or b"")[:5] == b"%PDF-":
        return True
    low = (url or "").lower().split("?", 1)[0]
    return low.endswith(".pdf")


def build_annotated_from_content(
    data: bytes | str,
    page_url: str = "",
) -> AnnotatedArticle:
    url = (page_url or "").strip()
    if isinstance(data, bytes) and _is_pdf_bytes(data, url):
        ann = build_annotated_pdf(data)
        ann.page_url = url
        return ann
    if isinstance(data, bytes):
        html = data.decode("utf-8", errors="replace")
    else:
        html = data or ""
    fmt = detect_source_format(html, url)
    if fmt == "markdown":
        ann = build_annotated_markdown(html)
        ann.page_url = url
        return ann
    return build_annotated_article(html, url)


def _triage_for_pipeline(
    annotated: AnnotatedArticle,
    raw: bytes | str | None,
) -> tuple[AnnotatedArticle, TriageOutcome | None]:
    fmt = detect_source_format(raw, annotated.page_url)
    pruned, outcome = triage_annotated_article(
        annotated,
        raw=raw,
        source_format=fmt,
    )
    return pruned, outcome


def _toc_nodes_for_annotated(
    annotated: AnnotatedArticle,
    raw: bytes | str | None,
    outcome: TriageOutcome | None,
) -> list[TOCNode]:
    if outcome is not None and outcome.structure.nodes:
        return list(outcome.structure.nodes)
    fmt = detect_source_format(raw, annotated.page_url)
    tree = UniversalTOCExtractor().extract(annotated, fmt, raw)
    return list(tree.nodes)


def _document_summary_from_final(
    final: FinalArticleSummaryResponse,
    *,
    title: str,
    url: str,
    registry: object | None = None,
) -> DocumentSummary:
    from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
        normalize_final_knowledge,
    )

    final = normalize_final_knowledge(final)
    diagram_notes: list[str] = []
    if registry is not None:
        from knowledge_engine.services.article_ingestion.figure_registry_service import (
            FigureRegistry,
        )

        if isinstance(registry, FigureRegistry):
            for ent in registry.entries.values():
                if ent.vlm_summary or ent.mermaid_code:
                    diagram_notes.append(
                        f"{ent.internal_id}: {(ent.vlm_summary or ent.caption)[:400]}"
                    )
    if not diagram_notes:
        diagram_notes = [
            f"{d.figure_id}: {d.reason[:400]}"
            for d in (final.target_diagrams_for_vlm or [])
        ]
    return DocumentSummary(
        title=(title or final.executive_summary[:120] or url)[:300],
        url=url,
        key_takeaways=list(final.key_takeaways or [])[:12],
        failure_modes=[],
        cs_concepts=[],
        diagram_descriptions=diagram_notes,
    )


def _document_summary_from_spatial(
    spatial: BlogArticleSummaryResponse,
    *,
    title: str,
    url: str,
) -> DocumentSummary:
    diagram_notes = [
        f"{loc.figure_id}: {loc.semantic_reason[:400]}"
        for loc in (spatial.critical_diagram_locations or [])
    ]
    return DocumentSummary(
        title=(title or spatial.summary[:120] or url)[:300],
        url=url,
        key_takeaways=list(spatial.key_takeaways or [])[:12],
        failure_modes=[],
        cs_concepts=[],
        diagram_descriptions=diagram_notes,
    )


def _figure_sort_key(fid: str) -> tuple[int, int, str]:
    m = re.match(r"^FIG_(\d+)$", (fid or "").strip(), re.I)
    if m:
        return (0, int(m.group(1)), fid.upper())
    m = re.match(r"^FIG_SEQ_(\d+)$", (fid or "").strip(), re.I)
    if m:
        return (1, int(m.group(1)), fid.upper())
    return (2, 0, (fid or "").upper())


def _figure_ids(annotated: AnnotatedArticle) -> list[str]:
    return sorted(
        set(annotated.fig_map.keys()) | set(annotated.fig_bytes.keys()),
        key=_figure_sort_key,
    )


def _fallback_vlm_targets_from_figures(
    fig_ids: list[str],
) -> list[WindowDiagramCheck]:
    reason = (
        "Техническая схема из разметки статьи (MAP не пометил FIG; fallback для VLM)."
    )
    out: list[WindowDiagramCheck] = []
    for fid in fig_ids:
        f = (fid or "").strip().upper()
        if not f.startswith("FIG"):
            continue
        if not f.startswith("FIG_"):
            f = f"FIG_{f[3:].lstrip('_')}"
        out.append(
            WindowDiagramCheck(
                figure_id=f,
                referenced_paragraphs=[],
                reason=reason,
            )
        )
    return out


def _ensure_vlm_targets(
    final: FinalArticleSummaryResponse,
    annotated: AnnotatedArticle,
    page_url: str,
) -> FinalArticleSummaryResponse:
    if final.target_diagrams_for_vlm:
        return final
    if not BLOG_SPATIAL_VLM_FALLBACK_ALL_FIGURES:
        return final
    fig_ids = _figure_ids(annotated)
    if not fig_ids:
        return final
    final.target_diagrams_for_vlm = _fallback_vlm_targets_from_figures(fig_ids)
    trace(
        f"BLOG_SPATIAL vlm fallback | MAP empty → {len(fig_ids)} FIG | "
        f"{page_url[:55]}"
    )
    return final


def _academic_pdf_fallback_bytes(page_url: str) -> bytes | None:
    """ACM/IEEE + Cloudflare: Sci-Hub PDF когда manifest/Playwright не дали байты."""
    from knowledge_engine.src.fetcher.academic import (
        _fetch_scihub_pdf_bytes,
        extract_doi,
        is_academic_url,
    )
    from knowledge_engine.src.fetcher.context import fast_academic_fetch_enabled

    if not is_academic_url(page_url):
        return None
    if fast_academic_fetch_enabled():
        trace("BLOG_SPATIAL academic fallback ⊘ | fast_academic scope")
        return None
    target = extract_doi(page_url) or page_url
    pdf_bytes = _fetch_scihub_pdf_bytes(target)
    if pdf_bytes and pdf_bytes[:5] == b"%PDF-":
        trace(f"BLOG_SPATIAL academic fallback ✓ | Sci-Hub | {page_url[:55]}")
        return pdf_bytes
    return None


@dataclass
class SpatialDiagramIngestJob:
    source_id: str
    page_url: str
    annotated: AnnotatedArticle
    article_title: str = ""
    toc_nodes: list[TOCNode] = field(default_factory=list)
    trust_score: float = 1.0


def _maybe_filter_annotated_academic_pdf(
    annotated: AnnotatedArticle,
    page_url: str,
    title: str,
    content_raw: bytes | str | None,
) -> AnnotatedArticle:
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        get_cached_prepared_paper_body,
        is_academic_pdf_url,
        prepare_paper_body_for_gemma,
        try_fetch_pdf_bytes_for_url,
    )

    if not is_academic_pdf_url(page_url):
        return annotated
    cached_filtered = get_cached_prepared_paper_body(page_url)
    if cached_filtered and len(cached_filtered.strip()) >= 80:
        trace(f"PAPER_STRUCTURE spatial cache ✓ | skip analyze | {page_url[:55]}")
        annotated.annotated_markdown = cached_filtered
        return annotated
    pdf_b: bytes | None = None
    if isinstance(content_raw, bytes) and content_raw[:5] == b"%PDF-":
        pdf_b = content_raw
    else:
        pdf_b = try_fetch_pdf_bytes_for_url(page_url)
    plain = (
        "\n\n".join(annotated.paragraph_map.values())
        if annotated.paragraph_map
        else (annotated.annotated_markdown or "")
    )
    topic = (title or page_url).strip() or "scientific paper"
    filtered = prepare_paper_body_for_gemma(
        plain,
        topic,
        pdf_bytes=pdf_b,
        label=page_url[:48],
        page_url=page_url,
    )
    if filtered and len(filtered.strip()) >= 80:
        trace(f"PAPER_STRUCTURE spatial ✓ | annotated body filtered | {page_url[:55]}")
        annotated.annotated_markdown = filtered
    return annotated


def prepare_spatial_diagram_job(
    source_id: str,
    url: str,
    *,
    raw_html: str | None = None,
    raw_bytes: bytes | None = None,
) -> SpatialDiagramIngestJob | None:
    page_url = (url or "").strip()
    if page_url.startswith("http") and "/doi/" in page_url.lower():
        from knowledge_engine.services.parsers.pdf_bytes import prefer_acm_pdf_endpoint

        page_url = prefer_acm_pdf_endpoint(page_url)
    if not page_url.startswith("http") and raw_bytes is None:
        return None
    sid = (source_id or "").strip()
    cached = get_cached_manifest(page_url)
    if cached is not None:
        manifest = cached
    elif raw_bytes is not None and raw_html is None:
        from knowledge_engine.src.fetcher.academic import extract_doi

        manifest = ArticleResourceManifest(
            source_id=sid,
            canonical_url=page_url,
            doi=extract_doi(page_url),
        )
        if raw_bytes[:5] == b"%PDF-" and is_parseable_pdf(raw_bytes):
            manifest.fetched_pdf_bytes = raw_bytes
        store_manifest(manifest)
    else:
        manifest = discover_article_resources(
            page_url,
            sid,
            html_content=raw_html,
        )
    boost_manifest_pdf_candidates(manifest)
    fetch_url = page_url
    raw: bytes | str | None = raw_bytes
    if isinstance(raw, bytes) and raw[:5] == b"%PDF-" and not is_parseable_pdf(raw):
        trace(
            f"BLOG_SPATIAL ingest ⊘ | corrupt pdf ({len(raw)} B, 0 pages) | "
            f"{page_url[:55]}"
        )
        raw = None

    if raw is None:
        if raw_html is not None:
            raw = raw_html
        else:
            pdf_bytes = manifest.fetched_pdf_bytes
            if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
                pdf_bytes = fetch_pdf_from_manifest(manifest)
            if pdf_bytes and is_parseable_pdf(pdf_bytes):
                raw = pdf_bytes
                fetch_url = page_url
                if is_acm_doi_pdf_url(page_url):
                    fetch_url = page_url
                else:
                    fetch_url = manifest.selected_pdf_url or page_url
            elif manifest.html_snapshot:
                raw = manifest.html_snapshot
    elif isinstance(raw, bytes) and is_parseable_pdf(raw):
        manifest.fetched_pdf_bytes = raw
        store_manifest(manifest)
    elif isinstance(raw, str):
        manifest.html_snapshot = raw
        store_manifest(manifest)

    if isinstance(raw, bytes) and len(raw) < 40:
        return None
    if isinstance(raw, str) and len(raw.strip()) < 200:
        trace(f"BLOG_SPATIAL ingest ⊘ | thin content | {page_url[:60]}")
        return None

    annotated = build_annotated_from_content(raw, fetch_url)
    content_raw: bytes | str | None = raw
    triage_outcome: TriageOutcome | None = None
    annotated, triage_outcome = _triage_for_pipeline(annotated, raw)
    if not annotated.fig_map and not annotated.fig_bytes:
        pdf_bytes = manifest.fetched_pdf_bytes
        if not pdf_bytes or pdf_bytes[:5] != b"%PDF-":
            pdf_bytes = fetch_pdf_from_manifest(manifest)
        if pdf_bytes and is_parseable_pdf(pdf_bytes):
            trace(f"BLOG_SPATIAL ingest ▶ | manifest PDF retry | {page_url[:55]}")
            annotated = build_annotated_from_content(pdf_bytes, fetch_url)
            content_raw = pdf_bytes
            annotated, triage_outcome = _triage_for_pipeline(annotated, pdf_bytes)
    if not annotated.fig_map and not annotated.fig_bytes:
        pdf_bytes = _academic_pdf_fallback_bytes(page_url)
        if pdf_bytes:
            annotated = build_annotated_from_content(pdf_bytes, fetch_url)
            content_raw = pdf_bytes
            annotated, triage_outcome = _triage_for_pipeline(annotated, pdf_bytes)
            manifest.fetched_pdf_bytes = pdf_bytes
            store_manifest(manifest)
    if not annotated.fig_map and not annotated.fig_bytes:
        pdf_bytes = manifest.fetched_pdf_bytes or annotated.source_pdf_bytes
        if is_parseable_pdf(pdf_bytes):
            from knowledge_engine.services.parsers.pymupdf_figure_extract import (
                extract_figures_pymupdf,
            )

            trace(f"BLOG_SPATIAL ingest ▶ | pymupdf figure extract | {page_url[:55]}")
            extra = extract_figures_pymupdf(pdf_bytes)
            if extra:
                annotated.source_pdf_bytes = pdf_bytes
                for k, v in extra.items():
                    annotated.fig_bytes[k] = v
                    annotated.fig_map[k] = f"embedded:{k}"
                content_raw = pdf_bytes
                annotated, triage_outcome = _triage_for_pipeline(annotated, pdf_bytes)
    if not annotated.fig_map and not annotated.fig_bytes:
        trace(f"BLOG_SPATIAL ingest ⊘ | no figures after annotate | {page_url[:60]}")
        return None
    toc_nodes = _toc_nodes_for_annotated(annotated, content_raw, triage_outcome)
    article_title = infer_article_title(
        annotated=annotated,
        toc_nodes=toc_nodes,
        source_id=sid,
        page_url=page_url,
    )
    annotated = _maybe_filter_annotated_academic_pdf(
        annotated,
        page_url,
        article_title,
        content_raw,
    )
    return SpatialDiagramIngestJob(
        source_id=sid,
        page_url=fetch_url,
        annotated=annotated,
        article_title=article_title,
        toc_nodes=toc_nodes,
    )


def _map_job_from_ingest(
    job: SpatialDiagramIngestJob,
    *,
    figure_registry: object | None = None,
) -> MapReduceArticleJob:
    ann = job.annotated
    fig_ids = _figure_ids(ann)
    body = (ann.annotated_markdown or "").strip()
    title = job.article_title or infer_article_title(
        annotated=ann,
        toc_nodes=job.toc_nodes,
        source_id=job.source_id,
        page_url=job.page_url,
    )
    windows = split_annotated_text_by_tokens(
        body,
        title=title,
        all_figure_ids=fig_ids,
        figure_registry=figure_registry,  # type: ignore[arg-type]
        toc_nodes=job.toc_nodes,
        paragraph_map=ann.paragraph_map,
    )
    if not windows:
        windows = [TokenWindowChunk(window_index=0, body=body)]
    return MapReduceArticleJob(
        job_id=job.page_url,
        title=title,
        url=job.page_url,
        windows=windows,
        all_figure_ids=fig_ids,
        figure_registry=figure_registry,
        trust_score=float(getattr(job, "trust_score", 1.0) or 1.0),
    )


async def run_spatial_diagram_ingest_jobs_async(
    jobs: list[SpatialDiagramIngestJob],
) -> dict[str, int]:
    if not jobs:
        return {}
    from knowledge_engine.services.article_ingestion.figure_registry_service import (
        persist_figure_registry,
        run_vlm_on_registry,
    )
    from knowledge_engine.src.services.openalex_evaluator import (
        prefetch_trust_scores_async,
    )

    urls = [j.page_url for j in jobs]
    trust_by_url = await prefetch_trust_scores_async(urls)
    for j in jobs:
        j.trust_score = float(trust_by_url.get(j.page_url, 1.0))

    registries: dict[str, object] = {}
    map_jobs: list[MapReduceArticleJob] = []
    for j in jobs:
        aid = canonical_article_id(j.source_id, j.page_url)
        reg = persist_figure_registry(aid, j.annotated)
        run_vlm_on_registry(
            aid,
            j.annotated,
            reg,
            source_id=j.source_id,
            page_url=j.page_url,
        )
        registries[j.page_url] = reg
        map_jobs.append(_map_job_from_ingest(j, figure_registry=reg))

    # Prefer high-trust articles earlier in the shared MAP pool ordering
    map_jobs.sort(key=lambda m: float(m.trust_score or 1.0), reverse=True)

    trace(
        f"BLOG_SPATIAL pool ▶ | articles={len(map_jobs)} "
        f"chunks={sum(len(m.windows) for m in map_jobs)} | VLM→MAP order"
    )
    finals = await map_reduce_jobs_pooled_async(map_jobs)
    saved: dict[str, int] = {}
    by_url = {j.page_url: j for j in jobs}
    store = VectorStore()
    for url, outcome in finals.items():
        ingest_job = by_url.get(url)
        final = outcome.final if outcome else None
        if ingest_job is None or final is None:
            saved[url] = 0
            continue
        reg = registries.get(url)
        summary = _document_summary_from_final(
            final,
            title=ingest_job.article_title or url,
            url=url,
            registry=reg,
        )
        store.save_summary(summary)
        map_job = next((m for m in map_jobs if m.job_id == url or m.url == url), None)
        if map_job is not None and outcome is not None:
            window_texts = [(w.body or "").strip() for w in map_job.windows]
            window_summaries: list[str | None] = []
            for m in outcome.map_results:
                if m is None:
                    window_summaries.append(None)
                else:
                    window_summaries.append((m.window_summary or "").strip() or None)
            while len(window_summaries) < len(window_texts):
                window_summaries.append(None)
            if any(window_texts):
                store.upsert_rag_academic_map_windows(
                    url,
                    ingest_job.article_title or url,
                    window_texts,
                    summary,
                    window_summaries=window_summaries[: len(window_texts)],
                )
        n_atoms = store.upsert_knowledge_atoms(url, list(final.knowledge_atoms or []))
        trace(f"BLOG_SPATIAL LanceDB ✓ | atoms={n_atoms} | {url[:55]}")
        saved[url] = len(getattr(reg, "entries", {}) or {})
        trace(f"ARTICLE_AUTO_INGEST spatial ✓ | registry={saved[url]} | {url[:60]}")
    return saved


def run_spatial_diagram_ingest_jobs(
    jobs: list[SpatialDiagramIngestJob],
) -> dict[str, int]:
    import asyncio

    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                run_spatial_diagram_ingest_jobs_async(jobs),
            ).result()
    except RuntimeError:
        return asyncio.run(run_spatial_diagram_ingest_jobs_async(jobs))


def run_blog_spatial_diagram_ingest(
    source_id: str,
    url: str,
    *,
    raw_html: str | None = None,
    raw_bytes: bytes | None = None,
) -> int:
    job = prepare_spatial_diagram_job(
        source_id,
        url,
        raw_html=raw_html,
        raw_bytes=raw_bytes,
    )
    if job is None:
        return 0
    saved = run_spatial_diagram_ingest_jobs([job])
    return int(saved.get(job.page_url, 0))


def run_blog_spatial_diagram_ingest_batch(
    items: list[tuple[str, str]],
    *,
    raw_by_url: dict[str, bytes | str] | None = None,
) -> dict[str, int]:
    """Несколько URL: общий MAP-пул по чанкам, REDUCE по готовности статьи."""
    jobs: list[SpatialDiagramIngestJob] = []
    raw_by_url = raw_by_url or {}
    for source_id, url in items:
        u = (url or "").strip()
        raw = raw_by_url.get(u)
        kwargs: dict[str, object] = {}
        if isinstance(raw, bytes):
            kwargs["raw_bytes"] = raw
        elif isinstance(raw, str):
            kwargs["raw_html"] = raw
        prepared = prepare_spatial_diagram_job(source_id, u, **kwargs)
        if prepared is not None:
            jobs.append(prepared)
    if not jobs:
        return {}
    return run_spatial_diagram_ingest_jobs(jobs)


def ingest_blog_with_spatial_mapping(
    title: str,
    url: str,
    source_id: str = "",
    *,
    raw_html: str | None = None,
    raw_bytes: bytes | None = None,
    save_lancedb: bool = True,
) -> tuple[AnnotatedArticle | None, DocumentSummary | None, int]:
    page_url = (url or "").strip()
    raw: bytes | str | None = raw_bytes
    if raw is None:
        if raw_html is not None:
            raw = raw_html
        else:
            html, _method = smart_fetch_page_html(page_url)
            raw = html
    if isinstance(raw, bytes) and len(raw) < 40:
        return None, None, 0
    if isinstance(raw, str) and len(raw.strip()) < 200:
        trace(f"BLOG_SPATIAL pipeline ⊘ | thin content | {page_url[:60]}")
        return None, None, 0

    annotated = build_annotated_from_content(raw, page_url)
    annotated, _triage_outcome = _triage_for_pipeline(annotated, raw)
    if not annotated.annotated_markdown:
        return annotated, None, 0

    annotated = _maybe_filter_annotated_academic_pdf(
        annotated,
        page_url,
        title or page_url,
        raw,
    )

    fig_ids = _figure_ids(annotated)
    aid = canonical_article_id(source_id, page_url)
    from knowledge_engine.services.article_ingestion.figure_registry_service import (
        persist_figure_registry,
        run_vlm_on_registry,
    )

    registry = persist_figure_registry(aid, annotated)
    vlm_saved = run_vlm_on_registry(
        aid,
        annotated,
        registry,
        source_id=source_id,
        page_url=page_url,
    )

    final = map_reduce_summarize_blog(
        annotated.annotated_markdown,
        title=title or page_url,
        url=page_url,
        all_figure_ids=fig_ids,
        figure_registry=registry,
    )
    if final is None:
        legacy = summarize_blog_article_spatial(
            annotated.annotated_markdown,
            title=title or page_url,
            url=page_url,
            all_figure_ids=fig_ids,
        )
        if legacy is None:
            return annotated, None, 0
        summary = _document_summary_from_spatial(legacy, title=title, url=page_url)
        saved = vlm_saved
    else:
        summary = _document_summary_from_final(
            final,
            title=title,
            url=page_url,
            registry=registry,
        )
        saved = vlm_saved

    if save_lancedb:
        VectorStore().save_summary(summary)
        trace(f"BLOG_SPATIAL pipeline ✓ | LanceDB saved | {page_url[:60]}")

    return annotated, summary, saved
