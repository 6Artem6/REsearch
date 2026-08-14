"""VLM только для FIG_X из spatial mapping (Map-Reduce)."""

from __future__ import annotations

import logging

from knowledge_engine.services.article_diagram_context import canonical_article_id
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    BlogArticleSummaryResponse,
    FinalArticleSummaryResponse,
    TargetDiagramLocation,
    WindowDiagramCheck,
)
from knowledge_engine.services.article_ingestion.pipeline import (
    ArticleIngestionPipeline,
)
from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle
from knowledge_engine.services.parsers.image_bytes import load_image_bytes
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)


def _normalize_fig_id(raw: str) -> str:
    t = (raw or "").strip().upper()
    if t.startswith("FIG_"):
        return t
    if t.startswith("FIG"):
        return f"FIG_{t[3:].lstrip('_')}"
    return t


def _norm_p(raw: str) -> str:
    t = (raw or "").strip().upper()
    if t.startswith("P_"):
        return t
    return f"P_{t.lstrip('P_')}"


def _paragraph_context_from_check(
    loc: WindowDiagramCheck | TargetDiagramLocation,
    paragraph_map: dict[str, str],
) -> str:
    reason = getattr(loc, "reason", None) or getattr(loc, "semantic_reason", "")
    parts: list[str] = [f"semantic_reason: {(reason or '').strip()}"]
    paras = getattr(loc, "referenced_paragraphs", None) or getattr(
        loc, "relevant_paragraphs", []
    )
    for pid in paras or []:
        key = _norm_p(pid)
        text = (paragraph_map.get(key) or paragraph_map.get(pid) or "").strip()
        if text:
            parts.append(f"[{key}] {text[:1200]}")
    return "\n".join(parts)[:4000]


def _sanitize_figure_for_vlm(
    fid: str,
    data: bytes,
    mime: str,
    annotated: AnnotatedArticle,
) -> tuple[bytes, str] | None:
    source = (annotated.fig_extract_source or {}).get(fid, "")
    if source.startswith("invalid:"):
        logger.warning(
            "BLOG_SPATIAL vlm rejected | %s (reason: %s)",
            fid,
            source,
        )
        trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: {source})")
        return None

    from knowledge_engine.services.parsers.vector_pdf_cropper import (
        VectorPDFCropper,
        is_invalid_vlm_crop,
    )

    if not is_invalid_vlm_crop(data):
        return data, mime

    pdf = annotated.source_pdf_bytes or b""
    if pdf[:5] != b"%PDF-":
        trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: bad_aspect_pixels)")
        return None

    expanded = VectorPDFCropper().resolve_figure_expanded(pdf, fid)
    if expanded is None or not expanded.is_renderable:
        trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: expanded_crop_failed)")
        return None
    if is_invalid_vlm_crop(expanded.data):
        trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: expanded_still_flat)")
        return None

    annotated.fig_bytes[fid] = expanded.as_payload()
    annotated.fig_extract_source[fid] = expanded.source
    if expanded.topology:
        annotated.fig_extract_topology[fid] = dict(expanded.topology)
    trace(f"BLOG_SPATIAL vlm crop fallback ✓ | {fid} | source={expanded.source}")
    return expanded.data, expanded.mime


def _load_figure_bytes(
    fid: str,
    annotated: AnnotatedArticle,
    base_url: str,
) -> tuple[bytes, str] | None:
    source = (annotated.fig_extract_source or {}).get(fid, "")
    if source.startswith("invalid:"):
        trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: {source})")
        return None
    if fid in annotated.fig_bytes:
        return annotated.fig_bytes[fid]
    src = (annotated.fig_map.get(fid) or "").strip()
    if not src or src.startswith("embedded:"):
        if fid in annotated.fig_bytes:
            return annotated.fig_bytes[fid]
    else:
        loaded = load_image_bytes(src, base_url=base_url)
        if loaded is not None:
            return loaded

    pdf = annotated.source_pdf_bytes or b""
    if pdf[:5] == b"%PDF-":
        from knowledge_engine.services.parsers.vector_pdf_cropper import (
            VectorPDFCropper,
        )

        crop = VectorPDFCropper().resolve_figure_detailed(pdf, fid)
        if crop is not None:
            annotated.fig_extract_source[fid] = crop.source
            if crop.topology:
                annotated.fig_extract_topology[fid] = dict(crop.topology)
            if crop.is_renderable:
                annotated.fig_bytes[fid] = crop.as_payload()
                annotated.fig_map[fid] = f"embedded:{fid}"
                trace(f"BLOG_SPATIAL vlm vector ✓ | {fid}")
                return crop.as_payload()
            trace(f"BLOG_SPATIAL vlm ⊘ | {fid} rejected (reason: {crop.source})")
            return None

    if src and not src.startswith("embedded:"):
        loaded = load_image_bytes(src, base_url=base_url)
        if loaded is not None:
            return loaded
    return None


def _build_target_images(
    annotated: AnnotatedArticle,
    diagrams: list[WindowDiagramCheck],
    *,
    page_url: str = "",
) -> list[ExtractedImage]:
    base_url = page_url or annotated.page_url
    images: list[ExtractedImage] = []
    for loc in diagrams:
        fid = _normalize_fig_id(loc.figure_id)
        loaded = _load_figure_bytes(fid, annotated, base_url)
        if loaded is None:
            trace(f"BLOG_SPATIAL vlm ⊘ | {fid} not loadable")
            continue
        data, mime = loaded
        sanitized = _sanitize_figure_for_vlm(fid, data, mime, annotated)
        if sanitized is None:
            continue
        data, mime = sanitized
        ctx = _paragraph_context_from_check(loc, annotated.paragraph_map)
        reason = (loc.reason or "").strip()
        caption = f"{fid}: {reason[:300]}"
        images.append(
            ExtractedImage(
                image_bytes=data,
                caption=caption[:500],
                context_text=ctx,
                page_or_pos=len(images) + 1,
                mime=mime,
            )
        )
    return images


def ingest_target_diagrams(
    article_id: str,
    annotated: AnnotatedArticle,
    diagrams: list[WindowDiagramCheck],
    *,
    source_id: str = "",
    page_url: str = "",
) -> int:
    aid = (article_id or "").strip()
    if not aid:
        aid = canonical_article_id(source_id, page_url or annotated.page_url)
    if not aid:
        trace("BLOG_SPATIAL vlm ⊘ | empty article_id")
        return 0
    if not diagrams:
        trace("BLOG_SPATIAL vlm ⊘ | no target diagrams")
        return 0

    base_url = page_url or annotated.page_url
    images = _build_target_images(
        annotated,
        diagrams,
        page_url=base_url,
    )

    if not images:
        trace("BLOG_SPATIAL vlm ⊘ | no images loaded")
        return 0

    trace(f"BLOG_SPATIAL vlm ▶ | article={aid[:40]} targets={len(images)}")
    pipeline = ArticleIngestionPipeline(batch_min=1, batch_max=min(5, len(images)))
    saved = pipeline.ingest_vlm_targets(aid, images)
    trace(f"BLOG_SPATIAL vlm ✓ | saved={len(saved)}")
    return len(saved)


def ingest_critical_diagrams_from_spatial_map(
    article_id: str,
    annotated: AnnotatedArticle,
    spatial: BlogArticleSummaryResponse | FinalArticleSummaryResponse,
    *,
    source_id: str = "",
    page_url: str = "",
) -> int:
    if isinstance(spatial, FinalArticleSummaryResponse):
        return ingest_target_diagrams(
            article_id,
            annotated,
            list(spatial.target_diagrams_for_vlm or []),
            source_id=source_id,
            page_url=page_url,
        )
    checks = [
        WindowDiagramCheck(
            figure_id=loc.figure_id,
            referenced_paragraphs=list(loc.relevant_paragraphs),
            reason=loc.semantic_reason,
        )
        for loc in spatial.critical_diagram_locations or []
    ]
    return ingest_target_diagrams(
        article_id,
        annotated,
        checks,
        source_id=source_id,
        page_url=page_url,
    )


def ingest_spatial_maps_batch_vlm(
    entries: list[
        tuple[
            str,
            AnnotatedArticle,
            BlogArticleSummaryResponse | FinalArticleSummaryResponse,
            str,
            str,
        ]
    ],
) -> dict[str, int]:
    """MAP уже выполнен: один VLM-пул на все FIG всех статей."""
    works: list[tuple[str, ExtractedImage]] = []
    for article_id, annotated, spatial, source_id, page_url in entries:
        aid = (article_id or "").strip()
        if not aid:
            aid = canonical_article_id(source_id, page_url or annotated.page_url)
        if not aid:
            continue
        if isinstance(spatial, FinalArticleSummaryResponse):
            diagrams = list(spatial.target_diagrams_for_vlm or [])
        else:
            diagrams = [
                WindowDiagramCheck(
                    figure_id=loc.figure_id,
                    referenced_paragraphs=list(loc.relevant_paragraphs),
                    reason=loc.semantic_reason,
                )
                for loc in spatial.critical_diagram_locations or []
            ]
        imgs = _build_target_images(
            annotated,
            diagrams,
            page_url=page_url or annotated.page_url,
        )
        for img in imgs:
            works.append((aid, img))
    if not works:
        trace("BLOG_SPATIAL vlm ⊘ | batch empty")
        return {}
    trace(f"BLOG_SPATIAL vlm pool ▶ | cross-article images={len(works)}")
    pipeline = ArticleIngestionPipeline()
    counts = pipeline.ingest_vlm_targets_multi_article(works)
    trace(f"BLOG_SPATIAL vlm pool ✓ | saved_by_article={counts}")
    return counts
