"""DocumentTriageEngine: TOC + prune перед Map-Reduce."""

from __future__ import annotations

import re

from knowledge_engine.config import (
    BLOG_SPATIAL_TRIAGE_ENABLED,
    BLOG_SPATIAL_TRIAGE_KEEP_FIGURES,
)
from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    kept_p_id_set,
    prune_annotated_article,
    restore_figures_after_text_prune,
    sorted_p_ids,
)
from knowledge_engine.services.article_ingestion.article_pruner import (
    ArticleSectionPruner,
)
from knowledge_engine.services.article_ingestion.triage_schemas import TriageOutcome
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle
from knowledge_engine.services.parsers.toc_extractor import (
    SourceFormat,
    UniversalTOCExtractor,
)
from knowledge_engine.ui.run_log import trace


def detect_source_format(
    raw: bytes | str | None,
    page_url: str = "",
) -> SourceFormat:
    if isinstance(raw, bytes) and raw[:5] == b"%PDF-":
        return "pdf"
    text = (
        raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
    )
    low_url = (page_url or "").lower()
    if low_url.endswith(".md") or low_url.endswith(".markdown"):
        return "markdown"
    from knowledge_engine.services.article_ingestion.raw_source import (
        is_code_or_raw_source,
    )

    if is_code_or_raw_source(page_url, text):
        return "markdown"
    head = text[:8000].lower()
    if "<html" in head or "<body" in head or "<article" in head:
        return "html"
    if re.search(r"^#{1,6}\s+\S", text, re.M):
        return "markdown"
    return "html"


class DocumentTriageEngine:
    def triage(
        self,
        annotated: AnnotatedArticle,
        *,
        source_format: SourceFormat | None = None,
        raw: bytes | str | None = None,
    ) -> tuple[AnnotatedArticle, TriageOutcome | None]:
        if not BLOG_SPATIAL_TRIAGE_ENABLED:
            from knowledge_engine.ingest.pipeline_audit import pipeline_audit

            pipeline_audit(
                "Triage",
                annotated.page_url,
                annotated.annotated_markdown,
                extra="DOC_TRIAGE disabled",
            )
            return annotated, None
        order = sorted_p_ids(annotated.paragraph_map)
        if len(order) < 4:
            return annotated, None

        fmt = source_format or detect_source_format(raw, annotated.page_url)
        trace(
            f"DOC_TRIAGE ▶ | format={fmt} P={len(order)} "
            f"FIG={len(annotated.fig_map)}"
        )

        extractor = UniversalTOCExtractor()
        structure = extractor.extract(annotated, fmt, raw)
        decision = ArticleSectionPruner().prune(structure, annotated)
        kept = kept_p_id_set(annotated.paragraph_map, decision.keep_paragraph_ranges)
        if not kept or kept == set(order):
            if decision.pruned_sections_reason:
                trace(
                    f"DOC_TRIAGE ✓ | toc_nodes={len(structure.nodes)} "
                    f"explicit_toc={structure.has_explicit_toc} | no shrink"
                )
            outcome = TriageOutcome(
                structure=structure,
                decision=decision,
                kept_p_ids=list(order),
            )
            return annotated, outcome

        pruned = prune_annotated_article(annotated, kept)
        fig_before = len(annotated.fig_map)
        fig_after_prune = len(pruned.fig_map)
        if (
            BLOG_SPATIAL_TRIAGE_KEEP_FIGURES
            and fig_before
            and fig_after_prune < fig_before
        ):
            pruned = restore_figures_after_text_prune(annotated, pruned)
            trace(
                f"DOC_TRIAGE restore FIG | text prune dropped "
                f"{fig_before - fig_after_prune} → kept {len(pruned.fig_map)} for VLM"
            )
        from knowledge_engine.ingest.pipeline_audit import pipeline_audit

        pipeline_audit(
            "Triage",
            annotated.page_url,
            pruned.annotated_markdown,
            extra=f"DOC_TRIAGE P {len(order)}→{len(pruned.paragraph_map)}",
        )
        trace(
            f"DOC_TRIAGE ✓ | P {len(order)}→{len(pruned.paragraph_map)} "
            f"FIG {fig_before}→{len(pruned.fig_map)} "
            f"toc_nodes={len(structure.nodes)} explicit={structure.has_explicit_toc}"
        )
        outcome = TriageOutcome(
            structure=structure,
            decision=decision,
            kept_p_ids=sorted_p_ids(pruned.paragraph_map),
        )
        return pruned, outcome


def triage_annotated_article(
    annotated: AnnotatedArticle,
    *,
    raw: bytes | str | None = None,
    source_format: SourceFormat | None = None,
) -> tuple[AnnotatedArticle, TriageOutcome | None]:
    return DocumentTriageEngine().triage(
        annotated,
        source_format=source_format,
        raw=raw,
    )
