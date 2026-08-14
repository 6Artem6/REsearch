"""Source Evaluator — whitelist matrix и Gemini Lite аудит."""

from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    cap_collectible_items,
    cap_collectible_urls,
    is_collectible_article_url,
    is_fast_trusted_source,
    normalize_site_host,
    register_curriculum_source,
    resolve_source_provenance,
)
from knowledge_engine.src.source_evaluator.evaluator import (
    SourceEvaluatorResult,
    evaluate_source,
    format_whitelist_for_reasoner_prompt,
    lite_curriculum_hit_approved,
    match_whitelist,
)
from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST

__all__ = [
    "APPROVED_SOURCES_WHITELIST",
    "cap_collectible_items",
    "cap_collectible_urls",
    "evaluate_source",
    "format_whitelist_for_reasoner_prompt",
    "is_collectible_article_url",
    "is_fast_trusted_source",
    "lite_curriculum_hit_approved",
    "match_whitelist",
    "normalize_site_host",
    "register_curriculum_source",
    "resolve_source_provenance",
    "SourceEvaluatorResult",
]
