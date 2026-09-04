"""Утилиты валидации цитат (opt-in, не ломают [S*]/[R*])."""

from knowledge_engine.services.validators.anchor_validator import (
    UNVERIFIED_ANCHOR_SUFFIX,
    validate_anchor_citations,
    validate_and_annotate_anchors,
)

__all__ = [
    "UNVERIFIED_ANCHOR_SUFFIX",
    "validate_anchor_citations",
    "validate_and_annotate_anchors",
]
