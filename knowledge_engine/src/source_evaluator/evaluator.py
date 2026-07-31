"""Валидация источников: whitelist + Gemini Lite (Source Evaluator)."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.source_evaluator.evaluator_prompt import (
    build_evaluator_system_instruction,
    build_evaluator_user_message,
)
from knowledge_engine.src.source_evaluator.whitelist import (
    APPROVED_SOURCES_WHITELIST,
    format_whitelist_detailed,
)
from knowledge_engine.ui.run_log import trace

_MD_URL_RE = re.compile(r"\((https?://[^)\s]+)\)")


class SourceEvaluatorResult(BaseModel):
    status: Literal["APPROVED", "REJECTED"]
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    suggested_action: Literal[
        "RETRY_WITH_NEW_SOURCE", "REMOVE_LINK", "KEEP"
    ] = "KEEP"
    whitelist_match: bool = False


def _flatten_whitelist_patterns() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for category, entries in APPROVED_SOURCES_WHITELIST.items():
        for raw in entries:
            pat = raw.strip().lower()
            if pat:
                out.append((pat, category))
    return out


def _normalize_url_for_match(url: str) -> tuple[str, str]:
    u = (url or "").strip()
    if not u:
        return "", ""
    m = _MD_URL_RE.search(u)
    if m:
        u = m.group(1)
    if not u.startswith("http"):
        u = f"https://{u}"
    parsed = urlparse(u)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").strip("/")
    host_path = host
    if path:
        host_path = f"{host}/{path}"
    return host, host_path


def match_whitelist(url: str) -> tuple[bool, str]:
    host, host_path = _normalize_url_for_match(url)
    if not host:
        return False, ""
    patterns = _flatten_whitelist_patterns()
    for pattern, category in patterns:
        if host == pattern or host_path == pattern:
            return True, category
        if host_path.startswith(pattern) or pattern.startswith(host_path):
            return True, category
        base = pattern.split("/")[0]
        if host == base:
            return True, category
        if host.endswith(f".{base}"):
            return True, category
    return False, ""


def evaluate_source(
    url: str,
    thesis: str,
    excerpt: str = "",
    global_anchor: str = "",
) -> dict[str, Any]:
    """Оценить источник: instant pass по whitelist или Gemini Lite."""
    thesis_clean = (thesis or "").strip()[:800]
    excerpt_clean = (excerpt or "").strip()[:1200]
    url_clean = (url or "").strip()

    matched, category = match_whitelist(url_clean)
    if matched:
        from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
            register_curriculum_source,
        )

        register_curriculum_source(
            url_clean,
            thesis_clean,
            category=category,
            trust_score=0.92,
            status="accepted",
            reason="static_whitelist",
        )
        result = SourceEvaluatorResult(
            status="APPROVED",
            confidence_score=1.0,
            reason=f"Домен в APPROVED_SOURCES_WHITELIST ({category}).",
            suggested_action="KEEP",
            whitelist_match=True,
        )
        trace(f"SOURCE_EVAL ✓ whitelist | {url_clean[:80]}")
        return result.model_dump()

    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    system = (
        f"{RUSSIAN_OUTPUT_RULE}\n"
        "Поле reason — на русском. JSON строго по схеме.\n\n"
        f"{build_evaluator_system_instruction()}"
    )
    user_msg = build_evaluator_user_message(url_clean, thesis_clean, excerpt_clean)
    trace(f"SOURCE_EVAL ▶ Lite | {url_clean[:80]}…")

    class _LiteOut(BaseModel):
        status: Literal["APPROVED", "REJECTED"]
        confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
        reason: str = ""
        suggested_action: Literal[
            "RETRY_WITH_NEW_SOURCE", "REMOVE_LINK", "KEEP"
        ] = "KEEP"

    lite = run_gemini_lite_structured(
        system,
        user_msg,
        global_anchor,
        _LiteOut,
        "source_evaluator_lite",
    )
    status = (lite.status or "REJECTED").strip().upper()
    if status not in ("APPROVED", "REJECTED"):
        status = "REJECTED"
    action = lite.suggested_action or "KEEP"
    if status == "REJECTED" and action == "KEEP":
        action = "RETRY_WITH_NEW_SOURCE"

    result = SourceEvaluatorResult(
        status=status,
        confidence_score=float(lite.confidence_score or 0.0),
        reason=(lite.reason or "").strip(),
        suggested_action=action,
        whitelist_match=False,
    )
    if status == "APPROVED":
        from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
            register_curriculum_source,
        )

        register_curriculum_source(
            url_clean,
            thesis_clean,
            category="lite_approved",
            trust_score=min(0.95, float(lite.confidence_score or 0.86)),
            reason=(lite.reason or "")[:400],
        )
    trace(
        f"SOURCE_EVAL ✓ {result.status} "
        f"conf={result.confidence_score:.2f} | {result.reason[:100]}"
    )
    return result.model_dump()


def format_whitelist_for_reasoner_prompt() -> str:
    return format_whitelist_detailed()


def lite_curriculum_hit_approved(
    url: str,
    learning_goal: str,
    title: str,
    snippet: str,
    anchor: str,
    *,
    source_tier: str = "",
    skip_trusted_archive_reuse: bool = True,
) -> bool:
    """
    Единый Lite-gate для curriculum hits → evaluate_source + архив.
    Не дублировать run_gemini_lite_structured в pipeline.
    """
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        is_fast_trusted_source,
    )

    url_clean = (url or "").strip()
    if (
        skip_trusted_archive_reuse
        and (source_tier or "").strip().lower() == "archive"
        and is_fast_trusted_source(url_clean)
    ):
        return True

    excerpt = "\n".join(
        x for x in [(title or "").strip(), (snippet or "").strip()] if x
    )
    raw = evaluate_source(url_clean, learning_goal, excerpt, anchor)
    status = (raw.get("status") or "").strip().upper()
    ok = status == "APPROVED"
    if not ok:
        reason = (raw.get("reason") or "")[:80]
        trace(f"CURRICULUM lite source ⊘ | {url_clean[:60]} | {reason}")
    return ok
