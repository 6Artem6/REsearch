"""Re-Act аудит источников в ответе Reasoner (Gemini Lite Source Evaluator)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

from knowledge_engine.ui.run_log import trace

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

MAX_REACT_SOURCE_ITERATIONS = 2
_MAX_CITATIONS_PER_PASS = 8


class SourceEvaluationResult(BaseModel):
    status: Literal["APPROVED", "REJECTED"]
    reason: str = ""
    confidence_score: float = 0.0
    suggested_action: str = "KEEP"
    whitelist_match: bool = False


class SourceCitationCandidate(BaseModel):
    statement: str
    source_info: str
    url: str = ""
    excerpt: str = ""


def _sentence_with_match(text: str, match_start: int) -> str:
    if not text:
        return ""
    left = text[:match_start]
    parts_left = _SENTENCE_SPLIT_RE.split(left)
    sentence = parts_left[-1] if parts_left else left
    right = text[match_start:]
    parts_right = _SENTENCE_SPLIT_RE.split(right, maxsplit=1)
    if parts_right:
        sentence += parts_right[0]
    return sentence.strip()[:600]


def extract_source_citation_candidates(
    answer: str,
    registry: list[dict[str, Any]] | None = None,
) -> list[SourceCitationCandidate]:
    text = (answer or "").strip()
    if not text:
        return []
    by_id = {
        str(e.get("id") or e.get("source_id") or "").upper(): e
        for e in (registry or [])
        if isinstance(e, dict)
    }
    seen: set[str] = set()
    out: list[SourceCitationCandidate] = []

    for m in _MD_LINK_RE.finditer(text):
        title = m.group(1).strip()
        url = m.group(2).strip()
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        statement = _sentence_with_match(text, m.start())
        if len(statement) < 15:
            statement = f"Утверждение рядом со ссылкой «{title}»."
        out.append(
            SourceCitationCandidate(
                statement=statement,
                source_info=f"{title} | {url}",
                url=url,
                excerpt=title,
            )
        )

    for m in _SOURCE_TAG_RE.finditer(text):
        sid = f"S{m.group(1)}".upper()
        key = f"tag:{sid}"
        if key in seen:
            continue
        entry = by_id.get(sid)
        if not entry:
            continue
        seen.add(key)
        url = str(entry.get("url") or "").strip()
        title = str(entry.get("title") or sid)
        statement = _sentence_with_match(text, m.start())
        if len(statement) < 15:
            statement = f"Тезис с опорой на источник [{sid}]."
        out.append(
            SourceCitationCandidate(
                statement=statement,
                source_info=f"[{sid}] {title} | {url or '—'}",
                url=url,
                excerpt=title,
            )
        )

    return out[:_MAX_CITATIONS_PER_PASS]


def evaluate_source(
    statement: str,
    source_info: str,
    global_anchor: str,
) -> SourceEvaluationResult:
    """Один источник (legacy); для Re-Act используйте build_react_feedback (batch)."""
    from knowledge_engine.src.curriculum.lite_search_pipeline import (
        batch_evaluate_sources_sync,
    )

    info = (source_info or "").strip()
    url = ""
    excerpt = ""
    if "|" in info:
        excerpt, url_part = info.split("|", 1)
        excerpt = excerpt.strip()
        url = url_part.strip()
    else:
        url = info
    if not url.startswith("http"):
        link_m = re.search(r"(https?://[^\s)]+)", info)
        if link_m:
            url = link_m.group(1)
    batch = batch_evaluate_sources_sync(
        (statement or "").strip(),
        [
            {
                "id": 1,
                "url": url or info,
                "title": excerpt[:400],
                "snippet": (
                    f"Тезис: {(statement or '')[:500]}\n" f"Источник: {info[:400]}"
                ),
            }
        ],
        anchor=global_anchor,
    )
    ev = batch[0] if batch else None
    if not ev:
        return SourceEvaluationResult(status="REJECTED", reason="batch empty")
    return SourceEvaluationResult(
        status=ev.status,
        reason=ev.reason,
        confidence_score=float(ev.confidence or 0.0),
        suggested_action=ev.suggested_action,
        whitelist_match=ev.reason == "whitelist instant pass",
    )


def build_react_feedback(
    candidates: list[SourceCitationCandidate],
    global_anchor: str,
) -> str:
    if not candidates:
        return ""
    from knowledge_engine.src.curriculum.lite_search_pipeline import (
        batch_evaluate_sources_sync,
    )

    batch_src: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        batch_src.append(
            {
                "id": i,
                "url": cand.url or "",
                "title": (cand.excerpt or cand.source_info)[:400],
                "snippet": (
                    f"Тезис: {(cand.statement or '')[:500]}\n"
                    f"Источник: {cand.source_info[:400]}"
                ),
            }
        )
    evals = batch_evaluate_sources_sync("", batch_src, anchor=global_anchor)
    by_id = {e.id: e for e in evals}
    lines: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        ev = by_id.get(i)
        if not ev or ev.status != "REJECTED":
            continue
        src = cand.url or cand.source_info[:120]
        action_hint = ""
        if ev.suggested_action == "RETRY_WITH_NEW_SOURCE":
            action_hint = " Замени ссылку на источник из Whitelist Matrix."
        elif ev.suggested_action == "REMOVE_LINK":
            action_hint = " Убери ссылку и объясни тезис без неё."
        lines.append(
            f"[Системный отклик: Источник отклонён ({src}). "
            f"Причина: {ev.reason}.{action_hint}]"
        )
    return "\n".join(lines)


def audit_answer_sources_react(
    answer: str,
    registry: list[dict[str, Any]] | None,
    global_anchor: str,
) -> str:
    candidates = extract_source_citation_candidates(answer, registry)
    if not candidates:
        trace("SOURCE_EVAL ⊘ нет ссылок для аудита в ответе")
        return ""
    return build_react_feedback(candidates, global_anchor)
