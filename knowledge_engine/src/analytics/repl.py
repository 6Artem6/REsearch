"""REPL follow-up по загруженным источникам и ConceptGraph."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.research_schemas import ReplFollowUpResponse
from knowledge_engine.src.analytics.gemini_v07 import run_gemini_flash_structured

_MAX_CONTEXT_CHARS = 48_000


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 20] + "\n… [truncated]"


def build_repl_context(state: Dict[str, Any]) -> str:
    """Собрать контекст для уточняющих вопросов из финального state v0.7."""
    parts: List[str] = []

    query = state.get("user_query") or ""
    if query:
        parts.append(f"## Исходный вопрос\n{query}")

    spec = state.get("query_spec")
    if isinstance(spec, dict) and spec.get("cs_formal_query"):
        parts.append(f"## CS formal query\n{spec['cs_formal_query']}")

    docs = state.get("documents") or []
    if docs:
        lines = []
        for d in docs[:20]:
            if isinstance(d, dict):
                lines.append(f"- {d.get('doc_id', '?')}: {d.get('source_url', '')}")
        parts.append("## Документы (doc_id → URL)\n" + "\n".join(lines))

    chunks = state.get("structured_chunks") or []
    if chunks:
        chunk_rows = []
        for ch in chunks[:24]:
            if not isinstance(ch, dict):
                continue
            chunk_rows.append(
                {
                    "chunk_id": ch.get("chunk_id"),
                    "doc_id": ch.get("doc_id"),
                    "concepts": (ch.get("concepts") or [])[:10],
                    "text": (ch.get("text") or "")[:700],
                }
            )
        parts.append(
            "## Structured chunks\n"
            + _truncate(json.dumps(chunk_rows, ensure_ascii=False, indent=2), 14_000)
        )

    cg = state.get("concept_graph")
    if cg:
        parts.append(
            "## ConceptGraph (L2a)\n"
            + _truncate(json.dumps(cg, ensure_ascii=False, indent=2), 14_000)
        )

    gap = state.get("profile_gap_map")
    if gap:
        parts.append(
            "## ProfileGapMap (L2b)\n"
            + _truncate(json.dumps(gap, ensure_ascii=False, indent=2), 10_000)
        )

    matrix = state.get("tradeoff_matrix")
    if matrix:
        parts.append(
            "## Tradeoff matrix (L2c)\n"
            + _truncate(json.dumps(matrix, ensure_ascii=False, indent=2), 10_000)
        )

    body = "\n\n".join(parts)
    return _truncate(body, _MAX_CONTEXT_CHARS)


_REPL_SYSTEM = (
    "You are an assistant for an already-completed Knowledge Engine v0.7 research run.\n"
    "Return strictly valid JSON matching ReplFollowUpResponse.\n"
    "Required key: answer. Optional: sources_cover_question (false if RESEARCH CONTEXT "
    "does not cover the question).\n"
    "Ground answer only in RESEARCH CONTEXT (chunks, ConceptGraph, gap map, matrix).\n"
    "If sources lack data, say so in answer and set sources_cover_question=false.\n"
    "Do not shrink the answer to UMA/Mac unless the question is about hardware.\n"
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "The answer field MUST be natural Russian.\n"
)


def answer_follow_up(
    question: str,
    repl_context: str,
    global_anchor: str,
) -> str:
    user = (
        f"## RESEARCH CONTEXT\n{repl_context}\n\n"
        f"## Follow-up question\n{question.strip()}"
    )
    parsed = run_gemini_flash_structured(
        _REPL_SYSTEM,
        user,
        global_anchor,
        ReplFollowUpResponse,
        "REPL follow-up",
    )
    return (parsed.answer or "").strip()
