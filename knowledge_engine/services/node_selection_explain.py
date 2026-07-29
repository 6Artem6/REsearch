"""Пояснение выделенного фрагмента в материале ноды (как v08 explain + registry)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.node_source_registry import registry_for_prompt
from knowledge_engine.src.processors.explainer import DEFAULT_EXPLAIN_QUESTION, ExplainSourceRef
from knowledge_engine.src.processors.source_anchors import strip_source_anchor_tags

_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)


class NodeExplainResult(BaseModel):
    explanation: str = ""
    source_ref: ExplainSourceRef = Field(default_factory=ExplainSourceRef)


_NODE_EXPLAIN_SYSTEM = f"""Ты — исследовательский аналитик. Пользователь выделил фрагмент в учебном материале ноды.

{RUSSIAN_OUTPUT_RULE}

ЗАДАЧА: углубить выделение, опираясь на SOURCE REGISTRY и материал ноды (не пересказывать summary).

ПРАВИЛА:
1. НЕ пересказывай выделенный текст.
2. Если в registry есть [Sx] — цитируй механику из snippet источника.
3. Структура:
   - 🔍 **Деталь из источника ([Sx])**: что описывают авторы/документация?
   - 💡 **Коротко**: смысл для инженера.
"""


def _extract_source_ids(text: str) -> list[str]:
    ids: list[str] = []
    for m in _SOURCE_TAG_RE.finditer(text or ""):
        sid = f"S{m.group(1)}"
        if sid not in ids:
            ids.append(sid)
    return ids


def _registry_entry(registry: list[dict[str, Any]], source_id: str) -> dict[str, Any] | None:
    want = (source_id or "").strip().upper()
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("id") or entry.get("source_id") or "").strip().upper()
        if sid == want:
            return entry
    return None


def _material_from_registry(
    registry: list[dict[str, Any]],
    hint_ids: list[str],
) -> tuple[str, ExplainSourceRef]:
    entry: dict[str, Any] | None = None
    for sid in hint_ids:
        entry = _registry_entry(registry, sid)
        if entry:
            break
    if not entry and registry:
        entry = registry[0]
    if not entry:
        return "(нет источников в registry)", ExplainSourceRef()

    sid = str(entry.get("id") or "")
    title = str(entry.get("title") or "Source")
    url = str(entry.get("url") or "")
    snippet = (entry.get("snippet") or "")[:4000]
    material = (
        f"--- SOURCE [{sid}] ---\n"
        f"Title: {title}\nURL: {url}\n\n{snippet}"
    )
    return material, ExplainSourceRef(title=title, url=url, source_id=sid)


def run_node_selection_explain(
    node_title: str,
    selected_text: str,
    user_question: str,
    surrounding_paragraph: str,
    summary_excerpt: str,
    rag_profile: str,
    source_registry: list[dict[str, Any]],
    anchor: str,
) -> NodeExplainResult:
    selected = strip_source_anchor_tags((selected_text or "").strip())
    if len(selected) < 2:
        raise ValueError("selected_text is too short")

    question = (user_question or "").strip() or DEFAULT_EXPLAIN_QUESTION
    hints = _extract_source_ids(selected) + _extract_source_ids(surrounding_paragraph)
    material, source_ref = _material_from_registry(source_registry, hints)

    registry_block = registry_for_prompt(source_registry)
    summary = strip_source_anchor_tags((summary_excerpt or "").strip()[:6000])
    rag = strip_source_anchor_tags((rag_profile or "").strip()[:2000])

    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_text

    user_payload = (
        f"### node_title\n{node_title}\n\n"
        f"### SOURCE REGISTRY\n{registry_block}\n\n"
        f"### node_summary_excerpt\n{summary or '(пусто)'}\n\n"
        f"### rag_profile\n{rag or '(пусто)'}\n\n"
        f"### highlighted_text\n{selected}\n\n"
        f"### user_question\n{question}\n\n"
        f"### context_paragraph\n{(surrounding_paragraph or '').strip()[:2000]}\n\n"
        f"### matched_source_material\n{material}"
    )
    explanation = run_gemini_lite_text(
        _NODE_EXPLAIN_SYSTEM,
        user_payload,
        anchor,
        "node_selection_explain",
    ).strip()
    return NodeExplainResult(explanation=explanation, source_ref=source_ref)
