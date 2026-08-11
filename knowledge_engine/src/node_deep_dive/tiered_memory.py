"""Сборка tiered memory и ротация скользящего окна."""

from __future__ import annotations

import re

from knowledge_engine.src.node_deep_dive.fact_manifest import format_fact_manifest_block
from knowledge_engine.src.node_deep_dive.lecture_coverage_registry import (
    format_coverage_registry_block,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    CoreConceptRecord,
    NodeStatus,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.term_registry import (
    format_already_explained_terms_block,
)

ACTIVE_WINDOW_MAX = 6  # 3 цикла tutor+user
ROLE_USER = "user"
ROLE_TUTOR = "tutor"

_MERMAID_FENCE_RE = re.compile(
    r"```\s*mermaid[\s\S]*?```",
    re.IGNORECASE,
)
_COLLAPSE_MIDDLE_MARKER = (
    "\n\n... [теоретический блок свернут, суть учтена в concept_map] ...\n\n"
)
_MIN_TUTOR_COMPRESS_CHARS = 520


def _strip_mermaid_blocks(text: str) -> str:
    return _MERMAID_FENCE_RE.sub("[Схема сгенерирована]", text or "")


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text.strip()] if text.strip() else []


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?…])\s+", (text or "").strip())
    return [c.strip() for c in chunks if c.strip()]


def _first_unit(text: str) -> str:
    paras = _paragraphs(text)
    if paras:
        return paras[0]
    sents = _sentences(text)
    return sents[0] if sents else text.strip()


def _last_unit(text: str) -> str:
    paras = _paragraphs(text)
    if paras:
        return paras[-1]
    sents = _sentences(text)
    return sents[-1] if sents else text.strip()


def compress_tutor_message_for_window(content: str) -> str:
    """
    Smart Head+Tail для прошлых реплик тьютора: первый и последний смысловые блоки,
    без обрезки посередине предложения.
    """
    raw = (content or "").strip()
    if not raw:
        return ""
    body = _strip_mermaid_blocks(raw)
    if len(body) <= _MIN_TUTOR_COMPRESS_CHARS:
        return body
    head = _first_unit(body)
    tail = _last_unit(body)
    if not head:
        return tail or body
    if not tail:
        return head
    if head.strip() == tail.strip():
        return head
    if len(body) <= len(head) + len(tail) + len(_COLLAPSE_MIDDLE_MARKER) + 40:
        return body
    return f"{head.strip()}{_COLLAPSE_MIDDLE_MARKER}{tail.strip()}"


def _format_role_content_for_window(role: str, content: str) -> str:
    r = (role or "user").strip().lower()
    if r in (ROLE_USER, "human"):
        return (content or "").strip()
    if r in (ROLE_TUTOR, "assistant", "model"):
        return compress_tutor_message_for_window(content)
    return (content or "").strip()


def format_window_for_llm(window: list[dict[str, str]]) -> str:
    if not window:
        return "(пусто — диалог ещё не начат)"
    lines: list[str] = []
    for m in window:
        role = str(m.get("role") or "user").strip()
        text = _format_role_content_for_window(role, str(m.get("content") or ""))
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else "(пусто — диалог ещё не начат)"


def compress_rag_profile_text(text: str, max_len: int = 1400) -> str:
    t = (text or "").strip()
    if not t:
        return "(нет персональных фактов RAG для этой ноды)"
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "…"


def init_session_memory(
    node: NodeDataInput,
    rag_facts_text: str,
    *,
    preserved_covered_subtopics: dict[str, str] | None = None,
    preserved_introduced_terms: list[str] | None = None,
) -> SessionMemory:
    matrix = [
        CoreConceptRecord(concept=str(c).strip())
        for c in node.core_concepts
        if str(c).strip()
    ]
    mem = SessionMemory(
        rag_profile_compressed=compress_rag_profile_text(rag_facts_text),
        concepts_matrix=matrix,
        rolling_dialogue_summary="",
        active_window=[],
        topic_mastery_score=0,
    )
    from knowledge_engine.src.node_deep_dive.concept_map import ensure_sub_concept_map

    ensure_sub_concept_map(mem, node)
    if preserved_covered_subtopics:
        from knowledge_engine.src.node_deep_dive.session_store import (
            carry_covered_subtopics,
        )

        carry_covered_subtopics(preserved_covered_subtopics, mem)
    if preserved_introduced_terms:
        from knowledge_engine.src.node_deep_dive.term_registry import (
            carry_introduced_terms,
        )

        carry_introduced_terms(preserved_introduced_terms, mem)
    return mem


def memory_from_blob(raw: dict | None) -> SessionMemory | None:
    if not raw or not isinstance(raw, dict):
        return None
    data = dict(raw)
    data.setdefault("covered_subtopics", {})
    data.setdefault("introduced_terms", [])
    try:
        return SessionMemory.model_validate(data)
    except Exception:
        return None


def memory_to_blob(memory: SessionMemory) -> dict:
    return memory.model_dump()


def _norm_concept_key(s: str) -> str:
    return " ".join((s or "").lower().split())


def find_concept_record(
    matrix: list[CoreConceptRecord],
    concept_ref: str,
) -> CoreConceptRecord | None:
    key = _norm_concept_key(concept_ref)
    if not key:
        return None
    for row in matrix:
        if _norm_concept_key(row.concept) == key:
            return row
    for row in matrix:
        if (
            key in _norm_concept_key(row.concept)
            or _norm_concept_key(row.concept) in key
        ):
            return row
    return None


def apply_concept_updates(
    memory: SessionMemory,
    updates: list,
) -> None:
    for upd in updates:
        concept = getattr(upd, "concept", "") or ""
        row = find_concept_record(memory.concepts_matrix, concept)
        if not row:
            continue
        status = getattr(upd, "status", None)
        if status:
            row.status = status
        evidence = (getattr(upd, "evidence", "") or "").strip()
        if evidence:
            row.evidence = evidence[:2000]
        score = getattr(upd, "mastery_score", None)
        if score is not None:
            row.mastery_score = int(score)


def compute_topic_mastery(memory: SessionMemory) -> int:
    rows = memory.concepts_matrix
    if not rows:
        return 0
    total = sum(int(r.mastery_score) for r in rows)
    return min(100, max(0, round(total / len(rows))))


def engagement_topic_mastery(memory: SessionMemory) -> int:
    """Прогресс для UI: матрица + фаза цикла и статусы концептов."""
    from_matrix = compute_topic_mastery(memory)
    phase_floor = {
        "intro_assessment": 12,
        "dense_material": 32,
        "checkpoint": 52,
        "pathway_decision": 68,
        "socratic_focus": 82,
    }.get(memory.learning_phase, 0)
    rows = memory.concepts_matrix
    status_floor = 0
    if rows:
        verified = sum(1 for r in rows if r.status == "verified")
        in_prog = sum(1 for r in rows if r.status == "in_progress")
        status_floor = min(92, verified * 22 + in_prog * 12)
    sub = memory.sub_concepts
    layer_score = 0
    if sub:
        sc_verified = sum(1 for s in sub if s.status == "verified")
        sc_partial = sum(1 for s in sub if s.status == "partial")
        sub_floor = min(
            95,
            round(100 * sc_verified / len(sub)) + sc_partial * 4,
        )
        status_floor = max(status_floor, sub_floor)
        # WHY/HOW/MECHANIC thirds (aligned with Threshold Engine + UI depth bar).
        why = sum(1 for s in sub if s.why_passed) / len(sub)
        how = sum(1 for s in sub if s.how_passed) / len(sub)
        mech = sum(1 for s in sub if s.mechanic_passed) / len(sub)
        layer_score = min(100, round(100.0 * (why + how + mech) / 3.0))
        if any(s.why_passed or s.how_passed or s.mechanic_passed for s in sub):
            # Prefer depth-layer score when micro-eval flags are present.
            return min(100, max(from_matrix, layer_score))
    return min(100, max(from_matrix, phase_floor, status_floor))


def sync_topic_mastery_score(memory: SessionMemory) -> int:
    score = engagement_topic_mastery(memory)
    memory.topic_mastery_score = score
    return score


def derive_node_status(
    memory: SessionMemory,
    critical_gap: str | None,
) -> NodeStatus:
    score = sync_topic_mastery_score(memory)
    rows = memory.concepts_matrix
    all_verified = bool(rows) and all(r.status == "verified" for r in rows)
    gap_text = (critical_gap or "").strip()
    if gap_text and score < 40:
        return "gap"
    if all_verified and score >= 100:
        return "mastered"
    if score >= 40:
        return "deep_understanding"
    return "in_progress"


def append_to_active_window(memory: SessionMemory, role: str, content: str) -> None:
    text = (content or "").strip()
    if not text:
        return
    r = role if role in (ROLE_USER, ROLE_TUTOR) else ROLE_TUTOR
    from knowledge_engine.src.node_deep_dive.dialog_ids import (
        dialog_message,
        next_msg_id,
    )

    msg_id = next_msg_id(memory)
    memory.active_window.append(dialog_message(r, text, msg_id))
    memory.dialog_seq = msg_id


def pop_evicted_message(memory: SessionMemory) -> dict[str, str] | None:
    if len(memory.active_window) <= ACTIVE_WINDOW_MAX:
        return None
    return memory.active_window.pop(0)


def format_matrix_for_llm(matrix: list[CoreConceptRecord]) -> str:
    if not matrix:
        return "(пусто)"
    lines = []
    for i, row in enumerate(matrix, 1):
        ev = (row.evidence or "").strip()
        ev_part = f" | evidence: {ev[:400]}" if ev else ""
        lines.append(
            f"{i}. {row.concept} | status={row.status} | "
            f"mastery={row.mastery_score}%{ev_part}"
        )
    return "\n".join(lines)


def format_evicted_for_llm(messages: list[dict[str, str]]) -> str:
    return format_window_for_llm(messages)


def build_handoff_summary(memory: SessionMemory) -> str:
    """Сжатое состояние для новой модели (manifest + matrix, без прозаического rolling)."""
    reg_block = format_coverage_registry_block(memory)
    terms_block = format_already_explained_terms_block(memory)
    parts: list[str] = []
    manifest_block = format_fact_manifest_block(memory)
    if manifest_block.strip() and "{}" not in manifest_block[-3:]:
        parts.append(manifest_block)
    roll = (memory.rolling_dialogue_summary or "").strip()
    if roll and ("[!]" in roll or "PENDING" in roll):
        parts.append(f"legacy_pending_hint:\n{roll[:1500]}")
    matrix = format_matrix_for_llm(memory.concepts_matrix)
    if matrix and matrix != "(пусто)":
        parts.append(f"concepts_matrix:\n{matrix[:2000]}")
    if memory.pathway_bridge.strip():
        parts.append(f"pathway_bridge: {memory.pathway_bridge.strip()[:800]}")
    parts.append(f"topic_mastery_score: {memory.topic_mastery_score}%")
    parts.append(f"learning_phase: {memory.learning_phase}")
    body = "\n\n".join(parts)
    reg_budget = min(len(reg_block), 2800) if reg_block else 0
    terms_budget = min(len(terms_block), 1200) if terms_block else 0
    rest_budget = max(
        0, 6000 - reg_budget - terms_budget - (8 if reg_block or terms_block else 0)
    )
    rest = body[:rest_budget]
    head_parts: list[str] = []
    if reg_block:
        head_parts.append(reg_block[:reg_budget])
    if terms_block:
        head_parts.append(terms_block[:terms_budget])
    if head_parts:
        return f"{chr(10).join(head_parts)}\n\n{rest}".strip()[:6000]
    return rest[:6000]


def build_tiered_context_payload(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    behavior_state_block: str,
    user_message: str,
) -> str:
    concepts_list = "\n".join(f"- {c}" for c in node.core_concepts)
    msg = (user_message or "").strip()
    user_block = f"\n\n### current_user_message\n{msg}" if msg else ""
    return (
        f"### user_action\n{action}\n"
        f"### node_id\n{node.node_id}\n"
        f"### title\n{node.title}\n"
        f"### layer\n{node.layer}\n"
        f"### category\n{node.category}\n"
        f"### core_concepts_list\n{concepts_list}\n"
        f"### brief_summary\n{node.brief_summary}\n\n"
        f"### layer_1_compressed_rag_profile\n{memory.rag_profile_compressed}\n\n"
        f"### layer_2_core_concepts_matrix\n{format_matrix_for_llm(memory.concepts_matrix)}\n"
        f"### topic_mastery_score\n{memory.topic_mastery_score}%\n\n"
        f"### learning_phase\n{memory.learning_phase}\n"
        f"### learning_mode\n{memory.learning_mode}\n\n"
        f"### detected_user_intent\n{intent}\n"
        f"{behavior_state_block}"
        f"{user_block}"
    )


def build_tiered_static_context(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    behavior_state_block: str,
) -> str:
    return build_tiered_context_payload(
        memory, node, intent, action, behavior_state_block, user_message=""
    )
