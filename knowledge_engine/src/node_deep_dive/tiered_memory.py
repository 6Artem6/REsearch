"""Сборка tiered memory и ротация скользящего окна."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    CoreConceptRecord,
    NodeStatus,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput

ACTIVE_WINDOW_MAX = 6  # 3 цикла tutor+user
ROLE_USER = "user"
ROLE_TUTOR = "tutor"


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
) -> SessionMemory:
    matrix = [
        CoreConceptRecord(concept=str(c).strip())
        for c in node.core_concepts
        if str(c).strip()
    ]
    return SessionMemory(
        rag_profile_compressed=compress_rag_profile_text(rag_facts_text),
        concepts_matrix=matrix,
        rolling_dialogue_summary="",
        active_window=[],
        topic_mastery_score=0,
    )


def memory_from_blob(raw: dict | None) -> SessionMemory | None:
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return SessionMemory.model_validate(raw)
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
        if key in _norm_concept_key(row.concept) or _norm_concept_key(row.concept) in key:
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


def derive_node_status(
    memory: SessionMemory,
    critical_gap: str | None,
) -> NodeStatus:
    score = compute_topic_mastery(memory)
    memory.topic_mastery_score = score
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
    memory.active_window.append({"role": r, "content": text})


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


def format_window_for_llm(window: list[dict[str, str]]) -> str:
    if not window:
        return "(пусто — диалог ещё не начат)"
    return "\n".join(
        f"{m.get('role', 'user')}: {(m.get('content') or '')[:2000]}"
        for m in window
    )


def format_evicted_for_llm(messages: list[dict[str, str]]) -> str:
    return format_window_for_llm(messages)


def build_handoff_summary(memory: SessionMemory) -> str:
    """Сжатое состояние для новой модели (без сырой multi-turn истории)."""
    parts: list[str] = []
    roll = (memory.rolling_dialogue_summary or "").strip()
    if roll:
        if "PENDING_ACTION" in roll or "[!]" in roll or "NEXT_ACTION_FOR_TUTOR" in roll:
            parts.append(
                "CRITICAL — выполни NEXT_ACTION из rolling_summary (pending материал):\n"
                f"{roll[:4000]}"
            )
        else:
            parts.append(f"rolling_summary:\n{roll[:3500]}")
    matrix = format_matrix_for_llm(memory.concepts_matrix)
    if matrix and matrix != "(пусто)":
        parts.append(f"concepts_matrix:\n{matrix[:2000]}")
    if memory.pathway_bridge.strip():
        parts.append(f"pathway_bridge: {memory.pathway_bridge.strip()[:800]}")
    parts.append(f"topic_mastery_score: {memory.topic_mastery_score}%")
    parts.append(f"learning_phase: {memory.learning_phase}")
    return "\n\n".join(parts)[:6000]


def build_tiered_context_payload(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    behavior_hint: str,
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
        f"### layer_3_rolling_dialogue_summary\n"
        f"{memory.rolling_dialogue_summary or '(пусто)'}\n"
        f"### learning_phase\n{memory.learning_phase}\n"
        f"### learning_mode\n{memory.learning_mode}\n\n"
        f"### detected_user_intent\n{intent}\n"
        f"### tutor_behavior_rules\n{behavior_hint}"
        f"{user_block}"
    )


def build_tiered_static_context(
    memory: SessionMemory,
    node: NodeDataInput,
    intent: str,
    action: str,
    behavior_hint: str,
) -> str:
    """Статический блок без active_window (окно — только delta в chat-сессии)."""
    return build_tiered_context_payload(
        memory, node, intent, action, behavior_hint, user_message=""
    )
