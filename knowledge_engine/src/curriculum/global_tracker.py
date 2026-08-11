"""Global Concept Registry: инкрементальная дельта-синхронизация из SessionMemory."""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime
from typing import TYPE_CHECKING

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.schemas.global_knowledge import (
    GlobalSubConceptDelta,
    GlobalUserKnowledgeState,
    GlobalVerifiedSubConcept,
    utc_now_iso,
)
from knowledge_engine.src.config.question_angles_loader import (
    get_angle_description,
    get_angle_keywords_map,
)
from knowledge_engine.src.node_deep_dive.concept_map import slug_sub_concept_id
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.session_store import _lock as _persist_lock
from knowledge_engine.src.node_deep_dive.session_store import (
    get_all_sessions_for_curriculum,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import memory_from_blob
from knowledge_engine.ui.run_log import trace

if TYPE_CHECKING:
    from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput

_STORE_DIR = PACKAGE_ROOT / ".runs"

# ~300–400 токенов для блока «уже изучено» (смешанный ru/en текст)
GLOBAL_LEARNED_TOKEN_BUDGET = 380
MAX_PROMPT_CONCEPTS = 7

_TOKEN_RE = re.compile(r"[\wа-яёА-ЯЁ]+", re.IGNORECASE)

DEFAULT_USER_ID = "default"


def _store_path(user_id: str, curriculum_id: str) -> pathlib.Path:
    from pathlib import Path

    uid = re.sub(r"[^\w\-]+", "_", (user_id or DEFAULT_USER_ID).strip())[:48]
    cid = re.sub(r"[^\w\-]+", "_", (curriculum_id or "").strip())[:80]
    return Path(_STORE_DIR) / f"global_knowledge_{uid}_{cid}.json"


def load_global_knowledge_state(
    user_id: str,
    curriculum_id: str,
) -> GlobalUserKnowledgeState:
    cid = (curriculum_id or "").strip()
    if not cid:
        return GlobalUserKnowledgeState(curriculum_id="")
    path = _store_path(user_id, cid)
    with _persist_lock:
        if not path.is_file():
            return GlobalUserKnowledgeState(
                user_id=(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID,
                curriculum_id=cid,
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return GlobalUserKnowledgeState.model_validate(raw)
        except Exception:
            return GlobalUserKnowledgeState(
                user_id=(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID,
                curriculum_id=cid,
            )


def save_global_knowledge_state(state: GlobalUserKnowledgeState) -> None:
    cid = (state.curriculum_id or "").strip()
    if not cid:
        return
    path = _store_path(state.user_id, cid)
    state.updated_at = utc_now_iso()
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    with _persist_lock:
        path.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _node_title_from_graph(curriculum_id: str, node_id: str) -> str:
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    graph = get_curriculum_graph(curriculum_id) or {}
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("node_id") or "").strip() == node_id:
            return str(raw.get("title") or node_id).strip()[:300]
    return node_id


def extract_verified_from_memory(
    curriculum_id: str,
    node_id: str,
    memory: SessionMemory,
    *,
    node_title: str = "",
) -> list[GlobalVerifiedSubConcept]:
    """Десериализованный SessionMemory → записи реестра (O(sub_concepts + matrix))."""
    cid = curriculum_id.strip()
    nid = node_id.strip()
    title = (node_title or _node_title_from_graph(cid, nid)).strip()[:300]
    now = utc_now_iso()
    out: list[GlobalVerifiedSubConcept] = []
    seen: set[str] = set()

    for sc in memory.sub_concepts or []:
        if sc.status != "verified":
            continue
        key = f"{nid}::{sc.id}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            GlobalVerifiedSubConcept(
                curriculum_id=cid,
                node_id=nid,
                node_title=title,
                sub_concept_id=sc.id,
                label=sc.label[:200],
                status="verified",
                is_verified=True,
                evidence=(sc.evidence or "")[:400],
                updated_at=(sc.updated_at or now)[:40],
            )
        )

    for i, row in enumerate(memory.concepts_matrix or []):
        verified = row.status == "verified" or int(row.mastery_score or 0) >= 60
        if not verified:
            continue
        sid = slug_sub_concept_id(row.concept, i)
        key = f"{nid}::{sid}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            GlobalVerifiedSubConcept(
                curriculum_id=cid,
                node_id=nid,
                node_title=title,
                sub_concept_id=sid,
                label=row.concept[:200],
                status="verified" if row.status == "verified" else "partial",
                is_verified=True,
                evidence=(row.evidence or "")[:400],
                mastery_score=int(row.mastery_score or 0),
                updated_at=now,
            )
        )
    return out


def merge_session_into_global_state(
    state: GlobalUserKnowledgeState,
    entries: list[GlobalVerifiedSubConcept],
    *,
    node_id: str,
    session_updated_at: str = "",
) -> GlobalSubConceptDelta:
    """Инкремент: только новые ключи относительно state.entries."""
    delta_new: list[GlobalVerifiedSubConcept] = []
    for ent in entries:
        key = ent.registry_key()
        prev = state.entries.get(key)
        if prev is not None and prev.is_verified:
            continue
        state.entries[key] = ent
        delta_new.append(ent)
    if session_updated_at:
        state.node_session_sync_at[node_id.strip()] = session_updated_at[:40]
    if delta_new:
        state.revision += 1
    return GlobalSubConceptDelta(
        new_entries=delta_new,
        revision=state.revision,
        total_entries=len(state.entries),
    )


def sync_global_registry_from_session(
    user_id: str,
    curriculum_id: str,
    node_id: str,
    memory: SessionMemory | None,
    *,
    session_updated_at: str = "",
    node_title: str = "",
) -> GlobalSubConceptDelta:
    """Коммит одной ноды в глобальный реестр (вызывать под session_store lock)."""
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    if not cid or not nid or memory is None:
        return GlobalSubConceptDelta()
    uid = (user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    state = load_global_knowledge_state(uid, cid)
    extracted = extract_verified_from_memory(cid, nid, memory, node_title=node_title)
    delta = merge_session_into_global_state(
        state,
        extracted,
        node_id=nid,
        session_updated_at=session_updated_at,
    )
    save_global_knowledge_state(state)
    if delta.new_entries:
        trace(
            f"GLOBAL_REGISTRY delta +{len(delta.new_entries)} | "
            f"{cid}/{nid} rev={state.revision}"
        )
    return delta


def rebuild_global_registry_from_all_sessions(
    user_id: str,
    curriculum_id: str,
) -> GlobalUserKnowledgeState:
    """Полная пересборка из JSON сессий (repair / migration)."""
    cid = (curriculum_id or "").strip()
    uid = (user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    state = GlobalUserKnowledgeState(user_id=uid, curriculum_id=cid, entries={})
    for nid, blob in get_all_sessions_for_curriculum(cid).items():
        mem = memory_from_blob(blob.get("memory"))
        if mem is None:
            continue
        extracted = extract_verified_from_memory(cid, nid, mem)
        merge_session_into_global_state(
            state,
            extracted,
            node_id=nid,
            session_updated_at=str(blob.get("updated_at") or "")[:40],
        )
    state.revision += 1
    save_global_knowledge_state(state)
    return state


def get_global_verified_subconcepts_delta(
    user_id: str,
    curriculum_id: str,
    current_node_id: str,
    *,
    current_node: NodeDataInput | None = None,
) -> tuple[GlobalUserKnowledgeState, GlobalSubConceptDelta]:
    """
    Состояние реестра + дельта для промпта (записи других нод, не current_node_id).

    O(K) по числу записей реестра; без чтения сырой истории чата.
    """
    cid = (curriculum_id or "").strip()
    uid = (user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    current = (current_node_id or "").strip()
    state = load_global_knowledge_state(uid, cid)
    if not state.entries:
        rebuild_global_registry_from_all_sessions(uid, cid)
        state = load_global_knowledge_state(uid, cid)

    prior_keys = set(state.entries.keys())
    # Лёгкий инкремент: подтянуть ноды, чьи сессии новее sync watermark
    for nid, blob in get_all_sessions_for_curriculum(cid).items():
        if nid == current:
            continue
        sess_at = str(blob.get("updated_at") or "")
        synced = state.node_session_sync_at.get(nid, "")
        if sess_at and synced and sess_at <= synced:
            continue
        mem = memory_from_blob(blob.get("memory"))
        if mem is None:
            continue
        extracted = extract_verified_from_memory(cid, nid, mem)
        merge_session_into_global_state(
            state,
            extracted,
            node_id=nid,
            session_updated_at=sess_at,
        )
    if set(state.entries.keys()) != prior_keys:
        save_global_knowledge_state(state)

    others = [
        e for e in state.entries.values() if e.is_verified and e.node_id != current
    ]
    ranked = rank_entries_for_prompt(
        others,
        curriculum_id=cid,
        current_node=current_node,
    )
    prompt_entries, omitted = select_prompt_concepts_for_budget(
        ranked,
        token_budget=GLOBAL_LEARNED_TOKEN_BUDGET,
        max_concepts=MAX_PROMPT_CONCEPTS,
    )
    return state, GlobalSubConceptDelta(
        new_entries=others,
        prompt_entries=prompt_entries,
        omitted_concept_count=omitted,
        revision=state.revision,
        total_entries=len(state.entries),
    )


def estimate_prompt_tokens(text: str) -> int:
    """Грубая оценка токенов (смешанный ru/en) без tiktoken."""
    t = (text or "").strip()
    if not t:
        return 0
    return max(1, int(len(t) / 3.2))


def _parse_iso_timestamp(iso: str) -> float:
    raw = (iso or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _node_relevance_terms(node: NodeDataInput) -> set[str]:
    terms: set[str] = set()
    for part in (node.title, node.brief_summary, node.category, node.learning_goal):
        for m in _TOKEN_RE.findall(str(part or "")):
            w = m.lower()
            if len(w) >= 3:
                terms.add(w)
    for c in node.core_concepts or []:
        for m in _TOKEN_RE.findall(str(c)):
            w = m.lower()
            if len(w) >= 3:
                terms.add(w)
    return terms


def _semantic_overlap_score(
    entry: GlobalVerifiedSubConcept,
    terms: set[str],
) -> int:
    if not terms:
        return 0
    blob = f"{entry.label} {entry.evidence}".lower()
    return sum(1 for t in terms if t in blob)


def _dag_parent_node_ids(curriculum_id: str, current_node_id: str) -> set[str]:
    from knowledge_engine.services.skill_tree_store import get_node_neighbors_context

    ctx = get_node_neighbors_context(curriculum_id, current_node_id)
    out: set[str] = set()
    for p in ctx.get("predecessors") or []:
        pid = str(p.get("node_id") or "").strip()
        if pid:
            out.add(pid)
    return out


def rank_entries_for_prompt(
    entries: list[GlobalVerifiedSubConcept],
    *,
    curriculum_id: str,
    current_node: NodeDataInput | None,
) -> list[GlobalVerifiedSubConcept]:
    """
    Top-K приоритет: 1) DAG parents, 2) recency (updated_at), 3) semantic overlap.
    """
    if not entries:
        return []
    nid = (current_node.node_id if current_node else "").strip()
    parents = _dag_parent_node_ids(curriculum_id, nid) if nid else set()
    terms = _node_relevance_terms(current_node) if current_node else set()

    def sort_key(ent: GlobalVerifiedSubConcept) -> tuple[int, float, int]:
        parent_rank = 0 if ent.node_id in parents else 1
        recency = _parse_iso_timestamp(ent.updated_at)
        overlap = _semantic_overlap_score(ent, terms)
        return (parent_rank, -recency, -overlap)

    return sorted(entries, key=sort_key)


def select_prompt_concepts_for_budget(
    ranked: list[GlobalVerifiedSubConcept],
    *,
    token_budget: int,
    max_concepts: int,
) -> tuple[list[GlobalVerifiedSubConcept], int]:
    """Отбор концептов с учётом лимита токенов и max_concepts."""
    if not ranked:
        return [], 0
    header = "[УЖЕ ИЗУЧЕНО В ПРЕДЫДУЩИХ НОДАХ]\n"
    selected: list[GlobalVerifiedSubConcept] = []

    for ent in ranked:
        if len(selected) >= max_concepts:
            break
        trial = selected + [ent]
        full = header + _format_learned_lines_from_entries(trial)
        if estimate_prompt_tokens(full) <= token_budget:
            selected = trial
        elif not selected:
            selected = [ent]
            break

    omitted = max(0, len(ranked) - len(selected))
    return selected, omitted


def _format_learned_lines_from_entries(
    entries: list[GlobalVerifiedSubConcept],
) -> str:
    order: list[str] = []
    by_node: dict[str, tuple[str, list[str]]] = {}
    for ent in entries:
        if ent.node_id not in by_node:
            order.append(ent.node_id)
            by_node[ent.node_id] = (
                (ent.node_title or ent.node_id).strip(),
                [],
            )
        labels = by_node[ent.node_id][1]
        lab = ent.label[:120]
        if lab not in labels:
            labels.append(lab)
    lines: list[str] = []
    for nid in order:
        title, labels = by_node[nid]
        if not labels:
            continue
        lines.append(f'- Нода "{title}": {", ".join(labels)}')
    return "\n".join(lines)


def format_global_learned_block(
    delta: GlobalSubConceptDelta,
    *,
    exclude_node_id: str = "",
    token_budget: int = GLOBAL_LEARNED_TOKEN_BUDGET,
) -> str:
    """Компактный блок [УЖЕ ИЗУЧЕНО В ПРЕДЫДУЩИХ НОДАХ] для промпта."""
    entries = list(delta.prompt_entries or [])
    if not entries:
        exclude = (exclude_node_id or "").strip()
        pool = [
            e
            for e in delta.new_entries
            if e.is_verified and (not exclude or e.node_id != exclude)
        ]
        entries, _ = select_prompt_concepts_for_budget(
            pool[: MAX_PROMPT_CONCEPTS * 2],
            token_budget=token_budget,
            max_concepts=MAX_PROMPT_CONCEPTS,
        )
    if not entries:
        return ""
    header = "[УЖЕ ИЗУЧЕНО В ПРЕДЫДУЩИХ НОДАХ]"
    body = _format_learned_lines_from_entries(entries)
    text = f"{header}\n{body}"
    omitted = delta.omitted_concept_count
    if omitted > 0:
        text += f"\n...и ещё {omitted} усвоенных концептов."
    if estimate_prompt_tokens(text) > token_budget:
        while entries and estimate_prompt_tokens(text) > token_budget:
            entries = entries[:-1]
            body = _format_learned_lines_from_entries(entries)
            extra = omitted + (
                len(delta.prompt_entries or delta.new_entries) - len(entries)
            )
            suffix = f"\n...и ещё {extra} усвоенных концептов." if extra > 0 else ""
            text = f"{header}\n{body}{suffix}"
    return text


def commit_session_and_global_registry(
    user_id: str,
    curriculum_id: str,
    node_id: str,
    memory: SessionMemory | None,
    *,
    session_updated_at: str,
    node_title: str = "",
) -> GlobalSubConceptDelta:
    """Атомарно с session_store: вызывать внутри session_store._lock."""
    return sync_global_registry_from_session(
        user_id,
        curriculum_id,
        node_id,
        memory,
        session_updated_at=session_updated_at,
        node_title=node_title,
    )


def infer_question_angle(tutor_message: str) -> str:
    """Эвристика угла последнего вопроса тьютора (keywords из question_angles.json)."""
    low = (tutor_message or "").lower()
    if not low:
        return ""
    scores: dict[str, int] = {}
    for angle, kws in get_angle_keywords_map().items():
        scores[angle] = sum(1 for k in kws if k in low)
    best = max(scores.values()) if scores else 0
    if best < 1:
        return ""
    for angle, score in scores.items():
        if score == best:
            return angle
    return ""


def format_last_question_angle_hint(last_angle: str) -> str:
    angle = (last_angle or "").strip()
    if not angle:
        return ""
    human = get_angle_description(angle)
    return (
        "### last_tutor_question_angle\n"
        f"Предыдущий вопрос тьютора в этой ноде был в угле: {human}. "
        "Следующий вопрос ДОЛЖЕН использовать другой угол (см. Question Angle Matrix).\n"
    )
