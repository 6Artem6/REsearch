"""Сквозной качественный профиль компетенций по curriculum (DAG)."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import (
    get_all_sessions_for_curriculum,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import memory_from_blob

_STORE_DIR = PACKAGE_ROOT / ".runs"
_lock = threading.Lock()

_MAX_PROVEN = 7
_MAX_BLIND = 4
_PINNED_CHAR_BUDGET = 620  # ~150 токенов

_ENTITY_RE = re.compile(r"[a-zA-ZА-Яа-я][a-zA-ZА-Яа-я0-9\-]{2,}")


class CompetencyDelta(BaseModel):
    new_proven_skills: list[str] = Field(default_factory=list)
    new_blind_spots: list[str] = Field(default_factory=list)
    resolved_blind_spots: list[str] = Field(default_factory=list)
    new_mastered_entities: list[str] = Field(default_factory=list)

    def has_updates(self) -> bool:
        return bool(
            self.new_proven_skills
            or self.new_blind_spots
            or self.resolved_blind_spots
            or self.new_mastered_entities
        )


class UserCompetencyProfile(BaseModel):
    proven_skills: list[str] = Field(default_factory=list, max_length=7)
    blind_spots: list[str] = Field(default_factory=list, max_length=4)
    mastered_entities: list[str] = Field(default_factory=list, max_length=48)

    @field_validator("proven_skills", "blind_spots", "mastered_entities", mode="before")
    @classmethod
    def _coerce_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []


def _store_path(curriculum_id: str) -> Any:
    safe = re.sub(r"[^\w\-]+", "_", (curriculum_id or "").strip())[:80]
    return _STORE_DIR / f"curriculum_user_mastery_{safe}.json"


def _normalize_entity(text: str) -> str:
    return " ".join((text or "").lower().split())[:120]


def _normalize_skill(text: str) -> str:
    t = " ".join((text or "").split())
    return t[:240]


def _similar_line(a: str, b: str) -> bool:
    al = a.lower()
    bl = b.lower()
    if al == bl:
        return True
    if len(al) > 20 and (al in bl or bl in al):
        return True
    return False


def _dedupe_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        line = _normalize_skill(raw)
        if not line:
            continue
        if any(_similar_line(line, x) for x in out):
            continue
        out.append(line)
    return out


def _consolidate_proven_skills(skills: list[str]) -> list[str]:
    """При >7 сжать старые записи в одну обобщённую формулировку."""
    deduped = _dedupe_lines(skills)
    if len(deduped) <= _MAX_PROVEN:
        return deduped
    overflow = deduped[: len(deduped) - (_MAX_PROVEN - 1)]
    kept = deduped[len(deduped) - (_MAX_PROVEN - 1) :]
    themes = "; ".join(s[:60] for s in overflow[:4])
    summary = f"Обобщённо усвоено ранее: {themes}"[:240]
    return _dedupe_lines([summary] + kept)


def _entities_from_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    found = _ENTITY_RE.findall(raw)
    out: list[str] = []
    seen: set[str] = set()
    for w in found:
        key = _normalize_entity(w)
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_raw_store(curriculum_id: str) -> dict[str, Any]:
    cid = (curriculum_id or "").strip()
    if not cid:
        return {}
    path = _store_path(cid)
    with _lock:
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def _save_raw_store(curriculum_id: str, raw: dict[str, Any]) -> None:
    cid = (curriculum_id or "").strip()
    if not cid:
        return
    path = _store_path(cid)
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _profile_from_legacy_map(mastery_map: dict[str, float]) -> UserCompetencyProfile:
    entities = [k for k, v in mastery_map.items() if float(v) >= 0.55]
    return UserCompetencyProfile(mastered_entities=entities[:48])


def get_curriculum_competency_profile(curriculum_id: str) -> UserCompetencyProfile:
    raw = _load_raw_store(curriculum_id)
    blob = raw.get("competency_profile")
    if isinstance(blob, dict):
        try:
            return UserCompetencyProfile.model_validate(blob)
        except Exception:
            pass
    legacy = raw.get("user_mastery_map")
    if isinstance(legacy, dict) and legacy:
        return _profile_from_legacy_map(
            {str(k): float(v) for k, v in legacy.items() if k}
        )
    return UserCompetencyProfile()


def save_curriculum_competency_profile(
    curriculum_id: str,
    profile: UserCompetencyProfile,
) -> None:
    cid = (curriculum_id or "").strip()
    if not cid:
        return
    raw = _load_raw_store(cid)
    raw["curriculum_id"] = cid
    raw["competency_profile"] = profile.model_dump()
    _save_raw_store(cid, raw)


def merge_competency_delta(
    curriculum_id: str,
    delta: CompetencyDelta,
) -> UserCompetencyProfile:
    profile = get_curriculum_competency_profile(curriculum_id)

    resolved = [_normalize_skill(s) for s in delta.resolved_blind_spots if s]
    for r in resolved:
        profile.blind_spots = [
            b for b in profile.blind_spots if not _similar_line(b, r)
        ]
        if r and not any(_similar_line(r, p) for p in profile.proven_skills):
            profile.proven_skills.append(r)

    for s in delta.new_proven_skills:
        line = _normalize_skill(s)
        if line and not any(_similar_line(line, p) for p in profile.proven_skills):
            profile.proven_skills.append(line)

    for b in delta.new_blind_spots:
        line = _normalize_skill(b)
        if not line:
            continue
        if any(_similar_line(line, x) for x in profile.blind_spots):
            continue
        if any(_similar_line(line, p) for p in profile.proven_skills):
            continue
        profile.blind_spots.append(line)

    entities: list[str] = list(profile.mastered_entities)
    for e in delta.new_mastered_entities:
        key = _normalize_entity(e)
        if key and key not in entities:
            entities.append(key)
    profile.mastered_entities = entities[:48]

    profile.proven_skills = _consolidate_proven_skills(profile.proven_skills)
    profile.blind_spots = _dedupe_lines(profile.blind_spots)[-_MAX_BLIND:]

    save_curriculum_competency_profile(curriculum_id, profile)
    return profile


def merge_entities_from_session_memory(
    curriculum_id: str,
    memory: SessionMemory | None,
    extra_entities: list[str] | None = None,
) -> UserCompetencyProfile:
    """Лёгкое обновление mastered_entities без Ollama (equivalence / verified concepts)."""
    profile = get_curriculum_competency_profile(curriculum_id)
    entities: list[str] = list(profile.mastered_entities)
    for ent in extra_entities or []:
        key = _normalize_entity(ent)
        if key and key not in entities:
            entities.append(key)
    if memory is not None:
        for row in memory.concepts_matrix:
            if row.status == "verified" or row.mastery_score >= 60:
                key = _normalize_entity(row.concept)
                if key and key not in entities:
                    entities.append(key)
    profile.mastered_entities = entities[:48]
    save_curriculum_competency_profile(curriculum_id, profile)
    return profile


def rebuild_competency_profile_from_sessions(
    curriculum_id: str,
) -> UserCompetencyProfile:
    """Re-eval: entities из сессий; proven/blind не пересобираем (история в файле)."""
    for blob in get_all_sessions_for_curriculum(curriculum_id).values():
        mem = memory_from_blob(blob.get("memory"))
        if mem is None:
            continue
        merge_entities_from_session_memory(curriculum_id, mem)
    return get_curriculum_competency_profile(curriculum_id)


# --- Legacy numeric map (fast-track fallback) ---


def get_curriculum_user_mastery_map(curriculum_id: str) -> dict[str, float]:
    profile = get_curriculum_competency_profile(curriculum_id)
    out: dict[str, float] = {}
    for ent in profile.mastered_entities:
        out[ent] = 0.85
    for skill in profile.proven_skills:
        for ent in _entities_from_text(skill):
            out[ent] = max(out.get(ent, 0.0), 0.75)
    return out


def mastered_entities_set(
    mastery_map: dict[str, float] | None = None,
    *,
    threshold: float = 0.55,
    profile: UserCompetencyProfile | None = None,
) -> set[str]:
    if profile is not None:
        return {_normalize_entity(e) for e in profile.mastered_entities if e}
    m = mastery_map or {}
    return {k for k, v in m.items() if v >= threshold}


def _node_relevance_terms(
    node: NodeDataInput,
    curriculum_id: str,
) -> set[str]:
    terms: set[str] = set()
    for part in (
        node.title,
        node.category,
        node.brief_summary,
        node.layer,
    ):
        for w in _entities_from_text(str(part or "")):
            terms.add(w)
    for c in node.core_concepts or []:
        for w in _entities_from_text(str(c)):
            terms.add(w)
    for tag in getattr(node, "tags", None) or []:
        for w in _entities_from_text(str(tag)):
            terms.add(w)
    if (curriculum_id or "").strip():
        from knowledge_engine.services.skill_tree_store import (
            get_node_neighbors_context,
        )

        ctx = get_node_neighbors_context(curriculum_id, node.node_id)
        for p in ctx.get("predecessors") or []:
            for field in ("title", "short_concepts"):
                for w in _entities_from_text(str(p.get(field) or "")):
                    terms.add(w)
    return {t for t in terms if len(t) >= 3}


def _line_relevant_to_terms(line: str, terms: set[str]) -> bool:
    low = line.lower()
    for t in terms:
        if t in low or low in t:
            return True
    words = set(_entities_from_text(line))
    return bool(words & terms)


def filter_competency_for_node(
    profile: UserCompetencyProfile,
    node: NodeDataInput,
    curriculum_id: str,
) -> tuple[list[str], list[str]]:
    terms = _node_relevance_terms(node, curriculum_id)
    if not terms:
        return profile.proven_skills[:4], profile.blind_spots[:3]
    proven = [p for p in profile.proven_skills if _line_relevant_to_terms(p, terms)]
    blind = [b for b in profile.blind_spots if _line_relevant_to_terms(b, terms)]
    if not proven and profile.proven_skills:
        proven = profile.proven_skills[:2]
    if not blind and profile.blind_spots:
        blind = profile.blind_spots[:2]
    return proven[:5], blind[:3]


def format_competency_pinned_block(
    node: NodeDataInput,
    curriculum_id: str,
    *,
    profile: UserCompetencyProfile | None = None,
) -> str:
    prof = profile or get_curriculum_competency_profile(curriculum_id)
    if not prof.proven_skills and not prof.blind_spots:
        return ""
    proven, blind = filter_competency_for_node(prof, node, curriculum_id)
    parts: list[str] = ["### competency_profile (релевантно ноде)"]
    if proven:
        parts.append(
            "Доказанные навыки в этой теме (НЕ задавай базовые вопросы по ним):\n"
            + "\n".join(f"- {p[:120]}" for p in proven)
        )
    if blind:
        parts.append(
            "Слепые зоны для точечной проверки (если сценарий позволяет):\n"
            + "\n".join(f"- {b[:120]}" for b in blind)
        )
    text = "\n\n".join(parts)
    if len(text) > _PINNED_CHAR_BUDGET:
        text = text[: _PINNED_CHAR_BUDGET - 1].rstrip() + "…"
    return text


def format_user_mastery_map_for_prompt(
    mastery_map: dict[str, float],
    limit: int = 40,
) -> str:
    """Legacy alias — для init payload без ноды."""
    if not mastery_map:
        return "(профиль компетенций ещё не накоплен)"
    rows = sorted(mastery_map.items(), key=lambda x: (-x[1], x[0]))
    lines = [f"- {name}" for name, _ in rows[:limit]]
    return "\n".join(lines)


def merge_mastery_from_session_memory(
    curriculum_id: str,
    memory: SessionMemory | None,
    extra_entities: list[str] | None = None,
    *,
    score: float = 0.85,
) -> dict[str, float]:
    merge_entities_from_session_memory(
        curriculum_id,
        memory,
        extra_entities=extra_entities,
    )
    return get_curriculum_user_mastery_map(curriculum_id)


def rebuild_mastery_map_from_curriculum_sessions(
    curriculum_id: str,
) -> dict[str, float]:
    rebuild_competency_profile_from_sessions(curriculum_id)
    return get_curriculum_user_mastery_map(curriculum_id)
