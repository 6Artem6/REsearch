"""Персистентный реестр покрытых микро-тем лекций (Coverage Registry)."""

from __future__ import annotations

import re
from collections.abc import Callable

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LectureExtractedConcept,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    DenseMaterialOutput,
    NodeDataInput,
)

COVERED_SUBTOPICS_MAX_KEYS = 40
_OVERLAP_THRESHOLD = 0.7
_TOKEN_RE = re.compile(r"[\wа-яёА-ЯЁ]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "и",
        "в",
        "на",
        "по",
        "для",
        "как",
        "что",
        "это",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "лекция",
        "лекцию",
        "разобрать",
        "расскажи",
        "объясни",
        "про",
        "ещё",
        "еще",
        "mode",
        "lecture",
    }
)


def slug_concept_key(label: str) -> str:
    raw = (label or "").strip()
    if not raw:
        return "topic"
    cleaned = re.sub(r"[^\w\s\-]", " ", raw, flags=re.UNICODE)
    parts = [p for p in re.split(r"[\s\-]+", cleaned) if p]
    key = "_".join(parts)[:64]
    return key or "topic"


def _topic_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for m in _TOKEN_RE.findall(text or ""):
        t = m.lower()
        if len(t) < 2 or t in _STOPWORDS:
            continue
        out.add(t)
    return out


def coverage_request_overlap(
    user_message: str,
    focus_text: str,
    registry: dict[str, str],
) -> float:
    """Доля токенов запроса, уже присутствующих в реестре (0..1)."""
    if not registry:
        return 0.0
    query = f"{focus_text or ''} {user_message or ''}".strip()
    qtok = _topic_tokens(query)
    if not qtok:
        return 0.0
    covered: set[str] = set()
    for key, summary in registry.items():
        covered |= _topic_tokens(key.replace("_", " "))
        covered |= _topic_tokens(summary)
    if not covered:
        return 0.0
    return len(qtok & covered) / len(qtok)


def matching_registry_keys(
    user_message: str,
    focus_text: str,
    registry: dict[str, str],
    *,
    min_key_overlap: float = 0.45,
) -> list[str]:
    query = f"{focus_text or ''} {user_message or ''}".strip()
    qtok = _topic_tokens(query)
    if not qtok:
        return []
    hits: list[tuple[float, str]] = []
    for key, summary in registry.items():
        rtok = _topic_tokens(key.replace("_", " ")) | _topic_tokens(summary)
        if not rtok:
            continue
        ratio = len(qtok & rtok) / len(qtok)
        if ratio >= min_key_overlap:
            hits.append((ratio, key))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [k for _, k in hits]


def merge_lecture_coverage_from_dense(
    memory: SessionMemory,
    dense: DenseMaterialOutput,
    *,
    focus_text: str = "",
    lecture_scope: str = "",
    persist: Callable[[], None] | None = None,
) -> None:
    """Дополнить memory.covered_subtopics после успешной плотной лекции."""
    concepts = list(dense.extracted_concepts or [])
    if not concepts:
        focus = (focus_text or "").strip()
        summary = (dense.summary or "").strip()
        if focus:
            concepts = [
                LectureExtractedConcept(
                    key=slug_concept_key(focus),
                    summary=(summary or focus)[:600],
                )
            ]
        elif summary and len(summary) >= 40:
            concepts = [
                LectureExtractedConcept(
                    key="last_lecture_block",
                    summary=summary[:600],
                )
            ]
    if not concepts:
        return

    reg = dict(memory.covered_subtopics or {})
    for item in concepts:
        key = (item.key or "").strip() or slug_concept_key(item.summary)
        key = key[:64]
        summary = (item.summary or "").strip()[:600]
        if not summary:
            continue
        prev = reg.get(key, "").strip()
        if prev and summary not in prev:
            merged = f"{prev}; {summary}"[:600]
            reg[key] = merged
        else:
            reg[key] = summary or prev

    scope = (lecture_scope or "").strip()
    focus = (focus_text or "").strip()
    if scope == "targeted_lecture" and focus:
        fk = slug_concept_key(focus)
        if fk not in reg:
            body_hint = (dense.summary or "").strip()[:400]
            reg[fk] = body_hint or focus[:600]

    if len(reg) > COVERED_SUBTOPICS_MAX_KEYS:
        # FIFO по порядку вставки (dict Py3.7+)
        while len(reg) > COVERED_SUBTOPICS_MAX_KEYS:
            reg.pop(next(iter(reg)))

    memory.covered_subtopics = reg
    if persist is not None:
        persist()


def format_coverage_registry_block(memory: SessionMemory) -> str:
    reg = memory.covered_subtopics or {}
    if not reg:
        return ""
    lines = [
        "### lecture_coverage_registry (persistent — не удалять при compact)",
        "Уже подробно разобранные микро-темы в этой сессии ноды:",
    ]
    for key, summary in reg.items():
        lines.append(f"- **{key}**: {summary}")
    return "\n".join(lines)


def format_registry_excerpt_for_notice(
    registry: dict[str, str],
    matching_keys: list[str] | None = None,
    *,
    max_items: int = 8,
) -> str:
    if not registry:
        return ""
    keys = matching_keys if matching_keys else list(registry.keys())
    lines: list[str] = []
    for key in keys[:max_items]:
        summary = (registry.get(key) or "").strip()
        if summary:
            lines.append(f"- **{key}**: {summary}")
        else:
            lines.append(f"- **{key}**")
    return "\n".join(lines)


def suggest_uncovered_deep_dive_topics(
    node: NodeDataInput,
    memory: SessionMemory,
) -> list[str]:
    """Темы для углубления, слабо пересекающиеся с реестром."""
    from knowledge_engine.src.node_deep_dive.lecture_coverage import (
        suggest_deep_dive_topics,
    )

    candidates = suggest_deep_dive_topics(node)
    reg = memory.covered_subtopics or {}
    if not reg:
        return candidates
    out: list[str] = []
    for topic in candidates:
        if coverage_request_overlap(topic, "", reg) < 0.55:
            out.append(topic)
        if len(out) >= 4:
            break
    return out or candidates[:4]


def overlap_threshold() -> float:
    return _OVERLAP_THRESHOLD
