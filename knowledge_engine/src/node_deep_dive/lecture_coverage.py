"""Проверка покрытия темы перед повторной генерацией плотной лекции."""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_engine.src.node_deep_dive.lecture_coverage_registry import (
    coverage_request_overlap,
    format_registry_excerpt_for_notice,
    matching_registry_keys,
    overlap_threshold,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import is_generic_lecture_stub
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.schemas import (
    DenseMaterialOutput,
    NodeDataInput,
)

LECTURE_COVERED_MIN_WORDS = 500
_MODE_LECTURE_RE = re.compile(r"\[mode:\s*lecture\]", re.I)


def count_words(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    parts = re.findall(r"[\wа-яёА-ЯЁ]+", t, flags=re.IGNORECASE)
    return len(parts) if parts else len(t.split())


def _history_tutor_bodies(history: list[dict[str, str]] | None) -> list[str]:
    out: list[str] = []
    for item in history or []:
        if (item.get("role") or "").strip() != "tutor":
            continue
        c = (item.get("content") or "").strip()
        if c:
            out.append(c)
    return out


def _history_user_lecture_requests(history: list[dict[str, str]] | None) -> int:
    n = 0
    for item in history or []:
        if (item.get("role") or "").strip() != "user":
            continue
        c = (item.get("content") or "").strip()
        if _MODE_LECTURE_RE.search(c) or is_generic_lecture_stub(c):
            n += 1
    return n


def prior_substantial_lecture_word_count(
    memory: SessionMemory,
    history: list[dict[str, str]] | None,
    content_summary: str = "",
) -> int:
    best = 0
    for body in _history_tutor_bodies(history):
        best = max(best, count_words(body))
    if memory.learning_phase in (
        "dense_material",
        "checkpoint",
        "pathway_decision",
        "socratic_focus",
    ):
        for m in memory.active_window:
            if (m.get("role") or "") == "tutor":
                best = max(best, count_words(m.get("content") or ""))
    summary_w = count_words(content_summary)
    if summary_w >= 120:
        best = max(best, summary_w)
    return best


def has_prior_full_lecture(
    memory: SessionMemory,
    history: list[dict[str, str]] | None,
    content_summary: str = "",
) -> bool:
    return (
        prior_substantial_lecture_word_count(memory, history, content_summary)
        >= LECTURE_COVERED_MIN_WORDS
    )


def suggest_deep_dive_topics(node: NodeDataInput) -> list[str]:
    concepts = [c.strip() for c in (node.core_concepts or []) if c.strip()]
    topics: list[str] = []
    for c in concepts[:4]:
        topics.append(f"Углубление: {c}")
    fallbacks = [
        "SIMD / AVX-512 при векторном поиске и batch query",
        "Математика Product Quantization и ошибки квантизации",
        "DiskANN, Vamana graph и дисковые ANN-индексы",
        "GPU-ускорение Faiss / batch HNSW и trade-offs latency",
        "Binary Quantization в Qdrant / pgvector",
        "Настройка Nprobe, Nlist и баланс Recall vs QPS",
    ]
    for fb in fallbacks:
        if len(topics) >= 4:
            break
        if fb not in topics:
            topics.append(fb)
    return topics[:4]


@dataclass(frozen=True)
class LectureCoverageAssessment:
    is_topic_already_covered: bool
    should_return_coverage_notice: bool
    prior_lecture_words: int
    lecture_request_count: int
    registry_overlap: float = 0.0
    matching_subtopic_keys: tuple[str, ...] = ()


def assess_lecture_coverage(
    memory: SessionMemory,
    history: list[dict[str, str]] | None,
    user_message: str,
    lecture_scope: str,
    focus_text: str,
    lecture_button_pressed: bool,
    content_summary: str = "",
) -> LectureCoverageAssessment:
    prior_words = prior_substantial_lecture_word_count(memory, history, content_summary)
    registry = memory.covered_subtopics or {}
    reg_overlap = (
        coverage_request_overlap(user_message, focus_text, registry)
        if registry
        else 0.0
    )
    matching = tuple(
        matching_registry_keys(user_message, focus_text, registry) if registry else ()
    )
    substantial_registry = len(registry) >= 2
    is_covered = prior_words >= LECTURE_COVERED_MIN_WORDS or substantial_registry
    req_count = _history_user_lecture_requests(history)
    if lecture_button_pressed or _MODE_LECTURE_RE.search(user_message or ""):
        req_count = max(req_count, 1)

    generic = is_generic_lecture_stub(user_message)
    scope = (lecture_scope or "").strip()
    threshold = overlap_threshold()

    should_notice = False
    if reg_overlap >= threshold:
        should_notice = True
    elif (
        scope == "full_node_lecture"
        and generic
        and (prior_words >= LECTURE_COVERED_MIN_WORDS or len(registry) >= 3)
    ):
        should_notice = True

    return LectureCoverageAssessment(
        is_topic_already_covered=is_covered,
        should_return_coverage_notice=should_notice,
        prior_lecture_words=prior_words,
        lecture_request_count=req_count,
        registry_overlap=reg_overlap,
        matching_subtopic_keys=matching,
    )


def build_coverage_short_message(
    node_title: str,
    topics: list[str],
    *,
    registry: dict[str, str] | None = None,
    matching_keys: list[str] | None = None,
) -> str:
    title = (node_title or "тема ноды").strip()
    lines = [
        f"Тема «{title}» и запрошенный фокус уже существенно покрыты в материале этой сессии. "
        "Чтобы не дублировать лекцию, выберите узкое углубление или уточните новый аспект.",
    ]
    excerpt = format_registry_excerpt_for_notice(
        registry or {},
        matching_keys=matching_keys,
    )
    if excerpt:
        lines.append("")
        lines.append("**Мы уже подробно разобрали:**")
        lines.append(excerpt)
    if topics:
        lines.append("")
        lines.append("**Точечные темы для углубления (ещё не в реестре):**")
        for t in topics:
            lines.append(f"- {t}")
    return "\n".join(lines).strip()


def build_coverage_dense_output(
    node: NodeDataInput,
    topics: list[str] | None = None,
) -> DenseMaterialOutput:
    picked = topics or suggest_deep_dive_topics(node)
    summary_lines = ["**Темы для углубления:**"]
    for t in picked:
        summary_lines.append(f"- {(t or '').strip()}")
    summary = "\n".join(summary_lines).strip() if picked else ""
    return DenseMaterialOutput(
        lecture_body="",
        summary=summary,
        referenced_diagram_id=None,
        references=[],
        code_snippets=[],
        bridge_to_next="",
        checkpoint_prompt="",
    )


def coverage_flag_payload_block(
    assessment: LectureCoverageAssessment,
    memory: SessionMemory | None = None,
) -> str:
    if not assessment.is_topic_already_covered and not (
        memory and memory.covered_subtopics
    ):
        return ""
    from knowledge_engine.src.node_deep_dive.lecture_coverage_registry import (
        format_coverage_registry_block,
    )

    reg_block = format_coverage_registry_block(memory) if memory else ""
    overlap_line = ""
    if assessment.registry_overlap > 0:
        overlap_line = (
            f"registry_overlap={assessment.registry_overlap:.2f} "
            f"(block repeat if ≥{overlap_threshold():.2f})\n"
        )
    keys_line = ""
    if assessment.matching_subtopic_keys:
        keys_line = (
            "matching_subtopics="
            + ", ".join(assessment.matching_subtopic_keys[:12])
            + "\n"
        )
    body = (
        "\n\n### COVERAGE_CONTEXT\n"
        f"IS_TOPIC_ALREADY_COVERED={assessment.is_topic_already_covered}\n"
        f"prior_lecture_words={assessment.prior_lecture_words}\n"
        f"{overlap_line}{keys_line}"
        "Режим: Deep Dive On-Demand — не базовый лонгрид; без повтора микро-тем из "
        "lecture_coverage_registry.\n"
    )
    if reg_block:
        body += f"\n{reg_block}\n"
    return body
