"""Fact Manifest: структурированная память диалога вместо прозаического rolling_compress."""

from __future__ import annotations

import json

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.adapters.llm_providers.gemini_stateless import (
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.config.settings import GEMINI_LITE_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.grounding.memory_schemas import (
    DialogueFactManifest,
    SessionMemory,
)
from knowledge_engine.src.domains.grounding.tutor import DialogueFactManifestContract
from knowledge_engine.src.domains.grounding.tutor_reply_sanitize import (
    sanitize_evicted_tutor_content_for_manifest,
)

_FACT_EXTRACT_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Extract engineering facts from one or more evicted dialogue turns (skill "
    "tree node), given as numbered ### evicted_message_N blocks — they were "
    "evicted from the SAME active window rotation and must be merged into "
    "ONE resulting patch (not judged independently).\n"
    "Structured JSON only — not prose summary.\n"
    "- agreed_concepts: accepted stack/algorithms/metrics.\n"
    "- rejected_options: rejected alternatives.\n"
    "- open_bottlenecks: open bottlenecks, latency, RAM, index.\n"
    "- stack_mentions: concrete technologies (Qdrant, HNSW, bge-reranker…).\n"
    "- current_subtopic: active sub-topic one line.\n"
    "Merge with previous_manifest; do not drop prior agreements without explicit rejection.\n"
    "CRITICAL: Do NOT extract potential concepts, bottlenecks, or topics from "
    "unanswered questions asked by the tutor in evicted tutor messages. Only "
    "extract confirmed facts, user answers, and active user misconceptions.\n"
)
"""
RU (пояснение): из evicted turn → DialogueFactManifest (без вопросов тьютора как фактов).
"""


def _merge_lists(existing: list[str], new: list[str], cap: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(existing or []) + list(new or []):
        s = (str(item) or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s[:400])
    return out[:cap]


def merge_manifest(
    prev: DialogueFactManifest,
    patch: DialogueFactManifest,
) -> DialogueFactManifest:
    sub = (patch.current_subtopic or "").strip() or prev.current_subtopic
    return DialogueFactManifest(
        agreed_concepts=_merge_lists(prev.agreed_concepts, patch.agreed_concepts),
        rejected_options=_merge_lists(prev.rejected_options, patch.rejected_options),
        open_bottlenecks=_merge_lists(prev.open_bottlenecks, patch.open_bottlenecks),
        stack_mentions=_merge_lists(prev.stack_mentions, patch.stack_mentions),
        current_subtopic=sub[:400],
    )


def _evicted_tutor_needs_question_strip(
    memory: SessionMemory,
    evicted: dict[str, str],
) -> bool:
    from knowledge_engine.src.domains.grounding.concept_map import (
        stored_pending_evaluation_id,
    )

    role = (evicted.get("role") or "").strip().lower()
    if role not in ("tutor", "model", "assistant"):
        return False
    content = (evicted.get("content") or "").strip()
    if not content or "?" not in content:
        return False
    pending = stored_pending_evaluation_id(memory)
    if not pending:
        return False
    last = (memory.last_tutor_sub_concept_id or "").strip()
    return last == pending


def prepare_evicted_for_manifest_extraction(
    memory: SessionMemory,
    evicted: dict[str, str],
) -> dict[str, str] | None:
    """Guard-проверки для ОДНОГО вытесненного сообщения — читают ЖИВОЕ
    состояние memory на момент eviction (их нельзя откладывать в фон, т.к. к
    моменту выполнения job'а pending_eval_kind/last_tutor_sub_concept_id могут
    уже измениться). Возвращает ``{"role", "content"}`` или None, если
    сообщение нужно молча пропустить.

    Batch-уровневые поля (anchor/prev_manifest/expected_manifest_version)
    собираются один раз на весь batch — см.
    ``prepare_evicted_batch_for_manifest_extraction``.
    """
    role = (evicted.get("role") or "").strip()
    content = (evicted.get("content") or "").strip()
    if not content:
        return None
    role_l = role.lower()
    # Deep Analysis / open Star Task tutor turns must not seed fact_manifest with
    # synthesized constraints / homework hypotheses — only later user answers may.
    from knowledge_engine.src.domains.grounding.star_task_fsm import (
        is_overlay_eval_kind,
        star_task_blocks_transition,
    )

    if role_l in ("tutor", "model", "assistant") and (
        is_overlay_eval_kind(memory.pending_eval_kind)
        or star_task_blocks_transition(memory)
    ):
        trace("NODE_DIVE fact_manifest skip | deep_analysis/star_task tutor eviction")
        return None
    if _evicted_tutor_needs_question_strip(memory, evicted):
        content = sanitize_evicted_tutor_content_for_manifest(content)
        trace("NODE_DIVE fact_manifest | sanitized unanswered tutor tail in evicted")
    return {"role": role, "content": content[:2500]}


def prepare_evicted_batch_for_manifest_extraction(
    memory: SessionMemory,
    evicted_list: list[dict[str, str]],
    anchor: str,
) -> dict | None:
    """Синхронная (без LLM) часть: guard-проверки по КАЖДОМУ вытесненному
    сообщению + сборка ОДНОГО payload для background-экстракции
    (context_compressor_worker.run_dialog_summarize_job).

    Раньше ``rotate_window_after_message`` ставил отдельную job (и отдельный
    Gemini-вызов) на каждое вытесненное сообщение — обычно 2 за ход в steady
    state (active_window заполнен → user- и tutor-эвикция одного хода), что
    удваивало стоимость и дублировало ``previous_manifest`` в промпте.
    Теперь все сообщения одной ротации собираются в ОДИН payload → один
    LLM-вызов, один merge, один CAS-коммит.

    Возвращает None, если после guard-фильтрации не осталось ни одного
    сообщения (тот же контракт, что раньше был у early-return внутри
    update_manifest_from_evicted).
    """
    messages: list[dict[str, str]] = []
    for evicted in evicted_list:
        prepared = prepare_evicted_for_manifest_extraction(memory, evicted)
        if prepared is not None:
            messages.append(prepared)
    if not messages:
        return None
    return {
        "anchor": anchor,
        "messages": messages,
        "prev_manifest": memory.fact_manifest.model_dump(),
        "learning_phase": memory.learning_phase,
        "learning_mode": memory.learning_mode,
        "expected_manifest_version": memory.manifest_version,
    }


def run_fact_manifest_extraction(payload: dict) -> DialogueFactManifest:
    """Чистая LLM-экстракция без доступа к live SessionMemory — единственная
    функция из этого модуля, которую можно безопасно звать из фонового
    воркера (см. services/context_compressor_worker.py). Не вызывать с hot
    path тьютора — именно этот вызов раньше блокировал ответ пользователю.

    ``payload["messages"]`` — список ``{"role", "content"}`` одной ротации
    (обычно 1-2 сообщения); все они идут в ОДИН промпт и ОДИН merge-проход.
    """
    prev = DialogueFactManifest.model_validate(payload.get("prev_manifest") or {})
    messages = payload.get("messages") or []
    anchor = str(payload.get("anchor") or "")
    evicted_block = "\n".join(
        f"### evicted_message_{i + 1}\n{m.get('role', '')}: {m.get('content', '')}"
        for i, m in enumerate(messages)
    )
    llm_payload = (
        f"### previous_manifest\n{prev.model_dump_json()}\n\n"
        f"{evicted_block}\n\n"
        f"### learning_phase\n{payload.get('learning_phase', '')}\n"
        f"### learning_mode\n{payload.get('learning_mode', '')}\n"
    )
    combined_content = " ".join(str(m.get("content", "")) for m in messages)
    try:
        patch = run_gemini_structured_with_chain(
            GEMINI_LITE_MODEL,
            _FACT_EXTRACT_SYSTEM,
            llm_payload,
            anchor,
            DialogueFactManifestContract,
            "node_deep_dive / fact_manifest",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        )
        return merge_manifest(
            prev,
            DialogueFactManifest.model_validate(patch.model_dump()),
        )
    except Exception as exc:
        trace(f"NODE_DIVE fact_manifest fallback | {exc}")
        return _heuristic_merge_manifest(prev, combined_content)


def update_manifest_from_evicted(
    memory: SessionMemory,
    evicted: dict[str, str],
    anchor: str,
) -> None:
    """Синхронный convenience-wrapper (guard + extraction + merge за один
    вызов) — оставлен для тестов/вызовов вне hot path тьютора. Hot path
    (rotate_window_after_message) больше НЕ вызывает эту функцию: он
    публикует ОДНУ batched job через context_compressor_worker (см.
    prepare_evicted_batch_for_manifest_extraction) и возвращается сразу.
    """
    payload = prepare_evicted_batch_for_manifest_extraction(memory, [evicted], anchor)
    if payload is None:
        return
    memory.fact_manifest = run_fact_manifest_extraction(payload)
    memory.manifest_version += 1


def _heuristic_merge_manifest(
    prev: DialogueFactManifest,
    evicted_content: str,
) -> DialogueFactManifest:
    text = (evicted_content or "").lower()
    stacks = list(prev.stack_mentions)
    for token in (
        "qdrant",
        "pgvector",
        "hnsw",
        "lancedb",
        "llamaindex",
        "langchain",
        "bge-reranker",
        "mmr",
        "cross-encoder",
        "llmlingua",
    ):
        if token in text and token not in stacks:
            stacks.append(token)
    return DialogueFactManifest(
        agreed_concepts=list(prev.agreed_concepts),
        rejected_options=list(prev.rejected_options),
        open_bottlenecks=list(prev.open_bottlenecks),
        stack_mentions=stacks[:24],
        current_subtopic=prev.current_subtopic,
    )


def format_fact_manifest_block(memory: SessionMemory) -> str:
    data = memory.fact_manifest.model_dump()
    if not any(data.values()) and not data.get("current_subtopic"):
        return "### fact_manifest\n{}"
    return "### fact_manifest\n" + json.dumps(data, ensure_ascii=False, indent=0)
