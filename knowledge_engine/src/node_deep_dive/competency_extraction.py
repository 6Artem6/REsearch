"""Фоновая экстракция компетенций через Gemma Cloud (fire-and-forget)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.llm import complete_structured_async
from knowledge_engine.src.locks import uma_resource_lock
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
    CompetencyDelta,
    merge_competency_delta,
)
from knowledge_engine.ui.run_log import trace

_EXTRACT_SYSTEM = (
    "You analyze a tutoring dialogue step and extract a competency delta for the user.\n"
    "Return strict JSON only:\n"
    "{\n"
    '  "new_proven_skills": ["short skill phrase in Russian"],\n'
    '  "new_blind_spots": ["gap or misconception"],\n'
    '  "resolved_blind_spots": ["blind spot text the user closed"],\n'
    '  "new_mastered_entities": ["key terms: RAG, LanceDB"]\n'
    "}\n"
    "Rules: max 2 proven, max 2 blind_spots, max 4 entities; only if explicit in user turn; "
    "no percentages or numeric scores; output field strings in Russian.\n"
)
"""
RU (пояснение): Gemma Cloud sidecar — JSON дельта компетенций из шага диалога.
"""


class _ExtractJson(BaseModel):
    new_proven_skills: list[str] = Field(default_factory=list)
    new_blind_spots: list[str] = Field(default_factory=list)
    resolved_blind_spots: list[str] = Field(default_factory=list)
    new_mastered_entities: list[str] = Field(default_factory=list)

    @field_validator(
        "new_proven_skills",
        "new_blind_spots",
        "resolved_blind_spots",
        "new_mastered_entities",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []


def _delta_from_extract(model: _ExtractJson) -> CompetencyDelta:
    return CompetencyDelta(
        new_proven_skills=model.new_proven_skills[:2],
        new_blind_spots=model.new_blind_spots[:2],
        resolved_blind_spots=model.resolved_blind_spots[:3],
        new_mastered_entities=model.new_mastered_entities[:4],
    )


async def _gemma_extract_delta(
    user_message: str,
    tutor_preview: str,
    node: NodeDataInput,
) -> CompetencyDelta | None:
    concepts = ", ".join(str(c) for c in (node.core_concepts or [])[:6])
    user = (
        f"Node: {node.title}\n"
        f"Concepts: {concepts}\n"
        f"User turn:\n{(user_message or '')[:1200]}\n\n"
        f"Tutor reply (start):\n{(tutor_preview or '')[:800]}"
    )
    try:
        model = await complete_structured_async(
            _ExtractJson,
            _EXTRACT_SYSTEM,
            user,
            label="competency_extract",
        )
    except Exception:
        return None
    if model is None:
        return None
    return _delta_from_extract(model)


async def _run_extract_with_uma_backoff(
    curriculum_id: str,
    user_message: str,
    tutor_preview: str,
    node: NodeDataInput,
) -> None:
    """Не ждать UMA lock — отложить, не блокировать CE/LanceDB."""
    for attempt in range(15):
        if uma_resource_lock.acquire(blocking=False):
            try:
                delta = await _gemma_extract_delta(user_message, tutor_preview, node)
                if delta and delta.has_updates():
                    merge_competency_delta(curriculum_id, delta)
                    trace(
                        f"COMPETENCY_EXTRACT ✓ | {curriculum_id}/{node.node_id} | "
                        f"proven+{len(delta.new_proven_skills)} "
                        f"blind+{len(delta.new_blind_spots)} "
                        f"resolved={len(delta.resolved_blind_spots)}"
                    )
                return
            except Exception as exc:
                trace(f"COMPETENCY_EXTRACT ✗ | {exc}")
                return
            finally:
                uma_resource_lock.release()
        await asyncio.sleep(1.5 + attempt * 0.2)


async def _background_collect_and_extract(
    curriculum_id: str,
    user_message: str,
    tutor_chunks: list[str],
    node: NodeDataInput,
) -> None:
    """Подождать начало ответа тьютора, затем экстракция."""
    await asyncio.sleep(0.15)
    for _ in range(40):
        preview = "".join(tutor_chunks)[:800]
        if len(preview) >= 60 or (preview and _ > 3):
            break
        await asyncio.sleep(0.25)
    preview = "".join(tutor_chunks)[:800]
    await _run_extract_with_uma_backoff(
        curriculum_id,
        user_message,
        preview,
        node,
    )


def schedule_competency_extraction(
    curriculum_id: str,
    user_message: str,
    node: NodeDataInput,
    *,
    tutor_preview: str = "",
    tutor_chunks: list[str] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """
    Fire-and-forget: не блокирует стрим и HTTP.
    """
    cid = (curriculum_id or "").strip()
    msg = (user_message or "").strip()
    if not cid or len(msg) < 4:
        return
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

    if tutor_chunks is not None:
        loop.create_task(
            _background_collect_and_extract(cid, msg, tutor_chunks, node),
            name=f"competency_extract:{node.node_id}",
        )
    else:
        loop.create_task(
            _run_extract_with_uma_backoff(cid, msg, tutor_preview[:800], node),
            name=f"competency_extract:{node.node_id}",
        )


def wrap_stream_callback_for_competency_extraction(
    stream_callback: Callable[[str], None] | None,
    curriculum_id: str,
    user_message: str,
    node: NodeDataInput,
) -> tuple[Callable[[str], None] | None, list[str]]:
    """На первом токене — фоновая задача; chunks накапливаются для preview."""
    chunks: list[str] = []
    scheduled = False

    if stream_callback is None:
        return None, chunks

    try:
        owner_loop = asyncio.get_running_loop()
    except RuntimeError:
        owner_loop = None

    def _schedule_on_loop() -> None:
        schedule_competency_extraction(
            curriculum_id,
            user_message,
            node,
            tutor_chunks=chunks,
            loop=owner_loop,
        )

    def wrapped(text: str) -> None:
        nonlocal scheduled
        if text:
            chunks.append(text)
        stream_callback(text)
        if scheduled or not text or owner_loop is None:
            return
        scheduled = True
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is owner_loop:
            _schedule_on_loop()
        else:
            owner_loop.call_soon_threadsafe(_schedule_on_loop)

    return wrapped, chunks
