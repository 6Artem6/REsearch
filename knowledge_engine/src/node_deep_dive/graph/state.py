"""LangGraph state for Node Deep-Dive tutor orchestration."""

from __future__ import annotations

from typing import Any, TypedDict

from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory, UserIntent
from knowledge_engine.src.node_deep_dive.prompt_types import InteractionPromptMode
from knowledge_engine.src.node_deep_dive.schemas import (
    DeepDiveLLMOutput,
    NodeContentBlock,
    NodeDeepDiveRequest,
    NodeDeepDiveResponse,
)

# Route targets after coverage_router (conditional edges).
TutorRoute = str  # "tutor" | "dense" | "coverage_notice" | "transition" | "skip_llm"


class TutorGraphState(TypedDict, total=False):
    """
    Graph state for one invoke of the tutor pipeline.

    Persistent session data lives only in ``memory`` (SessionMemory).
    Do not duplicate sub_concepts / pending / phase in parallel top-level fields.
    """

    memory: SessionMemory
    request: NodeDeepDiveRequest
    anchor: str
    content: NodeContentBlock
    intent: UserIntent
    pipeline_gap: str | None
    interaction_mode: InteractionPromptMode
    route: TutorRoute
    focus_sub_concept_id: str
    llm_out: DeepDiveLLMOutput | None
    tutor_message: str
    response: NodeDeepDiveResponse | None
    rag_facts_count: int
    rag_fact_labels: list[str]
    errors: list[str]
    response_verified_sub_concept_ids: list[str]
    session_history: list[dict[str, str]]
    _stream_callback: Any
    is_layer_just_completed: bool
