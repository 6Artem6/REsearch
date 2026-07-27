"""LangGraph state v0.3 — якорь задачи и ссылки на граф, без сырого HTML в state."""

from __future__ import annotations

from typing import Any, List

from typing_extensions import NotRequired, TypedDict


class ResearchGraphState(TypedDict):
    """Stateless Deep Research graph (v0.3)."""

    # Якорь (константы прогона)
    original_query: str
    constraints: str
    # Совместимость с CLI / EngineState
    user_problem: str
    context_constraints: str

    l0_summary: str
    l0_node_id: NotRequired[str]
    l1_node_ids: NotRequired[List[str]]

    pending_urls: NotRequired[List[str]]
    explored_urls: NotRequired[List[str]]
    depth: NotRequired[int]

    search_queries: NotRequired[List[str]]
    knowledge_node_ids: NotRequired[List[str]]

    gemini_raw_response: NotRequired[str]
    report: NotRequired[Any]
    selected_option_id: NotRequired[int]
    unraveled_details: NotRequired[str]
    abstractions: NotRequired[List[Any]]
