"""FSM stage-progress событий графа тьютора (см. prompt.txt: "Интеграция
FSM-статусов LangGraph c SSE-стримингом").

Стримятся через ТОТ ЖЕ SSE-канал, что уже несёт token/complete/error
(POST /node/chat-stream, /node/init-stream — job_stream.py, Redis List
relay), не отдельный HTTP-эндпоинт — по решению пользователя: одно
SSE-соединение на ход, не два параллельных.

TutorStage — 6 стадий per ТЗ, но реальные узлы графа
(knowledge_engine/src/node_deep_dive/graph/__init__.py: ingest, init,
lazy_intro, equivalence, step_analysis, sub_concept_eval, coverage_router,
tutor_generate, dense_lecture, commit_turn, persist, finalize_response) не
совпадают 1:1 с именами из ТЗ (там были плейсхолдеры вида
node_retrieve_context, INTENT_ANALYSIS как отдельный узел и т.п. — их нет).
Маппинг узел → стадия задан в graph/stage_events.py::NODE_STAGE_MAP,
обоснование см. там.

Никакого enforced timeout/abort на уровне стадии — по решению пользователя:
elapsed_sec чисто информационный (реальные tutor_generate/dense_lecture
LLM-вызовы стабильно занимают 40-90s, per-node wait_for с любым разумным
дефолтом ложно бы срабатывал на нормальных ходах).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FSMStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class TutorStage(str, Enum):
    INIT = "INIT"
    VECTOR_SEARCH = "VECTOR_SEARCH"
    INTENT_ANALYSIS = "INTENT_ANALYSIS"
    LLM_GENERATE = "LLM_GENERATE"
    SUMMARIZE = "SUMMARIZE"
    FINALIZE = "FINALIZE"


class StageProgressEvent(BaseModel):
    session_id: str
    stage: TutorStage
    status: FSMStatus
    message: str
    elapsed_sec: float
    payload: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sse_dict(self) -> dict[str, Any]:
        """ "type": "stage" — дискриминатор в одном потоке с существующими
        {"type": "token"|"complete"|"error"} событиями job_stream.py."""
        data = self.model_dump(mode="json")
        data["type"] = "stage"
        return data
