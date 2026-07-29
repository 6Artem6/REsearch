"""Skill Tree — локальное сохранение маршрутов и состояния."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from knowledge_engine.services.skill_tree_store import (
    get_active_curriculum_id,
    get_workspace_state,
    list_curriculum_summaries,
    set_active_curriculum,
)
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/skill-tree", tags=["skill-tree-persistence"])


class ActiveCurriculumBody(BaseModel):
    curriculum_id: str = Field(min_length=3, max_length=80)


@router.get("/curricula")
def get_curricula_list() -> dict[str, Any]:
    """Список сохранённых учебных маршрутов."""
    items = list_curriculum_summaries()
    active = get_active_curriculum_id()
    return {"active_curriculum_id": active, "curricula": items}


@router.get("/curricula/{curriculum_id}/workspace")
def get_curriculum_workspace(curriculum_id: str) -> dict[str, Any]:
    """
    Граф + статусы нод + сессии (summary, diagram, references, история чата).
    """
    trace(f"API ▶ GET /skill-tree/workspace | {curriculum_id}")
    state = get_workspace_state(curriculum_id.strip())
    if not state:
        raise HTTPException(status_code=404, detail="Маршрут не найден")
    return state


@router.post("/curricula/active")
def post_set_active(body: ActiveCurriculumBody) -> dict[str, Any]:
    ok = set_active_curriculum(body.curriculum_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Маршрут не найден")
    return {"active_curriculum_id": body.curriculum_id.strip()}
