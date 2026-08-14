"""Сквозной реестр подтверждённых субконцептов (Global Concept Registry)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from knowledge_engine.src.node_deep_dive.memory_schemas import SubConceptStatus

SubConceptPassStatus = SubConceptStatus


class ChatDialogTurn(BaseModel):
    """Контракт одной реплики истории чата (синхронизация с dialog_ids)."""

    role: Literal["user", "tutor"] = "tutor"
    content: str = Field(default="", max_length=12_000)
    msg_id: str = Field(default="", max_length=16)
    updated_at: str = Field(
        default="",
        max_length=40,
        description="ISO-8601 UTC; опционально для legacy-строк",
    )

    def to_storage_dict(self) -> dict[str, str]:
        row: dict[str, str] = {
            "role": self.role,
            "content": self.content.strip(),
        }
        if self.msg_id:
            row["msg_id"] = self.msg_id
        if self.updated_at:
            row["updated_at"] = self.updated_at
        return row

    @classmethod
    def from_storage_dict(cls, raw: dict) -> ChatDialogTurn:
        role = str(raw.get("role") or "tutor").strip()
        if role not in ("user", "tutor"):
            role = "tutor"
        return cls(
            role=role,
            content=str(raw.get("content") or "")[:12_000],
            msg_id=str(raw.get("msg_id") or raw.get("id") or "").strip()[:16],
            updated_at=str(raw.get("updated_at") or "").strip()[:40],
        )


class GlobalVerifiedSubConcept(BaseModel):
    """Зафиксированная подтема в глобальном реестре пользователя."""

    curriculum_id: str = Field(min_length=1, max_length=80)
    node_id: str = Field(min_length=2, max_length=80)
    node_title: str = Field(default="", max_length=300)
    sub_concept_id: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    status: SubConceptStatus = "verified"
    is_verified: bool = True
    evidence: str = Field(default="", max_length=400)
    mastery_score: int = Field(default=0, ge=0, le=100)
    updated_at: str = Field(default="", max_length=40)

    def registry_key(self) -> str:
        return f"{self.node_id.strip()}::{self.sub_concept_id.strip()}"


class GlobalUserKnowledgeState(BaseModel):
    """Персистентный снимок сквозного реестра по curriculum + user."""

    user_id: str = Field(default="default", max_length=64)
    curriculum_id: str = Field(min_length=1, max_length=80)
    entries: dict[str, GlobalVerifiedSubConcept] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)
    updated_at: str = Field(default="", max_length=40)
    node_session_sync_at: dict[str, str] = Field(
        default_factory=dict,
        description="node_id → ISO updated_at последней синхронизации сессии",
    )


class GlobalSubConceptDelta(BaseModel):
    """Инкремент реестра + отобранные для промпта записи (Top-K / token budget)."""

    new_entries: list[GlobalVerifiedSubConcept] = Field(
        default_factory=list,
        max_length=256,
        description="Все verified записи других нод (сырой пул)",
    )
    prompt_entries: list[GlobalVerifiedSubConcept] = Field(
        default_factory=list,
        max_length=12,
        description="Top-K после ранжирования и усечения по токен-бюджету",
    )
    omitted_concept_count: int = Field(
        default=0,
        ge=0,
        description="Число verified-концептов, не вошедших в блок промпта",
    )
    revision: int = 0
    total_entries: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
