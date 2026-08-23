"""Pydantic contracts for entity-guided claim consensus (Phase 3B)."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, field_validator

ConsensusStatus = Literal["consensus", "disputed", "unique"]
_STATUS_ALLOWED: tuple[str, ...] = ("consensus", "disputed", "unique")
_LOG = logging.getLogger(__name__)


def _unique_anchors(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        tag = str(raw or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _coerce_status(value: object) -> str:
    if value is None:
        _LOG.warning("ConsensusNode.status missing/None; defaulting to 'consensus'")
        return "consensus"
    if not isinstance(value, str):
        _LOG.warning(
            "ConsensusNode.status has non-string type %s; defaulting to 'consensus'",
            type(value).__name__,
        )
        return "consensus"
    raw = value.strip().lower()
    if not raw:
        _LOG.warning("ConsensusNode.status empty; defaulting to 'consensus'")
        return "consensus"
    if raw not in _STATUS_ALLOWED:
        _LOG.warning(
            "ConsensusNode.status=%r unknown; defaulting to 'consensus'",
            value,
        )
        return "consensus"
    return raw


def _nonempty(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
        return text or fallback
    text = str(value).strip()
    return text or fallback


class RawFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fact_id: str = Field(
        default="f-unknown",
        min_length=1,
        description="Stable id within the current REDUCE job.",
    )
    # RU: идентификатор факта в текущем проходе REDUCE.
    subject: str = Field(
        default="unknown",
        min_length=1,
        description="Entity / subject of the SPO triple.",
    )
    # RU: подлежащее (сущность) тройки SPO.
    entity_type: str = Field(
        default="general",
        description="Type bucket, e.g. python_library, architecture_pattern, general.",
    )
    # RU: тип сущности; general — общая fallback-группа.
    predicate: str = Field(
        default="states",
        min_length=1,
        description="Predicate / relation of the SPO triple.",
    )
    # RU: отношение (сказуемое).
    obj: str = Field(
        default="unspecified",
        min_length=1,
        description="Object / complement of the SPO triple.",
    )
    # RU: дополнение тройки SPO.
    anchor: str = Field(default="", description="Citation tag in the current context, e.g. A1.")
    # RU: анкор текущего окна, например A1.
    all_anchors: list[str] = Field(
        default_factory=list,
        description="Union of supporting citation tags after local merges.",
    )
    # RU: полный набор подтверждающих анкоров после локального слияния.

    @field_validator("entity_type", mode="before")
    @classmethod
    def _entity_type(cls, v: object) -> str:
        raw = str(v or "general").strip().lower() or "general"
        return raw

    @field_validator("fact_id", mode="before")
    @classmethod
    def _fact_id(cls, v: object) -> str:
        return _nonempty(v, "f-unknown")

    @field_validator("subject", mode="before")
    @classmethod
    def _subject(cls, v: object) -> str:
        return _nonempty(v, "unknown")

    @field_validator("predicate", mode="before")
    @classmethod
    def _predicate(cls, v: object) -> str:
        return _nonempty(v, "states")

    @field_validator("obj", mode="before")
    @classmethod
    def _obj(cls, v: object) -> str:
        return _nonempty(v, "unspecified")

    @field_validator("all_anchors", mode="before")
    @classmethod
    def _anchors(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return _unique_anchors([v])
        if isinstance(v, (list, tuple, set)):
            return _unique_anchors([str(x) for x in v])
        return []

    @computed_field
    @property
    def canonical_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}".strip()

    def merged_anchors(self) -> list[str]:
        return _unique_anchors([self.anchor, *self.all_anchors])


class ConsensusNode(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    node_id: str = Field(
        default="n-unknown",
        min_length=1,
        description="Stable consensus node id.",
    )
    # RU: идентификатор узла консенсуса.
    entity: str = Field(
        default="unknown",
        min_length=1,
        description="Entity label for this cluster.",
    )
    # RU: имя сущности кластера.
    summary_text: str = Field(
        default="(empty consensus summary)",
        min_length=1,
        validation_alias=AliasChoices("summary_text", "summary"),
        description="Learner-facing cluster summary.",
    )
    # RU: текстовый отчёт по кластеру (русский).
    primary_anchors: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Anti-bloat: at most 3 citation tags for the rendered report.",
    )
    # RU: не больше трёх анкоров в тексте отчёта.
    all_anchors: list[str] = Field(
        default_factory=list,
        description="Full supporting citation set (metadata, not truncated).",
    )
    # RU: полный список подтверждающих источников.
    status: ConsensusStatus = Field(
        default="consensus",
        description=(
            "Arbiter verdict: consensus, disputed, or unique. "
            "Missing/unknown values coerce to consensus (do not fail the batch)."
        ),
    )
    # RU: вердикт арбитра по кластеру; мягкий дефолт, чтобы не ронять батч.
    disputed_details: str | None = Field(
        default=None,
        description="Optional conflict note when status is disputed.",
    )
    # RU: пояснение конфликта, если status=disputed.

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: object) -> str:
        return _coerce_status(v)

    @field_validator("node_id", mode="before")
    @classmethod
    def _node_id(cls, v: object) -> str:
        return _nonempty(v, "n-unknown")

    @field_validator("entity", mode="before")
    @classmethod
    def _entity(cls, v: object) -> str:
        return _nonempty(v, "unknown")

    @field_validator("summary_text", mode="before")
    @classmethod
    def _summary_text(cls, v: object) -> str:
        return _nonempty(v, "(empty consensus summary)")

    @field_validator("all_anchors", mode="before")
    @classmethod
    def _all_anchors(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return _unique_anchors([v])
        if isinstance(v, (list, tuple, set)):
            return _unique_anchors([str(x) for x in v])
        return []

    @field_validator("primary_anchors", mode="before")
    @classmethod
    def _primary_anchors(cls, v: object) -> list[str]:
        return cls._all_anchors(v)[:3]


class ConsensusBatchResponse(BaseModel):
    """LLM contract for one token-bounded consensus batch."""

    model_config = ConfigDict(extra="ignore")

    nodes: list[ConsensusNode] = Field(
        default_factory=list,
        description="Consensus nodes for the facts in this batch.",
    )
