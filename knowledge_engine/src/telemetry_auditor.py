"""Host-turn telemetry: structured JSON, non-blocking on the P99 path."""

from __future__ import annotations

import logging
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_LOG = logging.getLogger("ke.host_telemetry")

IntentSource = Literal["exact", "vector", "fallback"]


class HostTurnTelemetry(BaseModel):
    """One host-layer turn (no LLM payload)."""

    session_id: str = Field(default="", max_length=160)
    node_id: str = Field(default="", max_length=80)
    intent_detected: str = Field(default="", max_length=64)
    intent_source: IntentSource = "fallback"
    active_overlay: str = Field(default="", max_length=64)
    weakness_tags: list[str] = Field(default_factory=list, max_length=12)
    latency_host_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("weakness_tags", mode="before")
    @classmethod
    def _tags(cls, v: object) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in v:  # type: ignore[union-attr]
            tag = str(item or "").strip()[:64]
            if not tag or tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
        return out[:12]


_RECENT: deque[HostTurnTelemetry] = deque(maxlen=128)


def emit_host_telemetry(row: HostTurnTelemetry) -> None:
    """Append to an in-process ring and log JSON if a handler is attached."""
    try:
        _RECENT.append(row)
        if _LOG.handlers:
            _LOG.info(row.model_dump_json())
    except Exception:
        pass


def recent_host_telemetry() -> list[HostTurnTelemetry]:
    """Tests / diagnostics: copy of the ring buffer."""
    return list(_RECENT)


def clear_host_telemetry_for_tests() -> None:
    _RECENT.clear()
