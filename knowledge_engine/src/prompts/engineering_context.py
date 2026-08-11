"""Глобальные инженерные критерии и опциональные контекстные оверрайды."""

from __future__ import annotations

import sys

from knowledge_engine.config import (
    KE_PROMPT_CONTEXT_OVERRIDE_FULLSTACK,
    KE_PROMPT_CONTEXT_OVERRIDE_LOCAL_MAC,
)

GLOBAL_ENGINEERING_CRITERIA = (
    "=== GLOBAL ENGINEERING CRITERIA (mandatory solution filters) ===\n"
    "1. **Trade-offs & No Magic:** explicit pros/cons; no marketing «works out of the box» claims.\n"
    "2. **Failure Modes & Edge Cases:** degradation, OOM, tail latency, races, split-brain, stale reads.\n"
    "3. **Observability & Determinism:** transparent logging, state checkpointing, reproducible pipelines.\n"
)
"""
RU (пояснение): глобальные фильтры решений — trade-offs, failure modes, observability.
"""

CONTEXT_OVERRIDE_OPERATIONAL_MAC = (
    "=== CONTEXT OVERRIDE (example: local Mac runtime) ===\n"
    "Operational environment: Apple Silicon CPU/unified memory load, local debugging, quantized models, "
    "RAM-conscious embedded stores vs heavy vector clusters when appropriate.\n"
)
"""
RU (пояснение): override для задач на локальном Mac (unified memory, quantized).
"""

CONTEXT_OVERRIDE_FULLSTACK_PRAGMATISM = (
    "=== CONTEXT OVERRIDE (example: fullstack) ===\n"
    "Fullstack pragmatism: end-to-end design — backend contracts with clean client integration "
    "(API, types, responsibility boundaries).\n"
)
"""
RU (пояснение): override fullstack — API, типы, границы ответственности.
"""


def format_optional_context_overrides(
    *,
    local_mac_runtime: bool | None = None,
    fullstack_task: bool | None = None,
) -> str:
    """Необязательные оверрайды — только когда флаги явно включены."""
    mac = (
        KE_PROMPT_CONTEXT_OVERRIDE_LOCAL_MAC
        if local_mac_runtime is None
        else local_mac_runtime
    )
    fs = (
        KE_PROMPT_CONTEXT_OVERRIDE_FULLSTACK
        if fullstack_task is None
        else fullstack_task
    )
    parts: list[str] = []
    if mac:
        parts.append(CONTEXT_OVERRIDE_OPERATIONAL_MAC)
    if fs:
        parts.append(CONTEXT_OVERRIDE_FULLSTACK_PRAGMATISM)
    return "\n".join(parts).strip()


def default_local_mac_runtime() -> bool:
    return sys.platform == "darwin"
