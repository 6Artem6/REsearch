"""Host-layer parallel prep: chip routing, prompt factory, ledger context."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from knowledge_engine.context_drift_manager import ContextDriftManager
from knowledge_engine.src.node_deep_dive.control_intent import (
    classify_control_chip_detailed,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    TutorFactoryMode,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.telemetry_auditor import (
    HostTurnTelemetry,
    emit_host_telemetry,
)


@dataclass(frozen=True)
class HostPrep:
    chip: str
    system_prompt: str
    factory_mode: TutorFactoryMode | str
    cleaned_user_message: str
    ledger_block: str
    intent_source: str = "fallback"


def _ledger_block(curriculum_id: str, exclude_node_id: str) -> str:
    cid = (curriculum_id or "").strip()
    if not cid:
        return ""
    return ContextDriftManager(cid, persist=False).build_cross_node_prompt_context(
        exclude_node_id=exclude_node_id
    )


def _emit_prep_telemetry(
    *,
    session_id: str,
    node_id: str,
    exclude_node_id: str,
    curriculum_id: str,
    persist_ledger: bool,
    chip: str,
    source: str,
    active_overlay: str,
    t0: float,
) -> None:
    import time

    try:
        cid = (curriculum_id or "").strip()
        tags: list[str] = []
        if cid:
            tags = ContextDriftManager(cid, persist=persist_ledger).open_weakness_tags()
        src = source if source in ("exact", "vector", "fallback") else "fallback"
        emit_host_telemetry(
            HostTurnTelemetry(
                session_id=session_id or cid,
                node_id=node_id or exclude_node_id,
                intent_detected=chip,
                intent_source=src,  # type: ignore[arg-type]
                active_overlay=active_overlay,
                weakness_tags=tags,
                latency_host_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
        )
    except Exception:
        pass


async def gather_host_prep(
    user_text: str,
    *,
    curriculum_id: str = "",
    exclude_node_id: str = "",
    default_system_prompt: str = "",
    persist_ledger: bool = False,
    session_id: str = "",
    node_id: str = "",
    active_overlay: str = "",
) -> HostPrep:
    """Run independent host reads concurrently (no LLM / network)."""
    import time

    t0 = time.perf_counter()

    def _chip() -> tuple[str, str]:
        return classify_control_chip_detailed(user_text)

    def _prompt() -> tuple[str, TutorFactoryMode | str, str]:
        return select_system_prompt_and_mode(
            user_text,
            default_system_prompt=default_system_prompt,
        )

    def _ledger() -> str:
        cid = (curriculum_id or "").strip()
        if not cid:
            return ""
        return ContextDriftManager(
            cid, persist=persist_ledger
        ).build_cross_node_prompt_context(exclude_node_id=exclude_node_id)

    chip_src, prompt, ledger = await asyncio.gather(
        asyncio.to_thread(_chip),
        asyncio.to_thread(_prompt),
        asyncio.to_thread(_ledger),
    )
    chip, source = chip_src
    system, mode, cleaned = prompt
    _emit_prep_telemetry(
        session_id=session_id,
        node_id=node_id,
        exclude_node_id=exclude_node_id,
        curriculum_id=curriculum_id,
        persist_ledger=persist_ledger,
        chip=chip,
        source=source,
        active_overlay=active_overlay,
        t0=t0,
    )
    return HostPrep(
        chip=chip,
        system_prompt=system,
        factory_mode=mode,
        cleaned_user_message=cleaned,
        ledger_block=ledger,
        intent_source=source,
    )


def run_host_prep_sync(
    user_text: str,
    *,
    curriculum_id: str = "",
    exclude_node_id: str = "",
    default_system_prompt: str = "",
    persist_ledger: bool = False,
    session_id: str = "",
    node_id: str = "",
    active_overlay: str = "",
) -> HostPrep:
    """Sync wrapper around ``gather_host_prep`` (safe when no event loop is running)."""
    kwargs = dict(
        curriculum_id=curriculum_id,
        exclude_node_id=exclude_node_id,
        default_system_prompt=default_system_prompt,
        persist_ledger=persist_ledger,
        session_id=session_id,
        node_id=node_id,
        active_overlay=active_overlay,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(gather_host_prep(user_text, **kwargs))
    import time

    t0 = time.perf_counter()
    chip, source = classify_control_chip_detailed(user_text)
    system, mode, cleaned = select_system_prompt_and_mode(
        user_text,
        default_system_prompt=default_system_prompt,
    )
    _emit_prep_telemetry(
        session_id=session_id,
        node_id=node_id,
        exclude_node_id=exclude_node_id,
        curriculum_id=curriculum_id,
        persist_ledger=persist_ledger,
        chip=chip,
        source=source,
        active_overlay=active_overlay,
        t0=t0,
    )
    return HostPrep(
        chip=chip,
        system_prompt=system,
        factory_mode=mode,
        cleaned_user_message=cleaned,
        ledger_block=_ledger_block(curriculum_id, exclude_node_id)
        if not persist_ledger
        else ContextDriftManager(
            curriculum_id, persist=True
        ).build_cross_node_prompt_context(exclude_node_id=exclude_node_id),
        intent_source=source,
    )
