"""Stage 0 personal context orchestration."""

from __future__ import annotations

from knowledge_engine.src.guardrails.personal_context import (
    PersonalContext,
    run_personal_context_stage,
)
from knowledge_engine.ui.run_log import trace


async def run_stage_0(user_query: str, user_profile_md: str = "") -> PersonalContext:
    trace("STAGE 0 ▶ personal context (Ollama 7B)")
    ctx = await run_personal_context_stage(user_query, user_profile_md)
    trace("STAGE 0 ✓ personal context")
    return ctx
