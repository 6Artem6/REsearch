"""Stateless Gemini API (v0.3). См. gemini_stateless.run_stateless_gemini."""

from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    is_gemini_available,
    run_stateless_gemini,
)

__all__ = ["run_stateless_gemini", "global_anchor_from_state", "is_gemini_available"]
