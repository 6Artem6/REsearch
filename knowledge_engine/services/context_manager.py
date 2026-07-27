"""FrugalGPT-style context: SLM сжатие + Sandwich payload для heavy Gemini."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import (
    ROLLING_SUMMARY_MAX_CHARS,
    ROUTER_MODEL,
    SUMMARIZER_MAX_PROFILE_CHARS,
    USER_PROFILE_PATH,
)
from knowledge_engine.llm import invoke_logged, structured_chat
from knowledge_engine.llm_locale import RUSSIAN_ROUTER_RULE
from knowledge_engine.schemas import DialogueRollingSummary, EngineState
from knowledge_engine.services.context_blocks import (
    assemble_gemini_payload,
    blocks_from_state_dicts,
    build_context_blocks,
    default_selections,
)
from knowledge_engine.ui.logger import set_status


def _load_user_profile() -> str:
    path = Path(USER_PROFILE_PATH)
    if not path.is_file():
        return "(user_profile.md не найден)"
    return path.read_text(encoding="utf-8")[:SUMMARIZER_MAX_PROFILE_CHARS]


def load_user_profile() -> str:
    return _load_user_profile()


def rolling_summarize_dialogue(
    state: EngineState,
    new_user_line: str | None = None,
) -> str:
    """Rolling Context Summarization (1.5B router)."""
    history_lines: list[str] = []
    if state.dialogue_rolling_summary:
        history_lines.append(
            f"Предыдущее сжатие:\n{state.dialogue_rolling_summary[:800]}"
        )
    for turn in state.external_ai_dialogue_history[-6:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "")[:500]
        history_lines.append(f"{role}: {content}")
    if new_user_line:
        history_lines.append(f"user (новое): {new_user_line}")

    if not history_lines:
        return state.dialogue_rolling_summary or "(нет уточняющего диалога)"

    set_status("[context_manager] 1.5B rolling summary…")
    structured = structured_chat(ROUTER_MODEL, DialogueRollingSummary, temperature=0.1)
    system = SystemMessage(
        content=(
            f"{RUSSIAN_ROUTER_RULE} "
            "Сжать историю уточнений в 5–8 буллетов для heavy LLM. "
            "Только факты, стек, ограничения, без воды."
        )
    )
    human = HumanMessage(content="\n".join(history_lines)[:4000])
    result = invoke_logged(
        structured, [system, human], "context_manager / rolling summary"
    )
    if result is None:
        return state.dialogue_rolling_summary
    return result.summary[:ROLLING_SUMMARY_MAX_CHARS]


def build_gemini_payload(state: EngineState) -> str:
    """Sandwich payload: детерминированная сборка из блоков и галочек."""
    blocks = (
        blocks_from_state_dicts(state.context_blocks)
        if state.context_blocks
        else build_context_blocks(state)
    )
    selections = state.context_block_selections or default_selections(blocks)
    return assemble_gemini_payload(blocks, selections)
