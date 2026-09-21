"""Hybrid Context Buffer — Layer 3 (BLOCK COLLAPSING / GLOBAL PROGRESS).

Gated by ``DIALOG_BLOCK_COLLAPSE_ENABLED`` (default off). When the
"compressed timeline" portion of a chat session's ``api_turns`` (everything
past the raw tail — see ``RAW_HISTORY_DEPTH`` — kept as
``message_bullet_summary`` by Windowed History) grows to
``COMPRESSED_BLOCK_SIZE`` messages, the OLDEST such block is collapsed in a
single call to a Gemma model (GEMMA_PRIMARY_MODEL, falling back to
GEMMA_FALLBACK_MODEL — whichever is available/under quota, see
gemma_client.py::build_gemma_model_slots) into a short
``LEARNER_PROGRESS_SUMMARY`` digest, and removed from ``api_turns``
entirely — never dropped silently, unlike the prior hard
``CHAT_SESSION_API_TURNS_MAX`` cutoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from knowledge_engine.src.core.run_log import trace

if TYPE_CHECKING:
    from knowledge_engine.src.domains.grounding.chat_session_manager import (
        StoredChatSession,
    )
    from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory


class LearnerProgressBlockSummary(BaseModel):
    summary: str = Field(
        ...,
        min_length=8,
        max_length=800,
        description=(
            "2-3 sentences, natural Russian: what the learner covered and "
            "how well, in this chunk of dialogue. General progress digest, "
            "not a verbatim recap of every line."
        ),
    )


_BLOCK_SUMMARY_SYSTEM = (
    "You compress one aging chunk of a tutoring dialogue into a short "
    "progress digest for later reference — the raw messages below will be "
    "deleted from working history right after this; your summary is all "
    "that survives.\n"
    "Write 2-3 sentences in natural Russian: what topics/sub-concepts the "
    "learner engaged with in this chunk, and the general quality of their "
    "understanding (solid / partial / struggled) — no verbatim quotes, no "
    "line-by-line recap, no meta-commentary about being a summary.\n"
)


def _format_block_for_summary(block: list[dict]) -> str:
    lines: list[str] = []
    for turn in block:
        role = "Ученик" if (turn.get("role") or "") == "user" else "Тьютор"
        bullets = turn.get("bullet_summary")
        if bullets:
            text = "; ".join(str(b).strip() for b in bullets if str(b).strip())
        else:
            text = (turn.get("content") or "").strip()[:400]
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _fallback_block_digest(block: list[dict]) -> str:
    n_user = sum(1 for t in block if (t.get("role") or "") == "user")
    return f"Пройден блок диалога ({n_user} реплик учащегося); сводка недоступна."


async def _summarize_block_via_gemma(text_blob: str) -> LearnerProgressBlockSummary | None:
    """Try GEMMA_PRIMARY_MODEL, then GEMMA_FALLBACK_MODEL (whichever is

    available/under quota) — reuses build_gemma_model_slots() so this
    respects the SAME shared TPM/RPM budget as the ingestion pipeline
    instead of a second, uncoordinated Gemma client."""
    from knowledge_engine.src.config.settings import (
        HISTORY_BLOCK_SUMMARY_MAX_OUTPUT_TOKENS,
    )

    from knowledge_engine.src.adapters.llm_providers.gemma_client import (
        build_gemma_model_slots,
    )

    slots = build_gemma_model_slots()
    if not slots:
        trace("WARN history_block_collapse | no Gemma model configured")
        return None
    max_tokens = HISTORY_BLOCK_SUMMARY_MAX_OUTPUT_TOKENS
    for slot in slots:
        try:
            input_tokens = slot.client.estimate_input_tokens(
                _BLOCK_SUMMARY_SYSTEM, text_blob, LearnerProgressBlockSummary
            )
            await slot.limiter.acquire(
                input_tokens + max_tokens, model=slot.model, priority=True
            )
            result = await slot.client.complete_structured(
                _BLOCK_SUMMARY_SYSTEM,
                text_blob,
                LearnerProgressBlockSummary,
                label="node_deep_dive/history_block_collapse",
                limiter=slot.limiter,
                max_tokens=max_tokens,
            )
            if result is not None:
                return result
            trace(
                f"WARN history_block_collapse | {slot.label} {slot.model} "
                "returned no result, trying next slot"
            )
        except Exception as exc:
            trace(
                f"WARN history_block_collapse | {slot.label} {slot.model} "
                f"failed: {exc}, trying next slot"
            )
            continue
    return None


def _summarize_block(block: list[dict]) -> str:
    text_blob = _format_block_for_summary(block)
    if not text_blob.strip():
        return _fallback_block_digest(block)
    try:
        import asyncio

        result = asyncio.run(_summarize_block_via_gemma(text_blob))
        if result is not None:
            return (result.summary or "").strip() or _fallback_block_digest(block)
        return _fallback_block_digest(block)
    except Exception as exc:
        trace(f"WARN history_block_collapse summary failed | {exc}")
        return _fallback_block_digest(block)


def maybe_collapse_old_block(
    stored: "StoredChatSession",
    memory: "SessionMemory",
) -> bool:
    """Collapse the oldest COMPRESSED_BLOCK_SIZE messages once the

    compressed-timeline portion (everything past the RAW_HISTORY_DEPTH raw
    tail) reaches that size. Mutates ``stored.api_turns`` and
    ``memory.learner_progress_blocks`` / ``learner_progress_verified_ids``
    in place. Returns True if a block was collapsed."""
    from knowledge_engine.src.config.settings import (
        COMPRESSED_BLOCK_SIZE,
        DIALOG_BLOCK_COLLAPSE_ENABLED,
        MAX_COMPRESSED_BLOCKS,
        RAW_HISTORY_DEPTH,
    )

    if not DIALOG_BLOCK_COLLAPSE_ENABLED:
        return False
    turns = stored.api_turns
    older_count = len(turns) - max(2, RAW_HISTORY_DEPTH)
    if older_count < max(1, COMPRESSED_BLOCK_SIZE):
        return False

    block = turns[:COMPRESSED_BLOCK_SIZE]
    remaining = turns[COMPRESSED_BLOCK_SIZE:]

    from knowledge_engine.src.domains.grounding.concept_map_state import (
        list_verified_sub_concept_ids,
    )

    summary_text = _summarize_block(block)
    verified_now = list_verified_sub_concept_ids(memory)
    merged_ids = list(
        dict.fromkeys([*(memory.learner_progress_verified_ids or []), *verified_now])
    )

    blocks = list(memory.learner_progress_blocks or []) + [summary_text]
    if len(blocks) > MAX_COMPRESSED_BLOCKS:
        blocks = blocks[-MAX_COMPRESSED_BLOCKS:]

    memory.learner_progress_verified_ids = merged_ids
    memory.learner_progress_blocks = blocks
    stored.api_turns = remaining
    trace(
        "NODE_DIVE history_block_collapse | "
        f"collapsed={len(block)} remaining={len(remaining)} "
        f"blocks={len(blocks)} verified_ids={len(merged_ids)}"
    )
    return True


def build_learner_progress_summary_block(memory: "SessionMemory") -> str:
    """``LEARNER_PROGRESS_SUMMARY`` text for the system prompt — empty string

    when nothing has been collapsed yet (add-only, never injects a header
    for a session that never hit the block-collapse threshold)."""
    blocks = list(memory.learner_progress_blocks or [])
    if not blocks:
        return ""
    ids = list(memory.learner_progress_verified_ids or [])
    lines = ["[LEARNER_PROGRESS_SUMMARY]"]
    lines.extend(blocks)
    if ids:
        lines.append("Закреплённые подтемы: " + ", ".join(ids))
    return "\n".join(lines)
