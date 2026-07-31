"""Полный trace промптов и сырых ответов LLM (KE_LLM_FULL_TRACE)."""

from __future__ import annotations

from threading import Lock

from knowledge_engine.config import KE_LLM_FULL_TRACE

_lock = Lock()
_step = 0


def _emit_block(step_name: str, incoming: str, raw_output: str) -> None:
    if not KE_LLM_FULL_TRACE:
        return
    global _step
    with _lock:
        _step += 1
        n = _step
    block = (
        "\n"
        + "=" * 50
        + f"\n>>> [STEP {n}: {step_name}]\n"
        + "--- INCOMING PROMPT ---\n"
        + (incoming or "")
        + "\n\n--- RAW OUTPUT ---\n"
        + (raw_output or "")
        + "\n"
        + "=" * 50
        + "\n"
    )
    from knowledge_engine.ui.run_log import trace

    trace(block)


def reset_llm_trace_steps() -> None:
    global _step
    with _lock:
        _step = 0


def trace_llm_exchange(
    step_name: str,
    system_instruction: str,
    user_payload: str,
    raw_output: str,
    model: str = "",
) -> None:
    title = f"{step_name} / {model}".strip(" /")
    incoming = ""
    if (system_instruction or "").strip():
        incoming += "=== SYSTEM ===\n" + system_instruction.strip() + "\n\n"
    incoming += "=== USER ===\n" + (user_payload or "")
    _emit_block(title, incoming, raw_output)


def trace_llm_messages(
    step_name: str,
    messages: list,
    raw_output: str,
    model: str = "",
) -> None:
    parts: list[str] = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", "message")
        content = getattr(m, "content", str(m))
        parts.append(f"[{role}]\n{content}")
    title = f"{step_name} / {model}".strip(" /")
    _emit_block(title, "\n\n".join(parts), raw_output)


def trace_plain_io(step_name: str, incoming: str, raw_output: str) -> None:
    _emit_block(step_name, incoming, raw_output)
