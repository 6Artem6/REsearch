"""Локальный structured JSON: Gemma Cloud SSOT (бывший Ollama helper)."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from knowledge_engine.llm import invoke_logged, structured_chat

T = TypeVar("T", bound=BaseModel)


def run_local_structured(
    model: str,
    schema: type[T],
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    temperature: float = 0.1,
    num_predict: int | None = None,
) -> T:
    """Изолированный structured-вызов Gemma Cloud (без chat history)."""
    llm = structured_chat(
        model, schema, temperature=temperature, num_predict=num_predict
    )
    human = f"GLOBAL ANCHOR:\n{global_anchor.strip()}\n\n{user_payload.strip()}"
    result = invoke_logged(
        llm,
        [SystemMessage(content=system_instruction), HumanMessage(content=human)],
        f"local / {label}",
    )
    if result is None:
        raise RuntimeError(f"Локальная модель вернула None ({label})")
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        return schema.model_validate(result.model_dump())
    raise RuntimeError(f"Неверный тип ответа локальной модели ({label})")


def run_local_text(
    model: str,
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    temperature: float = 0.2,
) -> str:
    from knowledge_engine.llm import chat_ollama

    llm = chat_ollama(model, temperature)
    human = f"GLOBAL ANCHOR:\n{global_anchor.strip()}\n\n{user_payload.strip()}"
    msg = invoke_logged(
        llm,
        [SystemMessage(content=system_instruction), HumanMessage(content=human)],
        f"local / {label}",
    )
    content = msg.content if hasattr(msg, "content") else str(msg)
    text = content if isinstance(content, str) else str(content)
    if not text.strip():
        raise RuntimeError(f"Пустой локальный ответ ({label})")
    return text.strip()
