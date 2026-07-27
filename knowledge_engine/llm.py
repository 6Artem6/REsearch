"""Shared Ollama LLM helpers."""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from knowledge_engine.config import (
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from knowledge_engine.ui.logger import append_stream_token, clear_stream
from knowledge_engine.ui.run_log import ollama_invoke

T = TypeVar("T", bound=BaseModel)


def chat_ollama(
    model: str,
    temperature: float = 0.2,
    num_predict: int | None = None,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=num_predict if num_predict is not None else OLLAMA_NUM_PREDICT,
    )


def structured_chat(
    model: str,
    schema: type[T],
    temperature: float = 0.2,
    num_predict: int | None = None,
) -> ChatOllama:
    return chat_ollama(model, temperature, num_predict).with_structured_output(
        schema, method="json_schema"
    )


def invoke_logged(llm: ChatOllama, messages: list[BaseMessage], label: str) -> Any:
    return ollama_invoke(llm, messages, label)


def stream_chat(
    model: str,
    messages: list[BaseMessage],
    temperature: float = 0.2,
    label: str = "LLM",
) -> str:
    """Потоковая генерация с отображением токенов в UI."""
    clear_stream()
    llm = chat_ollama(model, temperature)
    chunks: list[str] = []
    for chunk in llm.stream(messages):
        part = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        if part:
            chunks.append(part)
            append_stream_token(part)
    return "".join(chunks)
