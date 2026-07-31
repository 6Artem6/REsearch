"""Shared Ollama LLM helpers."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from knowledge_engine.config import (
    CONTEXT_EVAL_MODEL,
    LOCAL_ROUTER_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_HEAVY_KEEP_ALIVE,
    OLLAMA_HEAVY_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_ROUTER_KEEP_ALIVE,
    OLLAMA_ROUTER_NUM_CTX,
    REACT_EVAL_MODEL,
    ROUTER_MODEL,
    SELECTION_PROMPTS_OLLAMA_MODEL,
)
from knowledge_engine.ui.logger import append_stream_token, clear_stream
from knowledge_engine.ui.run_log import ollama_invoke

T = TypeVar("T", bound=BaseModel)

OllamaTier = Literal["router", "heavy", "auto"]


def _router_model_names() -> frozenset[str]:
    return frozenset(
        {
            ROUTER_MODEL,
            LOCAL_ROUTER_MODEL,
            REACT_EVAL_MODEL,
            SELECTION_PROMPTS_OLLAMA_MODEL,
            CONTEXT_EVAL_MODEL,
        }
    )


def resolve_ollama_tier(model: str) -> Literal["router", "heavy"]:
    name = (model or "").strip()
    if name in _router_model_names():
        return "router"
    low = name.lower()
    if ":1.5b" in low or "-1.5b" in low:
        return "router"
    return "heavy"


def chat_ollama(
    model: str,
    temperature: float = 0.2,
    num_predict: int | None = None,
    tier: OllamaTier = "auto",
) -> ChatOllama:
    resolved = tier if tier != "auto" else resolve_ollama_tier(model)
    if resolved == "router":
        num_ctx = OLLAMA_ROUTER_NUM_CTX
        keep_alive = OLLAMA_ROUTER_KEEP_ALIVE
    else:
        num_ctx = OLLAMA_HEAVY_NUM_CTX
        keep_alive = OLLAMA_HEAVY_KEEP_ALIVE
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict if num_predict is not None else OLLAMA_NUM_PREDICT,
        keep_alive=keep_alive,
    )


def structured_chat(
    model: str,
    schema: type[T],
    temperature: float = 0.2,
    num_predict: int | None = None,
    tier: OllamaTier = "auto",
) -> ChatOllama:
    return chat_ollama(
        model, temperature, num_predict, tier=tier
    ).with_structured_output(schema, method="json_schema")


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
