"""Gemma Cloud SSOT: structured + text LLM (replaces ChatOllama)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Iterator, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from knowledge_engine.config import GEMMA_PRIMARY_MODEL, gemma_cloud_api_key_available
from knowledge_engine.ui.run_log import gemma_cloud_invoke

T = TypeVar("T", bound=BaseModel)

# Back-compat alias: callers still pass model names; cloud slot is always Gemma primary.
OllamaTier = str


class _GemmaTextEnvelope(BaseModel):
    text: str = Field(default="", description="Plain assistant reply.")


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _messages_to_system_user(messages: list[BaseMessage]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        name = type(msg).__name__.lower()
        if "system" in name:
            system_parts.append(content)
        else:
            user_parts.append(content)
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()


def _client(model: str | None = None):
    from knowledge_engine.services.llm.gemma_client import GemmaCloudClient

    chosen = (model or "").strip()
    low = chosen.lower()
    if (
        not chosen
        or "qwen" in low
        or "ollama" in low
        or ":1.5b" in low
        or ":7b" in low
    ):
        chosen = GEMMA_PRIMARY_MODEL
    return GemmaCloudClient(model=chosen)


async def complete_structured_async(
    schema: type[T],
    system: str,
    user: str,
    *,
    label: str = "gemma_structured",
    model: str | None = None,
) -> T | None:
    if not gemma_cloud_api_key_available():
        raise RuntimeError("Gemma Cloud API key missing (GEMINI_API_KEY / GEMMA_API_KEY)")
    return await _client(model).complete_structured(system, user, schema, label=label)


def complete_structured_sync(
    schema: type[T],
    system: str,
    user: str,
    *,
    label: str = "gemma_structured",
    model: str | None = None,
) -> T | None:
    return _run_coro(
        complete_structured_async(schema, system, user, label=label, model=model)
    )


class GemmaStructuredRunnable:
    """LangChain-shaped ``.invoke(messages)`` over Gemma Cloud structured JSON."""

    def __init__(
        self,
        schema: type[T],
        *,
        model: str,
        temperature: float = 0.2,
        label: str = "gemma_structured",
    ) -> None:
        self.schema = schema
        self.model = model or GEMMA_PRIMARY_MODEL
        self.temperature = temperature
        self._label = label

    def invoke(self, messages: list[BaseMessage], **_kwargs: Any) -> T | None:
        system, user = _messages_to_system_user(messages)
        return complete_structured_sync(
            self.schema,
            system,
            user,
            label=self._label,
            model=self.model,
        )


class GemmaTextRunnable:
    """LangChain-shaped text invoke/stream over Gemma Cloud."""

    def __init__(self, *, model: str, temperature: float = 0.2) -> None:
        self.model = model or GEMMA_PRIMARY_MODEL
        self.temperature = temperature

    def invoke(self, messages: list[BaseMessage], **_kwargs: Any) -> Any:
        system, user = _messages_to_system_user(messages)
        client = _client(self.model)

        async def _go() -> _GemmaTextEnvelope | None:
            return await client.complete_structured(
                system,
                user,
                _GemmaTextEnvelope,
                label="gemma_text",
            )

        parsed = _run_coro(_go())
        text = (parsed.text if parsed is not None else "") or ""

        class _Msg:
            content = text

        return _Msg()

    def stream(self, messages: list[BaseMessage], **_kwargs: Any) -> Iterator[Any]:
        msg = self.invoke(messages)
        yield msg


def chat_llm(
    model: str,
    temperature: float = 0.2,
    num_predict: int | None = None,
    tier: OllamaTier = "auto",
) -> GemmaTextRunnable:
    _ = (num_predict, tier)
    return GemmaTextRunnable(model=model, temperature=temperature)


# Historical name — no ChatOllama underneath.
chat_ollama = chat_llm


def structured_chat(
    model: str,
    schema: type[T],
    temperature: float = 0.2,
    num_predict: int | None = None,
    tier: OllamaTier = "auto",
) -> GemmaStructuredRunnable:
    _ = (num_predict, tier)
    return GemmaStructuredRunnable(
        schema,
        model=model,
        temperature=temperature,
        label=f"structured/{getattr(schema, '__name__', 'schema')}",
    )


def invoke_logged(llm: Any, messages: list[BaseMessage], label: str) -> Any:
    return gemma_cloud_invoke(llm, messages, label)


def stream_chat(
    model: str,
    messages: list[BaseMessage],
    temperature: float = 0.2,
    label: str = "LLM",
) -> str:
    from knowledge_engine.ui.logger import append_stream_token, clear_stream

    clear_stream()
    llm = chat_llm(model, temperature)
    msg = gemma_cloud_invoke(llm, messages, label)
    text = msg.content if hasattr(msg, "content") else str(msg)
    if text:
        append_stream_token(text)
    return text or ""
