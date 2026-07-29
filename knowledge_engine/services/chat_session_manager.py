"""Изолированные Gemini chat-сессии: модель, история, delta без дублирования."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from knowledge_engine.ui.run_log import trace

ContextType = Literal["Fresh", "Summary"]


class StoredChatSession(BaseModel):
    """Персистентное состояние (без live Chat объекта SDK)."""

    session_id: str = Field(min_length=8, max_length=32)
    model_name: str = Field(min_length=2, max_length=120)
    label: str = Field(min_length=1, max_length=200)
    context_type: ContextType = "Fresh"
    turns: int = Field(default=0, ge=0, le=500)
    api_turns: list[dict[str, str]] = Field(default_factory=list, max_length=40)


class ChatSessionManager:
    """
    Одна логическая сессия = (user_scope, label) → session_id + model_name + api_turns.
    Смена model_name → новый session_id, только summary, не сырая история другой модели.
    """

    def __init__(
        self,
        user_scope: str,
        sessions: dict[str, StoredChatSession] | None = None,
    ) -> None:
        self._user_scope = (user_scope or "default").strip()
        self._sessions: dict[str, StoredChatSession] = dict(sessions or {})
        self._live: dict[str, Any] = {}

    @classmethod
    def from_memory_blob(
        cls,
        user_scope: str,
        blob: dict[str, dict] | None,
    ) -> ChatSessionManager:
        parsed: dict[str, StoredChatSession] = {}
        for key, raw in (blob or {}).items():
            try:
                parsed[str(key)] = StoredChatSession.model_validate(raw)
            except Exception:
                continue
        return cls(user_scope, parsed)

    def to_memory_blob(self) -> dict[str, dict]:
        return {k: v.model_dump() for k, v in self._sessions.items()}

    def clear_all(self, reason: str = "reset") -> None:
        if self._sessions:
            trace(f"CHAT_SESSION clear all | scope={self._user_scope} | {reason}")
        self._sessions.clear()
        self._live.clear()

    def get(self, label: str) -> StoredChatSession | None:
        return self._sessions.get(label)

    def create_new_session(
        self,
        model_name: str,
        label: str,
        initial_summary: str = "",
        context_type: ContextType = "Fresh",
    ) -> StoredChatSession:
        model = (model_name or "").strip()
        lab = (label or "default").strip()
        session_id = uuid.uuid4().hex[:12]
        stored = StoredChatSession(
            session_id=session_id,
            model_name=model,
            label=lab,
            context_type=context_type,
            turns=0,
            api_turns=[],
        )
        if context_type == "Summary" and (initial_summary or "").strip():
            stored.api_turns.append(
                {
                    "role": "user",
                    "content": (
                        "### context_summary_from_previous_session\n"
                        f"{initial_summary.strip()[:6000]}"
                    ),
                }
            )
        self._sessions[lab] = stored
        self._live.pop(lab, None)
        trace(
            f"[Session Created] ID: {session_id} | Model: {model} | "
            f"Context Type: {context_type} | label={lab} | scope={self._user_scope}"
        )
        return stored

    def resolve_for_model(
        self,
        label: str,
        requested_model: str,
        summary_for_handoff: str,
    ) -> StoredChatSession:
        lab = (label or "default").strip()
        model = (requested_model or "").strip()
        cur = self._sessions.get(lab)
        if cur is None:
            return self.create_new_session(model, lab, "", "Fresh")
        if cur.model_name != model:
            return self.create_new_session(
                model,
                lab,
                summary_for_handoff,
                "Summary",
            )
        return cur

    def record_turn(
        self,
        label: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        stored = self._sessions.get(label)
        if not stored:
            return
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if u:
            stored.api_turns.append({"role": "user", "content": u[:8000]})
        if a:
            stored.api_turns.append({"role": "model", "content": a[:8000]})
        stored.turns += 1
        if len(stored.api_turns) > 40:
            stored.api_turns = stored.api_turns[-40:]

    def build_user_payload(
        self,
        label: str,
        full_context: str,
        delta_user_message: str,
    ) -> str:
        """Первый turn — полный контекст; далее — только delta (без active_window)."""
        stored = self._sessions.get(label)
        if stored is None or stored.turns == 0:
            return (full_context or "").strip()
        delta = (delta_user_message or "").strip()
        if delta:
            return f"### current_user_message\n{delta}"
        return (full_context or "").strip()

    def get_or_create_live_chat(
        self,
        client: Any,
        label: str,
        model: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
    ) -> tuple[StoredChatSession, Any]:
        from google.genai import types

        stored = self.resolve_for_model(label, model, summary_for_handoff)
        if label in self._live and self._live[label].get("session_id") == stored.session_id:
            return stored, self._live[label]["chat"]

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        history: list[Any] = []
        for turn in stored.api_turns:
            role = turn.get("role") or "user"
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            gemini_role = "model" if role == "model" or role == "tutor" else "user"
            history.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

        chat = client.chats.create(
            model=stored.model_name,
            config=types.GenerateContentConfig(**config_kwargs),
            history=history,
        )
        self._live[label] = {"session_id": stored.session_id, "chat": chat}
        return stored, chat

    def send_chat_message(
        self,
        client: Any,
        label: str,
        model: str,
        message: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
    ) -> str:
        stored, chat = self.get_or_create_live_chat(
            client,
            label,
            model,
            system_instruction,
            response_schema,
            summary_for_handoff,
        )
        response = chat.send_message((message or "").strip())
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini chat: пустой ответ")
        self.record_turn(label, message, text)
        return text
