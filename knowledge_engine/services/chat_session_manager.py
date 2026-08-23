"""Изолированные Gemini chat-сессии: модель, история, delta без дублирования."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from knowledge_engine.config import CHAT_SESSION_API_TURNS_MAX
from knowledge_engine.ui.run_log import trace

ContextType = Literal["Fresh", "Summary"]


def _with_generation_config(
    config_kwargs: dict[str, Any],
    *,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    return config_kwargs


class StoredChatSession(BaseModel):
    """Персистентное состояние (без live Chat объекта SDK)."""

    session_id: str = Field(min_length=8, max_length=32)
    model_name: str = Field(min_length=2, max_length=120)
    label: str = Field(min_length=1, max_length=200)
    context_type: ContextType = "Fresh"
    turns: int = Field(default=0, ge=0, le=500)
    api_turns: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=CHAT_SESSION_API_TURNS_MAX,
    )
    last_pinned_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hex последнего pinned, отправленного в Gemini",
    )
    explicit_cache_name: str = Field(default="", max_length=256)
    explicit_cache_digest: str = Field(default="", max_length=128)
    last_layer2_hash: str = Field(default="", max_length=128)


@dataclass(frozen=True)
class UserPayloadBuildMeta:
    layer1_omitted: bool = False
    layer2_in_request: bool = False
    layer2_bytes: int = 0
    explicit_cache_mode: str = ""
    cache_name: str = ""
    digest: str = ""


_PINNED_REFRESH_HEADER = (
    "[PINNED_NODE_CONTEXT_REFRESH]\n"
    "Контекст ноды обновлён (concept_map, прогресс, источники). "
    "Используй блок ниже вместе с историей чата.\n\n"
)


def _pinned_context_digest(pinned: str) -> str:
    return hashlib.sha256((pinned or "").encode("utf-8")).hexdigest()


def _resolve_pinned_for_turn(
    stored: StoredChatSession | None,
    pinned: str,
) -> str:
    body = (pinned or "").strip()
    if not body:
        return ""
    if stored is None:
        return body
    digest = _pinned_context_digest(body)
    if stored.turns <= 0:
        stored.last_pinned_hash = digest
        return body
    prev = (stored.last_pinned_hash or "").strip()
    if prev and prev == digest:
        return ""
    stored.last_pinned_hash = digest
    if prev:
        return f"{_PINNED_REFRESH_HEADER}{body}"
    return body


_LAYER2_REFRESH_HEADER = (
    "[LAYER2_SESSION_STATE_REFRESH]\n"
    "Состояние сессии обновлено (manifest, concept_map, прогресс).\n\n"
)


def _body_digest(body: str) -> str:
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def _resolve_layer2_for_turn(
    stored: StoredChatSession | None,
    layer2: str,
) -> str:
    body = (layer2 or "").strip()
    if not body:
        return ""
    if stored is None:
        return body
    digest = _body_digest(body)
    if stored.turns <= 0:
        stored.last_layer2_hash = digest
        return body
    prev = (stored.last_layer2_hash or "").strip()
    if prev and prev == digest:
        return ""
    stored.last_layer2_hash = digest
    if prev:
        return f"{_LAYER2_REFRESH_HEADER}{body}"
    return body


def _fallback_pinned_blob(
    pinned_context: str,
    layer1_context: str,
    layer2_context: str,
) -> str:
    pinned = (pinned_context or "").strip()
    if pinned:
        return pinned
    from knowledge_engine.src.node_deep_dive.dialog_context import (
        PINNED_CONTEXT_TAG,
        combine_node_context_layers,
    )

    body = combine_node_context_layers(layer1_context, layer2_context)
    if not body:
        return ""
    if PINNED_CONTEXT_TAG in body:
        return body
    return f"{PINNED_CONTEXT_TAG}\n\n{body}"


def _clear_explicit_cache_fields(stored: StoredChatSession) -> None:
    stored.explicit_cache_name = ""
    stored.explicit_cache_digest = ""


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
            last_pinned_hash="",
            explicit_cache_name="",
            explicit_cache_digest="",
            last_layer2_hash="",
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

    def _trim_api_turns(self, stored: StoredChatSession) -> None:
        cap = max(2, CHAT_SESSION_API_TURNS_MAX)
        if len(stored.api_turns) > cap:
            stored.api_turns = stored.api_turns[-cap:]

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
        self._trim_api_turns(stored)

    def replace_last_model_turn(self, label: str, compact_text: str) -> bool:
        """
        Rewrite the last model api_turn with a compact digest (asterisk-question cache hygiene).

        Leaves user turns untouched. Returns True if a model turn was replaced.
        """
        lab = (label or "").strip()
        stored = self._sessions.get(lab)
        body = (compact_text or "").strip()
        if not stored or not body:
            return False
        for i in range(len(stored.api_turns) - 1, -1, -1):
            turn = stored.api_turns[i]
            if (turn.get("role") or "").strip() == "model":
                stored.api_turns[i] = {
                    "role": "model",
                    "content": body[:8000],
                }
                return True
        return False

    def compact_dialog_session(self, label: str, handoff_summary: str) -> None:
        """Сброс api_turns в summary (после dense / переполнения окна)."""
        lab = (label or "").strip()
        stored = self._sessions.get(lab)
        if not stored:
            return
        summary = (handoff_summary or "").strip()[:6000]
        stored.api_turns = []
        stored.turns = 0
        stored.last_pinned_hash = ""
        stored.last_layer2_hash = ""
        _clear_explicit_cache_fields(stored)
        stored.context_type = "Summary"
        if summary:
            stored.api_turns.append(
                {
                    "role": "user",
                    "content": (
                        "### context_summary_from_previous_session\n" f"{summary}"
                    ),
                }
            )
        self._live.pop(lab, None)
        trace(f"CHAT_SESSION compact | label={lab} | summary_len={len(summary)}")

    def build_user_payload(
        self,
        label: str,
        pinned_context: str,
        movable_context: str,
        delta_user_message: str,
        *,
        layer1_context: str = "",
        layer2_context: str = "",
        explicit_cache: Any | None = None,
    ) -> tuple[str, str, UserPayloadBuildMeta]:
        """
        Полный user payload для Gemini и текст только диалога для record_turn.
        При active explicit cache layer1 не включается в текст (запечён в cached_content).
        """
        from knowledge_engine.services.gemini_cache_manager import (
            ExplicitCacheResult,
            explicit_cache_is_active,
        )

        cache = (
            explicit_cache if isinstance(explicit_cache, ExplicitCacheResult) else None
        )
        cache_active = explicit_cache_is_active(cache)
        movable = (movable_context or "").strip()
        delta = (delta_user_message or "").strip()
        stored = self._sessions.get(label)

        context_prefix = ""
        layer2_bytes = 0
        layer2_in_request = False

        if cache_active:
            layer2_out = _resolve_layer2_for_turn(stored, layer2_context)
            context_prefix = layer2_out
            layer2_bytes = len(layer2_out)
            layer2_in_request = bool(layer2_out)
            if stored is not None and cache:
                stored.explicit_cache_name = cache.cache_name
                stored.explicit_cache_digest = (cache.digest or "").strip()
        else:
            pinned_blob = _fallback_pinned_blob(
                pinned_context,
                layer1_context,
                layer2_context,
            )
            context_prefix = _resolve_pinned_for_turn(stored, pinned_blob)
            if stored is not None:
                _clear_explicit_cache_fields(stored)

        dialog_parts: list[str] = []
        if stored is None or stored.turns == 0:
            if movable:
                dialog_parts.append(movable)
            if delta:
                dialog_parts.append(delta)
            dialog_user = "\n\n".join(dialog_parts).strip()
            full_parts = [p for p in [context_prefix, dialog_user] if p]
            meta = UserPayloadBuildMeta(
                layer1_omitted=cache_active,
                layer2_in_request=layer2_in_request,
                layer2_bytes=layer2_bytes,
                explicit_cache_mode=(cache.mode if cache else ""),
                cache_name=(cache.cache_name if cache else ""),
                digest=(cache.digest if cache else ""),
            )
            return "\n\n".join(full_parts), dialog_user, meta

        if delta:
            dialog_user = delta
        else:
            dialog_user = movable
        full_parts = [p for p in [context_prefix, dialog_user] if p]
        meta = UserPayloadBuildMeta(
            layer1_omitted=cache_active,
            layer2_in_request=layer2_in_request,
            layer2_bytes=layer2_bytes,
            explicit_cache_mode=(cache.mode if cache else ""),
            cache_name=(cache.cache_name if cache else ""),
            digest=(cache.digest if cache else ""),
        )
        return "\n\n".join(full_parts), (dialog_user or "").strip(), meta

    def _log_prompt_trace_before_send(
        self,
        label: str,
        stored: StoredChatSession,
        *,
        model: str,
        system_instruction: str,
        message: str,
        dialog_user_text: str,
        prompt_trace: Any | None,
    ) -> None:
        from knowledge_engine.services.session_prompt_trace import (
            PromptTraceContext,
            write_session_prompt_trace,
        )

        ctx = prompt_trace if isinstance(prompt_trace, PromptTraceContext) else None
        return write_session_prompt_trace(
            session_id=stored.session_id,
            turn_number=stored.turns + 1,
            chat_label=label,
            model_name=model,
            system_instruction=system_instruction,
            user_payload=message,
            api_turns=list(stored.api_turns),
            dialog_user_text=dialog_user_text,
            trace_context=ctx,
        )

    def _merge_prompt_trace(
        self,
        prompt_trace: Any | None,
        payload_meta: UserPayloadBuildMeta,
    ) -> Any | None:
        from knowledge_engine.services.session_prompt_trace import (
            ExplicitCacheTraceMeta,
            PromptTraceContext,
        )

        if not isinstance(prompt_trace, PromptTraceContext):
            return prompt_trace
        mode = payload_meta.explicit_cache_mode or "—"
        return PromptTraceContext(
            metrics=prompt_trace.metrics,
            node_session_key=prompt_trace.node_session_key,
            explicit_cache=ExplicitCacheTraceMeta(
                explicit_cache_mode=mode,
                cache_name=payload_meta.cache_name,
                digest=payload_meta.digest,
                layer1_omitted=payload_meta.layer1_omitted,
                layer2_in_request=payload_meta.layer2_in_request,
                layer2_bytes=payload_meta.layer2_bytes,
            ),
        )

    def invalidate_live_chat(self, label: str) -> None:
        """Drop the live SDK chat so the next send rebuilds history with a new schema."""
        self._live.pop(label, None)

    def _invalidate_live_chat(self, label: str) -> None:
        self.invalidate_live_chat(label)

    def get_or_create_live_chat(
        self,
        client: Any,
        label: str,
        model: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        explicit_cache: Any | None = None,
        force_recreate: bool = False,
    ) -> tuple[StoredChatSession, Any]:
        from google.genai import types

        from knowledge_engine.services.gemini_cache_manager import (
            ExplicitCacheResult,
            apply_explicit_cache_to_generation_config,
            explicit_cache_is_active,
        )

        cache = (
            explicit_cache if isinstance(explicit_cache, ExplicitCacheResult) else None
        )
        cache_digest = (
            (cache.digest or "").strip() if explicit_cache_is_active(cache) else ""
        )

        stored = self.resolve_for_model(label, model, summary_for_handoff)
        if (
            not force_recreate
            and label in self._live
            and self._live[label].get("session_id") == stored.session_id
            and self._live[label].get("explicit_cache_digest", "") == cache_digest
        ):
            return stored, self._live[label]["chat"]

        config_kwargs: dict[str, Any] = {}
        apply_explicit_cache_to_generation_config(
            config_kwargs,
            system_instruction=system_instruction,
            cache=cache if not force_recreate else None,
        )
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        _with_generation_config(
            config_kwargs,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

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
        if explicit_cache_is_active(cache) and cache is not None:
            stored.explicit_cache_name = cache.cache_name
            stored.explicit_cache_digest = cache_digest
        elif force_recreate:
            _clear_explicit_cache_fields(stored)
        self._live[label] = {
            "session_id": stored.session_id,
            "chat": chat,
            "explicit_cache_digest": cache_digest,
        }
        return stored, chat

    def _send_message_with_cache_fallback(
        self,
        *,
        client: Any,
        label: str,
        model: str,
        message: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
        explicit_cache: Any | None,
        pinned_context: str,
        movable_context: str,
        delta_user_message: str,
        layer1_context: str,
        layer2_context: str,
        max_output_tokens: int | None,
        temperature: float | None,
        stream: bool,
        stream_callback: Callable[[str], None] | None = None,
        payload_meta: UserPayloadBuildMeta | None = None,
    ) -> tuple[str, StoredChatSession, Any, str, UserPayloadBuildMeta]:
        from knowledge_engine.services.gemini_cache_manager import (
            ExplicitCacheResult,
            explicit_cache_is_active,
            is_cache_resource_error,
            registry_delete,
        )

        cache = (
            explicit_cache if isinstance(explicit_cache, ExplicitCacheResult) else None
        )
        msg = (message or "").strip()
        base_meta = payload_meta or UserPayloadBuildMeta()
        out_meta = UserPayloadBuildMeta(
            layer1_omitted=base_meta.layer1_omitted or explicit_cache_is_active(cache),
            layer2_in_request=base_meta.layer2_in_request,
            layer2_bytes=base_meta.layer2_bytes,
            explicit_cache_mode=base_meta.explicit_cache_mode
            or (cache.mode if cache else ""),
            cache_name=base_meta.cache_name or (cache.cache_name if cache else ""),
            digest=base_meta.digest or (cache.digest if cache else ""),
        )
        stored, chat = self.get_or_create_live_chat(
            client,
            label,
            model,
            system_instruction,
            response_schema,
            summary_for_handoff,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            explicit_cache=cache,
        )
        try:
            if stream:
                return self._stream_from_chat(chat, msg), stored, chat, msg, out_meta
            response = chat.send_message(msg)
            return (response.text or "").strip(), stored, chat, msg, out_meta
        except Exception as exc:
            if not explicit_cache_is_active(cache) or not is_cache_resource_error(exc):
                raise
            digest = (cache.digest if cache else "") or ""
            trace(
                f"CHAT_SESSION explicit cache fallback | {label} | "
                f"cache 404/invalid → full pinned | {digest[:12]}"
            )
            if digest:
                registry_delete(digest)
            stored = self._sessions.get(label)
            if stored:
                _clear_explicit_cache_fields(stored)
            self._invalidate_live_chat(label)
            message, _dialog, fb_meta = self.build_user_payload(
                label,
                pinned_context,
                movable_context,
                delta_user_message,
                layer1_context=layer1_context,
                layer2_context=layer2_context,
                explicit_cache=None,
            )
            fb_meta = UserPayloadBuildMeta(
                layer1_omitted=fb_meta.layer1_omitted,
                layer2_in_request=fb_meta.layer2_in_request,
                layer2_bytes=fb_meta.layer2_bytes,
                explicit_cache_mode="error_fallback",
                cache_name="",
                digest=digest,
            )
            msg = (message or "").strip()
            stored, chat = self.get_or_create_live_chat(
                client,
                label,
                model,
                system_instruction,
                response_schema,
                summary_for_handoff,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                explicit_cache=None,
                force_recreate=True,
            )
            if stream:
                return (
                    self._stream_from_chat(chat, msg),
                    stored,
                    chat,
                    msg,
                    fb_meta,
                )
            response = chat.send_message(msg)
            return (response.text or "").strip(), stored, chat, msg, fb_meta

    def _stream_from_chat(
        self,
        chat: Any,
        message: str | None = None,
    ) -> str:
        msg = (message or "").strip()
        cum_text = ""
        stream = chat.send_message_stream(msg)
        for chunk in stream:
            piece = getattr(chunk, "text", None) or ""
            if not piece:
                continue
            if cum_text and piece.startswith(cum_text):
                cum_text = piece
            else:
                cum_text += piece
        return cum_text.strip()

    def send_chat_message(
        self,
        client: Any,
        label: str,
        model: str,
        message: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
        *,
        record_user_text: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        prompt_trace: Any | None = None,
        explicit_cache: Any | None = None,
        pinned_context: str = "",
        movable_context: str = "",
        delta_user_message: str = "",
        layer1_context: str = "",
        layer2_context: str = "",
        payload_meta: UserPayloadBuildMeta | None = None,
    ) -> str:
        stored = self.resolve_for_model(label, model, summary_for_handoff)
        dialog_user = (record_user_text or message or "").strip()
        text, stored, _chat, sent_msg, sent_meta = (
            self._send_message_with_cache_fallback(
                client=client,
                label=label,
                model=model,
                message=message,
                system_instruction=system_instruction,
                response_schema=response_schema,
                summary_for_handoff=summary_for_handoff,
                explicit_cache=explicit_cache,
                pinned_context=pinned_context,
                movable_context=movable_context,
                delta_user_message=delta_user_message,
                layer1_context=layer1_context,
                layer2_context=layer2_context,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=False,
                payload_meta=payload_meta,
            )
        )
        meta = sent_meta or payload_meta or UserPayloadBuildMeta()
        trace_ctx = self._merge_prompt_trace(prompt_trace, meta)
        self._log_prompt_trace_before_send(
            label,
            stored,
            model=model,
            system_instruction=system_instruction,
            message=sent_msg,
            dialog_user_text=dialog_user,
            prompt_trace=trace_ctx,
        )
        finish_reason = ""
        try:
            from knowledge_engine.services.gemini_stateless import trace_gemini_io_sizes

            trace_gemini_io_sizes(
                label or "gemini_chat",
                system_instruction=system_instruction,
                user_payload=sent_msg,
                output_text=text,
                max_output_tokens=max_output_tokens,
                finish_reason=finish_reason,
                model_name=stored.model_name,
            )
        except Exception:
            pass
        if not text:
            raise RuntimeError("Gemini chat: пустой ответ")
        from knowledge_engine.ui.llm_trace import trace_llm_exchange

        trace_llm_exchange(
            label or "gemini_chat",
            system_instruction,
            sent_msg,
            text,
            model=stored.model_name,
        )
        to_record = (record_user_text or message or "").strip()
        self.record_turn(label, to_record, text)
        return text

    def send_chat_message_stream(
        self,
        client: Any,
        label: str,
        model: str,
        message: str,
        system_instruction: str,
        response_schema: type | None,
        summary_for_handoff: str,
        stream_callback: Callable[[str], None],
        *,
        record_user_text: str | None = None,
        stream_text_field: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        prompt_trace: Any | None = None,
        explicit_cache: Any | None = None,
        pinned_context: str = "",
        movable_context: str = "",
        delta_user_message: str = "",
        layer1_context: str = "",
        layer2_context: str = "",
        payload_meta: UserPayloadBuildMeta | None = None,
    ) -> str:
        from knowledge_engine.services.gemini_cache_manager import (
            ExplicitCacheResult,
            explicit_cache_is_active,
            is_cache_resource_error,
            registry_delete,
        )
        from knowledge_engine.services.gemini_json_stream import (
            DRILL_ACTIVE_STREAM_FIELDS,
            DRILL_COMPLETE_STREAM_FIELDS,
            TUTOR_EXPLAIN_STREAM_FIELDS,
            JsonFieldStreamFilter,
            wrap_stream_callback_for_json_field,
            wrap_stream_callback_for_tutor_dialogue_fields,
        )

        schema_name = getattr(response_schema, "__name__", "") or ""
        field = (stream_text_field or "").strip()
        field_filter: JsonFieldStreamFilter | None = None
        if (
            schema_name in ("DeepDiveTutorContract", "DeepDiveDeepAnalysisContract")
            and stream_callback is not None
        ):
            field_filter = wrap_stream_callback_for_tutor_dialogue_fields(
                stream_callback
            )
        elif schema_name == "DeepDiveExplainContract" and stream_callback is not None:
            field_filter = wrap_stream_callback_for_tutor_dialogue_fields(
                stream_callback, fields=TUTOR_EXPLAIN_STREAM_FIELDS
            )
        elif schema_name == "ActiveDrillStepResponse" and stream_callback is not None:
            field_filter = wrap_stream_callback_for_tutor_dialogue_fields(
                stream_callback, fields=DRILL_ACTIVE_STREAM_FIELDS
            )
        elif schema_name == "LayerCompletionTutorOutput" and stream_callback is not None:
            field_filter = wrap_stream_callback_for_tutor_dialogue_fields(
                stream_callback, fields=DRILL_COMPLETE_STREAM_FIELDS
            )
        elif field and response_schema is not None:
            field_filter = wrap_stream_callback_for_json_field(field, stream_callback)
        msg = (message or "").strip()
        dialog_user = (record_user_text or msg or "").strip()
        meta = payload_meta or UserPayloadBuildMeta()
        cache = (
            explicit_cache if isinstance(explicit_cache, ExplicitCacheResult) else None
        )

        def _run_stream(chat: Any, stream_msg: str) -> tuple[str, Any]:
            parts: list[str] = []
            cum_text = ""
            last_chunk: Any = None
            stream = chat.send_message_stream((stream_msg or "").strip())
            for chunk in stream:
                last_chunk = chunk
                piece = getattr(chunk, "text", None) or ""
                if not piece:
                    continue
                if cum_text and piece.startswith(cum_text):
                    delta_raw = piece[len(cum_text) :]
                    cum_text = piece
                else:
                    delta_raw = piece
                    cum_text += piece
                if not delta_raw:
                    continue
                parts.append(delta_raw)
                if field_filter is not None:
                    field_filter.feed(delta_raw)
                elif stream_callback is not None:
                    stream_callback(delta_raw)
            if field_filter is not None:
                field_filter.flush()
            return (cum_text or "".join(parts)).strip(), last_chunk

        sent_msg = msg
        sent_meta = meta
        stored, chat = self.get_or_create_live_chat(
            client,
            label,
            model,
            system_instruction,
            response_schema,
            summary_for_handoff,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            explicit_cache=cache,
        )
        try:
            text, last_chunk = _run_stream(chat, msg)
        except Exception as exc:
            if not explicit_cache_is_active(cache) or not is_cache_resource_error(exc):
                raise
            digest = (cache.digest if cache else "") or ""
            trace(f"CHAT_SESSION explicit cache stream fallback | {label}")
            if digest:
                registry_delete(digest)
            st = self._sessions.get(label)
            if st:
                _clear_explicit_cache_fields(st)
            self._invalidate_live_chat(label)
            msg, _d, fb_meta = self.build_user_payload(
                label,
                pinned_context,
                movable_context,
                delta_user_message,
                layer1_context=layer1_context,
                layer2_context=layer2_context,
                explicit_cache=None,
            )
            sent_msg = (msg or "").strip()
            sent_meta = UserPayloadBuildMeta(
                layer1_omitted=fb_meta.layer1_omitted,
                layer2_in_request=fb_meta.layer2_in_request,
                layer2_bytes=fb_meta.layer2_bytes,
                explicit_cache_mode="error_fallback",
                cache_name="",
                digest=digest,
            )
            stored, chat = self.get_or_create_live_chat(
                client,
                label,
                model,
                system_instruction,
                response_schema,
                summary_for_handoff,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                explicit_cache=None,
                force_recreate=True,
            )
            text, last_chunk = _run_stream(chat, sent_msg)
        trace_ctx = self._merge_prompt_trace(
            prompt_trace,
            sent_meta or meta,
        )
        self._log_prompt_trace_before_send(
            label,
            stored,
            model=model,
            system_instruction=system_instruction,
            message=sent_msg,
            dialog_user_text=dialog_user,
            prompt_trace=trace_ctx,
        )
        try:
            from knowledge_engine.services.gemini_stateless import (
                finish_reason_from_gemini_response,
                trace_gemini_io_sizes,
            )

            finish_reason = finish_reason_from_gemini_response(last_chunk)
            trace_gemini_io_sizes(
                label or "gemini_chat_stream",
                system_instruction=system_instruction,
                user_payload=sent_msg,
                output_text=text,
                max_output_tokens=max_output_tokens,
                finish_reason=finish_reason,
                model_name=stored.model_name,
            )
        except Exception:
            pass
        if not text:
            raise RuntimeError("Gemini chat stream: пустой ответ")
        from knowledge_engine.ui.llm_trace import trace_llm_exchange

        trace_llm_exchange(
            label or "gemini_chat_stream",
            system_instruction,
            sent_msg,
            text,
            model=stored.model_name,
        )
        to_record = (record_user_text or message or "").strip()
        self.record_turn(label, to_record, text)
        return text
