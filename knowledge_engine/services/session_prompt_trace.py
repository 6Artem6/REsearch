"""Файловые дампы промптов Gemini (и fallback LLM) для отладки."""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from knowledge_engine.services.gemini_stateless import estimate_llm_tokens

_log = logging.getLogger(__name__)

_NODE_DIVE_MARKER = "node_deep_dive"

_pending_trace: contextvars.ContextVar[PromptTraceRef | None] = contextvars.ContextVar(
    "session_prompt_trace_pending",
    default=None,
)


@dataclass(frozen=True)
class PromptTraceMetrics:
    pinned_len: int = 0
    behavior_state_len: int = 0
    recency_tail_len: int = 0
    user_message_len: int = 0


@dataclass(frozen=True)
class ExplicitCacheTraceMeta:
    explicit_cache_mode: str = ""
    cache_name: str = ""
    digest: str = ""
    layer1_omitted: bool = False
    layer2_in_request: bool = False
    layer2_bytes: int = 0


@dataclass(frozen=True)
class PromptTraceContext:
    """Опциональный контекст для breakdown (обычно из _invoke_tutor)."""

    metrics: PromptTraceMetrics | None = None
    node_session_key: str = ""
    explicit_cache: ExplicitCacheTraceMeta | None = None


@dataclass(frozen=True)
class PromptTraceRef:
    path: Path
    session_id: str
    turn_number: int
    chat_label: str


def normalize_trace_label(label: str) -> str:
    s = re.sub(r"\s+", " ", (label or "").strip())
    s = s.replace("node_deep_dive /", "node_deep_dive/")
    s = s.replace("node_deep_dive/ ", "node_deep_dive/")
    return s


def should_trace_label(label: str) -> bool:
    from knowledge_engine.config import ENABLE_PROMPT_TRACE_LOGS, PROMPT_TRACE_ALL_LLM

    if not ENABLE_PROMPT_TRACE_LOGS:
        return False
    lab = normalize_trace_label(label)
    if not lab:
        return False
    if PROMPT_TRACE_ALL_LLM:
        return True
    low = lab.lower()
    return _NODE_DIVE_MARKER in low


def should_trace_chat_label(label: str) -> bool:
    return should_trace_label(label)


def _label_slug(label: str) -> str:
    lab = normalize_trace_label(label)
    if "/" in lab:
        lab = lab.split("/")[-1]
    slug = re.sub(r"[^\w.-]+", "_", lab).strip("_")
    return slug or "request"


def _md_fence(text: str) -> str:
    body = text or ""
    ticks = 3
    while f"{'`' * ticks}" in body:
        ticks += 1
    fence = "`" * ticks
    return f"{fence}\n{body}\n{fence}"


def _format_api_turns(api_turns: list[dict[str, str]]) -> str:
    if not api_turns:
        return "*(нет предыдущих ходов в ChatSession)*\n"
    lines: list[str] = []
    for turn in api_turns:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        display_role = "MODEL" if role in ("model", "assistant", "tutor") else "USER"
        lines.append(f"- **Role: {display_role}**")
        lines.append(_md_fence(content))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _trace_path(session_id: str, turn_number: int, chat_label: str) -> Path:
    from knowledge_engine.config import PROMPT_TRACE_DIR

    slug = _label_slug(chat_label)
    fname = f"turn_{turn_number:02d}_{slug}_exchange.md"
    return PROMPT_TRACE_DIR / session_id / fname


def write_session_prompt_trace(
    *,
    session_id: str,
    turn_number: int,
    chat_label: str,
    model_name: str,
    system_instruction: str,
    user_payload: str,
    api_turns: list[dict[str, str]],
    dialog_user_text: str = "",
    trace_context: PromptTraceContext | None = None,
) -> PromptTraceRef | None:
    """
    Markdown-дамп входа перед LLM-вызовом. Выход — append_session_prompt_output /
    trace_gemini_io_sizes (pending ref).
    """
    norm_label = normalize_trace_label(chat_label)
    if not should_trace_label(norm_label):
        return None
    sid = (session_id or "").strip()
    if not sid:
        return None

    try:
        pass

        system = system_instruction or ""
        payload = user_payload or ""
        est_in = estimate_llm_tokens(system, model_name) + estimate_llm_tokens(
            payload, model_name
        )
        total_sym = len(system) + len(payload)
        metrics = (
            trace_context.metrics
            if trace_context and trace_context.metrics
            else PromptTraceMetrics()
        )
        node_key = (trace_context.node_session_key if trace_context else "") or ""
        cache_meta = trace_context.explicit_cache if trace_context else None
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        header_lines = [
            f"# Session Trace: {sid} | Turn: {turn_number}",
            f"**Timestamp:** {ts}",
            f"**Chat label:** {norm_label}",
        ]
        if node_key:
            header_lines.append(f"**Node session:** {node_key}")
        if cache_meta and (cache_meta.explicit_cache_mode or cache_meta.cache_name):
            header_lines.extend(
                [
                    "",
                    "## 0. CACHE METADATA",
                    f"**Explicit cache:** {cache_meta.explicit_cache_mode or '—'}"
                    + (f" | `{cache_meta.cache_name}`" if cache_meta.cache_name else "")
                    + (
                        f" | digest=`{cache_meta.digest[:16]}…`"
                        if cache_meta.digest
                        else ""
                    ),
                    f"**Layer1 omitted from payload:** "
                    f"{'yes' if cache_meta.layer1_omitted else 'no'}",
                    f"**Layer2 in request:** "
                    f"{'yes' if cache_meta.layer2_in_request else 'no'}"
                    + (
                        f" ({cache_meta.layer2_bytes} bytes)"
                        if cache_meta.layer2_in_request
                        else ""
                    ),
                    "",
                ]
            )
        header_lines.extend(
            [
                f"**Model:** {model_name}",
                f"**Total Estimated Input Tokens:** {est_in}",
                f"**Raw Input Length:** {total_sym} chars",
                "",
                "---",
                "",
                f"## 1. SYSTEM INSTRUCTION (Len: {len(system)} sym)",
                "",
                _md_fence(system),
                "",
                "---",
                "",
                "## 2. API TURNS (Native Gemini History)",
                "*(История до текущего запроса; текущий user payload — в секции 3)*",
                "",
                _format_api_turns(list(api_turns or [])),
                "",
                "---",
                "",
                f"## 3. CURRENT USER PAYLOAD / DELTA (Len: {len(payload)} sym)",
                "",
                _md_fence(payload),
                "",
            ]
        )
        dialog_only = (dialog_user_text or "").strip()
        if dialog_only and dialog_only != payload.strip():
            header_lines.extend(
                [
                    f"### Dialog-only payload (Len: {len(dialog_only)} sym)",
                    "",
                    _md_fence(dialog_only),
                    "",
                ]
            )
        header_lines.extend(
            [
                "---",
                "",
                "## 4. METRICS BREAKDOWN",
                f"- Pinned Context Length: {metrics.pinned_len} sym",
                f"- Behavior State Length: {metrics.behavior_state_len} sym",
                f"- Recency Tail Length: {metrics.recency_tail_len} sym",
                f"- User Message Length: {metrics.user_message_len} sym",
                "",
                "---",
                "",
                "## 5. MODEL OUTPUT",
                "",
                "*(ожидание ответа модели…)*",
                "",
            ]
        )
        content = "\n".join(header_lines)

        out_path = _trace_path(sid, turn_number, norm_label)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(out_path)
        ref = PromptTraceRef(
            path=out_path,
            session_id=sid,
            turn_number=turn_number,
            chat_label=norm_label,
        )
        _pending_trace.set(ref)
        return ref
    except Exception as exc:
        _log.warning(
            "session_prompt_trace: не удалось записать %s: %s",
            chat_label,
            exc,
            exc_info=True,
        )
        return None


def write_stateless_gemini_prompt_trace(
    *,
    trace_label: str,
    model_name: str,
    system_instruction: str,
    user_payload: str,
) -> PromptTraceRef | None:
    """Одиночный stateless-вызов (generate_content без ChatSession)."""
    if not should_trace_label(trace_label):
        return None
    sid = f"call-{uuid.uuid4().hex[:12]}"
    return write_session_prompt_trace(
        session_id=sid,
        turn_number=1,
        chat_label=trace_label,
        model_name=model_name,
        system_instruction=system_instruction,
        user_payload=user_payload,
        api_turns=[],
    )


def append_session_prompt_output(
    ref: PromptTraceRef | None,
    output_text: str,
    *,
    finish_reason: str = "",
    model_name: str = "",
) -> None:
    if ref is None or not ref.path.is_file():
        _pending_trace.set(None)
        return
    try:
        body = output_text or ""
        est_out = estimate_llm_tokens(body, model_name)
        lines = [
            f"**Response timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        if model_name:
            lines.append(f"**Model:** {model_name}")
        lines.append(f"**Estimated Output Tokens:** {est_out}")
        lines.append(f"**Output Length:** {len(body)} sym")
        if finish_reason:
            lines.append(f"**Finish reason:** {finish_reason}")
        lines.extend(["", _md_fence(body), ""])
        section = "\n".join(lines)

        existing = ref.path.read_text(encoding="utf-8")
        marker = "## 5. MODEL OUTPUT"
        if marker in existing:
            head, _sep, _tail = existing.partition(marker)
            new_content = f"{head.rstrip()}\n\n{marker}\n\n{section}"
        else:
            new_content = f"{existing.rstrip()}\n\n---\n\n{marker}\n\n{section}"
        tmp = ref.path.with_suffix(".md.tmp")
        tmp.write_text(new_content, encoding="utf-8")
        tmp.replace(ref.path)
    except Exception as exc:
        _log.warning(
            "session_prompt_trace: не удалось дописать output %s: %s",
            ref.path,
            exc,
            exc_info=True,
        )
    finally:
        _pending_trace.set(None)


def consume_pending_trace_output(
    output_text: str,
    *,
    finish_reason: str = "",
    model_name: str = "",
) -> None:
    """Вызывается из trace_gemini_io_sizes после ответа модели."""
    ref = _pending_trace.get()
    if ref is None:
        return
    append_session_prompt_output(
        ref,
        output_text,
        finish_reason=finish_reason,
        model_name=model_name,
    )


def log_local_llm_exchange(
    *,
    trace_label: str,
    model_name: str,
    system_instruction: str,
    user_payload: str,
    output_text: str,
) -> None:
    """Fallback Ollama / local structured — один файл in+out."""
    if not should_trace_label(trace_label):
        return
    sid = f"local-{uuid.uuid4().hex[:12]}"
    ref = write_session_prompt_trace(
        session_id=sid,
        turn_number=1,
        chat_label=trace_label,
        model_name=model_name,
        system_instruction=system_instruction,
        user_payload=user_payload,
        api_turns=[],
    )
    append_session_prompt_output(ref, output_text, model_name=model_name)
