"""Очистка JSON из ответов Gemini и извлечение текстового поля из стрима."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
JSON_CLEAN_PATTERN = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*\})\s*```|(\{[\s\S]*\})",
    re.DOTALL,
)


def _extract_balanced_json_object(raw: str, start: int) -> str:
    """Первый JSON-объект с учётом строк (не обрезать по `}` внутри текста)."""
    if start < 0 or start >= len(raw):
        return ""
    depth = 0
    in_str = False
    esc = False
    i = start
    n = len(raw)
    while i < n:
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
        i += 1
    return raw[start:]


def extract_clean_json(text: str) -> str:
    """Убрать markdown-обёртку и посторонний текст вокруг JSON-объекта."""
    raw = (text or "").strip()
    if not raw:
        return ""
    match = JSON_CLEAN_PATTERN.search(raw)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    m = JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    if start >= 0:
        return _extract_balanced_json_object(raw, start).strip()
    return raw


def _decode_json_string(raw: str, start: int) -> tuple[str, bool]:
    """Decode a JSON string value starting after the opening quote.

    Returns (decoded_text, complete) where complete is True iff the closing
    quote was seen in ``raw``.
    """
    out: list[str] = []
    i = start
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            return "".join(out), True
        if ch == "\\" and i + 1 < n:
            esc = raw[i + 1]
            mapping = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                '"': '"',
                "\\": "\\",
                "/": "/",
            }
            if esc in mapping:
                out.append(mapping[esc])
                i += 2
                continue
            if esc == "u" and i + 5 < n:
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), False


def partial_json_string_field_state(raw: str, field: str) -> tuple[str, str]:
    """Partial JSON string field: (value, status).

    status is ``missing`` (key not in buffer yet), ``partial`` (opened, no
    closing quote), or ``complete``.
    """
    pat = re.compile(rf'"{re.escape(field)}"\s*:\s*"', re.IGNORECASE)
    m = pat.search(raw or "")
    if not m:
        return "", "missing"
    value, complete = _decode_json_string(raw, m.end())
    return value, "complete" if complete else "partial"


def partial_json_string_field(raw: str, field: str) -> str:
    """Текст поля из неполного JSON (стрим)."""
    value, _status = partial_json_string_field_state(raw, field)
    return value


def resolved_json_field_text(raw: str, field: str) -> str:
    """Сначала полный parse, иначе partial string extractor."""
    cleaned = extract_clean_json(raw)
    if cleaned:
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and field in obj:
                val = obj.get(field)
                return str(val) if val is not None else ""
        except json.JSONDecodeError:
            pass
    return partial_json_string_field(raw, field)


def streaming_json_field_text(raw: str, field: str) -> str:
    """Только partial decode — без json.loads (стрим может «скакать» при parse)."""
    return partial_json_string_field(raw, field)


class JsonFieldStreamFilter:
    """Пробрасывает в UI только дельты текстового поля структурированного JSON."""

    def __init__(self, field: str, on_delta: Callable[[str], None]) -> None:
        self._field = (field or "").strip()
        self._on_delta = on_delta
        self._buffer = ""
        self._emitted_value = ""

    def _emit_value_delta(self, value: str) -> None:
        prev = self._emitted_value
        if not value:
            return
        if value.startswith(prev):
            delta = value[len(prev) :]
        elif prev.startswith(value):
            return
        else:
            common = 0
            lim = min(len(prev), len(value))
            while common < lim and prev[common] == value[common]:
                common += 1
            delta = value[common:]
        if delta:
            self._on_delta(delta)
        self._emitted_value = value

    def feed(self, chunk: str) -> None:
        if not chunk or not self._field:
            return
        self._buffer += chunk
        value = streaming_json_field_text(self._buffer, self._field)
        self._emit_value_delta(value)

    def flush(self) -> None:
        if not self._field:
            return
        value = streaming_json_field_text(self._buffer, self._field)
        self._emit_value_delta(value)

    @property
    def buffer(self) -> str:
        return self._buffer


TUTOR_DIALOGUE_STREAM_FIELDS: tuple[str, ...] = (
    "confirmation",
    "correction_breakdown",
    "technical_explanation",
    "follow_up_question",
)

TUTOR_EXPLAIN_STREAM_FIELDS: tuple[str, ...] = (
    "technical_explanation",
    "follow_up_question",
)

DRILL_ACTIVE_STREAM_FIELDS: tuple[str, ...] = (
    "status_header",
    "confirmation",
    "correction_breakdown",
    "theory_body",
    "next_question",
)

DRILL_COMPLETE_STREAM_FIELDS: tuple[str, ...] = (
    "praise",
    "layer_summary",
    "transition_framing",
)

# Mutually exclusive audit branches — skip if omitted/empty, never block later fields.
_OPTIONAL_STREAM_FIELDS = frozenset({"confirmation", "correction_breakdown"})


class TutorDialogueFieldsStreamFilter:
    """Стримит в UI склейку полей structured JSON в порядке UI, не JSON.

    Gemini emits nested ``audit.confirmation`` before ``status_header``. If we
    joined whatever keys are already visible, SSE deltas (append-only) would
    glue confirmation+header and then reprint confirmation after the header
    lands. Compose only a prefix-stable layout: wait for earlier required
    fields to finish before appending later ones.
    """

    def __init__(
        self,
        on_delta: Callable[[str], None],
        fields: tuple[str, ...] | None = None,
    ) -> None:
        self._on_delta = on_delta
        self._fields = fields or TUTOR_DIALOGUE_STREAM_FIELDS
        self._buffer = ""
        self._emitted_value = ""

    def _format_field(self, field: str, value: str) -> str:
        text = (value or "").strip()
        if not text:
            return ""
        if field == "next_question" and not text.startswith("**Вопрос:**"):
            return f"**Вопрос:** {text}"
        return text

    def _compose_partial(self) -> str:
        parts: list[str] = []
        for field in self._fields:
            raw_value, status = partial_json_string_field_state(self._buffer, field)
            optional = field in _OPTIONAL_STREAM_FIELDS
            if status == "missing":
                if optional:
                    continue
                break
            formatted = self._format_field(field, raw_value)
            if status == "partial":
                if formatted:
                    parts.append(formatted)
                break
            if formatted:
                parts.append(formatted)
        return "\n\n".join(parts)

    def _emit_value_delta(self, value: str) -> None:
        prev = self._emitted_value
        if not value:
            return
        if value.startswith(prev):
            delta = value[len(prev) :]
        elif prev.startswith(value):
            return
        else:
            common = 0
            lim = min(len(prev), len(value))
            while common < lim and prev[common] == value[common]:
                common += 1
            delta = value[common:]
        if delta:
            self._on_delta(delta)
        self._emitted_value = value

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk
        self._emit_value_delta(self._compose_partial())

    def flush(self) -> None:
        self._emit_value_delta(self._compose_partial())


def wrap_stream_callback_for_tutor_dialogue_fields(
    stream_callback: Callable[[str], None] | None,
    fields: tuple[str, ...] | None = None,
) -> TutorDialogueFieldsStreamFilter | None:
    if not stream_callback:
        return None
    return TutorDialogueFieldsStreamFilter(stream_callback, fields=fields)


def wrap_stream_callback_for_json_field(
    field: str,
    stream_callback: Callable[[str], None] | None,
) -> JsonFieldStreamFilter | None:
    if not stream_callback or not (field or "").strip():
        return None
    return JsonFieldStreamFilter(field.strip(), stream_callback)


def structured_stream_text_field(response_schema: type | None) -> str | None:
    if response_schema is None:
        return None
    name = getattr(response_schema, "__name__", "") or ""
    if name == "DenseMaterialOutput" or name == "StructuredLectureResponse":
        return "lecture_body"
    if name in (
        "DeepDiveTutorContract",
        "DeepDiveDeepAnalysisContract",
        "DeepDiveExplainContract",
    ):
        return None
    if name == "ActiveDrillStepResponse":
        return None
    if name == "LayerCompletionTutorOutput":
        return None
    if name in ("DeepDiveLLMOutput", "IntroAssessmentOutput"):
        return "tutor_message"
    fields = getattr(response_schema, "model_fields", None) or {}
    if "lecture_body" in fields:
        return "lecture_body"
    if "explanation" in fields:
        return "explanation"
    if "tutor_message" in fields:
        return "tutor_message"
    return None
