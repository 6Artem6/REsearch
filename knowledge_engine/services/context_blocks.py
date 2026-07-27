"""Блочный контекст: парсинг MD, галочки, сборка payload без перегенерации текста."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from knowledge_engine.config import (
    GEMINI_PAYLOAD_MAX_CHARS,
    SUMMARIZER_MAX_PROFILE_CHARS,
    USER_PROFILE_PATH,
)
from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE
from knowledge_engine.schemas import ContextBlock, DocumentSummary, EngineState

_SLUG_RE = re.compile(r"[^\w\-]+", re.UNICODE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_ARTIFACT_MARKERS = (
    "gemini-heavy",
    "gemini-research-bundle",
    "gemini-research",
)
_JUNK_MARKERS = (
    "svg icon",
    "how-to-get-help",
    "support.microsoft.com",
    "windows/how-to",
)
_MATRIX_FACT_RE = re.compile(
    r"^(Классика|SOTA|Минимализм)\s*\(",
    re.IGNORECASE,
)
_PROFILE_NOISE_MARKERS = (
    "yandex practicum",
    "practicum",
    "moodle",
    "wordpress",
    "kaspi",
    "e-commerce",
    "ценообразования",
)
_ABSTRACTION_WATER_MARKERS = (
    "можно использовать",
    "для решения задачи",
    "это позволит",
    "будет состоять",
    "будут выполняться",
)


def _slug(text: str, max_len: int = 48) -> str:
    s = _SLUG_RE.sub("-", text.strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] or "block"


def _url_block_id(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"source:{digest}"


def _is_empty_source(summary: DocumentSummary) -> bool:
    takeaways = [t.strip() for t in summary.key_takeaways if t and t.strip()]
    failures = [f.strip() for f in summary.failure_modes if f and f.strip()]
    concepts = [c.strip() for c in summary.cs_concepts if c and c.strip()]
    return not takeaways and not failures and not concepts


def _is_artifact_source(summary: DocumentSummary) -> bool:
    blob = f"{summary.url} {summary.title}".lower()
    return any(m in blob for m in _ARTIFACT_MARKERS)


def _is_likely_junk_source(summary: DocumentSummary) -> bool:
    blob = f"{summary.url} {summary.title}".lower()
    return any(m in blob for m in _JUNK_MARKERS)


def _source_hints(summary: DocumentSummary) -> tuple[list[str], bool]:
    hints: list[str] = []
    default_include = True
    if _is_empty_source(summary):
        hints.append("empty_source")
        default_include = False
    if _is_artifact_source(summary):
        hints.append("artifact")
        default_include = False
    if _is_likely_junk_source(summary):
        hints.append("likely_junk")
        default_include = False
    return hints, default_include


def _fact_hints(fact: str) -> tuple[list[str], bool]:
    hints: list[str] = []
    default_include = True
    if _MATRIX_FACT_RE.match(fact.strip()):
        hints.append("matrix_strawman")
        default_include = False
    lower = fact.lower()
    if "unified memory" in lower and "apple silicon" in lower:
        hints.append("en_takeaway_duplicate")
        default_include = False
    if (
        "cache invalidation" in lower
        and "stale reads" in lower
        and "semantic cache" in lower
    ):
        hints.append("en_takeaway_duplicate")
        default_include = False
    return hints, default_include


def _abstraction_hints(description: str) -> tuple[list[str], bool]:
    lower = description.lower()
    hits = sum(1 for m in _ABSTRACTION_WATER_MARKERS if m in lower)
    if hits >= 2:
        return ["textbook_water"], False
    return [], True


def _profile_section_hints(block_id: str, content: str) -> tuple[list[str], bool]:
    lower = content.lower()
    if block_id == "profile:введение":
        return ["meta_only"], False
    if block_id == "profile:личные-данные-и-стек":
        if any(m in lower for m in _PROFILE_NOISE_MARKERS):
            return ["profile_background_noise"], False
    if block_id == "profile:проекты-в-разработке":
        return ["optional_projects"], False
    if block_id == "profile:аппаратное-окружение-hardware-ecosystem":
        return [], True
    if block_id == "profile:критерии-отбора-решений-knowledge-engine-rules":
        return [], True
    return [], True


def _clean_profile_preamble(chunk: str) -> str:
    chunk = _HTML_COMMENT_RE.sub("", chunk)
    lines: list[str] = []
    for line in chunk.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue
        if stripped:
            lines.append(line)
    return "\n".join(lines).strip()


def parse_markdown_profile_blocks(profile_text: str) -> list[ContextBlock]:
    """Разбить user_profile.md на блоки по ## (без дублирования всего файла)."""
    text = _HTML_COMMENT_RE.sub("", profile_text[:SUMMARIZER_MAX_PROFILE_CHARS])
    blocks: list[ContextBlock] = []
    sections = re.split(r"(?=^## )", text.strip(), flags=re.MULTILINE)
    for raw in sections:
        chunk = raw.strip()
        if not chunk:
            continue
        if chunk.startswith("## "):
            title_line, _, body = chunk.partition("\n")
            title = title_line[3:].strip()
            content = body.strip()
        else:
            title = "Введение"
            content = _clean_profile_preamble(chunk)
            if not content:
                continue
        if not content:
            continue
        block_id = f"profile:{_slug(title)}"
        hints, default_include = _profile_section_hints(block_id, content)
        blocks.append(
            ContextBlock(
                block_id=block_id,
                kind="profile",
                title=title,
                content=content,
                always_include=False,
                default_include=default_include,
                hints=hints,
            )
        )
    return blocks


def load_profile_blocks() -> list[ContextBlock]:
    path = Path(USER_PROFILE_PATH)
    if not path.is_file():
        return [
            ContextBlock(
                block_id="profile:missing",
                kind="profile",
                title="Профиль",
                content="(user_profile.md не найден)",
                always_include=False,
                default_include=False,
            )
        ]
    return parse_markdown_profile_blocks(path.read_text(encoding="utf-8"))


def _format_source_content(summary: DocumentSummary) -> str:
    return (
        f"- [{summary.title}] {summary.url}\n"
        f"  CS: {', '.join(summary.cs_concepts[:6])}\n"
        f"  Takeaways: {'; '.join(summary.key_takeaways[:5])}\n"
        f"  Failure modes: {'; '.join(summary.failure_modes[:4])}"
    ).strip()


def build_context_blocks(state: EngineState) -> list[ContextBlock]:
    """Стабильные блоки из профиля (MD), LanceDB summaries и state — без LLM."""
    blocks: list[ContextBlock] = []

    system_text = (
        f"{GEMINI_RUSSIAN_ROLE} "
        "Синтез + Trade-off матрица (Классика / SOTA / Минимализм), failure modes, RAM/latency на Apple Silicon."
    )
    blocks.append(
        ContextBlock(
            block_id="system:role",
            kind="system",
            title="System Role",
            content=system_text,
            always_include=False,
            default_include=True,
        )
    )

    blocks.extend(load_profile_blocks())

    for summary in state.found_summaries:
        content = _format_source_content(summary)
        hints, default_include = _source_hints(summary)
        blocks.append(
            ContextBlock(
                block_id=_url_block_id(summary.url),
                kind="source",
                title=summary.title or summary.url[:80],
                content=content,
                always_include=False,
                default_include=default_include,
                hints=hints,
            )
        )

    for idx, fact in enumerate(state.found_facts[:24]):
        text = fact.strip()
        if not text:
            continue
        hints, default_include = _fact_hints(text)
        blocks.append(
            ContextBlock(
                block_id=f"fact:{idx}:{_slug(text[:36])}",
                kind="fact",
                title=f"Факт #{idx + 1}",
                content=f"- {text}",
                always_include=False,
                default_include=default_include,
                hints=hints,
            )
        )

    for idx, abs_item in enumerate(state.abstractions):
        line = (
            f"- {abs_item.title} ({abs_item.cs_concept}): {abs_item.description[:200]}"
        )
        hints, default_include = _abstraction_hints(abs_item.description)
        blocks.append(
            ContextBlock(
                block_id=f"abstraction:{idx}:{_slug(abs_item.title)}",
                kind="abstraction",
                title=abs_item.title,
                content=line,
                always_include=False,
                default_include=default_include,
                hints=hints,
            )
        )

    dialogue = (state.dialogue_rolling_summary or "").strip()
    if dialogue and dialogue != "(нет уточняющего диалога)":
        blocks.append(
            ContextBlock(
                block_id="dialogue:rolling",
                kind="dialogue",
                title="История уточнений",
                content=dialogue,
                always_include=False,
                default_include=False,
                hints=["empty_dialogue"],
            )
        )

    task_text = (
        f"Задача: {state.user_problem}\n"
        f"Ограничения: {state.context_constraints or '(не указаны)'}"
    )
    blocks.append(
        ContextBlock(
            block_id="user_task:main",
            kind="user_task",
            title="Задача пользователя",
            content=task_text,
            always_include=True,
            default_include=True,
        )
    )

    return blocks


def apply_hard_hint_exclusions(
    blocks: list[ContextBlock],
    selections: dict[str, bool],
) -> dict[str, bool]:
    """Жёстко выключить блоки с hints (даже если SLM включил всё)."""
    out = dict(selections)
    force_off = {
        "empty_source",
        "artifact",
        "likely_junk",
        "matrix_strawman",
        "en_takeaway_duplicate",
        "textbook_water",
        "meta_only",
        "profile_background_noise",
        "empty_dialogue",
    }
    for b in blocks:
        if b.always_include:
            out[b.block_id] = True
            continue
        if any(h in force_off for h in b.hints):
            out[b.block_id] = False
    return out


def _include_block(block: ContextBlock, selections: dict[str, bool]) -> bool:
    if block.always_include:
        return True
    if block.block_id in selections:
        return bool(selections[block.block_id])
    return block.default_include


def assemble_gemini_payload(
    blocks: list[ContextBlock],
    selections: dict[str, bool],
    max_chars: int = GEMINI_PAYLOAD_MAX_CHARS,
) -> str:
    """Собрать Sandwich payload только из отмеченных блоков (без LLM)."""
    parts: list[str] = []

    system_blocks = [
        b for b in blocks if b.kind == "system" and _include_block(b, selections)
    ]
    if system_blocks:
        parts.append("[SYSTEM ROLE]\n" + "\n\n".join(b.content for b in system_blocks))

    profile_in = [
        b
        for b in blocks
        if b.kind == "profile"
        and b.block_id != "profile:введение"
        and _include_block(b, selections)
    ]
    if profile_in:
        profile_body = "\n\n".join(f"## {b.title}\n{b.content}" for b in profile_in)
        parts.append("[DEVELOPER PROFILE]\n" + profile_body)

    source_in = [
        b for b in blocks if b.kind == "source" and _include_block(b, selections)
    ]
    fact_in = [b for b in blocks if b.kind == "fact" and _include_block(b, selections)]
    abs_in = [
        b for b in blocks if b.kind == "abstraction" and _include_block(b, selections)
    ]
    if source_in or fact_in or abs_in:
        section_parts: list[str] = []
        if source_in:
            section_parts.append("\n\n".join(b.content for b in source_in))
        if fact_in:
            section_parts.append(
                "Доп. факты:\n" + "\n".join(b.content for b in fact_in)
            )
        if abs_in:
            section_parts.append(
                "CS-абстракции (SLM/7B):\n" + "\n".join(b.content for b in abs_in)
            )
        parts.append(
            "[RELEVANT SOURCES & KNOWLEDGE (LANCEDB + SEARCH)]\n"
            + "\n\n".join(section_parts)
        )

    dialogue_in = [
        b for b in blocks if b.kind == "dialogue" and _include_block(b, selections)
    ]
    if dialogue_in:
        parts.append(
            "[CONCISE DIALOGUE HISTORY]\n" + "\n\n".join(b.content for b in dialogue_in)
        )

    task_in = [
        b for b in blocks if b.kind == "user_task" and _include_block(b, selections)
    ]
    if task_in:
        parts.append("[USER TASK]\n" + "\n\n".join(b.content for b in task_in))

    payload = "\n\n".join(parts)
    if len(payload) <= max_chars:
        return payload
    head = parts[0] if parts else ""
    tail = parts[-1] if parts else ""
    mid = "\n\n".join(parts[1:-1]) if len(parts) > 2 else ""
    budget = max_chars - len(head) - len(tail) - 4
    if budget < 0:
        return (head + "\n\n" + tail)[:max_chars]
    mid_trim = mid[:budget] + "\n…[truncated]"
    return head + "\n\n" + mid_trim + "\n\n" + tail


def blocks_from_state_dicts(items: list[dict]) -> list[ContextBlock]:
    return [ContextBlock.model_validate(x) for x in items]


def default_selections(blocks: list[ContextBlock]) -> dict[str, bool]:
    return {b.block_id: b.default_include for b in blocks}


def catalog_for_evaluator(blocks: list[ContextBlock], preview_chars: int = 320) -> str:
    """Короткий каталог для оценки — каждый block_id отдельно."""
    lines: list[str] = []
    for b in blocks:
        preview = b.content.replace("\n", " ")[:preview_chars]
        if b.always_include:
            flag = "always_include"
        else:
            flag = f"default_include={b.default_include}"
        hint_txt = f" hints={','.join(b.hints)}" if b.hints else ""
        lines.append(
            f"- id={b.block_id} | kind={b.kind} | {flag}{hint_txt} | {b.title}\n"
            f"  preview: {preview}"
        )
    return "\n".join(lines)
