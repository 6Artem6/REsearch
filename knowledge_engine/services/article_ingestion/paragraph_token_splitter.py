"""Токеновая нарезка Annotated Markdown по целым блокам [P_n] / [FIG_n]."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowledge_engine.config import (
    BLOG_SPATIAL_MAP_MAX_TOKENS,
    BLOG_SPATIAL_MAP_USER_OVERHEAD_TOKENS,
    BLOG_SPATIAL_OVERLAP_TOKENS,
)

if TYPE_CHECKING:
    from knowledge_engine.services.article_ingestion.figure_registry_service import (
        FigureRegistry,
    )
    from knowledge_engine.services.article_ingestion.triage_schemas import TOCNode

_BLOCK_START_RE = re.compile(r"^\[(P_\d+|FIG(?:_SEQ)?_\d+[^\]]*)\]", re.M)
_FIG_TAG_RE = re.compile(r"\[(FIG(?:_SEQ)?_\d+)", re.I)
_TIKTOKEN_SAFETY_FACTOR = 1.15

_QWEN_TOKENIZER: object | None = None
_QWEN_TOKENIZER_TRIED = False

_QWEN_HF_IDS = (
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
)


def _load_qwen_tokenizer() -> object | None:
    global _QWEN_TOKENIZER, _QWEN_TOKENIZER_TRIED
    if _QWEN_TOKENIZER_TRIED:
        return _QWEN_TOKENIZER
    _QWEN_TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    for model_id in _QWEN_HF_IDS:
        try:
            _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            return _QWEN_TOKENIZER
        except Exception:
            continue
    return None


def _count_tokens(text: str) -> int:
    raw = text or ""
    if not raw:
        return 0
    tok = _load_qwen_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(raw))
        except Exception:
            pass
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, int(len(enc.encode(raw)) * _TIKTOKEN_SAFETY_FACTOR))
    except Exception:
        return max(1, int(len(raw) // 4 * _TIKTOKEN_SAFETY_FACTOR))


@dataclass
class TokenWindowChunk:
    window_index: int
    body: str
    paragraph_ids: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    attached_diagrams: str = ""
    section_heading: str = ""


_LEGACY_ANCHOR_PREFIX_RE = re.compile(r"^\[ANCHOR:[^\]]*\]\n?")
_ORDINAL_ANCHOR_PREFIX_RE = re.compile(r"^\[A\d+\]\n?")


def strip_context_anchor_prefix(body: str) -> str:
    """Снять ``[A{i}]`` / legacy ``[ANCHOR: …]`` с первой строки (для перенумерации)."""
    text = (body or "").lstrip("\n")
    text = _LEGACY_ANCHOR_PREFIX_RE.sub("", text, count=1)
    text = _ORDINAL_ANCHOR_PREFIX_RE.sub("", text, count=1)
    return text


def maybe_prepend_chunk_anchor(body: str, ordinal: int) -> str:
    """Opt-in паспорт ``[A{i}]`` в начало тела чанка (i — номер в текущем контексте).

    При ``CHUNK_ANCHOR_INJECTION=false`` возвращает ``body`` без изменений.
    Повторный вызов идемпотентен для того же ordinal.
    """
    from knowledge_engine import config as ke_config

    if not ke_config.CHUNK_ANCHOR_INJECTION:
        return body
    try:
        idx = int(ordinal)
    except (TypeError, ValueError):
        return body or ""
    if idx < 1:
        return body or ""
    marker = f"[A{idx}]"
    stripped = strip_context_anchor_prefix(body or "")
    return f"{marker}\n{stripped}"


def build_anchor_index_map(
    windows: list[TokenWindowChunk],
    *,
    url: str,
) -> dict[str, dict[str, object]]:
    """Локальный ``index_map``: ``A1`` → ``chunk_id`` ``{doc_id}_map_{n}`` + позиция."""
    from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
    from knowledge_engine.services.vector_store import VectorStore

    index_map: dict[str, dict[str, object]] = {}
    if not windows:
        return index_map
    doc_id = VectorStore.doc_id_for_url(url)
    for position, win in enumerate(windows, start=1):
        key = f"A{position}"
        cid = map_window_chunk_id(doc_id, win.window_index)
        index_map[key] = {
            "chunk_id": cid,
            "window_index": win.window_index,
            "position": position,
        }
    return index_map


def ordinal_for_window(
    window_index: int,
    index_map: dict[str, dict[str, object]] | None,
) -> int:
    """1-based ordinal в текущем ``index_map``; иначе ``window_index + 1``."""
    for key, meta in (index_map or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            if int(meta.get("window_index", -1)) != int(window_index):
                continue
        except (TypeError, ValueError):
            continue
        pos = meta.get("position")
        try:
            if pos is not None:
                return int(pos)
        except (TypeError, ValueError):
            pass
        if str(key).startswith("A") and str(key)[1:].isdigit():
            return int(str(key)[1:])
    try:
        return int(window_index) + 1
    except (TypeError, ValueError):
        return 1


def apply_chunk_anchors_to_windows(
    windows: list[TokenWindowChunk],
    *,
    url: str,
) -> dict[str, dict[str, object]]:
    """Собрать ``index_map``; при флаге препендить ``[A{{i}}]`` к ``body``."""
    from knowledge_engine import config as ke_config

    index_map = build_anchor_index_map(windows, url=url)
    if not ke_config.CHUNK_ANCHOR_INJECTION or not windows:
        return index_map
    for key, meta in index_map.items():
        try:
            win_idx = int(meta.get("window_index", -1))
            position = int(meta.get("position") or int(str(key)[1:]))
        except (TypeError, ValueError):
            continue
        for win in windows:
            if win.window_index == win_idx:
                win.body = maybe_prepend_chunk_anchor(win.body, position)
                break
    return index_map


class ParagraphTokenSplitter:
    def split_annotated_text_by_tokens(
        self,
        annotated_markdown: str,
        *,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        parent_header: str = "",
        all_figure_tags: list[str] | None = None,
        figure_registry: FigureRegistry | None = None,
        map_user_overhead_tokens: int | None = None,
        toc_nodes: list[TOCNode] | None = None,
        paragraph_map: dict[str, str] | None = None,
    ) -> list[TokenWindowChunk]:
        from knowledge_engine.services.article_ingestion.map_diagram_attach import (
            build_attached_diagrams_block,
        )
        from knowledge_engine.services.article_ingestion.section_context import (
            resolve_section_heading_for_paragraph_ids,
        )

        max_t = max_tokens or BLOG_SPATIAL_MAP_MAX_TOKENS
        overlap_t = overlap_tokens or BLOG_SPATIAL_OVERLAP_TOKENS
        user_overhead = (
            map_user_overhead_tokens
            if map_user_overhead_tokens is not None
            else BLOG_SPATIAL_MAP_USER_OVERHEAD_TOKENS
        )
        blocks = _split_into_blocks(annotated_markdown)
        if not blocks:
            return []

        header = (parent_header or "").strip()
        fig_tags = all_figure_tags or []
        toc = list(toc_nodes or [])
        para_map = paragraph_map or {}

        def _compose_body(chunk_blocks: list[str]) -> str:
            body_parts: list[str] = []
            if header:
                body_parts.append(header)
            if fig_tags:
                body_parts.append(
                    "Available figures in document: " + ", ".join(fig_tags[:40])
                )
            body_parts.extend(chunk_blocks)
            return "\n\n".join(body_parts)

        def _ids_from_blocks(chunk_blocks: list[str]) -> tuple[list[str], list[str]]:
            chunk_p: list[str] = []
            chunk_f: list[str] = []
            for blk in chunk_blocks:
                _collect_ids(blk, chunk_p, chunk_f)
            return list(dict.fromkeys(chunk_p)), list(dict.fromkeys(chunk_f))

        def _attached_for(body: str, figure_ids: list[str]) -> str:
            if figure_registry is None:
                return ""
            return build_attached_diagrams_block(
                body,
                figure_registry,
                extra_figure_ids=figure_ids,
            )

        def _user_prompt_tokens(body: str, figure_ids: list[str]) -> tuple[int, str]:
            attached = _attached_for(body, figure_ids)
            total = _count_tokens(body) + _count_tokens(attached) + user_overhead
            return total, attached

        windows: list[TokenWindowChunk] = []
        i = 0
        win_idx = 0
        while i < len(blocks):
            chunk_blocks: list[str] = []
            j = i
            while j < len(blocks):
                blk = blocks[j]
                trial_blocks = chunk_blocks + [blk]
                trial_body = _compose_body(trial_blocks)
                trial_p, trial_f = _ids_from_blocks(trial_blocks)
                total, _ = _user_prompt_tokens(trial_body, trial_f)
                if chunk_blocks and total > max_t:
                    break
                if not chunk_blocks and total > max_t:
                    chunk_blocks = trial_blocks
                    j += 1
                    break
                chunk_blocks = trial_blocks
                j += 1

            if not chunk_blocks:
                break

            while len(chunk_blocks) > 1:
                body = _compose_body(chunk_blocks)
                chunk_p, chunk_f = _ids_from_blocks(chunk_blocks)
                total, _ = _user_prompt_tokens(body, chunk_f)
                if total <= max_t:
                    break
                chunk_blocks.pop()

            body = _compose_body(chunk_blocks)
            chunk_p, chunk_f = _ids_from_blocks(chunk_blocks)
            total, attached = _user_prompt_tokens(body, chunk_f)
            section = resolve_section_heading_for_paragraph_ids(
                chunk_p,
                toc,
                paragraph_map=para_map,
            )

            windows.append(
                TokenWindowChunk(
                    window_index=win_idx,
                    body=body,
                    paragraph_ids=chunk_p,
                    figure_ids=chunk_f,
                    attached_diagrams=attached,
                    section_heading=section,
                )
            )
            win_idx += 1

            if j >= len(blocks):
                break

            overlap_used = 0
            k = j - 1
            while k >= i and overlap_used < overlap_t:
                overlap_used += _count_tokens(blocks[k])
                k -= 1
            if k >= i:
                i = k + 1
            else:
                i = j

        return windows


def _split_into_blocks(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    matches = list(_BLOCK_START_RE.finditer(raw))
    if not matches:
        return [raw]
    blocks: list[str] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        blk = raw[start:end].strip()
        if blk:
            blocks.append(blk)
    return blocks


def _collect_ids(block: str, p_ids: list[str], f_ids: list[str]) -> None:
    for m in re.finditer(r"\[(P_\d+)\]", block):
        p_ids.append(m.group(1))
    for m in _FIG_TAG_RE.finditer(block):
        f_ids.append(m.group(1).upper())


def split_annotated_text_by_tokens(
    annotated_markdown: str,
    *,
    max_tokens: int = BLOG_SPATIAL_MAP_MAX_TOKENS,
    overlap_tokens: int = BLOG_SPATIAL_OVERLAP_TOKENS,
    title: str = "",
    all_figure_ids: list[str] | None = None,
    figure_registry: FigureRegistry | None = None,
    map_user_overhead_tokens: int | None = None,
    toc_nodes: list | None = None,
    paragraph_map: dict[str, str] | None = None,
) -> list[TokenWindowChunk]:
    header = f"Article: {(title or '').strip()[:300]}" if title else ""
    fig_tags = all_figure_ids or []
    return ParagraphTokenSplitter().split_annotated_text_by_tokens(
        annotated_markdown,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        parent_header=header,
        all_figure_tags=[f"[{fid}]" for fid in fig_tags],
        figure_registry=figure_registry,
        map_user_overhead_tokens=map_user_overhead_tokens,
        toc_nodes=toc_nodes,
        paragraph_map=paragraph_map,
    )


def estimate_text_tokens(text: str) -> int:
    """Оценка токенов для rate limit (Qwen HF / tiktoken ×1.15)."""
    return _count_tokens(text)
