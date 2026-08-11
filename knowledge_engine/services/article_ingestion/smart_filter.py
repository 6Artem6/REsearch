"""Pre-filter изображений: batch Ollama → Top-K для VLM."""

from __future__ import annotations

import asyncio
import json
import re
from typing import List

import httpx
from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import (
    ARTICLE_DIAGRAM_FILTER_NUM_CTX,
    ARTICLE_DIAGRAM_FILTER_NUM_PREDICT,
    ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL,
    ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC,
    ARTICLE_MAX_DIAGRAMS_PER_ARTICLE,
    OLLAMA_BASE_URL,
    SELECTION_PROMPTS_KEEP_ALIVE,
)
from knowledge_engine.services.ollama_runtime import ensure_ollama_server
from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.ui.run_log import trace

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

MAX_CAPTION_CHARS = 200
MAX_CONTEXT_CHARS = 400
MAX_BATCH_PROMPT_CHARS = 12000
MAX_ITEMS_PER_BATCH = 10
FALLBACK_MIN_BATCH_SIZE = 5

DIAGRAM_KEYWORDS = frozenset(
    {
        "схема",
        "диаграмма",
        "архитектура",
        "пайплайн",
        "pipeline",
        "architecture",
        "flow",
        "diagram",
        "структура",
        "модель",
        "граф",
        "benchmark",
        "бенчмарк",
        "qps",
        "recall",
        "latency",
        "throughput",
        "алгоритм",
        "chart",
        "figure",
        "quantization",
        "квантование",
        "fusion",
        "ranker",
        "вектор",
        "vector",
        "search",
        "поиск",
    }
)

_SYSTEM = (
    "You are a software architecture and technical documentation filter.\n"
    "Analyze the list of image context items (captions, surrounding text) "
    "from a technical article.\n\n"
    "Select items that represent:\n"
    "- Architecture diagrams, system designs, component flowcharts\n"
    "- Data structures, UML graphs, algorithm flow\n"
    "- Performance benchmark charts (QPS vs Recall, latency curves) — "
    "these are diagrams too; downstream VLM will convert them to xychart-beta, "
    "not flowcharts\n"
    "- Technical schemas or mathematical/search models\n\n"
    "Ignore strictly non-technical items:\n"
    "- Article covers, logos, author avatars, decorative UI photos, "
    "promotional banners.\n\n"
    "Output strictly valid JSON with no prose, following this schema:\n"
    '{"approved": [{"id": <candidate id from prompt>, "importance": 1-5}]}\n'
    "importance 5 = key diagram; omit ids that are not technical diagrams."
)


class ApprovedDiagram(BaseModel):
    id: int = Field(ge=0, le=500)
    importance: int = Field(default=3, ge=1, le=5)


class BatchFilterResult(BaseModel):
    approved: List[ApprovedDiagram] = Field(default_factory=list)

    @field_validator("approved", mode="before")
    @classmethod
    def _coerce_approved(cls, value: object) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return []


def _truncate_caption(text: str) -> str:
    return (text or "").strip()[:MAX_CAPTION_CHARS]


def _truncate_context(text: str) -> str:
    return (text or "").strip()[:MAX_CONTEXT_CHARS]


def _candidate_block(global_id: int, img: ExtractedImage) -> str:
    cap = _truncate_caption(img.caption)
    ctx = _truncate_context(img.context_text)
    return (
        f"--- candidate id={global_id} ---\n"
        f"caption: {cap or '(empty)'}\n"
        f"context: {ctx or '(empty)'}\n"
    )


def _fallback_heuristic_check(
    indices: list[int],
    images: list[ExtractedImage],
) -> list[ApprovedDiagram]:
    """Если Qwen отсеял всё — второй шанс по ключевым словам в caption/context."""
    recovered: list[ApprovedDiagram] = []
    for idx in indices:
        img = images[idx]
        text = (
            f"{_truncate_caption(img.caption)} "
            f"{_truncate_context(img.context_text)}"
        ).lower()
        if any(kw in text for kw in DIAGRAM_KEYWORDS):
            recovered.append(ApprovedDiagram(id=idx, importance=3))
    return recovered


def _split_ollama_batches(images: list[ExtractedImage]) -> list[list[int]]:
    if not images:
        return []
    groups: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    header_reserve = 200

    for idx in range(len(images)):
        block = _candidate_block(idx, images[idx])
        block_len = len(block)
        would_exceed_chars = (
            current_chars + block_len + header_reserve > MAX_BATCH_PROMPT_CHARS
        )
        would_exceed_count = len(current) >= MAX_ITEMS_PER_BATCH
        if current and (would_exceed_chars or would_exceed_count):
            groups.append(current)
            current = []
            current_chars = 0
        current.append(idx)
        current_chars += block_len

    if current:
        groups.append(current)
    return groups


def _parse_batch_result(text: str) -> BatchFilterResult | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return BatchFilterResult.model_validate(data)
    except Exception:
        return None


async def _ollama_batch_filter(
    indices: list[int],
    images: list[ExtractedImage],
) -> BatchFilterResult | None:
    if not indices:
        return BatchFilterResult()
    if not await ensure_ollama_server():
        return None

    lines = [
        "Select candidate ids suitable for architecture diagram / VLM extraction.",
        f"Candidates in this batch: {len(indices)}.",
        "",
    ]
    for idx in indices:
        lines.append(_candidate_block(idx, images[idx]))
    prompt = "\n".join(lines)
    trace(
        f"ARTICLE_SMART_FILTER ollama ▶ | batch_ids={len(indices)} "
        f"prompt_chars={len(prompt)} num_ctx={ARTICLE_DIAGRAM_FILTER_NUM_CTX}"
    )

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": ARTICLE_DIAGRAM_FILTER_OLLAMA_MODEL,
        "system": _SYSTEM,
        "prompt": prompt,
        "stream": False,
        "keep_alive": SELECTION_PROMPTS_KEEP_ALIVE,
        "options": {
            "num_predict": ARTICLE_DIAGRAM_FILTER_NUM_PREDICT,
            "num_ctx": ARTICLE_DIAGRAM_FILTER_NUM_CTX,
            "temperature": 0.05,
        },
    }
    timeout = httpx.Timeout(ARTICLE_DIAGRAM_FILTER_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    parsed = _parse_batch_result(str(data.get("response") or ""))
    if parsed is None:
        trace("ARTICLE_SMART_FILTER ollama ✗ | JSON parse failed")
    return parsed


def _apply_top_k(
    images: list[ExtractedImage],
    approved: list[ApprovedDiagram],
) -> list[ExtractedImage]:
    seen: set[int] = set()
    valid: list[ApprovedDiagram] = []
    for a in approved:
        if a.id < 0 or a.id >= len(images) or a.id in seen:
            continue
        seen.add(a.id)
        valid.append(a)
    valid.sort(key=lambda a: (-a.importance, a.id))
    cap = max(1, int(ARTICLE_MAX_DIAGRAMS_PER_ARTICLE))
    picked = valid[:cap]
    return [images[a.id] for a in picked]


async def _filter_all_batches(images: list[ExtractedImage]) -> list[ApprovedDiagram]:
    groups = _split_ollama_batches(images)
    if not groups:
        return []
    merged: list[ApprovedDiagram] = []
    for group in groups:
        result = await _ollama_batch_filter(group, images)
        batch_approved: list[ApprovedDiagram] = []
        if result is not None:
            batch_approved = list(result.approved)
        if len(batch_approved) == 0 and len(group) >= FALLBACK_MIN_BATCH_SIZE:
            recovered = _fallback_heuristic_check(group, images)
            if recovered:
                trace(
                    f"ARTICLE_SMART_FILTER | Fallback triggered for batch "
                    f"| ids={len(group)} recovered={len(recovered)}"
                )
                batch_approved = recovered
        merged.extend(batch_approved)
    return merged


def filter_structural_diagram_candidates(
    images: list[ExtractedImage],
) -> list[ExtractedImage]:
    """
    Batch Qwen → Top-K по importance (хронология при равном score).
    При недоступности Ollama — пустой список (VLM не вызывается).
    """
    n = len(images)
    if n == 0:
        return []

    groups = _split_ollama_batches(images)
    try:
        approved = asyncio.run(_filter_all_batches(images))
    except Exception as exc:
        trace(f"ARTICLE_SMART_FILTER ollama ✗ | {exc}")
        return []

    if not approved:
        trace(
            f"ARTICLE_SMART_FILTER | candidates={n} ollama_batches={len(groups)} "
            "qwen_approved=0 vlm_next=0"
        )
        return []

    before_cap = len({a.id for a in approved if 0 <= a.id < n})
    out = _apply_top_k(images, approved)
    trace(
        f"ARTICLE_SMART_FILTER | candidates={n} ollama_batches={len(groups)} "
        f"qwen_approved={before_cap} top_k={len(out)} "
        f"cap={ARTICLE_MAX_DIAGRAMS_PER_ARTICLE} vlm_next={len(out)}"
    )
    return out
