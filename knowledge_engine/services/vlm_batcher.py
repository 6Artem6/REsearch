"""VLM: одно изображение → отдельный Gemini Lite multimodal запрос; пул с RPM/TPM/RPD."""

from __future__ import annotations

import asyncio
import re
from typing import Literal

from knowledge_engine.config import (
    refresh_vlm_gemini_env_from_dotenv,
    vlm_gemini_concurrency_live,
    vlm_gemini_est_tokens_live,
    vlm_gemini_rate_limits_live,
)
from knowledge_engine.schemas.llm_contracts.vlm import (
    VlmBatchResponseContract,
    VlmDiagramItemContract,
)
from knowledge_engine.services.gemini_stateless import run_stateless_gemini_multimodal
from knowledge_engine.services.mermaid_validate import (
    is_misclassified_benchmark_flowchart,
    sanitize_mermaid_from_vlm,
)
from knowledge_engine.services.parsers.base import ExtractedImage
from knowledge_engine.services.vlm_gemini_pool import get_vlm_gemini_pool
from knowledge_engine.ui.run_log import trace

VlmBatchResponse = VlmBatchResponseContract
VlmDiagramItem = VlmDiagramItemContract

# Hard Mermaid generation rules shared by batch/single VLM system prompts.
VLM_MERMAID_GENERATION_RULES = (
    "=== MERMAID GENERATION RULES ===\n"
    "1. CHART TYPES ISOLATION:\n"
    "   - NEVER embed `xychart-beta` inside `flowchart`. They are separate diagram types!\n"
    "   - For charts, use pure `xychart-beta` syntax without `flowchart LR/TD` wrappers.\n"
    "2. NODE ID & LABEL SANITATION:\n"
    "   - Node IDs must contain ONLY alphanumeric characters and underscores "
    '(e.g., `Server_3`, NOT `Server3[ "MCP Server"]`).\n'
    '   - NEVER add spaces inside node bracket declarations like `[ "Text"]`. '
    'Use `["Text"]`.\n'
    "   - Node labels with special characters or newlines MUST be strictly enclosed "
    'in double quotes: `ID["Label with <br/>"]`.\n'
    "3. ESCAPING & QUOTES:\n"
    "   - Do NOT use unescaped single or double quotes inside labels. "
    "Replace single quotes with standard text or clean quotes.\n"
    '   - Every opening quote `"` MUST have exactly one closing quote `"`.\n'
    "4. FLOWCHART INTEGRITY:\n"
    "   - Do NOT skip step numbers in sequence arrows.\n"
    "   - Do NOT draw self-looping arrows on the same node for external requests "
    "(e.g., avoid `Client --> Client` for incoming user queries).\n"
)

VLM_BATCH_SYSTEM = (
    "You are an architect-engineer. You receive a batch of N images from a technical "
    "article with their context.\n\n"
    "For EACH image:\n"
    "1. Classify the visual (field diagram_kind):\n"
    "   - architecture — architecture diagram, data pipeline, components, ER/UML, "
    "dependency graph.\n"
    "   - benchmark_chart — comparison/benchmark chart: scatter, line, bar; "
    "X/Y axes with metrics (QPS, Recall, latency, throughput); curves of systems.\n"
    "   - none — UI screenshot, cover, formula, code, decoration -> is_diagram: false.\n"
    "2. If is_diagram: true:\n"
    "   - diagram_kind=architecture: exact Mermaid flowchart TD/LR, sequenceDiagram "
    "or classDiagram. Do NOT use for charts.\n"
    "   - diagram_kind=benchmark_chart: STRICTLY Mermaid xychart-beta "
    "(lines/bars per system). FORBIDDEN: flowchart/graph with rectangular nodes for "
    "X/Y axes or data points; FORBIDDEN: a horizontal row of blocks labeled "
    "«Ось Y», «Ось X», «QPS», «Recall».\n"
    "     Example syntax:\n"
    "     xychart-beta\n"
    '         title "Performance vs Recall"\n'
    '         x-axis "Recall" [0.88, 0.90, 0.93, 0.96, 0.99]\n'
    '         y-axis "QPS" 0 --> 4200\n'
    '         line "SystemA" [1050, 2000, 3000, 3957, 3957]\n'
    "     Recover numbers from the chart when possible; if unreadable, estimate "
    "from article context.\n"
    "3. Mermaid format: NO ``` and NO %%{init:…}%% (server injects styles); "
    "each section on a new line; never mix flowchart and xychart-beta.\n"
    "   - sequenceDiagram/flowchart: message label, Note, and edge label MUST be "
    "on a single line. Never break labels across multiple lines.\n"
    '   - sequenceDiagram: quoted messages: A->>B: "label".\n'
    "   - summary: 2–3 sentences in natural Russian (user-facing).\n\n"
    f"{VLM_MERMAID_GENERATION_RULES}\n"
    "Return STRICT JSON: items with index 0..N-1 in image order."
)

VLM_SINGLE_SYSTEM = (
    "You are an architect-engineer. One image from a technical article with context.\n\n"
    "1. Classify the visual (diagram_kind): architecture | benchmark_chart | none.\n"
    "2. If is_diagram: true — Mermaid (architecture: flowchart/sequence/class; "
    "benchmark_chart: xychart-beta only, never flowchart for axes).\n"
    "   No ``` and no %%{init}%% (server adds init); each Mermaid line separate.\n"
    '   Message/Note/edge labels on one line; A->>B: "label" with quotes.\n'
    "3. summary: 2–3 sentences in natural Russian (user-facing).\n\n"
    f"{VLM_MERMAID_GENERATION_RULES}\n"
    "Return STRICT JSON: items — one element with index=0."
)

DEFAULT_BATCH_MIN = 3
DEFAULT_BATCH_MAX = 5

_VLM_CAPTION_MAX = 200
_VLM_CONTEXT_MAX = 400

DiagramKind = Literal["architecture", "benchmark_chart", "none"]


def _truncate_caption(text: str) -> str:
    return (text or "").strip()[:_VLM_CAPTION_MAX]


def _truncate_context(text: str) -> str:
    return (text or "").strip()[:_VLM_CONTEXT_MAX]


def chunk_batches(
    images: list[ExtractedImage],
    min_size: int = DEFAULT_BATCH_MIN,
    max_size: int = DEFAULT_BATCH_MAX,
) -> list[list[ExtractedImage]]:
    if not images:
        return []
    max_size = max(1, min(max_size, 5))
    min_size = max(1, min(min_size, max_size))
    batches: list[list[ExtractedImage]] = []
    i = 0
    n = len(images)
    while i < n:
        remaining = n - i
        if remaining <= max_size:
            batches.append(images[i:])
            break
        take = max_size
        if remaining - take < min_size and remaining > max_size:
            take = remaining - min_size
        batches.append(images[i : i + take])
        i += take
    return batches


def _build_user_payload(batch: list[ExtractedImage]) -> str:
    lines = [
        f"Пачка из {len(batch)} изображений. Контекст по index:",
    ]
    for idx, img in enumerate(batch):
        cap = _truncate_caption(img.caption) or "(без подписи)"
        ctx = _truncate_context(img.context_text)
        if ctx:
            cap = f"{cap}\nКонтекст: {ctx}"
        lines.append(f"[{idx}] page/pos={img.page_or_pos} | {cap}")
    return "\n".join(lines)


def _build_single_payload(img: ExtractedImage) -> str:
    cap = _truncate_caption(img.caption) or "(без подписи)"
    ctx = _truncate_context(img.context_text)
    if ctx:
        cap = f"{cap}\nКонтекст: {ctx}"
    return f"Изображение index=0 | page/pos={img.page_or_pos} | {cap}"


def _sanitize_vlm_item(item: VlmDiagramItem, img: ExtractedImage) -> VlmDiagramItem:
    if not item.is_diagram:
        return item
    caption = f"{item.title} {img.caption} {img.context_text}"
    if is_misclassified_benchmark_flowchart(item.mermaid, caption):
        trace(
            f"VLM_BATCH ✗ benchmark misclassified as flowchart | "
            f"index={item.index} title={item.title[:40]!r}"
        )
        return item.model_copy(update={"mermaid": ""})
    if item.diagram_kind == "benchmark_chart" and item.mermaid:
        low = item.mermaid.lower()
        if "flowchart" in low or re.match(r"graph\s", low.strip()):
            trace(
                f"VLM_BATCH ⚠ benchmark_chart with flowchart mermaid — raw kept | "
                f"index={item.index}"
            )
    cleaned = sanitize_mermaid_from_vlm(item.mermaid or "")
    if cleaned:
        item = item.model_copy(update={"mermaid": cleaned})
    return item


def _run_vlm_single_sync(
    img: ExtractedImage,
    *,
    label: str,
    model: str,
) -> tuple[VlmDiagramItem | None, int]:
    """Один запрос на одну модель (слот пула)."""
    m = (model or "").strip()
    if not m:
        return None, 0
    inp_est, out_est = vlm_gemini_est_tokens_live()
    est_total = inp_est + out_est
    image_parts: list[tuple[bytes, str]] = [
        (img.image_bytes, img.mime or "image/png"),
    ]
    user_payload = _build_single_payload(img)
    raw = run_stateless_gemini_multimodal(
        VLM_SINGLE_SYSTEM,
        user_payload,
        "Article diagram extraction single",
        image_parts,
        response_schema=VlmBatchResponse,
        label=f"{label}/{m}",
        rpm_pause=False,
        models=[m],
    )
    # RPM/RPD accounted inside gemini_stateless (_call_with_model_fallback)
    if isinstance(raw, VlmBatchResponse):
        response = raw
    else:
        response = VlmBatchResponse.model_validate_json(str(raw))
    actual_out = max(out_est, len(str(raw)) // 4)
    est_total = inp_est + actual_out
    item = next((i for i in response.items if i.index == 0), None)
    if item is None and response.items:
        item = response.items[0].model_copy(update={"index": 0})
    if item is None:
        return None, est_total
    return _sanitize_vlm_item(item.model_copy(update={"index": 0}), img), est_total


async def run_vlm_images_parallel_async(
    images: list[ExtractedImage],
    *,
    label: str = "article_ingestion/vlm",
) -> list[tuple[ExtractedImage, VlmDiagramItem | None]]:
    """Параллельные stateless multimodal запросы (разные «чаты»), лимиты Flash Lite."""
    if not images:
        return []

    refresh_vlm_gemini_env_from_dotenv()
    pool = await get_vlm_gemini_pool()
    if not pool.model_ids:
        trace(
            "VLM skipped due to local quota guard | "
            "no models in pool (quota / config)"
        )
        return [(img, None) for img in images]

    concurrency = vlm_gemini_concurrency_live()
    inp_est, out_est = vlm_gemini_est_tokens_live()
    est = max(1, inp_est + out_est)
    rpm, tpm, rpd = vlm_gemini_rate_limits_live()
    sem = asyncio.Semaphore(concurrency)
    max_attempts = max(1, len(pool.model_ids))

    trace(
        f"VLM pool ▶ | images={len(images)} "
        f"models={pool.label_chain()} "
        f"limits rpm={rpm} tpm={tpm} rpd={rpd} per model "
        f"concurrency={concurrency} est_tpm_per_req≈{est}"
    )

    async def _one(
        idx: int, img: ExtractedImage
    ) -> tuple[int, ExtractedImage, VlmDiagramItem | None]:
        item: VlmDiagramItem | None = None
        async with sem:
            for attempt in range(max_attempts):
                slot = await pool.acquire_slot(est)
                if slot is None:
                    trace(f"VLM ✗ | image {idx + 1}/{len(images)} | no slot")
                    break
                trace(
                    f"VLM ▶ | image {idx + 1}/{len(images)} "
                    f"model={slot.model} attempt={attempt + 1}/{max_attempts}"
                )
                try:
                    item, actual_tpm = await asyncio.to_thread(
                        _run_vlm_single_sync,
                        img,
                        label=f"{label}/img{idx}",
                        model=slot.model,
                    )
                    await slot.limiter.reconcile_last_tpm(actual_tpm)
                    break
                except Exception as exc:
                    trace(
                        f"VLM failover | {slot.model} ✗ {exc!s:.120} " f"→ next in pool"
                    )
                    item = None
                    continue
        trace(
            f"VLM ✓ ready | image {idx + 1}/{len(images)} "
            f"diagram={bool(item and item.is_diagram)}"
        )
        return idx, img, item

    tasks = [asyncio.create_task(_one(i, img)) for i, img in enumerate(images)]
    by_idx: dict[int, tuple[ExtractedImage, VlmDiagramItem | None]] = {}
    for coro in asyncio.as_completed(tasks):
        idx, img, item = await coro
        by_idx[idx] = (img, item)

    return [by_idx[i] for i in range(len(images))]


def run_vlm_images_parallel(
    images: list[ExtractedImage],
    *,
    label: str = "article_ingestion/vlm",
) -> list[tuple[ExtractedImage, VlmDiagramItem | None]]:
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                run_vlm_images_parallel_async(images, label=label),
            ).result()
    except RuntimeError:
        return asyncio.run(run_vlm_images_parallel_async(images, label=label))


def run_vlm_batch(
    batch: list[ExtractedImage],
    *,
    label: str = "article_ingestion/vlm_batch",
) -> VlmBatchResponse:
    if not batch:
        return VlmBatchResponse()
    pairs = run_vlm_images_parallel(batch, label=label)
    sanitized: list[VlmDiagramItem] = []
    for idx, (img, item) in enumerate(pairs):
        if item is None:
            continue
        sanitized.append(
            _sanitize_vlm_item(item.model_copy(update={"index": idx}), img)
        )
    return VlmBatchResponse(items=sanitized)
