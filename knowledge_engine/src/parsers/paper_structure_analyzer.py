"""Two-stage paper structure analysis via Gemini Flash Lite before Gemma Map-Reduce."""

from __future__ import annotations

import json

from knowledge_engine.config import (
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
    refresh_vlm_gemini_env_from_dotenv,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_lite_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.services.vlm_gemini_pool import resolve_vlm_pool_model_ids
from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    arxiv_abs_url_to_pdf_url as _arxiv_abs_url_to_pdf_url,
)
from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    coerce_arxiv_url_to_pdf,
)
from knowledge_engine.src.parsers.paper_input_json import (
    build_input_paper_json_from_pdf_bytes,
    build_input_paper_json_from_plain_text,
    find_references_range_from_paper,
    input_paper_json_for_llm,
    is_references_section_start,
    ordered_paragraphs,
    references_range_from_index,
)
from knowledge_engine.src.parsers.paper_structure_schema import (
    InputPaperJson,
    PaperStructureAnalysis,
    ParagraphAnalysis,
    ParagraphPriority,
)
from knowledge_engine.ui.run_log import trace

_PREPARED_BODY_BY_URL: dict[str, str] = {}
_PREFETCH_PDF_BY_URL: dict[str, bytes] = {}


def paper_ingest_url_key(url: str) -> str:
    u = coerce_arxiv_url_to_pdf((url or "").strip())
    return u.lower()


def cache_prepared_paper_body(url: str, body: str) -> None:
    key = paper_ingest_url_key(url)
    text = (body or "").strip()
    if key and len(text) >= 80:
        _PREPARED_BODY_BY_URL[key] = text


def get_cached_prepared_paper_body(url: str) -> str | None:
    key = paper_ingest_url_key(url)
    if not key:
        return None
    hit = _PREPARED_BODY_BY_URL.get(key)
    return hit if hit and len(hit.strip()) >= 80 else None


def cache_prefetch_pdf_bytes(url: str, pdf_bytes: bytes) -> None:
    key = paper_ingest_url_key(url)
    if key and pdf_bytes and pdf_bytes[:5] == b"%PDF-":
        _PREFETCH_PDF_BY_URL[key] = pdf_bytes


def get_cached_prefetch_pdf_bytes(url: str) -> bytes | None:
    key = paper_ingest_url_key(url)
    if not key:
        return None
    data = _PREFETCH_PDF_BY_URL.get(key)
    return data if data and data[:5] == b"%PDF-" else None


# Static (cache-friendly) system instruction — no per-request topic or document JSON.
_PAPER_STRUCTURE_SYSTEM_STATIC = """
You are a scientific PDF structure analyst preparing text for a downstream Map-Reduce summarizer.

Evaluate the document in ONE pass along three axes:
1) Utility: locate References/Bibliography and technical appendices; list their page numbers in drop_pages
   and set references_start_page when the references section begins.
2) Relevance: score each paragraph against the user's target_topic (0–10).
3) Importance: assign each paragraph exactly one priority:
   - CORE: main math, model architecture, key formulas, algorithms, pseudocode, headline experimental results.
   - CONTEXT: introductory context, domain overview, related work, general discussion.
   - DROP: footnotes, layout noise, headers/footers, figure captions, licenses, acknowledgments.

Output MUST be valid JSON matching the PaperStructureAnalysis schema:
- references_start_page: optional int
- drop_pages: list of int page numbers to omit entirely
- paragraphs: array of objects with paragraph_id, page_number, section_title, priority (CORE|CONTEXT|DROP),
  topic_relevance (0–10), reason (short string).

Include one entry per paragraph id present in input_paper_json. Do not invent paragraph ids.
""".strip()


def _resolve_flash_lite_models() -> list[str]:
    refresh_vlm_gemini_env_from_dotenv()
    pool = resolve_vlm_pool_model_ids()
    if pool:
        return pool
    return gemini_lite_model_chain(GEMINI_LITE_MODEL)


def _build_dynamic_user_payload(target_topic: str, paper_llm: dict) -> str:
    topic = (target_topic or "").strip() or "general scientific understanding"
    doc_json = json.dumps(paper_llm, ensure_ascii=False, separators=(",", ":"))
    return f"target_topic:\n{topic}\n\n" f"input_paper_json:\n{doc_json}"


class PaperStructureAnalyzer:
    """Gemini Flash Lite analyzer with regex fallback."""

    def analyze(
        self,
        target_topic: str,
        input_paper: InputPaperJson | dict,
        *,
        label: str = "paper_structure",
        anchor: str = "",
    ) -> PaperStructureAnalysis:
        if isinstance(input_paper, dict):
            paper = InputPaperJson.model_validate(input_paper)
        else:
            paper = input_paper
        paper_llm = input_paper_json_for_llm(paper)
        if not paper_llm.get("pages"):
            return PaperStructureAnalysis()

        dynamic_payload = _build_dynamic_user_payload(target_topic, paper_llm)
        try:
            if not is_gemini_available():
                raise GeminiUnavailableError("Gemini unavailable")
            models = _resolve_flash_lite_models()
            primary = models[0] if models else GEMINI_LITE_MODEL
            result = run_gemini_structured_with_chain(
                primary,
                _PAPER_STRUCTURE_SYSTEM_STATIC,
                dynamic_payload,
                anchor or "paper_structure",
                PaperStructureAnalysis,
                f"paper structure / {label}",
                rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
                models=models,
            )
            trace(
                f"PAPER_STRUCTURE ✓ | Gemini | paras={len(result.paragraphs)} "
                f"drop_pages={len(result.drop_pages)}"
            )
            return result
        except Exception as exc:
            trace(f"PAPER_STRUCTURE fallback | {exc}")
            return local_fallback_analysis(paper, target_topic)


def local_fallback_analysis(
    paper: InputPaperJson,
    target_topic: str,
) -> PaperStructureAnalysis:
    """Regex references range; first 60% of remaining paras → CORE."""
    ref_range = _resolve_references_range(paper, PaperStructureAnalysis())
    refs_page: int | None = None
    if ref_range is not None:
        for page in paper.pages:
            for para in page.paragraphs:
                if para.paragraph_id == ref_range[0]:
                    refs_page = page.page_number
                    break

    remaining: list[tuple[int, int, str, str]] = []
    for page in paper.pages:
        for para in page.paragraphs:
            if _paragraph_in_references_range(para.paragraph_id, ref_range):
                continue
            remaining.append(
                (
                    para.paragraph_id,
                    page.page_number,
                    para.section_title,
                    para.text,
                )
            )

    drop_pages: list[int] = []
    n = len(remaining)
    core_cut = int(n * 0.6) if n else 0
    paragraphs: list[ParagraphAnalysis] = []
    for i, (pid, pno, sec, _text) in enumerate(remaining):
        if i < core_cut:
            pri = ParagraphPriority.CORE
            rel = 5
            reason = "fallback: early body paragraph (CORE)"
        else:
            pri = ParagraphPriority.CONTEXT
            rel = 5
            reason = "fallback: later body paragraph (CONTEXT)"
        paragraphs.append(
            ParagraphAnalysis(
                paragraph_id=pid,
                page_number=pno,
                section_title=sec[:200] or "Body",
                priority=pri,
                topic_relevance=rel,
                reason=reason,
            )
        )
    return PaperStructureAnalysis(
        references_start_page=refs_page,
        drop_pages=drop_pages,
        paragraphs=paragraphs,
    )


def _paragraph_text_map(paper: InputPaperJson) -> dict[int, tuple[int, str, str]]:
    out: dict[int, tuple[int, str, str]] = {}
    for page in paper.pages:
        for para in page.paragraphs:
            out[para.paragraph_id] = (
                page.page_number,
                para.section_title,
                para.text,
            )
    return out


def _page_for_paragraph_id(paper: InputPaperJson, pid: int) -> int | None:
    for page in paper.pages:
        for para in page.paragraphs:
            if para.paragraph_id == pid:
                return page.page_number
    return None


def _resolve_references_range(
    paper: InputPaperJson,
    analysis: PaperStructureAnalysis,
) -> tuple[int, int] | None:
    """Inclusive id range of references only (appendices after refs are excluded)."""
    rng = find_references_range_from_paper(paper)
    if rng is not None:
        return rng

    refs_page = analysis.references_start_page
    if refs_page is None:
        return None
    paras = ordered_paragraphs(paper)
    start_index: int | None = None
    for i, para in enumerate(paras):
        if _page_for_paragraph_id(paper, para.paragraph_id) != refs_page:
            continue
        if is_references_section_start(para.section_title, para.text):
            start_index = i
            break
    if start_index is None:
        for i, para in enumerate(paras):
            if _page_for_paragraph_id(paper, para.paragraph_id) == refs_page:
                start_index = i
                break
    if start_index is None:
        return None
    return references_range_from_index(paras, start_index)


def _paragraph_in_references_range(
    pid: int,
    ref_range: tuple[int, int] | None,
) -> bool:
    if ref_range is None:
        return False
    start, end = ref_range
    return start <= pid <= end


def _min_relevance_for_priority(
    priority: ParagraphPriority,
    min_topic_relevance: int,
) -> int:
    if priority == ParagraphPriority.CORE:
        return min_topic_relevance
    if priority == ParagraphPriority.CONTEXT:
        return max(min_topic_relevance + 2, 6)
    return min_topic_relevance


def _paragraph_passes_filter(
    row: ParagraphAnalysis | None,
    min_topic_relevance: int,
) -> bool:
    if row is None:
        return True
    if row.priority == ParagraphPriority.DROP:
        return False
    need = _min_relevance_for_priority(row.priority, min_topic_relevance)
    return row.topic_relevance >= need


def apply_structure_filter(
    paper: InputPaperJson,
    analysis: PaperStructureAnalysis,
    *,
    min_topic_relevance: int = 4,
) -> str:
    """Compose cleaned body: references range cut, DROP, tiered relevance; CORE first."""
    ref_range = _resolve_references_range(paper, analysis)
    text_by_id = _paragraph_text_map(paper)
    by_id = {p.paragraph_id: p for p in analysis.paragraphs}

    kept: list[tuple[int, int, str]] = []
    for pid in sorted(text_by_id.keys()):
        page_no, _sec, text = text_by_id[pid]
        if _paragraph_in_references_range(pid, ref_range):
            continue
        row = by_id.get(pid)
        if row is not None and row.priority == ParagraphPriority.DROP:
            continue
        if row is not None and not _paragraph_passes_filter(row, min_topic_relevance):
            continue
        if row is None:
            kept.append((1, 5, text))
            continue
        tier = 0 if row.priority == ParagraphPriority.CORE else 1
        kept.append((tier, -row.topic_relevance, text))

    if not kept:
        return ""

    kept.sort(key=lambda x: (x[0], x[1]))
    parts = [t for _, _, t in kept if (t or "").strip()]
    return "\n\n".join(parts).strip()


def prepare_paper_body_for_gemma(
    body: str,
    target_topic: str,
    *,
    pdf_bytes: bytes | None = None,
    label: str = "",
    page_url: str = "",
) -> str:
    """Build JSON, analyze, filter; returns original body if analysis yields nothing."""
    cache_url = (page_url or label or "").strip()
    cached_body = get_cached_prepared_paper_body(cache_url) if cache_url else None
    if cached_body:
        trace(f"PAPER_STRUCTURE cache ✓ | skip Gemini analyze | {cache_url[:55]}")
        return cached_body

    raw = (body or "").strip()
    if pdf_bytes:
        paper = build_input_paper_json_from_pdf_bytes(pdf_bytes)
        if not paper.pages or sum(len(p.paragraphs) for p in paper.pages) < 2:
            paper = build_input_paper_json_from_plain_text(raw)
    else:
        paper = build_input_paper_json_from_plain_text(raw)

    if sum(len(p.paragraphs) for p in paper.pages) < 2:
        return raw

    analyzer = PaperStructureAnalyzer()
    analysis = analyzer.analyze(
        target_topic,
        paper,
        label=label or "gemma_prep",
        anchor=f"paper_structure:{label[:40]}",
    )
    filtered = apply_structure_filter(paper, analysis)
    if len(filtered) < 80:
        trace("PAPER_STRUCTURE ⊘ | filtered body too short — using raw body")
        return raw
    trace(
        f"PAPER_STRUCTURE filter ✓ | chars {len(raw)} → {len(filtered)} "
        f"blocks_kept={max(1, filtered.count(chr(10) + chr(10)) + 1) if filtered.strip() else 0}"
    )
    if cache_url:
        cache_prepared_paper_body(cache_url, filtered)
    return filtered


def is_academic_pdf_url(url: str) -> bool:
    """arXiv PDF and direct .pdf links → paper-structure ingest path."""
    low = (url or "").strip().lower()
    if not low:
        return False
    if "arxiv.org/pdf/" in low:
        return True
    if low.endswith(".pdf") or ".pdf?" in low or ".pdf#" in low:
        return True
    return False


def arxiv_abs_url_to_pdf_url(url: str) -> str | None:
    """https://arxiv.org/abs/ID → https://arxiv.org/pdf/ID.pdf"""
    return _arxiv_abs_url_to_pdf_url(url)


def try_fetch_pdf_bytes_for_url(url: str) -> bytes | None:
    """Optional PDF bytes for structure extraction (best-effort)."""
    u = (url or "").strip()
    if not u:
        return None
    candidates: list[str] = []
    arxiv_pdf = arxiv_abs_url_to_pdf_url(u)
    if arxiv_pdf:
        candidates.append(arxiv_pdf)
    low = u.lower()
    if ".pdf" in low or "arxiv.org/pdf" in low:
        candidates.append(u)
    seen: set[str] = set()
    for cand in candidates:
        key = cand.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            import httpx

            with httpx.Client(
                timeout=httpx.Timeout(45.0, connect=12.0),
                follow_redirects=True,
            ) as client:
                resp = client.get(cand)
            data = resp.content
            if data and data[:5] == b"%PDF-":
                return data
        except Exception:
            continue
    return None


async def prepare_paper_body_for_gemma_async(
    body: str,
    target_topic: str,
    *,
    pdf_bytes: bytes | None = None,
    label: str = "",
    page_url: str = "",
) -> str:
    import asyncio

    return await asyncio.to_thread(
        prepare_paper_body_for_gemma,
        body,
        target_topic,
        pdf_bytes=pdf_bytes,
        label=label,
        page_url=page_url or label,
    )
