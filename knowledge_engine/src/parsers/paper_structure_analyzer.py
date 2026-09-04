"""Two-stage paper structure analysis via Gemini Flash Lite before Gemma Map-Reduce."""

from __future__ import annotations

import json
import re

from knowledge_engine.config import (
    GEMINI_LITE_MAX_OUTPUT_TOKENS,
    GEMINI_LITE_MODEL,
    GEMINI_RPM_PAUSE_SEC,
    INGEST_GATE_BLOG_QUALITY_MIN,
    INGEST_GATE_ENABLED,
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
from knowledge_engine.src.parsers.ingest_gate import (
    INGEST_GATE_REJECT_REASON,
    IngestGateResult,
    decide_ingest_gate,
    missing_pass2_placeholder,
)
from knowledge_engine.src.parsers.paper_structure_schema import (
    ExtractMode,
    InputPaperJson,
    PaperCredibilityAnalysis,
    PaperStructureAnalysis,
    ParagraphAnalysis,
    ParagraphPriority,
    TechnicalCorrectness,
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
Do not judge semantic_level, technical_correctness, information_density, or extract_mode in this pass.
""".strip()

# Pass 2 — same chat, remaining paragraphs only. Intrinsic traits; no docs/file-name cues.
_PAPER_CREDIBILITY_SYSTEM_STATIC = """
You are the second pass of an inbound ingest gate. Pass 1 already labeled structure
(CORE / CONTEXT / DROP) and topic_relevance. DROP paragraphs were removed.

=== PARAGRAPH CHARACTERISTICS AUDIT ===
Analyze each remaining paragraph independently and evaluate its intrinsic characteristics.
Judge only the substance of the paragraph text. Do not look for citations, URLs,
filenames, or source-code paths — those are stripped by extractors and are not evidence.

1. semantic_level:
   - SPEC_EXACT: Explains precise runtime mechanics, internal flags, exact data structures, or formal specs.
   - CONCEPTUAL_MODEL: Explains high-level system interactions accurately without low-level detail.
   - METAPHOR_ONLY: Relies strictly on non-technical everyday analogies or metaphors.

2. technical_correctness:
   - VERIFIED: Claims are technically sound for the given technology.
   - SIMPLIFIED: Minor pedagogical simplification, but fundamentally correct.
   - CONTRADICTION: Contains direct technical errors (e.g. confusing user-space bytecode
     dispatch / eval_breaker with hardware CPU interrupts, misrepresenting fundamental
     concurrency primitives, claiming parallel bytecode on multiple cores under a GIL).

3. information_density:
   - HIGH: Dense technical facts, algorithms, state transitions.
   - NEUTRAL: Informational context, explanations.
   - WATER_OR_OPINION: Pure fluff, author intros, unverified commentary.

4. extract_mode (volume of sentences, NOT importance):
   Pass 1 already decided paragraph priority (CORE / CONTEXT / DROP). Do not re-judge whether
   the paragraph belongs in the document. Evaluate only the structural density of sentences:

   - full: The technical content is distributed across the whole paragraph (e.g., multi-step algorithms,
     formulas with explanations, continuous reasoning). Keep every sentence.
   - head_1: The primary load-bearing fact/thesis is complete in sentence 1; remaining sentences
     merely restate, motivate, or provide obvious examples.
   - head_2: Sentence 1 states the core thesis; sentence 2 adds a mandatory qualifier/bound/condition;
     remaining sentences are secondary elaboration.

   CRITICAL RULE: extract_mode is orthogonal to priority and information_density.
   - A CORE + HIGH density paragraph CAN be head_1 if sentence 1 contains the complete fact.
   - A CONTEXT + NEUTRAL density paragraph CAN be full if the context requires the full text.
   Do NOT use heuristics like CORE->full or CONTEXT->head_1.

Do not rewrite the source. Do not drop paragraph ids. Include one row per remaining id.
JSON: paragraphs[].paragraph_id, semantic_level, technical_correctness, information_density, extract_mode, reason.
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
    return f"PASS: STRUCTURE\n\ntarget_topic:\n{topic}\n\ninput_paper_json:\n{doc_json}"


def _build_credibility_user_payload(
    remaining: list[dict],
) -> str:
    body = json.dumps(
        {"remaining_paragraphs": remaining},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "PASS: CREDIBILITY\n"
        "Paragraph characteristics audit of remaining paragraphs only. "
        "DROP rows were already removed. Judge intrinsic traits of each paragraph text.\n\n"
        f"{body}"
    )


def _as_paragraph_analysis(row: object) -> ParagraphAnalysis:
    if isinstance(row, ParagraphAnalysis):
        return row
    if hasattr(row, "model_dump"):
        return ParagraphAnalysis.model_validate(row.model_dump())
    return ParagraphAnalysis.model_validate(row)


def merge_credibility_into_paragraphs(
    structure_rows: list[object],
    audit: PaperCredibilityAnalysis | None,
    remaining_ids: set[int],
) -> list[ParagraphAnalysis]:
    """Attach pass-2 grades onto remaining pass-1 rows. No audit → leave unscored (fail-open)."""
    by_id = {row.paragraph_id: row for row in (audit.paragraphs if audit else [])}
    merged: list[ParagraphAnalysis] = []
    for raw in structure_rows:
        host = _as_paragraph_analysis(raw)
        if audit is None or host.paragraph_id not in remaining_ids:
            merged.append(host)
            continue
        verdict = by_id.get(host.paragraph_id)
        if verdict is None:
            placeholder = missing_pass2_placeholder(host.paragraph_id)
            host.semantic_level = placeholder.semantic_level
            host.technical_correctness = placeholder.technical_correctness
            host.information_density = placeholder.information_density
            host.extract_mode = placeholder.extract_mode
            host.accuracy_reason = placeholder.reason
        else:
            host.semantic_level = verdict.semantic_level
            host.technical_correctness = verdict.technical_correctness
            host.information_density = verdict.information_density
            host.extract_mode = verdict.extract_mode
            host.accuracy_reason = (verdict.reason or "").strip()
        merged.append(host)
    return merged


class PaperStructureAnalyzer:
    """Gemini Flash Lite two-pass inbound gate (structure then parametric credibility)."""

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

    def analyze_two_pass(
        self,
        target_topic: str,
        input_paper: InputPaperJson | dict,
        *,
        label: str = "paper_structure",
        anchor: str = "",
    ) -> tuple[PaperStructureAnalysis, list[ParagraphAnalysis]]:
        """Pass 1 structure + pass 2 credibility in one Flash Lite chat session."""
        if isinstance(input_paper, dict):
            paper = InputPaperJson.model_validate(input_paper)
        else:
            paper = input_paper
        paper_llm = input_paper_json_for_llm(paper)
        if not paper_llm.get("pages"):
            return PaperStructureAnalysis(), []

        structure = self._analyze_pass1_in_session(
            target_topic,
            paper,
            paper_llm,
            label=label,
            anchor=anchor,
        )
        remaining_ids = remaining_paragraph_ids(paper, structure)
        remaining_payload = remaining_paragraphs_for_llm(paper, remaining_ids)
        audit: PaperCredibilityAnalysis | None = None
        if remaining_payload and is_gemini_available():
            try:
                audit = self._analyze_pass2_in_session(
                    remaining_payload,
                    label=label,
                    anchor=anchor,
                )
                n_c = sum(
                    1
                    for row in audit.paragraphs
                    if row.technical_correctness == TechnicalCorrectness.CONTRADICTION
                )
                trace(
                    f"PAPER_CREDIBILITY ✓ | Gemini | remaining={len(remaining_payload)} "
                    f"contradiction={n_c}"
                )
            except Exception as exc:
                trace(f"PAPER_CREDIBILITY fallback | {exc}")
                audit = None
        merged = merge_credibility_into_paragraphs(
            list(structure.paragraphs),
            audit,
            remaining_ids,
        )
        return structure, merged

    def _session_label(self, label: str, anchor: str) -> str:
        raw = (anchor or label or "paper_structure").strip()
        return f"ingest_gate:{raw[:80]}"

    def _ensure_gate_session(self) -> object:
        mgr = getattr(self, "_gate_chat", None)
        if mgr is None:
            from knowledge_engine.services.chat_session_manager import ChatSessionManager

            mgr = ChatSessionManager(user_scope="ingest_gate")
            self._gate_chat = mgr
        return mgr

    def _analyze_pass1_in_session(
        self,
        target_topic: str,
        paper: InputPaperJson,
        paper_llm: dict,
        *,
        label: str,
        anchor: str,
    ) -> PaperStructureAnalysis:
        dynamic_payload = _build_dynamic_user_payload(target_topic, paper_llm)
        if not is_gemini_available():
            return local_fallback_analysis(paper, target_topic)
        models = _resolve_flash_lite_models()
        primary = models[0] if models else GEMINI_LITE_MODEL
        chat_label = self._session_label(label, anchor)
        chat_mgr = self._ensure_gate_session()
        try:
            result = run_gemini_structured_with_chain(
                primary,
                _PAPER_STRUCTURE_SYSTEM_STATIC,
                dynamic_payload,
                anchor or "paper_structure",
                PaperStructureAnalysis,
                f"paper structure / {label}",
                rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
                models=models,
                chat_manager=chat_mgr,
                chat_label=chat_label,
                max_output_tokens=GEMINI_LITE_MAX_OUTPUT_TOKENS,
            )
            trace(
                f"PAPER_STRUCTURE ✓ | Gemini | paras={len(result.paragraphs)} "
                f"drop_pages={len(result.drop_pages)}"
            )
            return result
        except Exception as exc:
            trace(f"PAPER_STRUCTURE fallback | {exc}")
            return local_fallback_analysis(paper, target_topic)

    def _analyze_pass2_in_session(
        self,
        remaining: list[dict],
        *,
        label: str,
        anchor: str,
    ) -> PaperCredibilityAnalysis:
        models = _resolve_flash_lite_models()
        primary = models[0] if models else GEMINI_LITE_MODEL
        chat_label = self._session_label(label, anchor)
        chat_mgr = self._ensure_gate_session()
        chat_mgr.invalidate_live_chat(chat_label)
        payload = _build_credibility_user_payload(remaining)
        return run_gemini_structured_with_chain(
            primary,
            _PAPER_CREDIBILITY_SYSTEM_STATIC,
            payload,
            anchor or "paper_credibility",
            PaperCredibilityAnalysis,
            f"paper credibility / {label}",
            rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
            models=models,
            chat_manager=chat_mgr,
            chat_label=chat_label,
            max_output_tokens=GEMINI_LITE_MAX_OUTPUT_TOKENS,
        )


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


def remaining_paragraph_ids(
    paper: InputPaperJson,
    analysis: PaperStructureAnalysis,
    *,
    min_topic_relevance: int = 4,
) -> set[int]:
    """Ids that survive pass-1 structure filter (refs / DROP / relevance)."""
    ref_range = _resolve_references_range(paper, analysis)
    text_by_id = _paragraph_text_map(paper)
    by_id = {p.paragraph_id: p for p in analysis.paragraphs}
    kept: set[int] = set()
    for pid in text_by_id:
        if _paragraph_in_references_range(pid, ref_range):
            continue
        row = by_id.get(pid)
        if row is not None and row.priority == ParagraphPriority.DROP:
            continue
        if row is not None and not _paragraph_passes_filter(row, min_topic_relevance):
            continue
        kept.add(pid)
    return kept


def remaining_paragraphs_for_llm(
    paper: InputPaperJson,
    remaining_ids: set[int],
    *,
    max_chars_per_para: int = 600,
) -> list[dict]:
    rows: list[dict] = []
    for page in paper.pages:
        for para in page.paragraphs:
            if para.paragraph_id not in remaining_ids:
                continue
            rows.append(
                {
                    "paragraph_id": para.paragraph_id,
                    "section_title": para.section_title,
                    "text": (para.text or "")[:max_chars_per_para],
                }
            )
    return rows


def _paragraph_is_contradiction(row: object | None) -> bool:
    grade = getattr(row, "technical_correctness", None)
    return grade == TechnicalCorrectness.CONTRADICTION


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_paragraph_sentences(text: str) -> list[str]:
    """Lightweight sentence split for extract_mode heads. Not a linguistic tokenizer."""
    raw = (text or "").strip()
    if not raw:
        return []
    return [part for part in _SENTENCE_SPLIT_RE.split(raw) if part.strip()]


def apply_extract_mode(text: str, mode: ExtractMode | str | None) -> str:
    """Keep full text, first sentence, or first two. Short paragraphs keep all sentences."""
    raw = text or ""
    if not raw.strip():
        return raw
    if mode is None or mode == ExtractMode.FULL or mode == ExtractMode.FULL.value:
        return raw
    if mode in (ExtractMode.HEAD_1, ExtractMode.HEAD_1.value):
        n = 1
    elif mode in (ExtractMode.HEAD_2, ExtractMode.HEAD_2.value):
        n = 2
    else:
        return raw
    sentences = split_paragraph_sentences(raw)
    if not sentences:
        return raw
    return " ".join(sentences[:n]).strip()


def apply_structure_filter(
    paper: InputPaperJson,
    analysis: PaperStructureAnalysis,
    *,
    min_topic_relevance: int = 4,
    drop_contradictions: bool = False,
    paragraph_overrides: list[ParagraphAnalysis] | None = None,
) -> str:
    """Compose cleaned body: references range cut, DROP, tiered relevance; CORE first."""
    ref_range = _resolve_references_range(paper, analysis)
    text_by_id = _paragraph_text_map(paper)
    rows = paragraph_overrides if paragraph_overrides is not None else analysis.paragraphs
    by_id = {p.paragraph_id: p for p in rows}

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
        if drop_contradictions and _paragraph_is_contradiction(row):
            continue
        if row is None:
            kept.append((1, 5, text))
            continue
        truncated = apply_extract_mode(text, getattr(row, "extract_mode", None))
        tier = 0 if row.priority == ParagraphPriority.CORE else 1
        kept.append((tier, -row.topic_relevance, truncated))

    if not kept:
        return ""

    kept.sort(key=lambda x: (x[0], x[1]))
    parts = [t for _, _, t in kept if (t or "").strip()]
    return "\n\n".join(parts).strip()


def quality_min_for_url(page_url: str) -> float | None:
    """Whole-article reject threshold. Blogs: 0.65. Academic PDFs: no whole-doc reject."""
    if is_academic_pdf_url(page_url):
        return None
    return float(INGEST_GATE_BLOG_QUALITY_MIN)


def _filtered_body_discards_source(raw: str, filtered: str) -> bool:
    """True when the gate kept a stub instead of the document (MAP would see ~100 words)."""
    raw_w = len((raw or "").split())
    fil_w = len((filtered or "").split())
    if raw_w < 800 or fil_w >= raw_w:
        return False
    if fil_w < 400:
        return True
    return fil_w < int(raw_w * 0.35)


def run_inbound_ingest_gate(
    body: str,
    target_topic: str,
    *,
    pdf_bytes: bytes | None = None,
    label: str = "",
    page_url: str = "",
    quality_min: float | None | object = ...,
) -> IngestGateResult:
    """Two-pass Flash Lite gate. Rejected blogs never proceed to Gemma Map-Reduce."""
    raw = (body or "").strip()
    if pdf_bytes:
        paper = build_input_paper_json_from_pdf_bytes(pdf_bytes)
        if not paper.pages or sum(len(p.paragraphs) for p in paper.pages) < 2:
            paper = build_input_paper_json_from_plain_text(raw)
    else:
        paper = build_input_paper_json_from_plain_text(raw)

    if sum(len(p.paragraphs) for p in paper.pages) < 2:
        return IngestGateResult(accepted=True, quality=1.0, body=raw)

    analyzer = PaperStructureAnalyzer()
    gate_label = label or "ingest_gate"
    structure, merged = analyzer.analyze_two_pass(
        target_topic,
        paper,
        label=gate_label,
        anchor=f"ingest_gate:{(page_url or label)[:40]}",
    )
    remaining_ids = remaining_paragraph_ids(paper, structure)
    scored = [p for p in merged if p.paragraph_id in remaining_ids]
    min_q = quality_min_for_url(page_url) if quality_min is ... else quality_min
    accepted, quality, reason = decide_ingest_gate(scored, quality_min=min_q)
    if not accepted:
        trace(
            f"INGEST_GATE ⊘ | {reason} | Q={quality:.3f} "
            f"remaining={len(scored)} | {(page_url or label)[:55]}"
        )
        return IngestGateResult(
            accepted=False,
            quality=quality,
            body="",
            reject_reason=reason or INGEST_GATE_REJECT_REASON,
            paragraphs=merged,
        )

    filtered = apply_structure_filter(
        paper,
        structure,
        drop_contradictions=True,
        paragraph_overrides=merged,
    )
    if len(filtered) < 80 or _filtered_body_discards_source(raw, filtered):
        from knowledge_engine.ingest.pipeline_audit import pipeline_audit

        why = (
            "filtered body too short — using raw body"
            if len(filtered) < 80
            else (
                f"filtered discarded source "
                f"({len(filtered.split())} words of {len(raw.split())}) — using raw"
            )
        )
        trace(f"INGEST_GATE ⊘ | {why}")
        pipeline_audit("Triage", page_url or label, raw, extra=f"gate_fail_open {why}")
        return IngestGateResult(
            accepted=True,
            quality=quality,
            body=raw,
            paragraphs=merged,
        )
    n_cut = sum(
        1
        for p in scored
        if p.technical_correctness == TechnicalCorrectness.CONTRADICTION
    )
    from knowledge_engine.ingest.pipeline_audit import pipeline_audit

    pipeline_audit(
        "Triage",
        page_url or label,
        filtered,
        extra=f"gate Q={quality:.3f} chars {len(raw)}→{len(filtered)}",
    )
    trace(
        f"INGEST_GATE ✓ | Q={quality:.3f} chars {len(raw)} → {len(filtered)} "
        f"contradiction_cut={n_cut}"
    )
    return IngestGateResult(
        accepted=True,
        quality=quality,
        body=filtered,
        paragraphs=merged,
    )


def prepare_paper_body_for_gemma(
    body: str,
    target_topic: str,
    *,
    pdf_bytes: bytes | None = None,
    label: str = "",
    page_url: str = "",
) -> str:
    """Two-pass gate + filter; returns original body if analysis yields nothing.

    Empty string means the article failed the credibility gate (do not Map-Reduce).
    """
    cache_url = (page_url or label or "").strip()
    cached_body = get_cached_prepared_paper_body(cache_url) if cache_url else None
    if cached_body:
        trace(f"PAPER_STRUCTURE cache ✓ | skip Gemini analyze | {cache_url[:55]}")
        return cached_body

    raw = (body or "").strip()
    if not INGEST_GATE_ENABLED:
        paper = (
            build_input_paper_json_from_pdf_bytes(pdf_bytes)
            if pdf_bytes
            else build_input_paper_json_from_plain_text(raw)
        )
        if pdf_bytes and (
            not paper.pages or sum(len(p.paragraphs) for p in paper.pages) < 2
        ):
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
            return raw
        if cache_url:
            cache_prepared_paper_body(cache_url, filtered)
        return filtered

    result = run_inbound_ingest_gate(
        raw,
        target_topic,
        pdf_bytes=pdf_bytes,
        label=label,
        page_url=page_url,
    )
    if not result.accepted:
        return ""
    if cache_url and len(result.body) >= 80:
        cache_prepared_paper_body(cache_url, result.body)
    return result.body


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
