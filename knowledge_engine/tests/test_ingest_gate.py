"""Inbound ingest gate: two-pass vector traits + Map-Reduce admit threshold."""

from __future__ import annotations

import pytest

from knowledge_engine.src.parsers.ingest_gate import (
    INGEST_GATE_REJECT_REASON,
    calculate_article_quality,
    calculate_paragraph_score,
    decide_ingest_gate,
)
from knowledge_engine.src.parsers.paper_input_json import (
    build_input_paper_json_from_plain_text,
)
from knowledge_engine.src.parsers.paper_structure_analyzer import (
    apply_structure_filter,
    merge_credibility_into_paragraphs,
    remaining_paragraph_ids,
    run_inbound_ingest_gate,
)
from knowledge_engine.src.parsers.paper_structure_schema import (
    ExtractMode,
    InformationDensity,
    PaperCredibilityAnalysis,
    PaperStructureAnalysis,
    ParagraphAnalysis,
    ParagraphCredibility,
    ParagraphPriority,
    SemanticLevel,
    TechnicalCorrectness,
)

_GIL_FALSE_MECHANIC = (
    "После Python 3.2 GIL переключается строго каждые 5 мс через семафор "
    "eval_tick_barrier: eval_breaker — это аппаратное прерывание CPU, которое "
    "останавливает все ядра; потоки без этого флага исполняют байт-код параллельно."
)
_GIL_METAPHOR = (
    "Представьте GIL как одну очередь в кассу супермаркета: пока один покупатель "
    "платит, остальные стоят и ждут своей очереди."
)
_GIL_OK_CONTEXT = (
    "CPython historically used a global interpreter lock so only one thread "
    "runs bytecode at a time; CPU-bound work often uses multiprocessing."
)


def _cred(
    pid: int,
    *,
    level: SemanticLevel = SemanticLevel.CONCEPTUAL_MODEL,
    correctness: TechnicalCorrectness = TechnicalCorrectness.VERIFIED,
    density: InformationDensity = InformationDensity.NEUTRAL,
    extract_mode: ExtractMode = ExtractMode.FULL,
    reason: str = "unit",
) -> ParagraphCredibility:
    return ParagraphCredibility(
        paragraph_id=pid,
        semantic_level=level,
        technical_correctness=correctness,
        information_density=density,
        extract_mode=extract_mode,
        reason=reason,
    )


def _para(
    pid: int,
    *,
    priority: ParagraphPriority = ParagraphPriority.CORE,
    relevance: int = 8,
    cred: ParagraphCredibility | None = None,
    reason: str = "test",
) -> ParagraphAnalysis:
    row = ParagraphAnalysis(
        paragraph_id=pid,
        page_number=1,
        section_title="GIL",
        priority=priority,
        topic_relevance=relevance,
        reason=reason,
    )
    if cred is not None:
        row.semantic_level = cred.semantic_level
        row.technical_correctness = cred.technical_correctness
        row.information_density = cred.information_density
        row.extract_mode = cred.extract_mode
        row.accuracy_reason = cred.reason
    return row


def test_metaphor_only_neutral_verified_is_032():
    cred = _cred(
        1,
        level=SemanticLevel.METAPHOR_ONLY,
        correctness=TechnicalCorrectness.VERIFIED,
        density=InformationDensity.NEUTRAL,
    )
    assert calculate_paragraph_score(cred) == 0.32


def test_contradiction_zeros_paragraph_score():
    cred = _cred(
        1,
        level=SemanticLevel.SPEC_EXACT,
        correctness=TechnicalCorrectness.CONTRADICTION,
        density=InformationDensity.HIGH,
    )
    assert calculate_paragraph_score(cred) == 0.0


def test_quality_contradiction_and_spec_average():
    rows = [
        _para(
            1,
            relevance=10,
            cred=_cred(1, level=SemanticLevel.SPEC_EXACT, density=InformationDensity.HIGH),
        ),
        _para(
            2,
            relevance=10,
            cred=_cred(2, correctness=TechnicalCorrectness.CONTRADICTION),
        ),
    ]
    # P=1.0 and P=0.0, equal weights → 0.5
    assert calculate_article_quality(rows) == 0.5


def test_quality_metaphor_heavy_fails_blog_threshold():
    rows = [
        _para(
            1,
            relevance=9,
            cred=_cred(
                1,
                level=SemanticLevel.METAPHOR_ONLY,
                density=InformationDensity.NEUTRAL,
            ),
        )
    ]
    q = calculate_article_quality(rows)
    assert q == 0.32
    accepted, quality, reason = decide_ingest_gate(rows, quality_min=0.65)
    assert quality == q
    assert accepted is False
    assert reason == INGEST_GATE_REJECT_REASON


def test_quality_unscored_fail_open():
    rows = [_para(1, relevance=9)]
    assert calculate_article_quality(rows) == 1.0
    accepted, _q, reason = decide_ingest_gate(rows, quality_min=0.65)
    assert accepted is True
    assert reason is None


def test_paper_url_has_no_whole_article_reject():
    rows = [
        _para(1, cred=_cred(1, correctness=TechnicalCorrectness.CONTRADICTION)),
    ]
    accepted, quality, reason = decide_ingest_gate(rows, quality_min=None)
    assert accepted is True
    assert quality == 0.0
    assert reason is None


def test_filter_cuts_contradiction_paragraphs():
    paper = build_input_paper_json_from_plain_text(
        f"## Internals\n\n{_GIL_FALSE_MECHANIC}\n\n## Aside\n\n{_GIL_OK_CONTEXT}"
    )
    paras = [p for pg in paper.pages for p in pg.paragraphs]
    assert len(paras) >= 2
    false_id = next(p.paragraph_id for p in paras if "eval_tick_barrier" in p.text)
    ok_id = next(p.paragraph_id for p in paras if "multiprocessing" in p.text)
    merged = [
        _para(
            false_id,
            cred=_cred(false_id, correctness=TechnicalCorrectness.CONTRADICTION),
        ),
        _para(
            ok_id,
            cred=_cred(
                ok_id,
                level=SemanticLevel.CONCEPTUAL_MODEL,
                density=InformationDensity.NEUTRAL,
            ),
        ),
    ]
    analysis = PaperStructureAnalysis(
        paragraphs=[row.model_copy(deep=True) for row in merged]
    )
    filtered = apply_structure_filter(
        paper,
        analysis,
        drop_contradictions=True,
        paragraph_overrides=merged,
    )
    assert "eval_tick_barrier" not in filtered
    assert "multiprocessing" in filtered


def test_extract_mode_defaults_to_full_when_omitted():
    parsed = ParagraphCredibility.model_validate(
        {
            "paragraph_id": 1,
            "semantic_level": "CONCEPTUAL_MODEL",
            "technical_correctness": "VERIFIED",
            "information_density": "NEUTRAL",
            "reason": "legacy row without extract_mode",
        }
    )
    assert parsed.extract_mode == ExtractMode.FULL

    body = (
        "Attention maps queries to keys with scaled dot-product. "
        "This restates the same identity for newcomers. "
        "A third sentence adds a toy example only."
    )
    paper = build_input_paper_json_from_plain_text(body)
    pid = next(p.paragraph_id for pg in paper.pages for p in pg.paragraphs)
    merged = [_para(pid, cred=parsed.model_copy(update={"paragraph_id": pid}))]
    filtered = apply_structure_filter(
        paper,
        PaperStructureAnalysis(paragraphs=[row.model_copy(deep=True) for row in merged]),
        paragraph_overrides=merged,
    )
    assert "scaled dot-product" in filtered
    assert "toy example" in filtered


def test_extract_mode_head_1_keeps_first_sentence_only():
    body = (
        "Scaled dot-product attention is softmax(QK^T / sqrt(d_k)) V. "
        "The rest of the paragraph only motivates why scaling helps. "
        "A closing example repeats the same formula in words."
    )
    paper = build_input_paper_json_from_plain_text(body)
    pid = next(p.paragraph_id for pg in paper.pages for p in pg.paragraphs)
    merged = [
        _para(
            pid,
            cred=_cred(pid, extract_mode=ExtractMode.HEAD_1),
        )
    ]
    filtered = apply_structure_filter(
        paper,
        PaperStructureAnalysis(paragraphs=[row.model_copy(deep=True) for row in merged]),
        paragraph_overrides=merged,
    )
    assert "softmax(QK^T / sqrt(d_k)) V." in filtered
    assert "motivates why scaling" not in filtered
    assert "closing example" not in filtered


def test_extract_mode_head_2_keeps_all_when_shorter():
    body = "The residual stream is the only communication bus between layers."
    paper = build_input_paper_json_from_plain_text(body)
    pid = next(p.paragraph_id for pg in paper.pages for p in pg.paragraphs)
    merged = [
        _para(
            pid,
            cred=_cred(pid, extract_mode=ExtractMode.HEAD_2),
        )
    ]
    filtered = apply_structure_filter(
        paper,
        PaperStructureAnalysis(paragraphs=[row.model_copy(deep=True) for row in merged]),
        paragraph_overrides=merged,
    )
    assert filtered == body


def test_merge_pass2_contradiction_onto_remaining():
    structure = [
        _para(1),
        _para(2, priority=ParagraphPriority.DROP, relevance=1),
    ]
    audit = PaperCredibilityAnalysis(
        paragraphs=[
            _cred(
                1,
                correctness=TechnicalCorrectness.CONTRADICTION,
                reason="eval_breaker treated as hardware IRQ",
            )
        ]
    )
    merged = merge_credibility_into_paragraphs(structure, audit, remaining_ids={1})
    by_id = {p.paragraph_id: p for p in merged}
    assert by_id[1].technical_correctness == TechnicalCorrectness.CONTRADICTION
    assert by_id[2].technical_correctness is None


def test_two_pass_gate_rejects_habr_like_gil_article(monkeypatch):
    body = (
        f"Как работает GIL\n\n{_GIL_FALSE_MECHANIC}\n\n"
        f"Аналогия\n\n{_GIL_METAPHOR}\n\n"
        f"Обход GIL\n\n{_GIL_OK_CONTEXT}\n\n"
        "Дополнительный абзац про multiprocessing.Queue и SharedMemory для обмена данными."
    )
    paper = build_input_paper_json_from_plain_text(body)
    ids = [p.paragraph_id for pg in paper.pages for p in pg.paragraphs]
    assert len(ids) >= 2
    false_ids = {
        p.paragraph_id
        for pg in paper.pages
        for p in pg.paragraphs
        if "eval_tick_barrier" in p.text or "кассу супермаркета" in p.text
    }

    calls: list[str] = []

    def fake_chain(_primary, _system, _user, _anchor, schema, _label, **kwargs):
        name = getattr(schema, "__name__", "")
        calls.append(name)
        assert kwargs.get("chat_manager") is not None
        assert str(kwargs.get("chat_label") or "").startswith("ingest_gate:")
        if schema is PaperStructureAnalysis:
            return PaperStructureAnalysis(
                paragraphs=[
                    ParagraphAnalysis(
                        paragraph_id=pid,
                        page_number=1,
                        section_title="GIL",
                        priority=ParagraphPriority.CORE,
                        topic_relevance=9,
                        reason="core internals",
                    )
                    for pid in ids
                ]
            )
        if schema is PaperCredibilityAnalysis:
            rows = []
            for pid in ids:
                if pid in false_ids:
                    text_hit = next(
                        p.text
                        for pg in paper.pages
                        for p in pg.paragraphs
                        if p.paragraph_id == pid
                    )
                    if "кассу супермаркета" in text_hit:
                        rows.append(
                            _cred(
                                pid,
                                level=SemanticLevel.METAPHOR_ONLY,
                                density=InformationDensity.NEUTRAL,
                                reason="supermarket queue metaphor",
                            )
                        )
                    else:
                        rows.append(
                            _cred(
                                pid,
                                correctness=TechnicalCorrectness.CONTRADICTION,
                                reason="eval_breaker as hardware interrupt",
                            )
                        )
                else:
                    rows.append(
                        _cred(
                            pid,
                            level=SemanticLevel.CONCEPTUAL_MODEL,
                            correctness=TechnicalCorrectness.SIMPLIFIED,
                            density=InformationDensity.NEUTRAL,
                        )
                    )
            return PaperCredibilityAnalysis(paragraphs=rows)
        raise AssertionError(name)

    monkeypatch.setattr(
        "knowledge_engine.src.parsers.paper_structure_analyzer.is_gemini_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.parsers.paper_structure_analyzer.run_gemini_structured_with_chain",
        fake_chain,
    )

    result = run_inbound_ingest_gate(
        body,
        "CPython GIL internals",
        page_url="https://habr.com/ru/articles/938980/",
        quality_min=0.65,
    )
    assert calls == ["PaperStructureAnalysis", "PaperCredibilityAnalysis"]
    assert result.accepted is False
    assert result.reject_reason == INGEST_GATE_REJECT_REASON
    assert result.quality < 0.65
    assert result.body == ""
    by_id = {p.paragraph_id: p for p in result.paragraphs}
    metaphor = next(
        p
        for pg in paper.pages
        for p in pg.paragraphs
        if "кассу супермаркета" in p.text
    )
    false = next(
        p
        for pg in paper.pages
        for p in pg.paragraphs
        if "eval_tick_barrier" in p.text
    )
    assert by_id[metaphor.paragraph_id].semantic_level == SemanticLevel.METAPHOR_ONLY
    assert calculate_paragraph_score(by_id[metaphor.paragraph_id].as_credibility()) == 0.32
    assert (
        by_id[false.paragraph_id].technical_correctness
        == TechnicalCorrectness.CONTRADICTION
    )
    assert calculate_paragraph_score(by_id[false.paragraph_id].as_credibility()) == 0.0


def test_blog_pipeline_skips_map_reduce_on_gate_reject(monkeypatch):
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        ingest_blog_with_spatial_mapping,
    )
    from knowledge_engine.src.parsers.ingest_gate import IngestGateResult

    html = (
        "<html><body><article>"
        + "".join(f"<p>{_GIL_FALSE_MECHANIC}</p>" for _ in range(3))
        + "</article></body></html>"
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline._apply_inbound_ingest_gate",
        lambda annotated, page_url, title, content_raw: (
            annotated,
            IngestGateResult(
                accepted=False,
                quality=0.12,
                body="",
                reject_reason=INGEST_GATE_REJECT_REASON,
            ),
        ),
    )
    map_calls: list[str] = []

    def _boom(*_a, **_k):
        map_calls.append("map")
        raise AssertionError("Map-Reduce must not run after ingest gate reject")

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.map_reduce_summarize_blog_outcome",
        _boom,
    )
    _ann, summary, saved = ingest_blog_with_spatial_mapping(
        "Как работает GIL и как от него избавиться",
        "https://habr.com/ru/articles/938980/",
        raw_html=html,
        save_lancedb=False,
    )
    assert summary is None
    assert saved == 0
    assert map_calls == []


def test_remaining_ids_drop_priority():
    paper = build_input_paper_json_from_plain_text(
        f"Core\n\n{_GIL_OK_CONTEXT}\n\nNoise\n\ncopyright footer license"
    )
    paras = [p for pg in paper.pages for p in pg.paragraphs]
    rows = []
    for i, p in enumerate(paras):
        pri = ParagraphPriority.DROP if i == len(paras) - 1 else ParagraphPriority.CORE
        rows.append(
            ParagraphAnalysis(
                paragraph_id=p.paragraph_id,
                page_number=1,
                section_title=p.section_title,
                priority=pri,
                topic_relevance=8,
                reason="t",
            )
        )
    analysis = PaperStructureAnalysis(paragraphs=rows)
    kept = remaining_paragraph_ids(paper, analysis)
    drop_ids = {r.paragraph_id for r in rows if r.priority == ParagraphPriority.DROP}
    assert drop_ids.isdisjoint(kept)
    assert kept


def test_spatial_jobs_skip_gemma_map_when_gate_rejected(monkeypatch):
    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        SpatialDiagramIngestJob,
        run_spatial_diagram_ingest_jobs_async,
    )
    from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle

    job = SpatialDiagramIngestJob(
        source_id="src",
        page_url="https://habr.com/ru/articles/938980/",
        annotated=AnnotatedArticle(annotated_markdown="x" * 200),
        article_title="GIL",
        gate_rejected=True,
        gate_reject_reason=INGEST_GATE_REJECT_REASON,
        gate_quality=0.1,
    )
    map_calls: list[int] = []

    async def _no_map(jobs, **_k):
        map_calls.append(len(jobs))
        return {}

    async def _trust(urls):
        return {u: 1.0 for u in urls}

    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_pipeline.map_reduce_jobs_pooled_async",
        _no_map,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.services.openalex_evaluator.prefetch_trust_scores_async",
        _trust,
    )
    import asyncio

    saved = asyncio.run(run_spatial_diagram_ingest_jobs_async([job]))
    assert saved == {"https://habr.com/ru/articles/938980/": 0}
    assert map_calls == []


@pytest.mark.skipif(
    not bool(
        __import__("os").environ.get("GEMINI_API_KEY")
        or __import__("os").environ.get("GOOGLE_API_KEY")
    ),
    reason="GEMINI_API_KEY required for live Habr ingest-gate check",
)
def test_live_habr_gil_article_rejected_before_map(monkeypatch):
    """Live Flash Lite: Habr GIL explainer should fail Q via vector traits, skip Gemma MAP."""
    import httpx

    from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
        build_annotated_from_content,
    )
    from knowledge_engine.src.parsers.paper_structure_analyzer import (
        run_inbound_ingest_gate,
    )

    url = "https://habr.com/ru/articles/938980/"
    try:
        html = httpx.get(url, timeout=30.0, follow_redirects=True).text
    except httpx.HTTPError as exc:
        pytest.skip(f"Habr fetch unavailable: {exc}")
    assert "GIL" in html

    annotated = build_annotated_from_content(html, url)
    body = "\n\n".join(annotated.paragraph_map.values()) or annotated.annotated_markdown
    assert len(body) > 400
    map_calls: list[str] = []
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.blog_spatial_summarizer.map_reduce_jobs_pooled_async",
        lambda *_a, **_k: map_calls.append("map") or (_ for _ in ()).throw(
            AssertionError("Gemma Map-Reduce must not start")
        ),
    )
    result = run_inbound_ingest_gate(
        body,
        "CPython GIL internals eval_breaker",
        page_url=url,
        quality_min=0.65,
    )
    scored = [p for p in result.paragraphs if p.as_credibility() is not None]
    assert scored, "pass 2 must grade remaining paragraphs"
    assert map_calls == []
    metaphor_scores = [
        calculate_paragraph_score(p.as_credibility())
        for p in scored
        if p.semantic_level == SemanticLevel.METAPHOR_ONLY
    ]
    contradiction_scores = [
        calculate_paragraph_score(p.as_credibility())
        for p in scored
        if p.technical_correctness == TechnicalCorrectness.CONTRADICTION
    ]
    if metaphor_scores:
        assert min(metaphor_scores) <= 0.32 + 1e-9
    if contradiction_scores:
        assert all(s == 0.0 for s in contradiction_scores)
    assert result.quality < 0.65
    assert result.accepted is False
    assert result.reject_reason == INGEST_GATE_REJECT_REASON
