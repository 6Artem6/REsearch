"""MAP/REDUCE prompt layout for blog spatial summarizer."""

from __future__ import annotations

from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.schemas.extraction import (
    AggregatedKnowledgeBase,
    KnowledgeAtom,
    ParagraphInspectionResult,
    ScopeType,
    format_takeaways_for_tutor,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    FinalArticleSummaryResponse,
    MapWindowResponse,
    normalize_final_knowledge,
    normalize_map_knowledge,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _CRITICAL_REDUCE_RULES,
    _MAP_SYSTEM,
    _REDUCE_DEDUP_SYSTEM,
    _REDUCE_SYNTHESIS_SYSTEM,
    _REDUCE_SYSTEM,
    MapReduceArticleJob,
    _build_reduce_user_prompt,
    _format_reduce_summaries_block,
    _prompt_for_window,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.article_ingestion.section_context import (
    infer_article_title,
    resolve_section_heading_for_paragraph_ids,
)
from knowledge_engine.services.article_ingestion.triage_schemas import TOCNode
from knowledge_engine.services.lecture_rag_context import _format_document_summary
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle


def test_map_prompt_window_text_before_diagram_context():
    job = MapReduceArticleJob(
        job_id="https://example.com/post",
        title="Transformer Attention",
        url="https://example.com/post",
        windows=[],
    )
    w = TokenWindowChunk(
        window_index=2,
        body="Article: Transformer Attention\n\n[P_1] Core idea.",
        section_heading="Architecture",
        attached_diagrams="[ATTACHED_DIAGRAMS]\n### FIG_1\nAxis labels",
    )
    prompt = _prompt_for_window(job, w)
    text_pos = prompt.index("<window_text>")
    diagram_pos = prompt.index("<diagram_context>")
    assert text_pos < diagram_pos
    assert "ARTICLE_TITLE: Transformer Attention" in prompt
    assert "SECTION: Architecture" in prompt
    assert "WINDOW_INDEX: 2" in prompt
    assert "CHUNK_ID:" in prompt
    assert "_map_3" in prompt  # window_index 2 → 1-based map_3
    assert "[ATTACHED_DIAGRAMS]" in prompt
    assert (
        "Article: Transformer Attention"
        not in prompt.split("<window_text>")[1].split("</window_text>")[0]
    )


def test_map_reduce_prompts_require_scope_tags():
    assert "[SCOPE: PRINCIPLE]" in _MAP_SYSTEM
    assert "[SCOPE: MECHANIC]" in _MAP_SYSTEM
    assert "[SCOPE: INSTANCE]" in _MAP_SYSTEM
    assert "knowledge_atoms" in _MAP_SYSTEM
    assert "DO NOT output <thought>" in _MAP_SYSTEM
    assert "knowledge_atoms" in _REDUCE_SYSTEM
    assert "[SCOPE: PRINCIPLE|MECHANIC|INSTANCE]" in _CRITICAL_REDUCE_RULES
    assert "Do NOT drop any unique fact" in _REDUCE_DEDUP_SYSTEM
    assert "source_chunk_ids" in _REDUCE_DEDUP_SYSTEM
    assert "Do NOT extract new facts" in _REDUCE_SYNTHESIS_SYSTEM


def test_reduce_summaries_use_window_index_and_role():
    windows = [
        TokenWindowChunk(window_index=0, body="", section_heading="Intro"),
        TokenWindowChunk(window_index=1, body="", section_heading="Intro"),
        TokenWindowChunk(window_index=2, body="", section_heading="Benchmarks"),
    ]
    maps = [
        MapWindowResponse(
            window_role="Введение",
            window_summary="Summary A",
            knowledge_atoms=[
                KnowledgeAtom(
                    scope=ScopeType.PRINCIPLE,
                    statement="Изоляция агента от хоста обязательна",
                )
            ],
        ),
        None,
        MapWindowResponse(window_role="Бенчмарки", window_summary="99% acc"),
    ]
    block = _format_reduce_summaries_block(windows, maps)
    assert "### Window 0 [Введение]" in block
    assert "### Window 2 [Бенчмарки]" in block
    assert "### Window 1" not in block
    assert "## Section: Intro" in block
    assert "## Section: Benchmarks" in block
    assert "[SCOPE: PRINCIPLE]" in block


def test_reduce_user_prompt_has_critical_rules_at_end():
    job = MapReduceArticleJob(
        job_id="u",
        title="Paper",
        url="https://x",
        windows=[],
    )
    prompt = _build_reduce_user_prompt(job, "### Window 0 [tag]\nbody")
    assert prompt.rstrip().endswith("</critical_reduce_rules>")
    assert "target_diagrams_for_vlm — always []" in prompt
    assert "<critical_reduce_rules>" in prompt
    rules_pos = prompt.index("<critical_reduce_rules>")
    window_pos = prompt.index("### Window 0")
    assert rules_pos > window_pos


def test_resolve_section_from_toc_nodes():
    toc = [
        TOCNode(title="Introduction", level=1, start_p_id="P_1"),
        TOCNode(title="Results", level=2, start_p_id="P_10"),
    ]
    para_map = {f"P_{i}": f"text {i}" for i in range(1, 20)}
    heading = resolve_section_heading_for_paragraph_ids(
        ["P_12", "P_13"],
        toc,
        paragraph_map=para_map,
    )
    assert heading == "Results"


def test_infer_article_title_prefers_toc_h1():
    ann = AnnotatedArticle(
        paragraph_map={"P_1": "Some long paragraph text."},
        annotated_markdown="[P_1] Some long paragraph text.",
    )
    toc = [TOCNode(title="Real Paper Title", level=1, start_p_id="P_1")]
    title = infer_article_title(
        annotated=ann,
        toc_nodes=toc,
        page_url="https://example.com/doc",
    )
    assert title == "Real Paper Title"


def test_paragraph_inspection_result_validates_atoms():
    raw = {
        "atoms": [
            {
                "scope": "PRINCIPLE",
                "statement": "Межсетевой перехват защищает от утечки tool-calls",
                "context_quote": "isolation boundary",
            },
            {
                "scope": "INSTANCE",
                "statement": "AJV даёт задержку 8.3 мс на 32 уровнях вложенности",
            },
        ]
    }
    result = ParagraphInspectionResult.model_validate(raw)
    assert len(result.atoms) == 2
    assert result.atoms[0].scope is ScopeType.PRINCIPLE
    assert result.atoms[1].scope is ScopeType.INSTANCE


def test_normalize_final_mirrors_atoms_into_tagged_takeaways():
    final = FinalArticleSummaryResponse(
        executive_summary="Обзор",
        key_takeaways=["сырой без тега достаточно длинный тезис для теста"],
        knowledge_atoms=[
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Валидация схем идёт до исполнения tool-call",
            ),
            KnowledgeAtom(
                scope=ScopeType.INSTANCE,
                statement="Библиотека AJV: 8.3 мс на глубокой вложенности",
            ),
        ],
    )
    out = normalize_final_knowledge(final)
    assert all(t.startswith("[SCOPE:") for t in out.key_takeaways)
    assert any("MECHANIC" in t for t in out.key_takeaways)
    assert any("INSTANCE" in t for t in out.key_takeaways)


def test_normalize_map_extracts_inline_scope_tags():
    mapped = MapWindowResponse(
        window_role="Архитектура",
        window_summary=(
            "[SCOPE: PRINCIPLE] Изоляция агента обязательна.\n" "Прочий текст без тега."
        ),
        knowledge_atoms=[],
    )
    out = normalize_map_knowledge(mapped)
    assert len(out.knowledge_atoms) >= 1
    assert out.knowledge_atoms[0].scope is ScopeType.PRINCIPLE


def test_tutor_context_splits_into_three_blocks():
    takeaways = [
        "[SCOPE: PRINCIPLE] Изоляция процесса агента от хоста",
        "[SCOPE: MECHANIC] Перехват tool-calls на границе сети",
        "[SCOPE: INSTANCE] AJV: 8.3 мс при 32 уровнях",
    ]
    block = format_takeaways_for_tutor(takeaways)
    assert "FUNDAMENTAL PRINCIPLES" in block
    assert "GENERALIZED MECHANICS" in block
    assert "PRACTICAL CASES" in block
    assert "AJV" in block
    kb = AggregatedKnowledgeBase.from_tagged_strings(takeaways)
    assert len(kb.principles) == 1
    assert len(kb.mechanics) == 1
    assert len(kb.evidence_cases) == 1


def test_format_document_summary_uses_triangulation_blocks():
    ds = DocumentSummary(
        title="AI Agents isolation",
        url="https://example.com/agents",
        key_takeaways=[
            "[SCOPE: PRINCIPLE] Изоляция и межсетевой перехват — базис безопасности агентов",
            "[SCOPE: INSTANCE] AJV даёт 8.3 мс на валидации схемы",
        ],
    )
    text = _format_document_summary(ds, 1)
    assert "FUNDAMENTAL PRINCIPLES" in text
    assert "PRACTICAL CASES" in text
    assert "8.3" in text
    assert "Выжимка:" not in text
