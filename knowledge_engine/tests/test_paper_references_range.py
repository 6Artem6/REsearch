"""References block range detection (not suffix cut through document end)."""

from knowledge_engine.src.parsers.paper_input_json import (
    build_input_paper_json_from_plain_text,
    find_references_range_from_paper,
)
from knowledge_engine.src.parsers.paper_structure_analyzer import (
    apply_structure_filter,
)
from knowledge_engine.src.parsers.paper_structure_schema import (
    PaperStructureAnalysis,
    ParagraphAnalysis,
    ParagraphPriority,
)


def _synthetic_paper_with_appendix():
    text = """
1 Introduction

We study GraphRAG systems for enterprise retrieval pipelines.

2 Methods

Our pipeline builds a knowledge graph and runs hybrid retrieval.

References

1. Patrick Lewis et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Proceedings of NeurIPS, pages 9459-9474, 2020.

2. Yunfan Gao et al. Retrieval-augmented generation survey. arXiv preprint arXiv:2312.10997, 2023.

Appendix A: Proofs

Here we prove the main theorem about reciprocal rank fusion bounds.

Appendix B

Additional lemmas for the dependency parsing extraction stage.
"""
    return build_input_paper_json_from_plain_text(text)


def test_references_range_stops_before_appendix():
    paper = _synthetic_paper_with_appendix()
    rng = find_references_range_from_paper(paper)
    assert rng is not None
    start, end = rng
    paras = [p for pg in paper.pages for p in pg.paragraphs]
    appendix_para = next(p for p in paras if "prove the main theorem" in p.text)
    assert appendix_para.paragraph_id > end
    assert start < end


def test_filter_keeps_appendix_after_references():
    paper = _synthetic_paper_with_appendix()
    paras = [p for pg in paper.pages for p in pg.paragraphs]
    rows = [
        ParagraphAnalysis(
            paragraph_id=p.paragraph_id,
            page_number=1,
            section_title=p.section_title,
            priority=ParagraphPriority.CORE,
            topic_relevance=8,
            reason="appendix/core body",
        )
        for p in paras
        if "prove the main theorem" in p.text
        or "dependency parsing extraction" in p.text
    ]
    analysis = PaperStructureAnalysis(paragraphs=rows)
    filtered = apply_structure_filter(paper, analysis)
    assert "prove the main theorem" in filtered
    assert "Patrick Lewis" not in filtered
    assert "dependency parsing extraction" in filtered


def test_manual_analysis_appendix_core():
    paper = _synthetic_paper_with_appendix()
    paras = [p for pg in paper.pages for p in pg.paragraphs]
    appendix_id = next(
        p.paragraph_id for p in paras if "prove the main theorem" in p.text
    )
    rows = [
        ParagraphAnalysis(
            paragraph_id=p.paragraph_id,
            page_number=1,
            section_title=p.section_title,
            priority=ParagraphPriority.CORE,
            topic_relevance=8,
            reason="test",
        )
        for p in paras
        if p.paragraph_id == appendix_id
    ]
    analysis = PaperStructureAnalysis(paragraphs=rows)
    filtered = apply_structure_filter(paper, analysis)
    assert "prove the main theorem" in filtered
