"""Unit tests for ArxivQueryBuilder precision query construction."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode

from knowledge_engine.services.search.arxiv_query_builder import (
    ArxivQueryBuilder,
    ArxivQueryParams,
    heuristic_arxiv_params_from_keywords,
)


def test_build_combines_ti_abs_cat_with_and_or():
    params = ArxivQueryParams(
        title_keywords=["transformer attention"],
        abstract_keywords=["multi-head", "self-attention"],
        categories=["cs.AI", "cs.CL", "cs.LG"],
        exclude_terms=["survey"],
        start_year=2023,
        end_year=2026,
        sort_by="relevance",
        sort_order="descending",
    )
    built = ArxivQueryBuilder(params).build(max_results=10)
    q = built.search_query
    assert 'ti:"transformer attention"' in q
    assert "(abs:multi-head OR abs:self-attention)" in q
    assert "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG)" in q
    assert " AND " in q
    assert "ANDNOT all:survey" in q
    assert "submittedDate:[202301010000 TO 202612312359]" in q
    assert built.sort_by == "relevance"
    assert built.sort_order == "descending"
    assert built.start == 0

    encoded = built.encode(max_results=10)
    parsed = parse_qs(encoded)
    assert parsed["search_query"][0] == q
    assert parsed["sortBy"][0] == "relevance"
    assert parsed["sortOrder"][0] == "descending"
    assert parsed["start"][0] == "0"
    assert parsed["max_results"][0] == "10"


def test_operators_are_uppercase_and_phrases_quoted():
    q = ArxivQueryBuilder(
        ArxivQueryParams(
            title_keywords=["graph neural network"],
            abstract_keywords=["GNN"],
            exclude_terms=["homework assignment"],
        )
    ).build_search_query()
    assert " AND " in q
    assert " OR " not in q or "abs:" in q  # single abs → no OR needed
    assert "ANDNOT" in q
    assert "andnot" not in q
    assert 'ANDNOT all:"homework assignment"' in q
    assert 'ti:"graph neural network"' in q


def test_pagination_start_and_sort_by_submitted_date():
    params = ArxivQueryParams(
        abstract_keywords=["retrieval augmented generation"],
        sort_by="submittedDate",
        sort_order="descending",
        start=20,
    )
    built = ArxivQueryBuilder(params).build(start=20, max_results=5)
    assert built.start == 20
    assert built.sort_by == "submittedDate"
    qs = urlencode(built.as_query_params(max_results=5))
    assert "start=20" in qs
    assert "sortBy=submittedDate" in qs


def test_free_text_fallback_when_params_empty():
    built = ArxivQueryBuilder(ArxivQueryParams()).build(
        free_text_fallback="electron thermal conductivity"
    )
    assert built.search_query.startswith("all:")
    assert "electron" in built.search_query


def test_heuristic_params_seed_keywords():
    params = heuristic_arxiv_params_from_keywords(
        ["transformers", "attention mechanism"],
        free_text="unused",
    )
    assert params.title_keywords
    assert params.abstract_keywords
    assert "survey" in params.exclude_terms
    q = ArxivQueryBuilder(params).build_search_query()
    assert "ti:" in q
    assert "abs:" in q


def test_invalid_category_dropped():
    q = ArxivQueryBuilder(
        ArxivQueryParams(categories=["cs.AI", "not a cat!", "stat.ML"])
    ).build_search_query()
    assert "cat:cs.AI" in q
    assert "cat:stat.ML" in q
    assert "not a cat" not in q
