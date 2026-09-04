"""Pydantic contracts for research-contour Unraveling, harvest, clarify, REPL."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from knowledge_engine.graph.nodes.unraveling import (
    render_unraveling_markdown,
    unraveling_node,
)
from knowledge_engine.graph.nodes.v04_unraveling import unraveling_node_v04
from knowledge_engine.schemas import AnalysisReport, EngineState, TradeOffOption
from knowledge_engine.schemas.research_schemas import (
    ClarificationConstraintsResponse,
    HarvestedLinkItem,
    HarvestedLinksResponse,
    ReplFollowUpResponse,
    render_clarification_constraints,
)
from knowledge_engine.schemas.unraveling_schemas import (
    TradeoffFailureMode,
    UnravelingNodeResponse,
)
from knowledge_engine.src.analytics.repl import answer_follow_up
from knowledge_engine.src.curriculum import gemini_web_blog_harvest
from knowledge_engine.src.curriculum.gemini_web_blog_harvest import (
    hits_from_harvest_response,
)


def _dense_breakdown() -> str:
    return (
        "алгоритм кэша инвалидации структур данных аллокация "
        "конфиг воркеров очередь backpressure "
    ) * 50


def _valid_unraveling() -> UnravelingNodeResponse:
    return UnravelingNodeResponse(
        summary=(
            "Выбранный вариант держит индекс в процессе и режет round-trip "
            "ценой пика RSS на Apple Silicon."
        ),
        ram_and_latency_impact=(
            "На Mac M-series unified memory делит бюджет с GPU: HNSW на 2M "
            "векторов плюс LanceDB mmap даёт пик RSS около 6–8 ГБ и p99 "
            "поиска 8–12 мс при тёплом кэше страниц."
        ),
        failure_modes=[
            TradeoffFailureMode(
                scenario="Холодный старт после eviction страниц mmap",
                impact="p99 latency вырастает в 5–10 раз, CPU в page-in",
                mitigation="Pinned warm set + prefetch горячих сегментов LanceDB",
            )
        ],
        technical_breakdown_markdown=_dense_breakdown(),
    )


def _option() -> TradeOffOption:
    return TradeOffOption(
        id=1,
        pattern_name="Локальный HNSW + LanceDB",
        category="SOTA",
        fundamental_idea="Держать индекс в процессе, диск как durability",
        pros=["низкий round-trip", "нет сети"],
        cons_and_risks=["пик RSS", "холодный mmap"],
        operational_cost="6–8 ГБ RSS на M-series",
    )


def test_unraveling_node_structure_validation() -> None:
    parsed = _valid_unraveling()
    assert parsed.failure_modes
    assert parsed.ram_and_latency_impact
    md = render_unraveling_markdown(parsed)
    assert "## Резюме" in md
    assert "## RAM и latency" in md
    assert "## Failure modes" in md
    assert "## Технический разбор" in md
    assert "Pinned warm set" in md

    state = EngineState(
        user_problem="Локальный RAG кэш на Mac",
        report=AnalysisReport(abstractions=[], options=[_option()]),
        selected_option_id=1,
    ).model_dump()
    store = MagicMock()
    store.hybrid_search_nodes.return_value = []
    store.get_hierarchical_context.return_value = ""
    with (
        patch(
            "knowledge_engine.graph.nodes.unraveling.VectorStore",
            return_value=store,
        ),
        patch(
            "knowledge_engine.graph.nodes.unraveling.run_structured_hybrid",
            return_value=parsed,
        ) as hybrid,
    ):
        out = unraveling_node(state)
    hybrid.assert_called_once()
    schema = hybrid.call_args.args[3]
    assert schema is UnravelingNodeResponse
    assert out["unraveled_details"] == md
    assert "RAM и latency" in out["unraveled_details"]
    assert unraveling_node_v04 is unraveling_node

    with pytest.raises(ValidationError):
        UnravelingNodeResponse(
            summary="короткое резюме варианта которое проходит min_length",
            ram_and_latency_impact="память и задержки описаны достаточно длинно",
            failure_modes=[
                TradeoffFailureMode(
                    scenario="нехватка RAM при росте корпуса",
                    impact="OOM killer режет процесс",
                    mitigation="квота mmap и вытеснение сегментов",
                )
            ],
            technical_breakdown_markdown="слишком коротко",
        )


def test_curriculum_harvest_pydantic_parsing() -> None:
    harvest_src = inspect.getsource(gemini_web_blog_harvest)
    assert "_MD_LINK_RE" not in harvest_src
    assert "re.compile" not in harvest_src
    assert "findall" not in harvest_src

    payload = HarvestedLinksResponse(
        items=[
            HarvestedLinkItem(
                title="Microservices",
                url="https://martinfowler.com/articles/microservices.html",
                relevance_reason="Канонический разбор границ сервисов и coupling.",
            ),
            HarvestedLinkItem(
                title="Homepage should drop",
                url="https://martinfowler.com/",
                relevance_reason="Домашняя страница без статьи, хост отфильтрует.",
            ),
        ]
    )
    hits = hits_from_harvest_response(payload, cap=8)
    assert len(hits) == 1
    assert hits[0].url == "https://martinfowler.com/articles/microservices.html"
    assert hits[0].title == "Microservices"
    assert "coupling" in hits[0].snippet
    assert hits[0].source_tier == "gemini_web"


def test_intent_constraints_schema_enforcement() -> None:
    with pytest.raises(ValidationError):
        ClarificationConstraintsResponse(constraints=["только Python", "LanceDB"])
    with pytest.raises(ValidationError):
        ClarificationConstraintsResponse(
            constraints=[f"ограничение номер {i}" for i in range(9)]
        )
    parsed = ClarificationConstraintsResponse(
        constraints=[
            "Python 3.14 локально",
            "LanceDB на диске, без облака",
            "Mac M-series 16 ГБ unified memory",
            "p99 поиска < 20 мс на тёплом кэше",
        ]
    )
    assert 3 <= len(parsed.constraints) <= 8
    rendered = render_clarification_constraints(parsed)
    assert rendered.startswith("- ")
    assert "LanceDB" in rendered


def test_repl_follow_up_uses_structured_schema() -> None:
    fake = ReplFollowUpResponse(
        answer="В контексте HNSW держит граф в RAM, диск только WAL.",
        sources_cover_question=True,
    )
    with patch(
        "knowledge_engine.src.analytics.repl.run_gemini_flash_structured",
        return_value=fake,
    ) as call:
        text = answer_follow_up("Как устроен индекс?", "chunks…", "anchor")
    assert text == fake.answer
    assert call.call_args.args[3] is ReplFollowUpResponse
