"""lecture_search_orchestrator Exa-ветка: Domain Discovery (Flash Lite,
по аналогии с DEEP) + round-robin по доменам (fair_domain_round_robin/
fill_round_robin_tail/merge_multi_vector_exa_hits, services.search.
exa_transform) + Async Fetch/Passage Extraction (lecture_passage_fetch.py) —
Fast & High-Quality lecture waterfall, БЕЗ Map-Reduce (никаких генеративных
LLM-проходов по документам).

Регресс на реальный баг: добор по «B-Tree индексы» из доменов postgresql.org
/ sqlite.org / github.com вернул 3 статьи исключительно с postgresql.org
разной версионности (v18/v17/v15) — round-robin по хосту в лёгком пути
отсутствовал, диверсификации не было.

Version-path dedup (/docs/18/ vs /docs/17/) сознательно НЕ реализуется (см.
решение пользователя — "без дедупликации версий, это лишнее"): postgresql.org
может занять несколько версионных URL внутри своей доменной квоты, но
round-robin гарантирует, что один домен не вытеснит остальные целиком.

Discovery и passage extraction застаблены по умолчанию (autouse-фикстура) —
оба реально бьют в сеть/LLM, а предмет большинства тестов здесь —
диверсификация; отдельные тесты ниже целятся именно в Discovery/passage-
extraction wiring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

import knowledge_engine.src.node_deep_dive.lecture_search_orchestrator as orch
from knowledge_engine.schemas.llm_contracts.exa_search import (
    ExaSearchContextExpansion,
)
from knowledge_engine.services.search.exa_client import ExaSearchHit
from knowledge_engine.services.search.exa_transform import ExaQuerySpec
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


@dataclass
class _FakeExaResponse:
    hits: list[ExaSearchHit]


class _FakeClient:
    def __init__(self, hits: list[ExaSearchHit]):
        self._hits = hits
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return True

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return _FakeExaResponse(hits=list(self._hits))


@dataclass
class _FakeQueryPlan:
    specs: list[ExaQuerySpec] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    core_theme: str = ""


def _hit(url: str, title: str, score: float) -> ExaSearchHit:
    return ExaSearchHit(
        url=url, title=title, highlights=[f"{title} snippet text."], score=score
    )


def _stub_postprocess(hits, *, cap):
    """Заглушка вместо реального URL/practical rank — сохраняем домены как
    есть, только капаем; предмет теста — диверсификация ДО этого шага."""
    out = []
    for h in hits[:cap]:
        out.append(
            CurriculumSearchHit(
                url=h.url,
                title=h.title,
                snippet=(h.highlights[0] if h.highlights else ""),
                exa_relevance_score=h.score,
            )
        )
    return out


@pytest.fixture(autouse=True)
def _no_op_discovery_and_passages(monkeypatch):
    """Discovery (Flash Lite expand + LanceDB + HTTP liveness + authority
    classify) и passage extraction (httpx + Trafilatura + BGE-M3) реально
    бьют в сеть/LLM — большинство тестов здесь не про них, поэтому по
    умолчанию оба no-op (пустой Discovery → include_domains=[] как раньше;
    пустой passages_by_url → снипеты остаются Exa-хайлайтами)."""
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda _context: ExaSearchContextExpansion(),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains",
        _async_return([]),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.filter_pass1_official_hosts",
        lambda hosts: [],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.absorb_new_exa_hosts",
        lambda urls, **kw: None,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.lecture_passage_fetch.fetch_and_extract_passages",
        _async_return({}),
    )


def test_domain_monoculture_gets_diversified(monkeypatch):
    """Реальный баг: сырой Exa-ответ = 3x postgresql.org (v18/v17/v15) +
    1x sqlite.org + 1x github.com. Без round-robin верхние по score 3 слота
    занял бы один postgresql.org. С диверсификацией — по одному с каждого
    домена (ровно как в примере задачи: 1x PG, 1x SQLite, 1x GitHub)."""
    raw_hits = [
        _hit("https://www.postgresql.org/docs/18/btree.html", "PG 18 B-Tree", 0.95),
        _hit("https://www.postgresql.org/docs/17/btree.html", "PG 17 B-Tree", 0.93),
        _hit("https://www.postgresql.org/docs/15/btree.html", "PG 15 B-Tree", 0.91),
        _hit("https://www.sqlite.org/btreemodule.html", "SQLite B-Tree", 0.80),
        _hit(
            "https://github.com/postgres/postgres/blob/master/src/backend/access/nbtree/README",
            "PG nbtree README",
            0.75,
        ),
    ]
    client = _FakeClient(raw_hits)

    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch,
        "build_exa_query_plan",
        _async_return(
            _FakeQueryPlan(
                specs=[
                    ExaQuerySpec(
                        role="en_technical",
                        query="B-Tree index internals",
                        highlight_query="internals",
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )
    monkeypatch.setattr(orch, "EXA_RECALL_MAX_PER_DOMAIN", 1)

    out = asyncio.run(
        orch._exa_sources_multi_vector("B-Tree индексы", 3, anchor="test")
    )

    domains = [_host(s.url) for s in out]
    assert len(out) == 3
    assert set(domains) == {"postgresql.org", "sqlite.org", "github.com"}
    assert len(set(domains)) == 3  # ни один домен не занял больше одного слота


def test_final_cap_does_not_reconcentrate_after_wide_diversification(monkeypatch):
    """Реальный баг: 2/3 источника с Habr. Широкий recall-пул диверсифицирован
    round-robin'ом (max_per_domain=EXA_RECALL_MAX_PER_DOMAIN=2 — рассчитан на
    DEEP cap~4), но ФИНАЛЬНОЕ сужение до лекционного cap=3 раньше шло только
    по composite score (postprocess_exa_hits_for_external_recall), без
    повторного round-robin — если топ-score занимали 2 статьи с одного
    домена, обе проходили в финал (2 из 3 = уже перекос, в отличие от 2 из 4
    у DEEP). Финальный срез должен применять более строгий
    EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN (=1 по умолчанию)."""
    raw_hits = [
        _hit("https://habr.com/ru/articles/1", "Habr 1", 0.99),
        _hit("https://habr.com/ru/articles/2", "Habr 2", 0.98),
        _hit("https://example-a.com/article", "A", 0.5),
        _hit("https://example-b.com/article", "B", 0.4),
        _hit("https://example-c.com/article", "C", 0.3),
    ]
    client = _FakeClient(raw_hits)
    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch,
        "build_exa_query_plan",
        _async_return(
            _FakeQueryPlan(
                specs=[
                    ExaQuerySpec(
                        role="en_technical",
                        query="some topic internals",
                        highlight_query="internals",
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )
    # Дефолты не переопределяю: EXA_RECALL_MAX_PER_DOMAIN=2 (широкий пул),
    # EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN=1 (финальный срез) — именно тот
    # разрыв, который воспроизводит баг без явного override.

    out = asyncio.run(orch._exa_sources_multi_vector("some topic", 3, anchor="test"))

    domains = [_host(s.url) for s in out]
    assert len(out) == 3
    assert len(set(domains)) == 3  # habr занял только 1 слот, не 2


def test_empty_query_plan_falls_back_to_single_synthetic_vector(monkeypatch):
    """Если Lite-план не построился (пустой context / ошибка) — не молчаливый
    сбой, а один синтетический вектор тем же query-текстом."""
    raw_hits = [
        _hit("https://www.sqlite.org/btreemodule.html", "SQLite B-Tree", 0.8),
    ]
    client = _FakeClient(raw_hits)
    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch, "build_exa_query_plan", _async_return(_FakeQueryPlan(specs=[]))
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )

    out = asyncio.run(
        orch._exa_sources_multi_vector("B-Tree индексы", 3, anchor="test")
    )

    assert len(out) == 1
    assert client.calls  # реально вызвали search
    assert client.calls[0]["include_domains"] == []  # Discovery no-op в этом тесте


def test_pass1_uses_validated_domains_as_include_domains(monkeypatch):
    """Domain Discovery (по аналогии с DEEP): validated_domains из Pass 1
    должны попасть в include_domains вызова Exa, а не игнорироваться."""
    raw_hits = [
        _hit("https://www.postgresql.org/docs/18/btree.html", "PG B-Tree", 0.9),
    ]
    client = _FakeClient(raw_hits)
    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch,
        "build_exa_query_plan",
        _async_return(
            _FakeQueryPlan(
                specs=[
                    ExaQuerySpec(
                        role="en_technical",
                        query="B-Tree index internals",
                        highlight_query="internals",
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.expand_search_context_with_flash_lite",
        lambda _context: ExaSearchContextExpansion(primary_domains=["postgresql.org"]),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_domain_validate.prepare_exa_pass1_domains",
        _async_return(["postgresql.org"]),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.exa_source_expand.filter_pass1_official_hosts",
        lambda hosts: list(hosts),
    )

    asyncio.run(orch._exa_sources_multi_vector("B-Tree индексы", 3, anchor="test"))

    assert client.calls
    assert client.calls[0]["include_domains"] == ["postgresql.org"]


def test_passage_extraction_replaces_snippet_when_available(monkeypatch):
    """Async Fetch + Passage Extraction: когда для URL нашлись извлечённые
    абзацы, они должны заменить сырой Exa-хайлайт в итоговом snippet."""
    raw_hits = [
        _hit("https://www.postgresql.org/docs/18/btree.html", "PG B-Tree", 0.9),
    ]
    client = _FakeClient(raw_hits)
    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch,
        "build_exa_query_plan",
        _async_return(
            _FakeQueryPlan(
                specs=[
                    ExaQuerySpec(
                        role="en_technical",
                        query="B-Tree index internals",
                        highlight_query="internals",
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )
    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.lecture_passage_fetch.fetch_and_extract_passages",
        _async_return(
            {
                "https://www.postgresql.org/docs/18/btree.html": [
                    "Leaf pages in a B-Tree are linked via a doubly linked list.",
                ]
            }
        ),
    )

    out = asyncio.run(
        orch._exa_sources_multi_vector("B-Tree индексы", 1, anchor="test")
    )

    assert len(out) == 1
    assert "doubly linked list" in out[0].snippet
    assert "PG B-Tree snippet text" not in out[0].snippet


def test_near_duplicate_dropped_and_backfilled_from_reserve(monkeypatch):
    """Реальный кейс: две почти одинаковые статьи (разные версии одной
    документации) не должны обе занимать финальные слоты — near-dup
    (BGE-M3 + Flash Lite Bulk Gate, lecture_passage_fetch.find_near_
    duplicate_urls) должен выбросить дубликат и добрать замену из резерва
    (reserve_cap = wide_cap + LECTURE_PASSAGE_BACKFILL_MARGIN), а не просто
    сократить итоговый набор, как ALIAS в DEEP."""
    raw_hits = [
        # RU: 4 РАЗНЫХ домена нарочно — round-robin с max_per_domain>=2 не
        # переставляет одиночные записи по доменам местами (обходит их в
        # исходном порядке по score), так что порядок processed/reserve
        # предсказуем и тест не зависит от деталей round-robin.
        _hit("https://a.example/btree-v18", "A v18", 0.95),
        _hit("https://b.example/btree-v17", "B v17 (dup of A)", 0.90),
        _hit("https://c.example/btree", "C", 0.5),
        _hit("https://d.example/btree", "D reserve", 0.3),
    ]
    client = _FakeClient(raw_hits)
    monkeypatch.setattr(orch, "ExaSearchClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        orch,
        "build_exa_query_plan",
        _async_return(
            _FakeQueryPlan(
                specs=[
                    ExaQuerySpec(
                        role="en_technical",
                        query="btree internals",
                        highlight_query="internals",
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        orch, "postprocess_exa_hits_for_external_recall", _stub_postprocess
    )
    monkeypatch.setattr(orch, "EXA_RERANK_LITE_THRESHOLD", 2)
    monkeypatch.setattr(orch, "LECTURE_PASSAGE_BACKFILL_MARGIN", 1)

    fetch_calls: list[list[str]] = []

    async def fake_fetch(urls, *, core_theme):
        fetch_calls.append(list(urls))
        return {u: [f"passage for {u}"] for u in urls}

    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.lecture_passage_fetch.fetch_and_extract_passages",
        fake_fetch,
    )
    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.lecture_passage_fetch.find_near_duplicate_urls",
        _async_return({"https://b.example/btree-v17": "https://a.example/btree-v18"}),
    )

    async def fake_lite_rerank(hits, *a, cap, **kw):
        return hits[:cap]

    monkeypatch.setattr(orch, "_lite_rerank_exa_hits", fake_lite_rerank)

    out = asyncio.run(orch._exa_sources_multi_vector("btree", 3, anchor="test"))

    urls = {s.url for s in out}
    assert "https://b.example/btree-v17" not in urls  # дубликат выброшен
    assert "https://d.example/btree" in urls  # добор из резерва
    assert len(fetch_calls) == 2  # основной фетч + доборный для backfill


def _host(url: str) -> str:
    from urllib.parse import urlparse

    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _async_return(value):
    async def _fn(*_a, **_kw):
        return value

    return _fn
