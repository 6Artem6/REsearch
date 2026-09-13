"""Node Grounding Gate — Этап 0 (единый профиль) + Этап 1 (сбор/дедуп/

паспорта) → Gate 1. См. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.

Эндпоинт-тесты (``test_post_node_grounding_discover_*``) используют уже
существующий в локальном dev-хранилище курс ``indexes_and_data_structures``
(создан ранее вручную при отладке SOTA-override) и мягко скипаются, если
его нет — чтобы тесты оставались рабочими на чистой машине/CI, а не
требовали писать тестовые данные в реальный skill_tree_store (там нет
публичной функции удаления записи — риск повторить прошлую ошибку с
замусориванием local-store тестовыми данными)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from knowledge_engine.src.adapters.search_providers.arxiv_query_builder import (
    ArxivQueryParams,
)
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    AcademicSearchPlan,
)
from knowledge_engine.src.domains.curriculum.services import (
    node_candidate_collection_service as ncs,
)
from knowledge_engine.src.domains.curriculum.services import node_digest_service as nds
from knowledge_engine.src.domains.grounding.schemas import NodeDataInput
from knowledge_engine.src.entrypoints.api.routes.node_skill import (
    NodeSessionBody,
    post_node_grounding_discover,
)
from knowledge_engine.src.shared.node_grounding import (
    node_grounding_finalize_service as ngf,
)
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NodeCandidatePassport,
    NodeDigestItem,
    NodeGate1ApprovePayload,
    NodeGate2ApprovePayload,
    NodeSearchProfile,
)
from knowledge_engine.src.shared.node_grounding.node_search_profile_service import (
    build_node_academic_plan,
    generate_node_search_profile,
)
from knowledge_engine.src.shared.skill_tree_store import get_curriculum_graph
from knowledge_engine.src.utils import translation_service as ts


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- Контракты ---------------------------------------------------------


def test_gate1_approve_payload_caps_at_four() -> None:
    ok = NodeGate1ApprovePayload(
        curriculum_id="cur1", node_id="n1", approved_urls=["a", "b", "c", "d"]
    )
    assert len(ok.approved_urls) == 4

    with pytest.raises(ValidationError):
        NodeGate1ApprovePayload(
            curriculum_id="cur1", node_id="n1", approved_urls=["a", "b", "c", "d", "e"]
        )


# --- Этап 0: профиль поиска ---------------------------------------------


@pytest.mark.anyio
async def test_profile_never_carries_academic_queries() -> None:
    """После ревью по docs/SOURCE_POOL.md ("не дублировать") этот Lite

    вызов больше не производит academic_queries вообще — за это отвечает
    build_node_academic_plan (реюз build_academic_search_plan) ниже.
    Промпт явно просит поле не заполнять, но проверяем ЖЁСТКО, а не
    доверяем инструкции — тот же принцип, что и раньше."""
    fake = NodeSearchProfile(habr_hubs=["x"], academic_queries=["should not survive"])
    with patch(
        "knowledge_engine.src.shared.node_grounding.node_search_profile_service._lite_structured",
        AsyncMock(return_value=fake),
    ):
        profile = await generate_node_search_profile(
            node_title="B-tree indexes",
            node_summary="",
            core_concepts=["btree"],
            include_academic=True,
        )
    assert profile.academic_queries == []
    assert profile.habr_hubs == ["x"]  # остальные поля не тронуты


@pytest.mark.anyio
async def test_build_node_academic_plan_reuses_existing_architect() -> None:
    """build_node_academic_plan — тонкая обёртка над УЖЕ СУЩЕСТВУЮЩИМ

    build_academic_search_plan (Academic Query Architect), не собственный
    промпт — проверяем, что goal реально доходит и arxiv_params не теряются."""
    fake_plan = AcademicSearchPlan(
        academic_query_en="cost-based query optimization index selection",
        arxiv_params=ArxivQueryParams(),
    )
    with patch(
        "knowledge_engine.src.shared.node_grounding.node_search_profile_service.build_academic_search_plan",
        AsyncMock(return_value=fake_plan),
    ) as mock_build:
        plan = await build_node_academic_plan(
            "Cost-based optimizer", "How the planner picks indexes."
        )
    assert plan.academic_query_en == "cost-based query optimization index selection"
    mock_build.assert_awaited_once()
    goal_arg = mock_build.call_args.args[0]
    assert "Cost-based optimizer" in goal_arg


@pytest.mark.anyio
async def test_profile_fail_open_heuristic_on_lite_error() -> None:
    with patch(
        "knowledge_engine.src.shared.node_grounding.node_search_profile_service._lite_structured",
        AsyncMock(side_effect=RuntimeError("lite down")),
    ):
        profile = await generate_node_search_profile(
            node_title="Cost based optimizer for indexes",
            node_summary="",
            core_concepts=["optimizer"],
            include_academic=True,
        )
    # Эвристика не падает и никогда не производит academic_queries.
    assert profile.academic_queries == []
    assert profile.habr_keywords  # что-то извлекло из title


# --- Habr: RSS-обнаружение (не Exa) -------------------------------------

_SAMPLE_HABR_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Sample</title>
  <item>
    <title><![CDATA[Индексы в PostgreSQL: полное руководство]]></title>
    <guid isPermaLink="true">https://habr.com/ru/companies/acme/articles/111/</guid>
    <link>https://habr.com/ru/companies/acme/articles/111/?utm_source=rss</link>
    <description><![CDATA[<p>Статья про B-tree и покрывающие индексы.</p>]]></description>
  </item>
  <item>
    <title><![CDATA[Погода в офисе Acme на этой неделе]]></title>
    <guid isPermaLink="true">https://habr.com/ru/companies/acme/articles/222/</guid>
    <link>https://habr.com/ru/companies/acme/articles/222/?utm_source=rss</link>
    <description><![CDATA[<p>Ничего технического.</p>]]></description>
  </item>
</channel></rss>"""


def test_rss_item_matches_keywords() -> None:
    assert ncs._rss_item_matches_keywords(
        "Индексы в PostgreSQL", "про B-tree", ["индекс"]
    )
    assert not ncs._rss_item_matches_keywords(
        "Погода в офисе", "ничего технического", ["индекс"]
    )
    # пустые keywords -> пропускаем всё (лучше нерелевантный кандидат, чем ни одного)
    assert ncs._rss_item_matches_keywords("Что угодно", "текст", [])


@pytest.mark.anyio
async def test_fetch_habr_company_rss_items_parses_guid_over_link() -> None:
    """guid (без utm_*) предпочитается ссылке с utm_* — прямая ссылка для

    Trafilatura должна быть канонической."""
    with patch.object(ncs, "fetch_html", AsyncMock(return_value=_SAMPLE_HABR_RSS)):
        items = await ncs._fetch_habr_company_rss_items("acme")
    assert items == [
        (
            "https://habr.com/ru/companies/acme/articles/111/",
            "Индексы в PostgreSQL: полное руководство",
            " Статья про B-tree и покрывающие индексы. ",
        ),
        (
            "https://habr.com/ru/companies/acme/articles/222/",
            "Погода в офисе Acme на этой неделе",
            " Ничего технического. ",
        ),
    ]


@pytest.mark.anyio
async def test_fetch_habr_company_rss_items_bad_xml_fails_open() -> None:
    with patch.object(ncs, "fetch_html", AsyncMock(return_value="not xml at all <<<")):
        items = await ncs._fetch_habr_company_rss_items("acme")
    assert items == []


@pytest.mark.anyio
async def test_collect_habr_filters_by_keyword_then_extracts_lead() -> None:
    """RSS отдаёт 2 статьи (одна по теме, одна нет) — keyword-фильтр должен

    оставить только релевантную ДО того, как для неё вызовется Trafilatura;
    для нерелевантной fetch_html вообще не должен вызываться."""
    profile = NodeSearchProfile(
        habr_hubs=["acme"], habr_tags=["postgresql"], habr_keywords=["индекс"]
    )
    fetched_urls: list[str] = []

    async def fake_fetch_html(url, *, timeout_sec):
        fetched_urls.append(url)
        if "rss/companies" in url:
            return _SAMPLE_HABR_RSS
        return (
            "<html><body><p>"
            + "x" * 100
            + " про индексы в базах данных подробно"
            + "</p></body></html>"
        )

    with patch.object(ncs, "fetch_html", fake_fetch_html):
        hits = await ncs._collect_habr(profile)

    assert len(hits) == 1
    assert hits[0].url == "https://habr.com/ru/companies/acme/articles/111/"
    assert hits[0].source_tier == "habr"
    # нерелевантная статья (222) не должна была дойти до полного fetch
    assert not any("222" in u for u in fetched_urls)


# --- Перевод для Gate 1 (translation_service + _translate_latin_snippets) --


class _FakeTranslateResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeTranslateClient:
    """Заглушка httpx.AsyncClient — перехватывает только .get(), остальное

    (async context manager протокол) делегирует напрямую."""

    def __init__(self, response_payload=None, raise_exc=None):
        self._response_payload = response_payload
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeTranslateResponse(self._response_payload)


@pytest.mark.anyio
async def test_translate_batch_to_russian_preserves_order() -> None:
    fake_client = _FakeTranslateClient(response_payload=["Привет", "Мир"])
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "World"])
    assert out == ["Привет", "Мир"]
    # ОДИН HTTP-запрос на весь батч — не по одному на текст.
    assert len(fake_client.calls) == 1
    q_values = [v for k, v in fake_client.calls[0]["params"] if k == "q"]
    assert q_values == ["Hello", "World"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_tolerates_nested_list_shape() -> None:
    fake_client = _FakeTranslateClient(response_payload=[["Привет"], ["Мир"]])
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "World"])
    assert out == ["Привет", "Мир"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_fail_open_on_network_error() -> None:
    fake_client = _FakeTranslateClient(raise_exc=RuntimeError("network down"))
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "World"])
    assert out == ["Hello", "World"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_fail_open_on_bad_shape() -> None:
    # HTML "automated queries" block (реальный ответ, полученный при
    # ручной проверке этого эндпоинта) — не JSON-список, парсинг честно
    # проваливается на .json() ИЛИ на isinstance(data, list).
    fake_client = _FakeTranslateClient(response_payload={"unexpected": "shape"})
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "World"])
    assert out == ["Hello", "World"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_fail_open_on_length_mismatch() -> None:
    fake_client = _FakeTranslateClient(response_payload=["Привет"])  # длина не совпала
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "World"])
    assert out == ["Hello", "World"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_skips_empty_strings() -> None:
    fake_client = _FakeTranslateClient(response_payload=["Привет"])
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian(["Hello", "", "  "])
    assert out == ["Привет", "", "  "]
    # Пустые строки не должны были уйти в q= вообще.
    q_values = [v for k, v in fake_client.calls[0]["params"] if k == "q"]
    assert q_values == ["Hello"]


@pytest.mark.anyio
async def test_translate_batch_to_russian_empty_input_makes_no_request() -> None:
    fake_client = _FakeTranslateClient(response_payload=[])
    with patch.object(ts.httpx, "AsyncClient", lambda **kw: fake_client):
        out = await ts.translate_batch_to_russian([])
    assert out == []
    assert fake_client.calls == []


def test_needs_ru_translation_detects_latin_vs_cyrillic() -> None:
    # Живёт в translation_service.py — общая эвристика с light_discovery_
    # service.py (Штурвал), не дублируется между двумя гейтами.
    assert ts.needs_ru_translation("This is an English abstract about caching.")
    assert not ts.needs_ru_translation("Это статья про кэширование на русском.")
    assert not ts.needs_ru_translation("")
    assert not ts.needs_ru_translation("   ")
    assert not ts.needs_ru_translation("12345 -- ...")  # нет букв вообще


@pytest.mark.anyio
async def test_translate_latin_snippets_only_sends_non_russian_hits() -> None:
    hits = [
        ncs._RawHit("https://habr.com/x", "X", "Статья про базы данных", "habr"),
        ncs._RawHit("https://exa/1", "Y", "An article about databases", "exa"),
    ]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(return_value=["Статья о базах данных (перевод)"]),
    ) as mock_translate:
        out = await ncs._translate_latin_snippets(hits)
    mock_translate.assert_awaited_once_with(["An article about databases"])
    assert out[0].snippet == "Статья про базы данных"  # Habr не тронут
    assert out[1].snippet == "Статья о базах данных (перевод)"


@pytest.mark.anyio
async def test_translate_latin_snippets_noop_when_nothing_needs_translation() -> None:
    hits = [ncs._RawHit("https://habr.com/x", "X", "Статья по-русски", "habr")]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(),
    ) as mock_translate:
        out = await ncs._translate_latin_snippets(hits)
    mock_translate.assert_not_called()
    assert out == hits


@pytest.mark.anyio
async def test_translate_latin_snippets_fail_open_on_error() -> None:
    hits = [ncs._RawHit("https://exa/1", "Y", "An English snippet", "exa")]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(side_effect=RuntimeError("down")),
    ):
        out = await ncs._translate_latin_snippets(hits)
    assert out == hits  # оригинал, без исключения наружу


# --- Этап 1: сбор/дедуп/выбор (чистые функции) --------------------------


def test_dedupe_by_url_case_and_trailing_slash() -> None:
    hits = [
        ncs._RawHit("https://a.com/x", "A", "s", "habr"),
        ncs._RawHit("https://a.com/x/", "A dup", "s", "habr"),  # trailing slash dup
        ncs._RawHit("https://b.com/y", "B", "s", "exa"),
    ]
    out = ncs._dedupe_by_url(hits)
    assert [h.url for h in out] == ["https://a.com/x", "https://b.com/y"]


def test_dedupe_by_embedding_collapses_near_duplicates() -> None:
    """Реальный локальный BGE-M3 (не мокаем — та же модель, что RAG_GATEWAY

    использует в проде; вызов быстрый на маленьком наборе кандидатов)."""
    hits = [
        ncs._RawHit(
            "https://a",
            "Caching",
            "an article about database caching strategies",
            "exa",
        ),
        ncs._RawHit(
            "https://b",
            "Caching v2",
            "another piece about database caching strategies",
            "habr",
        ),
        ncs._RawHit(
            "https://c",
            "Indexes",
            "a completely unrelated topic: B-tree indexes",
            "arxiv",
        ),
    ]
    out = ncs._dedupe_by_embedding(hits)
    urls = {h.url for h in out}
    assert "https://c" in urls
    assert len(out) == 2  # a/b схлопнулись в одного представителя


def test_select_balanced_pool_round_robins_tiers() -> None:
    hits = [
        ncs._RawHit("https://h1", "H1", "s", "habr"),
        ncs._RawHit("https://h2", "H2", "s", "habr"),
        ncs._RawHit("https://e1", "E1", "s", "exa"),
        ncs._RawHit("https://x1", "X1", "s", "arxiv"),
    ]
    pool = ncs._select_balanced_pool(hits)
    tiers_in_order = [h.source_tier for h in pool]
    assert tiers_in_order[:3] == [
        "habr",
        "exa",
        "arxiv",
    ]  # round-robin, не просто порядок сбора
    assert len(pool) == 4


@pytest.mark.anyio
async def test_collect_node_candidates_orchestration_is_fail_open() -> None:
    """Один канал падает НЕПРЕДВИДЕННО (не через свой внутренний try/except,

    а совсем неожиданно) — остальные всё равно доходят до Gate 1, а не
    роняют asyncio.gather целиком (return_exceptions=True в
    collect_node_candidates — было найдено и исправлено при написании
    этого теста: раньше один упавший канал ронял всю выборку)."""
    profile = NodeSearchProfile(
        habr_hubs=["yandex"], exa_keywords=["k"], academic_queries=["q"]
    )
    fake_passports = [
        NodeCandidatePassport(
            url="https://habr.com/x", title="X", source_tier="habr", topic="t", gist="g"
        ),
    ]
    with (
        patch.object(
            ncs,
            "_collect_habr",
            AsyncMock(
                return_value=[ncs._RawHit("https://habr.com/x", "X", "s", "habr")]
            ),
        ),
        patch.object(ncs, "_collect_exa", AsyncMock(return_value=[])),
        patch.object(
            ncs,
            "_collect_academic",
            AsyncMock(side_effect=RuntimeError("consensus down")),
        ),
        patch.object(
            ncs, "_generate_passports", AsyncMock(return_value=fake_passports)
        ),
        patch.object(
            ncs, "_translate_latin_snippets", AsyncMock(side_effect=lambda hits: hits)
        ),
    ):
        result = await ncs.collect_node_candidates(
            profile, curriculum_id="cur1", node_id="n1"
        )
    assert len(result) == 1
    assert result[0].url == "https://habr.com/x"


# --- Этап 2: BGE-M3 + Cross-Encoder + Gemma дайджест (Gate 2) -----------


def test_gate2_approve_payload_caps_at_four() -> None:
    ok = NodeGate2ApprovePayload(
        curriculum_id="cur1", node_id="n1", approved_urls=["a", "b", "c", "d"]
    )
    assert len(ok.approved_urls) == 4

    with pytest.raises(ValidationError):
        NodeGate2ApprovePayload(
            curriculum_id="cur1", node_id="n1", approved_urls=["a", "b", "c", "d", "e"]
        )


@pytest.mark.anyio
async def test_top_k_chunks_uses_coarse_filter_then_reranker() -> None:
    """>coarse_k абзацев → BGE-M3 отбирает coarse_k, реранкер — top_k из них."""
    paragraphs = [f"параграф про тему {i}" for i in range(20)]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Первый вектор — критерий; чем ближе индекс параграфа к 0, тем
        # больше косинус (упорядоченный по убыванию релевантности набор).
        out = [[1.0, 0.0]]
        for i in range(len(texts) - 1):
            out.append([1.0 - i * 0.01, 0.01 * i])
        return out

    def fake_score(criterion: str, texts: list[str]) -> list[float]:
        # Реранкер переворачивает порядок относительно грубого отбора —
        # проверяем, что именно ЕГО порядок побеждает в финальном top_k.
        return list(range(len(texts)))

    with (
        patch.object(nds, "embed_texts_bge_m3", fake_embed),
        patch.object(nds, "score_relevance_pairs", fake_score),
    ):
        result = await nds._top_k_chunks(paragraphs, "тема ноды", coarse_k=8, top_k=3)
    assert len(result) == 3
    # Реранкер отдал наибольший score последним элементам входного pool
    # (paragraphs[0:8]) — top_k по убыванию score: индексы 7, 6, 5.
    assert result == [paragraphs[7], paragraphs[6], paragraphs[5]]
    # И весь top_k — подмножество грубого BGE-M3 отбора (первые coarse_k=8
    # параграфов), а не «протёкшие» кандидаты за его пределами.
    assert set(result) <= set(paragraphs[:8])


@pytest.mark.anyio
async def test_top_k_chunks_fail_open_on_embed_and_rerank_errors() -> None:
    paragraphs = [f"параграф {i}" for i in range(10)]
    with (
        patch.object(
            nds, "embed_texts_bge_m3", side_effect=RuntimeError("bge_m3 down")
        ),
        patch.object(nds, "score_relevance_pairs", side_effect=RuntimeError("ce down")),
    ):
        result = await nds._top_k_chunks(paragraphs, "тема", coarse_k=4, top_k=2)
    # И грубый, и точный отбор упали — fail-open берёт первые top_k абзацев.
    assert result == paragraphs[:2]


@pytest.mark.anyio
async def test_generate_node_digests_builds_batch_from_top_chunks() -> None:
    fake_digest = NodeDigestItem(
        url="https://habr.com/art1",
        title="Индексы в PostgreSQL",
        architecture="B-tree с покрывающими индексами.",
        practical_case="Ускорение выборок в 10 раз на проде.",
        limitations="Не годится для низкоселективных колонок.",
    )

    class _FakeBatch:
        items = [fake_digest]

    with (
        patch.object(
            nds,
            "fetch_html",
            AsyncMock(return_value="<html><body><p>x</p></body></html>"),
        ),
        patch.object(
            nds,
            "_extract_paragraphs",
            return_value=["абзац про индексы " * 5, "абзац про кэши " * 5],
        ),
        patch.object(nds, "_title_from_html", return_value="Индексы в PostgreSQL"),
        patch.object(
            nds, "_top_k_chunks", AsyncMock(return_value=["релевантный абзац"])
        ),
        patch.object(
            nds.GemmaCloudClient,
            "complete_structured",
            AsyncMock(return_value=_FakeBatch()),
        ) as mock_gemma,
    ):
        result = await nds.generate_node_digests(
            ["https://habr.com/art1"],
            node_title="Индексы в базах данных",
            curriculum_id="cur1",
            node_id="n1",
        )

    mock_gemma.assert_awaited_once()
    assert result.curriculum_id == "cur1"
    assert result.node_id == "n1"
    assert len(result.digests) == 1
    assert result.digests[0].url == "https://habr.com/art1"


@pytest.mark.anyio
async def test_generate_node_digests_drops_hallucinated_urls() -> None:
    """Gemma вернула URL, которого не было среди articles — отбрасываем."""
    hallucinated = NodeDigestItem(
        url="https://not-in-input.example/",
        title="Придуманная статья",
        architecture="x",
        practical_case="y",
        limitations="z",
    )

    class _FakeBatch:
        items = [hallucinated]

    with (
        patch.object(nds, "fetch_html", AsyncMock(return_value="<html>x</html>")),
        patch.object(
            nds, "_extract_paragraphs", return_value=["абзац про индексы " * 5]
        ),
        patch.object(nds, "_title_from_html", return_value="T"),
        patch.object(
            nds, "_top_k_chunks", AsyncMock(return_value=["релевантный абзац"])
        ),
        patch.object(
            nds.GemmaCloudClient,
            "complete_structured",
            AsyncMock(return_value=_FakeBatch()),
        ),
    ):
        result = await nds.generate_node_digests(
            ["https://habr.com/art1"],
            node_title="Тема",
            curriculum_id="cur1",
            node_id="n1",
        )
    assert result.digests == []


@pytest.mark.anyio
async def test_generate_node_digests_fail_open_when_gemma_returns_none() -> None:
    with (
        patch.object(nds, "fetch_html", AsyncMock(return_value="<html>x</html>")),
        patch.object(
            nds, "_extract_paragraphs", return_value=["абзац про индексы " * 5]
        ),
        patch.object(nds, "_title_from_html", return_value="T"),
        patch.object(
            nds, "_top_k_chunks", AsyncMock(return_value=["релевантный абзац"])
        ),
        patch.object(
            nds.GemmaCloudClient,
            "complete_structured",
            AsyncMock(return_value=None),
        ),
    ):
        result = await nds.generate_node_digests(
            ["https://habr.com/art1"], node_title="Тема", curriculum_id="c", node_id="n"
        )
    assert result.digests == []


@pytest.mark.anyio
async def test_generate_node_digests_empty_urls_skips_all_network_calls() -> None:
    with patch.object(nds, "fetch_html", AsyncMock()) as mock_fetch:
        result = await nds.generate_node_digests(
            [], node_title="Тема", curriculum_id="c", node_id="n"
        )
    mock_fetch.assert_not_called()
    assert result.digests == []


# --- Этап 3: финальный инжест (Gate 2 approved → реюз Map-Reduce) -------


def _make_test_node(**overrides):
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumNode

    defaults = dict(
        node_id="deep_node_under_test",
        title="Продвинутая тема для теста",
        layer="advanced",
        category="Категория",
        brief_summary="Достаточно длинное краткое описание ноды для теста.",
        core_concepts=["концепция"],
        node_risk_kind="DEEP",
        grounding_status="pending_grounding",
    )
    defaults.update(overrides)
    return CurriculumNode(**defaults)


def _make_test_graph(node):
    from knowledge_engine.src.domains.curriculum.schemas import (
        CurriculumGraph,
        CurriculumNode,
    )

    filler_one = CurriculumNode(
        node_id="filler_one",
        title="Filler One",
        layer="foundation",
        category="Категория",
        brief_summary="Служебная BASE-нода для валидности графа.",
        core_concepts=["a"],
    )
    filler_two = CurriculumNode(
        node_id="filler_two",
        title="Filler Two",
        layer="foundation",
        category="Категория",
        brief_summary="Ещё одна служебная BASE-нода для валидности графа.",
        core_concepts=["b"],
    )
    return CurriculumGraph(
        curriculum_id="test_curriculum_ngf",
        title="Test Curriculum",
        description="Тестовый курс для юнит-тестов Этапа 3 Node Grounding Gate.",
        total_nodes=3,
        nodes=[node, filler_one, filler_two],
    )


def test_infer_source_tier_matches_canonical_taxonomy() -> None:
    assert ngf._infer_source_tier("https://habr.com/ru/companies/x/articles/1/") == (
        "whitelist_blog"
    )
    assert ngf._infer_source_tier("https://arxiv.org/abs/1234.5678") == "arxiv"
    assert (
        ngf._infer_source_tier("https://www.semanticscholar.org/paper/x")
        == "semantic_scholar"
    )
    assert ngf._infer_source_tier("https://consensus.app/papers/x") == "consensus"
    assert ngf._infer_source_tier("https://some-engineering-blog.example/p") == "exa"


@pytest.mark.anyio
async def test_finalize_node_grounding_empty_urls_is_noop() -> None:
    node = _make_test_node()
    graph = _make_test_graph(node)
    with patch.object(
        ngf, "summarize_whitelist_blog_hits_async", AsyncMock()
    ) as mock_summarize:
        out_graph, out_node = await ngf.finalize_node_grounding(graph, node, [])
    mock_summarize.assert_not_called()
    assert out_node is node
    assert out_node.grounding_status == "pending_grounding"


@pytest.mark.anyio
async def test_finalize_node_grounding_fail_open_when_ingest_empty() -> None:
    """Все URL сломаны/ingest не вернул ни одного хита — узел не трогаем."""
    node = _make_test_node()
    graph = _make_test_graph(node)
    with (
        patch.object(
            ngf, "summarize_whitelist_blog_hits_async", AsyncMock(return_value=[])
        ),
        patch.object(
            ngf, "persist_approved_curriculum_hits_to_lancedb_async", AsyncMock()
        ) as mock_persist,
    ):
        out_graph, out_node = await ngf.finalize_node_grounding(
            graph, node, ["https://habr.com/x"]
        )
    mock_persist.assert_not_called()
    assert out_node is node
    assert out_node.grounding_status == "pending_grounding"


@pytest.mark.anyio
async def test_finalize_node_grounding_attaches_hits_and_grounds_node() -> None:
    node = _make_test_node()
    graph = _make_test_graph(node)

    async def fake_summarize(hits, target_goal):
        # Симулируем реальный ingest: тир из стаба сохраняется, добавляются
        # key_extracts (как будто Map-Reduce отработал).
        return [
            h.model_copy(
                update={
                    "title": "Индексы в PostgreSQL",
                    "key_extracts": ["B-tree покрывающие индексы ускорили запросы"],
                    "snippet": "Кратко про индексы",
                }
            )
            for h in hits
        ]

    async def fake_enrich(hits, target_goal):
        return hits

    with (
        patch.object(ngf, "summarize_whitelist_blog_hits_async", fake_summarize),
        patch.object(ngf, "enrich_search_hits_with_extracts_async", fake_enrich),
        patch.object(
            ngf, "persist_approved_curriculum_hits_to_lancedb_async", AsyncMock()
        ) as mock_persist,
        patch(
            "knowledge_engine.src.domains.grounding.diagram_session."
            "refresh_node_session_diagrams_from_articles",
            return_value=0,
        ),
    ):
        out_graph, out_node = await ngf.finalize_node_grounding(
            graph, node, ["https://habr.com/ru/companies/x/articles/1/"]
        )

    assert out_node.grounding_status == "grounded"
    assert out_node.node_risk_kind == "DEEP"
    assert len(out_node.mapped_source_ids) == 1
    assert out_node.source_ref is not None
    assert out_node.source_ref.url == "https://habr.com/ru/companies/x/articles/1/"
    assert len(out_graph.curriculum_sources_registry) == 1
    assert out_graph.curriculum_sources_registry[0].source_tier == "whitelist_blog"
    # whitelist_blog уже "персистится" внутри summarize_whitelist_blog_hits_async
    # в бою — здесь просто убеждаемся, что известный blog/academic тир НЕ
    # уходит на повторный persist_approved_curriculum_hits_to_lancedb_async.
    mock_persist.assert_not_called()


@pytest.mark.anyio
async def test_finalize_node_grounding_persists_unknown_tier_hits() -> None:
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumSearchHit

    node = _make_test_node()
    graph = _make_test_graph(node)

    async def fake_summarize(hits, target_goal):
        # source_tier "exa" НЕ входит ни в _BLOG_SOURCE_TIERS, ни в
        # _ACADEMIC_SOURCE_TIERS в этом фейке — проверяем, что такой хит
        # уходит в persist_approved_curriculum_hits_to_lancedb_async.
        return [
            CurriculumSearchHit(
                url=h.url, title="Test Title", source_tier="totally_unknown_tier"
            )
            for h in hits
        ]

    async def fake_enrich(hits, target_goal):
        return hits

    with (
        patch.object(ngf, "summarize_whitelist_blog_hits_async", fake_summarize),
        patch.object(ngf, "enrich_search_hits_with_extracts_async", fake_enrich),
        patch.object(
            ngf, "persist_approved_curriculum_hits_to_lancedb_async", AsyncMock()
        ) as mock_persist,
        patch(
            "knowledge_engine.src.domains.grounding.diagram_session."
            "refresh_node_session_diagrams_from_articles",
            return_value=0,
        ),
    ):
        await ngf.finalize_node_grounding(graph, node, ["https://example.com/x"])
    mock_persist.assert_awaited_once()


# --- Эндпоинт (использует существующий dev-курс, мягкий skip) -----------


_DEV_CURRICULUM_ID = "indexes_and_data_structures"
_DEV_BASE_NODE_ID = "advanced_index_optimization"
_DEV_SOTA_NODE_ID = "cost_based_optimizer_and_indexes"


def _require_dev_curriculum() -> None:
    if not get_curriculum_graph(_DEV_CURRICULUM_ID):
        pytest.skip(
            f"dev-курс {_DEV_CURRICULUM_ID} отсутствует в локальном "
            "skill_tree_store — эндпоинт-тест пропущен (не пишем тестовые "
            "данные в реальный store, см. docstring файла)"
        )


@pytest.mark.anyio
async def test_endpoint_base_node_skips_gate_entirely() -> None:
    _require_dev_curriculum()
    body = NodeSessionBody(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_data=NodeDataInput(
            node_id=_DEV_BASE_NODE_ID, title="XX", layer="advanced", core_concepts=["c"]
        ),
    )
    result = await post_node_grounding_discover(body)
    assert result["candidates"] == []
    assert result["profile"]["habr_hubs"] == []


@pytest.mark.anyio
async def test_endpoint_sota_node_triggers_gate_with_academic() -> None:
    """Ровно та нода, что вчера ушла в 43-минутный харвест при отладке —

    здесь профиль/сбор/academic-план замоканы, проверяем SOTA-override +
    что academic_plan реально доходит до профиля и до collect_node_candidates
    (реюз build_academic_search_plan, не собственный дубль-промпт)."""
    _require_dev_curriculum()
    fake_profile = NodeSearchProfile(habr_hubs=["yandex"])
    fake_plan = AcademicSearchPlan(
        academic_query_en="cost based query optimization index selection",
        arxiv_params=ArxivQueryParams(),
    )
    fake_passports = [
        NodeCandidatePassport(
            url="https://habr.com/x",
            title="Habr X",
            source_tier="habr",
            topic="t",
            gist="g",
        )
    ]
    body = NodeSessionBody(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_data=NodeDataInput(
            node_id=_DEV_SOTA_NODE_ID,
            title="Оптимизатор",
            layer="sota",
            core_concepts=["c"],
        ),
    )
    captured: dict = {}

    async def fake_collect(
        profile, *, curriculum_id, node_id, node=None, academic_plan=None
    ):
        captured["academic_plan"] = academic_plan
        captured["node"] = node
        return fake_passports

    with (
        patch(
            "knowledge_engine.src.shared.node_grounding.node_search_profile_service._lite_structured",
            AsyncMock(return_value=fake_profile),
        ),
        patch(
            "knowledge_engine.src.shared.node_grounding.node_search_profile_service.build_academic_search_plan",
            AsyncMock(return_value=fake_plan),
        ),
        patch.object(ncs, "collect_node_candidates", fake_collect),
    ):
        result = await post_node_grounding_discover(body)
    assert result["profile"]["academic_queries"] == [
        "cost based query optimization index selection"
    ]
    assert len(result["candidates"]) == 1
    assert captured["academic_plan"] is fake_plan
    assert (
        captured["node"] is not None and captured["node"].node_id == _DEV_SOTA_NODE_ID
    )


@pytest.mark.anyio
async def test_endpoint_digest_calls_service_with_node_context() -> None:
    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        post_node_grounding_digest,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeDigestItem,
        NodeGate1ApprovePayload,
        NodeGate2DigestResponse,
    )

    _require_dev_curriculum()
    fake_response = NodeGate2DigestResponse(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_id=_DEV_SOTA_NODE_ID,
        digests=[
            NodeDigestItem(
                url="https://habr.com/x",
                title="Habr X",
                architecture="a",
                practical_case="p",
                limitations="l",
            )
        ],
    )
    captured: dict = {}

    async def fake_generate_node_digests(approved_urls, **kwargs):
        captured["approved_urls"] = approved_urls
        captured["kwargs"] = kwargs
        return fake_response

    payload = NodeGate1ApprovePayload(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_id=_DEV_SOTA_NODE_ID,
        approved_urls=["https://habr.com/x"],
    )
    with patch(
        "knowledge_engine.src.domains.curriculum.services.node_digest_service.generate_node_digests",
        fake_generate_node_digests,
    ):
        result = await post_node_grounding_digest(payload)

    assert result["digests"][0]["url"] == "https://habr.com/x"
    assert captured["approved_urls"] == ["https://habr.com/x"]
    # node_title приходит из РЕАЛЬНОГО узла графа (не из тела запроса —
    # у NodeGate1ApprovePayload нет поля title), поэтому проверяем факт
    # передачи контекста, а не конкретную строку.
    assert captured["kwargs"]["node_title"]
    assert captured["kwargs"]["curriculum_id"] == _DEV_CURRICULUM_ID
    assert captured["kwargs"]["node_id"] == _DEV_SOTA_NODE_ID


@pytest.mark.anyio
async def test_endpoint_digest_404_on_unknown_node() -> None:
    from fastapi import HTTPException

    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        post_node_grounding_digest,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate1ApprovePayload,
    )

    _require_dev_curriculum()
    payload = NodeGate1ApprovePayload(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_id="does_not_exist_node",
        approved_urls=["https://habr.com/x"],
    )
    with pytest.raises(HTTPException) as exc_info:
        await post_node_grounding_digest(payload)
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_endpoint_finalize_404_on_unknown_node() -> None:
    from fastapi import HTTPException

    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        post_node_grounding_finalize,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate2ApprovePayload,
    )

    _require_dev_curriculum()
    payload = NodeGate2ApprovePayload(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_id="does_not_exist_node",
        approved_urls=["https://habr.com/x"],
    )
    with pytest.raises(HTTPException) as exc_info:
        await post_node_grounding_finalize(payload)
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_endpoint_finalize_calls_service_and_persists_once() -> None:
    """save_curriculum_record ЗАМОКАН — этот тест никогда не пишет в

    реальный локальный skill_tree_store (см. docstring файла про прошлый
    инцидент с замусориванием local-хранилища тестовыми данными)."""
    from knowledge_engine.src.entrypoints.api.routes.node_skill import (
        post_node_grounding_finalize,
    )
    from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
        NodeGate2ApprovePayload,
    )

    _require_dev_curriculum()
    raw = get_curriculum_graph(_DEV_CURRICULUM_ID)
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumGraph

    graph = CurriculumGraph.model_validate(raw)
    node = next(n for n in graph.nodes if n.node_id == _DEV_SOTA_NODE_ID)
    updated_node = node.model_copy(update={"grounding_status": "grounded"})

    captured: dict = {}

    async def fake_finalize(graph_arg, node_arg, approved_urls, *, target_goal=""):
        captured["approved_urls"] = approved_urls
        captured["node_id"] = node_arg.node_id
        return graph_arg, updated_node

    payload = NodeGate2ApprovePayload(
        curriculum_id=_DEV_CURRICULUM_ID,
        node_id=_DEV_SOTA_NODE_ID,
        approved_urls=["https://habr.com/x"],
    )
    with (
        patch(
            "knowledge_engine.src.shared.node_grounding."
            "node_grounding_finalize_service.finalize_node_grounding",
            fake_finalize,
        ),
        patch(
            "knowledge_engine.src.shared.skill_tree_store.save_curriculum_record"
        ) as mock_save,
    ):
        result = await post_node_grounding_finalize(payload)

    mock_save.assert_called_once()
    assert captured["approved_urls"] == ["https://habr.com/x"]
    assert captured["node_id"] == _DEV_SOTA_NODE_ID
    assert result["grounding_status"] == "grounded"
    assert result["node_id"] == _DEV_SOTA_NODE_ID
