"""Штурвал, Gate 1 (light_discovery_service.py) — мгновенная батч-

локализация английских ``lead_paragraph`` перед показом пользователю.
Добавлено после того, как в реальном прогоне на Gate 1 Штурвала были
замечены непереведённые английские тексты (перевод изначально был
подключен только для Node Grounding Gate) — см.
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.

Использует ту же ``needs_ru_translation``/``translate_batch_to_russian``,
что и Node Grounding Gate (``translation_service.py``, не дублируется)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_engine.src.domains.steering.services import (
    light_discovery_service as lds,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    CandidateArticleMeta,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _candidate(url: str, lead: str) -> CandidateArticleMeta:
    return CandidateArticleMeta(
        url=url, title="T", source_hub="example.com", lead_paragraph=lead
    )


@pytest.mark.anyio
async def test_translate_latin_leads_translates_only_non_russian() -> None:
    candidates = [
        _candidate("https://ru.example/1", "Статья про базы данных на русском"),
        _candidate("https://en.example/2", "An article about databases in English"),
    ]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(return_value=["Статья про базы данных (перевод)"]),
    ) as mock_translate:
        out = await lds._translate_latin_leads(candidates)

    mock_translate.assert_awaited_once_with(["An article about databases in English"])
    assert out[0].lead_paragraph == "Статья про базы данных на русском"
    assert out[1].lead_paragraph == "Статья про базы данных (перевод)"


@pytest.mark.anyio
async def test_translate_latin_leads_noop_when_all_russian() -> None:
    candidates = [_candidate("https://ru.example/1", "Уже по-русски написанный текст")]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(),
    ) as mock_translate:
        out = await lds._translate_latin_leads(candidates)
    mock_translate.assert_not_called()
    assert out == candidates


@pytest.mark.anyio
async def test_translate_latin_leads_fail_open_on_translate_error() -> None:
    candidates = [_candidate("https://en.example/1", "An English introduction here")]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(side_effect=RuntimeError("network down")),
    ):
        out = await lds._translate_latin_leads(candidates)
    assert out == candidates  # оригинал, без исключения наружу


@pytest.mark.anyio
async def test_translate_latin_leads_fail_open_on_length_mismatch() -> None:
    candidates = [
        _candidate("https://en.example/1", "First English introduction here"),
        _candidate("https://en.example/2", "Second English introduction here"),
    ]
    with patch(
        "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
        AsyncMock(return_value=["Только один перевод"]),  # длина не совпала
    ):
        out = await lds._translate_latin_leads(candidates)
    assert out == candidates


@pytest.mark.anyio
async def test_discover_candidates_applies_translation_before_returning() -> None:
    """Полный ``discover_candidates`` (Exa замокан) — перевод должен

    попасть в ``TaxonomyDiscoveryResponse.candidate_articles`` до возврата
    пользователю на Gate 1, а не быть отдельным неиспользуемым шагом."""
    from knowledge_engine.src.adapters.search_providers.exa_client import ExaSearchHit
    from knowledge_engine.src.domains.steering.services.taxonomy_service import (
        TaxonomySeed,
    )

    seed = TaxonomySeed(
        tags=["базы данных"],
        company_hubs=["example.com"],
        keywords=["database"],
    )
    fake_hit = ExaSearchHit(
        url="https://example.com/article",
        title="An English Title",
        highlights=[],
    )

    async def fake_hit_to_candidate(hit, hub):
        return _candidate(hit.url, "An English lead paragraph about databases")

    with (
        patch.object(lds.ExaSearchClient, "is_configured", return_value=True),
        patch.object(lds, "_search_hub", AsyncMock(return_value=[fake_hit])),
        patch.object(lds, "_hit_to_candidate", fake_hit_to_candidate),
        patch(
            "knowledge_engine.src.utils.translation_service.translate_batch_to_russian",
            AsyncMock(return_value=["Английский заголовок про базы данных (перевод)"]),
        ),
    ):
        result = await lds.discover_candidates(seed)

    assert len(result.candidate_articles) == 1
    assert (
        result.candidate_articles[0].lead_paragraph
        == "Английский заголовок про базы данных (перевод)"
    )
