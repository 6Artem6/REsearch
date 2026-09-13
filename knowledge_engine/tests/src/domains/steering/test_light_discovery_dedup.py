"""Штурвал, Gate 1 (light_discovery_service.py) — дедуп кандидатов по URL.

Найдено при живой отладке реального прогона: один и тот же URL находился
через несколько ``company_hubs``, попадал в Gate 1 отдельными карточками,
пользователь отмечал обе — на Gate 2 ``approved_urls`` приходил с дублями
(9 approved, только 5 уникальных), а дайджест на дубли не строился заново
(тот же URL просто повторно фигурировал в ответе). См.
docs/STEERING_AND_TOPIC_QNA_ROADMAP.md."""

from __future__ import annotations

from knowledge_engine.src.domains.steering.services import (
    light_discovery_service as lds,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    CandidateArticleMeta,
)


def _candidate(url: str, hub: str = "example.com") -> CandidateArticleMeta:
    return CandidateArticleMeta(
        url=url, title="T", source_hub=hub, lead_paragraph="lead"
    )


def test_dedupe_candidates_by_url_collapses_exact_duplicates() -> None:
    candidates = [
        _candidate("https://a.com/x", "hub1"),
        _candidate("https://a.com/x", "hub2"),  # тот же URL, другой хаб
        _candidate("https://b.com/y", "hub1"),
    ]
    out = lds._dedupe_candidates_by_url(candidates)
    assert [c.url for c in out] == ["https://a.com/x", "https://b.com/y"]


def test_dedupe_candidates_by_url_normalizes_case_and_trailing_slash() -> None:
    candidates = [
        _candidate("https://A.com/X"),
        _candidate("https://a.com/x/"),
        _candidate("https://a.com/x"),
    ]
    out = lds._dedupe_candidates_by_url(candidates)
    assert len(out) == 1
    assert out[0].url == "https://A.com/X"  # первое вхождение — представитель


def test_dedupe_candidates_by_url_keeps_distinct_urls() -> None:
    candidates = [_candidate("https://a.com/1"), _candidate("https://a.com/2")]
    out = lds._dedupe_candidates_by_url(candidates)
    assert len(out) == 2
