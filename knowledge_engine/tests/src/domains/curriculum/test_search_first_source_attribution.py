"""Search-First Flash (`search_first_flash.py`) — атрибуция источника ноды

(`_resolve_source_ref`). Баг найден на реальном сгенерированном курсе
Штурвала: 5 нод с явно разными ``relevant_extracts`` (про разные approved-
статьи) получили один и тот же ``primary_source_id``. Причина:
``_norm_src_id(raw.source_id, 1)`` при нераспознанном формате
``source_id`` (LLM часто путает "src_N"/"SN") молча возвращала
захардкоженный ``"src_1"`` — РЕАЛЬНУЮ запись реестра, из-за чего url-based
фоллбэк (гейтился на ``if not entry``) никогда не успевал сработать.
Общий код — используется и Autopilot Search-First, и Штурвалом
(``steering_generator_bridge.py``). См. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md."""

from __future__ import annotations

from knowledge_engine.src.domains.curriculum.contracts.curriculum import (
    FlashSourceRefContract,
)
from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumSearchHit,
    CurriculumSourceRegistryEntry,
)
from knowledge_engine.src.domains.curriculum.search_first_flash import (
    _resolve_source_ref,
)


def _registry_entry(
    source_id: str, url: str, extract: str
) -> CurriculumSourceRegistryEntry:
    return CurriculumSourceRegistryEntry(
        source_id=source_id,
        title=f"Статья {source_id}",
        url=url,
        snippet=extract,
        key_extracts=[extract],
        source_tier="whitelist_blog",
    )


_URLS = [
    "https://habr.com/companies/yandex/articles/1/",
    "https://habr.com/companies/postgrespro/articles/2/",
    "https://habr.com/companies/avito/articles/3/",
    "https://habr.com/companies/ydb/articles/4/",
]
_EXTRACTS = ["Про Sharpei", "Про Citus", "Про шардирование Авито", "Про 2PC"]

# registry и hits строятся 1:1 из одних и тех же URL — как это и происходит
# в реальном пайплайне (_registry_from_hits строит реестр прямо из hits).
_REGISTRY = [
    _registry_entry(f"src_{i}", url, extract)
    for i, (url, extract) in enumerate(zip(_URLS, _EXTRACTS), start=1)
]
_HITS = [
    CurriculumSearchHit(url=url, title=f"Статья {i}", snippet=extract)
    for i, (url, extract) in enumerate(zip(_URLS, _EXTRACTS), start=1)
]


def test_resolve_source_ref_uses_url_when_source_id_is_malformed() -> None:
    """Регрессия на сам баг: 4 ноды с одинаково "плохим" source_id, но

    РАЗНЫМИ url — каждая должна получить СВОЙ источник, а не все src_1."""
    cases = [
        ("", _REGISTRY[0].url),  # пустой source_id
        ("unexpected_format", _REGISTRY[1].url),  # не src_N/SN
        ("2", _REGISTRY[2].url),  # голое число без префикса
        ("SRC-4", _REGISTRY[3].url),  # похоже, но не совпадает с regex
    ]
    resolved_ids = []
    for raw_sid, url in cases:
        raw = FlashSourceRefContract(source_id=raw_sid, url=url, relevant_extracts=[])
        ref = _resolve_source_ref(raw, hits=_HITS, registry=_REGISTRY)
        assert ref is not None
        resolved_ids.append(ref.source_id)

    # Раньше все 4 схлопывались в "src_1" — теперь должны быть 4 разных.
    assert resolved_ids == ["src_1", "src_2", "src_3", "src_4"]
    assert len(set(resolved_ids)) == 4


def test_resolve_source_ref_prefers_url_over_wrong_but_well_formed_source_id() -> None:
    """LLM указала src_1, но url явно указывает на src_3 (LLM ошиблась в

    id, но не в url) — url должен победить, раз он однозначно резолвится
    в известную запись реестра."""
    raw = FlashSourceRefContract(
        source_id="src_1", url=_REGISTRY[2].url, relevant_extracts=[]
    )
    ref = _resolve_source_ref(raw, hits=_HITS, registry=_REGISTRY)
    assert ref is not None
    assert ref.source_id == "src_3"


def test_resolve_source_ref_still_works_with_well_formed_source_id_and_no_url() -> None:
    """Обычный случай (не регрессия): корректный source_id, url пуст —

    должен резолвиться по id, как и раньше."""
    raw = FlashSourceRefContract(source_id="src_2", url="", relevant_extracts=[])
    ref = _resolve_source_ref(raw, hits=_HITS, registry=_REGISTRY)
    assert ref is not None
    assert ref.source_id == "src_2"


def test_resolve_source_ref_falls_back_to_first_registry_entry_as_last_resort() -> None:
    """И source_id, и url не резолвятся ни во что известное — последний

    резерв (первая запись реестра), как и раньше."""
    raw = FlashSourceRefContract(
        source_id="totally_unknown",
        url="https://not-in-registry.example/",
        relevant_extracts=[],
    )
    ref = _resolve_source_ref(raw, hits=_HITS, registry=_REGISTRY)
    assert ref is not None
    assert ref.source_id == "src_1"


def test_resolve_source_ref_returns_none_for_empty_registry() -> None:
    raw = FlashSourceRefContract(source_id="src_1", url="", relevant_extracts=[])
    ref = _resolve_source_ref(raw, hits=[], registry=[])
    assert ref is None


def test_resolve_source_ref_uses_provided_relevant_extracts_over_registry() -> None:
    raw = FlashSourceRefContract(
        source_id="src_2",
        url="",
        relevant_extracts=["Собственная выдержка LLM для этой ноды."],
    )
    ref = _resolve_source_ref(raw, hits=_HITS, registry=_REGISTRY)
    assert ref is not None
    assert ref.relevant_extracts == ["Собственная выдержка LLM для этой ноды."]
