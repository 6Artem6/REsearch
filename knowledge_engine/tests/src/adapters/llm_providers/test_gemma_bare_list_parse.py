"""Gemma вернула голый массив / один объект вместо обёртки схемы.

Живой случай (Gate 2, пакет из одной статьи): ответ ``[ {дайджест} ]``.
Срез «до первой `{`» терял `[`, массив из одного элемента становился
объектом дайджеста, а у ``_NodeDigestBatch.items`` был default ``[]`` —
пакет молча валидировался пустым, статья пропадала с Gate 2."""

from __future__ import annotations

import pytest

from knowledge_engine.src.adapters.llm_providers.gemma_client import _parse_structured
from knowledge_engine.src.domains.curriculum.services.node_digest_service import (
    _NodeDigestBatch,
)

_ITEM = (
    '{"url": "%s", "title": "T", "architecture": "a", '
    '"practical_case": "p", "limitations": "l"}'
)


@pytest.mark.parametrize(
    ("raw", "urls"),
    [
        ("[\n" + _ITEM % "u1" + "\n]", ["u1"]),
        ("[" + _ITEM % "u1" + "," + _ITEM % "u2" + "]", ["u1", "u2"]),
        (_ITEM % "u1", ["u1"]),
        ('{"items": [' + _ITEM % "u1" + "]}", ["u1"]),
        ("<thought>план [1]</thought>[" + _ITEM % "u1" + "]", ["u1"]),
    ],
    ids=["bare-one", "bare-two", "single-object", "wrapped", "thought-then-bare"],
)
def test_digest_batch_shapes_parse_to_items(raw: str, urls: list[str]) -> None:
    out = _parse_structured(raw, _NodeDigestBatch)
    assert out is not None
    assert [i.url for i in out.items] == urls


def test_empty_items_is_schema_failure_not_empty_batch() -> None:
    """Пустой пакет — ошибка схемы (→ штатный schema-retry), не «0 дайджестов»."""
    assert _parse_structured('{"items": []}', _NodeDigestBatch) is None
