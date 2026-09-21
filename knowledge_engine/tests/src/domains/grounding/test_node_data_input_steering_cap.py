"""Регрессия: NodeDataInput (node_deep_dive/schemas.py) — тело запросов

``/node/*`` (frontend сериализует ПОЛНЫЙ объект ноды в каждый запрос через
``toNodeDataInput()``, см. api.js) капало ``mapped_source_ids`` в 4 элемента
безусловно — из-за этого Штурвал Mode 2 (``node_kind="steering_standalone"``,
узел со ВСЕМИ Gate-2-approved источниками, см. steering_topic_node_service.py)
падал на валидации при открытии/чате: "List should have at most 4 items
after validation, not 8". Условная валидация в ``CurriculumNode``
(src/curriculum/schemas.py) уже это чинила для самого графа — этот файл
проверяет тот же mirror-фикс в ``NodeDataInput``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge_engine.src.config.settings import CURRICULUM_DEEP_NODE_MAX_HITS
from knowledge_engine.src.domains.grounding.schemas import NodeDataInput


def _base_kwargs(mapped_source_ids: list[str]) -> dict:
    return dict(
        node_id="n1",
        title="Test",
        layer="foundation",
        core_concepts=["c"],
        mapped_source_ids=mapped_source_ids,
    )


def test_standard_node_still_capped_at_max_hits() -> None:
    over = [f"src_{i}" for i in range(1, CURRICULUM_DEEP_NODE_MAX_HITS + 2)]
    with pytest.raises(ValidationError, match="mapped_source_ids"):
        NodeDataInput(**_base_kwargs(over))


def test_standard_node_at_exactly_max_hits_is_fine() -> None:
    exact = [f"src_{i}" for i in range(1, CURRICULUM_DEEP_NODE_MAX_HITS + 1)]
    node = NodeDataInput(**_base_kwargs(exact))
    assert len(node.mapped_source_ids) == CURRICULUM_DEEP_NODE_MAX_HITS


def test_steering_standalone_node_exceeds_max_hits_without_error() -> None:
    many = [f"src_{i}" for i in range(1, CURRICULUM_DEEP_NODE_MAX_HITS + 5)]
    node = NodeDataInput(**_base_kwargs(many), node_kind="steering_standalone")
    assert len(node.mapped_source_ids) == len(many)


def test_node_kind_defaults_to_standard() -> None:
    node = NodeDataInput(**_base_kwargs([]))
    assert node.node_kind == "standard"
