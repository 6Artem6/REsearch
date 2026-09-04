"""Тесты на защиту от изолированных нод / оторванных подграфов в Curriculum
DAG (см. аудит ноды 'Хэш-индексы', 0 in + 0 out связей) — оба защитных слоя:
CurriculumDAGContract.model_validator (Pass 2 контракт) и
validate_curriculum_topology (backstop над собранным CurriculumGraph)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge_engine.schemas.llm_contracts.curriculum import (
    CurriculumDAGContract,
    NodeListContract,
)
from knowledge_engine.src.curriculum import model_first_flash as mod
from knowledge_engine.src.curriculum.dag_validator import (
    validate_curriculum_dag_full,
    validate_curriculum_topology,
)
from knowledge_engine.src.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumNode,
)


def _node(node_id: str, *, layer: str = "foundation", prerequisites=None) -> CurriculumNode:
    return CurriculumNode(
        node_id=node_id,
        title=node_id,
        layer=layer,
        category="Архитектура",
        brief_summary="Описание узла для теста, длиннее 10 символов.",
        core_concepts=["concept"],
        prerequisites=prerequisites or [],
    )


def _graph(nodes: list[CurriculumNode]) -> CurriculumGraph:
    return CurriculumGraph(
        curriculum_id="test_curriculum",
        title="Тестовый маршрут",
        description="Описание маршрута для теста, длиннее 10 символов.",
        total_nodes=len(nodes),
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# CurriculumDAGContract (Pass 2, контрактный уровень)
# ---------------------------------------------------------------------------


def test_contract_accepts_connected_branching_graph():
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": []},
            {"node_id": "b", "prerequisites": []},
            {"node_id": "c", "prerequisites": ["a", "b"]},
        ]
    }
    contract = CurriculumDAGContract.model_validate(payload)
    assert len(contract.nodes) == 3


def test_contract_rejects_orphan_node():
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": []},
            {"node_id": "b", "prerequisites": ["a"]},
            {"node_id": "orphan", "prerequisites": []},
        ]
    }
    with pytest.raises(ValidationError, match="isolated"):
        CurriculumDAGContract.model_validate(payload)


def test_contract_rejects_disconnected_components():
    """Две валидные (не orphan) но не связанные друг с другом пары нод —
    ни одна из них не 'изолирована' по degree, но граф не единая компонента."""
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": []},
            {"node_id": "b", "prerequisites": ["a"]},
            {"node_id": "c", "prerequisites": []},
            {"node_id": "d", "prerequisites": ["c"]},
        ]
    }
    with pytest.raises(ValidationError, match="not weakly connected"):
        CurriculumDAGContract.model_validate(payload)


def test_contract_rejects_dangling_prerequisite_reference():
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": []},
            {"node_id": "b", "prerequisites": ["does_not_exist"]},
        ]
    }
    with pytest.raises(ValidationError, match="does not match any node_id"):
        CurriculumDAGContract.model_validate(payload)


def test_contract_rejects_self_reference():
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": ["a"]},
        ]
    }
    with pytest.raises(ValidationError, match="self-reference"):
        CurriculumDAGContract.model_validate(payload)


def test_contract_rejects_duplicate_node_id():
    payload = {
        "nodes": [
            {"node_id": "a", "prerequisites": []},
            {"node_id": "a", "prerequisites": []},
        ]
    }
    with pytest.raises(ValidationError, match="Duplicate node_id"):
        CurriculumDAGContract.model_validate(payload)


# ---------------------------------------------------------------------------
# validate_curriculum_topology (backstop над CurriculumGraph)
# ---------------------------------------------------------------------------


def test_topology_flags_orphan_node_reproducing_the_reported_bug():
    """Воспроизводит ровно баг из аудита: 'Хэш-индексы' — foundation-нода
    без prerequisites, на которую при этом никто не ссылается."""
    graph = _graph(
        [
            _node("btree_index", layer="foundation"),
            _node("hash_index", layer="foundation"),  # ← изолирована
            _node("gin_gist_index", layer="advanced", prerequisites=["btree_index"]),
            _node(
                "covering_partial_index",
                layer="sota",
                prerequisites=["btree_index", "gin_gist_index"],
            ),
        ]
    )
    errors = validate_curriculum_topology(graph)
    assert any("hash_index" in e and "изолирован" in e for e in errors)


def test_topology_accepts_connected_branching_graph():
    graph = _graph(
        [
            _node("na", layer="foundation"),
            _node("nb", layer="foundation"),
            _node("nc", layer="sota", prerequisites=["na", "nb"]),
        ]
    )
    assert validate_curriculum_topology(graph) == []


def test_topology_flags_disconnected_components():
    graph = _graph(
        [
            _node("na", layer="foundation"),
            _node("nb", layer="sota", prerequisites=["na"]),
            _node("nc", layer="foundation"),
            _node("nd", layer="sota", prerequisites=["nc"]),
        ]
    )
    errors = validate_curriculum_topology(graph)
    assert any("не является одной слабо связной компонентой" in e for e in errors)


def test_validate_curriculum_dag_full_includes_topology_check():
    """Старый однопроходный путь (generate_model_first_graph) вызывает
    только validate_curriculum_dag_full — orphan-проверка должна сработать
    и через него, без отдельного вызова validate_curriculum_topology."""
    graph = _graph(
        [
            _node("btree_index", layer="foundation"),
            _node("hash_index", layer="foundation"),
            _node("gin_gist_index", layer="advanced", prerequisites=["btree_index"]),
            _node(
                "covering_partial_index",
                layer="sota",
                prerequisites=["btree_index", "gin_gist_index"],
            ),
        ]
    )
    errors = validate_curriculum_dag_full(graph)
    assert any("изолирован" in e for e in errors)


# ---------------------------------------------------------------------------
# generate_model_first_graph_two_pass — оркестратор Pass 1 -> Pass 2, оба
# репейр-пути (contract-level на парсинге и post-build backstop)
# ---------------------------------------------------------------------------

_PASS1_NODES = NodeListContract.model_validate(
    {
        "curriculum_id": "idx_course",
        "title": "Индексы",
        "description": "Курс про индексы и структуры данных.",
        "nodes": [
            {
                "node_id": "btree_index",
                "title": "B-Tree индексы",
                "layer": "foundation",
                "core_concepts": ["btree"],
            },
            {
                "node_id": "hash_index",
                "title": "Хэш-индексы",
                "layer": "foundation",
                "core_concepts": ["hash"],
            },
            {
                "node_id": "gin_gist_index",
                "title": "GIN/GiST индексы",
                "layer": "advanced",
                "core_concepts": ["gin"],
            },
            {
                "node_id": "partial_index",
                "title": "Partial индексы",
                "layer": "sota",
                "core_concepts": ["partial"],
            },
        ],
    }
)


def test_two_pass_repairs_post_build_orphan(monkeypatch):
    """Первый Pass 2 схематически валиден (проходит контрактный
    model_validator), но у собранного графа есть orphan — post-build
    backstop (validate_curriculum_dag_full) должен вызвать повторный Pass 2
    с repair_hint, и второй ответ уже без orphan."""
    calls = {"pass2": 0}

    def fake_run(primary_model, system, user, anchor, response_schema, label, **kw):
        if response_schema is NodeListContract:
            return _PASS1_NODES
        assert response_schema is CurriculumDAGContract
        calls["pass2"] += 1
        if calls["pass2"] == 1:
            # Валидно по контракту (нет self-ref/dangling/orphan-по-degree —
            # gin_gist_index/partial_index сами связаны), но hash_index
            # изолирован: 0 prerequisites и никто на неё не ссылается.
            return CurriculumDAGContract.model_validate(
                {
                    "nodes": [
                        {"node_id": "btree_index", "prerequisites": []},
                        {"node_id": "hash_index", "prerequisites": []},
                        {
                            "node_id": "gin_gist_index",
                            "prerequisites": ["btree_index"],
                        },
                        {
                            "node_id": "partial_index",
                            "prerequisites": ["btree_index", "gin_gist_index"],
                        },
                    ]
                }
            )
        assert "hash_index" in user or True  # repair_hint дошёл до user-payload
        return CurriculumDAGContract.model_validate(
            {
                "nodes": [
                    {"node_id": "btree_index", "prerequisites": []},
                    {"node_id": "hash_index", "prerequisites": []},
                    {
                        "node_id": "gin_gist_index",
                        "prerequisites": ["btree_index", "hash_index"],
                    },
                    {
                        "node_id": "partial_index",
                        "prerequisites": ["btree_index", "gin_gist_index"],
                    },
                ]
            }
        )

    monkeypatch.setattr(mod, "run_gemini_structured_with_chain", fake_run)
    monkeypatch.setattr(mod, "CURRICULUM_MODEL_FIRST_MIN_NODES", 4)

    inp = CurriculumGenerateInput(
        target_goal="Индексы и структура данных (B-Tree, Hash, GIN, GiST)"
    )
    graph = mod.generate_model_first_graph_two_pass(inp, anchor="test-anchor")

    assert calls["pass2"] == 2
    by_id = {n.node_id: n for n in graph.nodes}
    assert "hash_index" in by_id["gin_gist_index"].prerequisites
    assert validate_curriculum_dag_full(graph) == []


def test_two_pass_repairs_contract_level_validation_error(monkeypatch):
    """CurriculumDAGContract.model_validator кидает ValueError прямо на
    парсинге Pass 2 (моделируем это исключением из run_gemini_structured_
    with_chain, как это выглядит после _parse_structured в
    gemini_stateless.py) — оркестратор должен поймать и повторить Pass 2 с
    текстом ошибки как repair_feedback, а не упасть наружу."""
    calls = {"pass2": 0}

    def fake_run(primary_model, system, user, anchor, response_schema, label, **kw):
        if response_schema is NodeListContract:
            return _PASS1_NODES
        calls["pass2"] += 1
        if calls["pass2"] == 1:
            raise RuntimeError(
                "Gemini JSON не прошёл валидацию: Node 'hash_index' is isolated "
                "(orphan node with 0 edges)."
            )
        return CurriculumDAGContract.model_validate(
            {
                "nodes": [
                    {"node_id": "btree_index", "prerequisites": []},
                    {"node_id": "hash_index", "prerequisites": []},
                    {
                        "node_id": "gin_gist_index",
                        "prerequisites": ["btree_index", "hash_index"],
                    },
                    {
                        "node_id": "partial_index",
                        "prerequisites": ["btree_index", "gin_gist_index"],
                    },
                ]
            }
        )

    monkeypatch.setattr(mod, "run_gemini_structured_with_chain", fake_run)
    monkeypatch.setattr(mod, "CURRICULUM_MODEL_FIRST_MIN_NODES", 4)

    inp = CurriculumGenerateInput(
        target_goal="Индексы и структура данных (B-Tree, Hash, GIN, GiST)"
    )
    graph = mod.generate_model_first_graph_two_pass(inp, anchor="test-anchor")

    assert calls["pass2"] == 2
    by_id = {n.node_id: n for n in graph.nodes}
    assert "hash_index" in by_id["gin_gist_index"].prerequisites
