"""Integration tests: PostgresVectorRepository + TutorGraphService против реального
Postgres (docker compose up postgres + alembic upgrade head — см. docker-compose.yml).

Пропускаются, если Postgres недоступен на POSTGRES_DSN (как test_gap_evaluator_real_llm.py
пропускается без GEMINI_API_KEY) — не ломают обычный targeted pytest-прогон без инфры.
"""

from __future__ import annotations

import random
import socket
import uuid
from typing import TypedDict
from urllib.parse import urlparse

import pytest

from knowledge_engine.config import POSTGRES_DSN


class _DemoState(TypedDict):
    """См. test_tutor_graph_service_resume_skips_already_paid_node."""

    step: str
    count: int


_DEMO_CALLS: dict[str, int] = {"a": 0, "b": 0}


def _demo_step_a(state: _DemoState) -> _DemoState:
    _DEMO_CALLS["a"] += 1
    return {"step": "a_done", "count": state.get("count", 0) + 1}


def _demo_step_b(state: _DemoState) -> _DemoState:
    _DEMO_CALLS["b"] += 1
    return {"step": "b_done", "count": state.get("count", 0) + 1}


def _postgres_reachable(dsn: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(dsn)
    try:
        with socket.create_connection(
            (parsed.hostname or "localhost", parsed.port or 5432), timeout=timeout
        ):
            return True
    except OSError:
        return False


_HAS_POSTGRES = _postgres_reachable(POSTGRES_DSN)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not _HAS_POSTGRES,
        reason=(
            "Postgres недоступен на POSTGRES_DSN — "
            "docker compose up -d postgres && alembic upgrade head"
        ),
    ),
]


@pytest.fixture
def unique_doc_id() -> str:
    return f"test_doc_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def two_random_vectors() -> tuple[list[float], list[float]]:
    random.seed(42)
    return (
        [random.random() for _ in range(1024)],
        [random.random() for _ in range(1024)],
    )


async def test_upsert_is_idempotent_on_repeated_run(unique_doc_id, two_random_vectors):
    """Регрессия: до фикса generate_vector_id (детерминированный UUIDv5) в
    vector_store.py:380 стоял uuid4() — повторный upsert плодил дубли вместо
    перезаписи. Здесь тот же payload дважды НЕ должен привести к 2 разным id."""
    from knowledge_engine.db.repositories.postgres_vector_repository import (
        PostgresVectorRepository,
        generate_vector_id,
    )

    repo = await PostgresVectorRepository.create(POSTGRES_DSN)
    try:
        v1, v2 = two_random_vectors
        rows = [
            {
                "id": generate_vector_id(unique_doc_id, 0, "hello world"),
                "payload": {
                    "doc_id": unique_doc_id,
                    "chunk_index": 0,
                    "text": "hello world",
                },
            },
            {
                "id": generate_vector_id(unique_doc_id, 1, "second chunk"),
                "payload": {
                    "doc_id": unique_doc_id,
                    "chunk_index": 1,
                    "text": "second chunk",
                },
            },
        ]
        n1 = await repo.upsert_documents(
            "knowledge_atoms", rows, [v1, v2], embed_model="BAAI/bge-m3"
        )
        n2 = await repo.upsert_documents(
            "knowledge_atoms", rows, [v1, v2], embed_model="BAAI/bge-m3"
        )
        assert n1 == 2
        assert n2 == 2

        results = await repo.search("knowledge_atoms", v1, limit=50)
        matching = [r for r in results if r["payload"].get("doc_id") == unique_doc_id]
        assert len(matching) == 2, "повторный upsert не должен плодить дубли строк"
    finally:
        await repo.delete_by_field("knowledge_atoms", "doc_id", unique_doc_id)
        await repo.close()


async def test_search_prefilter_does_not_collide_with_limit_placeholder(
    unique_doc_id, two_random_vectors
):
    """Регрессия: WHERE-клауза для where_payload_eq раньше нумеровала свой
    placeholder как $2 — та же позиция, что LIMIT — asyncpg падал с
    DatatypeMismatchError ('argument of LIMIT must be type bigint, not text')."""
    from knowledge_engine.db.repositories.postgres_vector_repository import (
        PostgresVectorRepository,
        generate_vector_id,
    )

    repo = await PostgresVectorRepository.create(POSTGRES_DSN)
    try:
        v1, v2 = two_random_vectors
        rows = [
            {
                "id": generate_vector_id(unique_doc_id, 0, "a"),
                "payload": {"doc_id": unique_doc_id, "text": "a"},
            },
        ]
        await repo.upsert_documents(
            "knowledge_atoms", rows, [v1], embed_model="BAAI/bge-m3"
        )

        matched = await repo.search(
            "knowledge_atoms", v1, limit=5, where_payload_eq={"doc_id": unique_doc_id}
        )
        not_matched = await repo.search(
            "knowledge_atoms", v1, limit=5, where_payload_eq={"doc_id": "no_such_doc"}
        )
        assert len(matched) == 1
        assert len(not_matched) == 0
    finally:
        await repo.delete_by_field("knowledge_atoms", "doc_id", unique_doc_id)
        await repo.close()


async def test_tutor_graph_service_resume_skips_already_paid_node():
    """Ядро ценности AsyncPostgresSaver: если граф прервался ПОСЛЕ дорогого
    узла, resume не должен пересчитывать этот узел заново. Это ровно то, чего
    в проекте не было нигде (см. аудит tutor Eval: MemorySaver никогда не
    читался обратно) — реальная демонстрация, а не только unit-mock.

    _DemoState/_demo_step_a/_demo_step_b — намеренно на уровне модуля, не
    внутри функции: из-за `from __future__ import annotations` в этом файле
    LangGraph резолвит аннотации узлов через globals() вызывающего модуля —
    локальный (внутрифункциональный) TypedDict здесь не находится
    (NameError: name 'DemoState' is not defined)."""
    from langgraph.graph import END, StateGraph

    from knowledge_engine.services.tutor_graph_service import TutorGraphService

    _DEMO_CALLS.update(a=0, b=0)
    builder = StateGraph(_DemoState)
    builder.add_node("step_a", _demo_step_a)
    builder.add_node("step_b", _demo_step_b)
    builder.set_entry_point("step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)

    thread_id = f"test-resume-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    async with TutorGraphService(POSTGRES_DSN) as svc:
        graph = builder.compile(checkpointer=svc._saver, interrupt_before=["step_b"])

        await svc.run_or_resume(graph, config, {"step": "init", "count": 0})
        assert _DEMO_CALLS["a"] == 1
        assert _DEMO_CALLS["b"] == 0

        out = await svc.run_or_resume(graph, config, {"step": "init", "count": 0})
        assert _DEMO_CALLS["a"] == 1, "step_a не должен повториться при resume"
        assert _DEMO_CALLS["b"] == 1
        assert out["step"] == "b_done"


async def test_real_tutor_graph_session_compiles_with_postgres_checkpointer():
    """graph.tutor_graph_session() (Phase 2 cutover, engine.py) — не toy-граф,
    а настоящий build_tutor_graph()/TutorGraphState: компилируется и
    переживает aget_state на Postgres-чекпоинтере без реального Gemini-хода
    (структурная проверка — полный живой ход слишком дорог/рискован для
    unit-прогона, см. отчёт)."""
    from knowledge_engine.src.node_deep_dive.graph import tutor_graph_session

    thread_id = f"test-real-graph-{uuid.uuid4().hex[:8]}"
    async with tutor_graph_session() as (svc, graph):
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        assert state.values == {}
        assert state.next == ()


async def test_vector_store_backend_factory_returns_postgres_adapter_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    import knowledge_engine.config as cfg
    import knowledge_engine.services.vector_store as vs
    from knowledge_engine.services.postgres_vector_store_adapter import (
        PostgresVectorStoreAdapter,
    )

    monkeypatch.setattr(cfg, "VECTOR_STORE_BACKEND", "postgres")
    monkeypatch.setattr(vs, "_postgres_store_adapter", None)
    store = await vs._get_active_vector_store()
    assert isinstance(store, PostgresVectorStoreAdapter)


async def test_upsert_knowledge_atoms_via_vector_store_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, unique_doc_id: str
):
    """Регрессия для фикса vector_store.py:412 (был uuid4(), теперь
    generate_vector_id) — проверяется через РЕАЛЬНЫЙ бизнес-путь
    (VectorStore.upsert_knowledge_atoms), не только голый repository."""
    import knowledge_engine.config as cfg
    import knowledge_engine.services.vector_store as vs
    from knowledge_engine.db.repositories.postgres_vector_repository import (
        PostgresVectorRepository,
    )
    from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType

    monkeypatch.setattr(cfg, "VECTOR_STORE_BACKEND", "postgres")
    monkeypatch.setattr(vs, "_postgres_store_adapter", None)

    store = vs.VectorStore()
    monkeypatch.setattr(store._embeddings, "embed_query", lambda text: [0.1] * 1024)

    atoms = [
        KnowledgeAtom(
            scope=ScopeType.PRINCIPLE,
            statement="idempotency regression test statement",
        )
    ]
    url = f"https://example.com/{unique_doc_id}"
    try:
        n1 = await store.upsert_knowledge_atoms(url, atoms, doc_id=unique_doc_id)
        n2 = await store.upsert_knowledge_atoms(url, atoms, doc_id=unique_doc_id)
        assert n1 == 1
        assert n2 == 1

        repo = await PostgresVectorRepository.create(POSTGRES_DSN)
        try:
            count = await repo.count_by_field(
                "knowledge_atoms", "doc_id", unique_doc_id
            )
            assert count == 1, (
                "повторный upsert через vector_store.py не должен плодить "
                "дубли — см. фикс uuid4()->generate_vector_id"
            )
        finally:
            await repo.close()
    finally:
        adapter = await vs._get_active_vector_store()
        await adapter.delete_by_field("knowledge_atoms", "doc_id", unique_doc_id)


@pytest.mark.skipif(
    not _HAS_POSTGRES,
    reason=(
        "Postgres недоступен на POSTGRES_DSN — "
        "docker compose up -d postgres && alembic upgrade head"
    ),
)
def test_vector_store_postgres_backend_survives_across_event_loops():
    """Регрессия (Phase 3): _get_postgres_backend() кэшировал движок как
    singleton НА ВЕСЬ ПРОЦЕСС, но воркер зовёт каждый node_deep_dive job
    через отдельный asyncio.run() (services/work_handlers.py) — второй job
    падал с asyncpg.exceptions.InterfaceError: 'cannot perform operation:
    another operation is in progress' при переиспользовании движка,
    созданного в закрытом event loop'е. НЕ async def — тест намеренно
    вызывает asyncio.run() дважды напрямую, воспроизводя ИМЕННО смену loop'а
    (pytest-anyio даёт один и тот же loop на тест, это здесь не годится)."""
    import asyncio

    import knowledge_engine.services.vector_store as vs

    async def _search_once() -> int:
        store = vs.VectorStore()
        results = await store.search_knowledge_atoms("database index", limit=1)
        return len(results)

    # Каждый asyncio.run() — новый event loop, как отдельный worker job.
    n1 = asyncio.run(_search_once())
    n2 = asyncio.run(_search_once())
    n3 = asyncio.run(_search_once())
    assert (
        n1 >= 0 and n2 >= 0 and n3 >= 0
    ), "все три 'job' должны отработать без InterfaceError"
