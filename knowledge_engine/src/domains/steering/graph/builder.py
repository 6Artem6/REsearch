"""SteeringGraphService — изолированный LangGraph-микрограф «Штурвала» с

двумя HITL-паузами (Gate 1 / Gate 2). Этап 3 мастер-плана, план/статус
этапов — docs/STEERING_AND_TOPIC_QNA_ROADMAP.md. Add-only: не импортирует и
не трогает Autopilot-граф (``src/node_deep_dive/graph``), его
``TutorGraphState`` или ``generator.py`` — узел ``node_finalize_steering``
только сохраняет ``final_approved_urls`` в результат WorkJob (см. докстринг
ниже), реальный запуск тяжёлого Map-Reduce по этим URL — Этап 5 (Gateway),
ещё не начат.

Checkpointer жёстко Postgres (``AsyncPostgresSaver``) — в отличие от
Tutor-графа (``src/node_deep_dive/graph/__init__.py``), у Штурвала НЕТ
MemorySaver dev-фолбэка: паузы Gate 1/2 могут стоять часами/днями, ждать
их в памяти одного процесса бессмысленно. ``compile_steering_graph()``
явно проверяет тип чекпоинтера и падает, если это не ``AsyncPostgresSaver``.

Открытие/закрытие пула на каждый вызов (не один долгоживущий инстанс на
процесс) — тот же паттерн и по той же причине, что ``TutorGraphService``
(см. ``services/tutor_graph_service.py``): воркер поднимает новый event
loop на job, пул asyncpg из одного loop нельзя использовать в другом.
Возобновление после Gate 1/2 работает поверх ЭТОГО же паттерна: checkpoint
лежит в Postgres, а не в памяти процесса, поэтому свежий
``SteeringGraphService`` в НОВОМ вызове (после того как пользователь нажал
approve) корректно резюмирует граф по тому же ``thread_id``
(``work_job_id``).
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from knowledge_engine.src.config.settings import POSTGRES_DSN
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.steering.graph.state import SteeringState
from knowledge_engine.src.domains.steering.services.batch_digest_service import (
    generate_surface_digests,
)
from knowledge_engine.src.domains.steering.services.light_discovery_service import (
    discover_candidates,
)
from knowledge_engine.src.domains.steering.services.taxonomy_service import (
    generate_taxonomy_seed,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    ControlAxis,
    Gate1ApprovePayload,
    Gate2ApprovePayload,
    TaxonomyDiscoveryResponse,
)
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobStatus,
    work_job_store,
)

# --- Ноды -------------------------------------------------------------------


async def node_taxonomy_and_discovery(state: SteeringState) -> dict[str, Any]:
    """Шаги 1-2: TaxonomyService → Light Discovery → статус AWAITING_GATE_1.

    ``source_policy`` (отладка "утечка Consensus/arXiv/S2 в Practical") —
    читается через ``.get`` с дефолтом ``"hybrid"``, чтобы не ломать
    вызовы/тесты, ещё не заполняющие это поле состояния."""
    policy = state.get("source_policy") or "hybrid"
    seed = await generate_taxonomy_seed(state["target_goal"], source_policy=policy)
    taxonomy = await discover_candidates(seed, source_policy=policy)
    work_job_store.set_status(state["work_job_id"], WorkJobStatus.AWAITING_GATE_1)
    trace(
        f"STEERING_GRAPH taxonomy_and_discovery ✓ | job={state['work_job_id']} "
        f"candidates={len(taxonomy.candidate_articles)}"
    )
    return {"taxonomy": taxonomy}


def node_gate1_interrupt(state: SteeringState) -> dict[str, Any]:
    """Gate 1: пауза до Command(resume=Gate1ApprovePayload-совместимый dict)."""
    taxonomy = state.get("taxonomy") or TaxonomyDiscoveryResponse()
    resume_value = interrupt({"kind": "gate1_approval", **taxonomy.model_dump()})
    payload = Gate1ApprovePayload.model_validate(resume_value)
    return {"approved_gate1_urls": payload.approved_urls}


async def node_generate_batch_digests(state: SteeringState) -> dict[str, Any]:
    """Шаг 3: Batch Digest Generator по approved_gate1_urls → AWAITING_GATE_2."""
    digests = await generate_surface_digests(state["approved_gate1_urls"])
    work_job_store.set_status(state["work_job_id"], WorkJobStatus.AWAITING_GATE_2)
    trace(
        f"STEERING_GRAPH generate_batch_digests ✓ | job={state['work_job_id']} "
        f"digests={len(digests.digests)}"
    )
    return {"surface_digests": digests}


def node_gate2_interrupt(state: SteeringState) -> dict[str, Any]:
    """Gate 2: пауза до Command(resume=Gate2ApprovePayload-совместимый dict)."""
    digests = state.get("surface_digests")
    resume_value = interrupt(
        {"kind": "gate2_approval", **(digests.model_dump() if digests else {})}
    )
    payload = Gate2ApprovePayload.model_validate(resume_value)
    return {"final_approved_urls": payload.final_approved_urls}


def node_finalize_steering(state: SteeringState) -> dict[str, Any]:
    """Шаг 5: WorkJob → COMPLETED с final_approved_urls в result.

    Сам запуск тяжёлого Map-Reduce по этим URL этот узел НЕ делает — это
    отдельный, ещё не начатый Этап 5 (Gateway): тот эндпоинт читает
    completed-джобу Штурвала и создаёт обычный CURRICULUM_GENERATE WorkJob
    (``work_job_store.create(...)``) с этими URL как источником. Прямой
    импорт ``generator.py``/Map-Reduce отсюда нарушил бы изоляцию этого
    графа от Autopilot-пайплайна."""
    final_urls = state["final_approved_urls"]
    work_job_store.complete(state["work_job_id"], {"final_approved_urls": final_urls})
    trace(
        f"STEERING_GRAPH finalize ✓ | job={state['work_job_id']} "
        f"final_approved_urls={len(final_urls)}"
    )
    return {}


# --- Граф ---------------------------------------------------------------


def build_steering_graph() -> StateGraph:
    """Линейный граф: TaxonomyService/Discovery → Gate 1 → Batch Digest →

    Gate 2 → финализация. Без conditional edges — оба гейта либо approve
    (продолжение), либо весь WorkJob остаётся висеть в AWAITING_GATE_* до
    approve (reject/cancel — забота вызывающего REST-слоя Этапа 5, не
    самого графа)."""
    workflow = StateGraph(SteeringState)

    workflow.add_node("node_taxonomy_and_discovery", node_taxonomy_and_discovery)
    workflow.add_node("node_gate1_interrupt", node_gate1_interrupt)
    workflow.add_node("node_generate_batch_digests", node_generate_batch_digests)
    workflow.add_node("node_gate2_interrupt", node_gate2_interrupt)
    workflow.add_node("node_finalize_steering", node_finalize_steering)

    workflow.set_entry_point("node_taxonomy_and_discovery")
    workflow.add_edge("node_taxonomy_and_discovery", "node_gate1_interrupt")
    workflow.add_edge("node_gate1_interrupt", "node_generate_batch_digests")
    workflow.add_edge("node_generate_batch_digests", "node_gate2_interrupt")
    workflow.add_edge("node_gate2_interrupt", "node_finalize_steering")
    workflow.add_edge("node_finalize_steering", END)

    return workflow


def compile_steering_graph(
    checkpointer: Any,
    *,
    control_axis: ControlAxis = "steering",
) -> CompiledStateGraph:
    """Компиляция с явной защитой от MemorySaver.

    Штурвал (в отличие от Tutor-графа) не имеет in-memory dev-фолбэка:
    Gate 1/2 могут стоять сколько угодно, RAM-чекпоинт умирает вместе с
    процессом воркера. ``control_axis`` — параметр (не хардкод), чтобы
    Этап 5 мог вызывать эту же функцию из общего диспетчера графов и
    получать проверку "для steering — только Postgres" без дублирования
    условия на своей стороне."""
    if control_axis == "steering" and not isinstance(checkpointer, AsyncPostgresSaver):
        raise ValueError(
            "Steering graph strictly requires Postgres checkpointer backend"
        )
    return build_steering_graph().compile(checkpointer=checkpointer)


class SteeringGraphService:
    """AsyncPostgresSaver lifecycle для Штурвала — тот же паттерн, что

    ``TutorGraphService`` (см. её докстринг про AsyncExitStack и открытие
    пула на каждый вызов), но ``compile()`` всегда идёт через
    ``compile_steering_graph()`` — то есть всегда с проверкой на
    MemorySaver, а не напрямую ``builder.compile(checkpointer=...)``."""

    def __init__(self, dsn: str = POSTGRES_DSN) -> None:
        self._dsn = dsn
        self._stack: AsyncExitStack | None = None
        self._saver: AsyncPostgresSaver | None = None

    @property
    def saver(self) -> AsyncPostgresSaver | None:
        return self._saver

    async def start(self) -> None:
        if self._saver is not None:
            return
        self._stack = AsyncExitStack()
        self._saver = await self._stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(self._dsn)
        )
        await self._saver.setup()

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._saver = None

    async def __aenter__(self) -> "SteeringGraphService":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def compile(self, *, control_axis: ControlAxis = "steering") -> CompiledStateGraph:
        if self._saver is None:
            raise RuntimeError(
                "SteeringGraphService.start() не вызван — нет активного AsyncPostgresSaver"
            )
        return compile_steering_graph(self._saver, control_axis=control_axis)

    async def run_or_resume(
        self,
        graph: CompiledStateGraph,
        config: dict[str, Any],
        initial_state: dict[str, Any] | None = None,
        *,
        resume: Any = None,
    ) -> dict[str, Any]:
        """``resume`` не None → ``Command(resume=...)`` поверх существующего

        checkpoint'а (продолжение после Gate 1/2 — ``resume`` это
        Gate1ApprovePayload/Gate2ApprovePayload как dict). Иначе — как
        ``TutorGraphService.run_or_resume``: незавершённый checkpoint
        резюмируется как есть, иначе свежий старт с initial_state."""
        if resume is not None:
            return await graph.ainvoke(Command(resume=resume), config=config)
        existing = await graph.aget_state(config)
        if existing and existing.next:
            return await graph.ainvoke(None, config=config)
        return await graph.ainvoke(initial_state or {}, config=config)


@asynccontextmanager
async def steering_graph_session() -> (
    AsyncIterator[tuple[SteeringGraphService, CompiledStateGraph]]
):
    """Postgres-checkpointed граф Штурвала на время ОДНОГО вызова — открывает

    и закрывает SteeringGraphService/AsyncPostgresSaver внутри, как
    ``tutor_graph_session()`` (см. её докстринг)."""
    async with SteeringGraphService() as svc:
        yield svc, svc.compile()


__all__ = [
    "node_taxonomy_and_discovery",
    "node_gate1_interrupt",
    "node_generate_batch_digests",
    "node_gate2_interrupt",
    "node_finalize_steering",
    "build_steering_graph",
    "compile_steering_graph",
    "SteeringGraphService",
    "steering_graph_session",
]
