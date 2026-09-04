"""TutorGraphService — async-безопасный lifecycle для AsyncPostgresSaver.

Phase 2 (см. prompt.txt): врезан в живой граф — graph.tutor_graph_session()
(node_deep_dive/graph/__init__.py) открывает TutorGraphService НА КАЖДЫЙ ход
и закрывает его в конце того же вызова, а не держит один долгоживущий
инстанс на весь процесс. Причина: воркер создаёт новый event loop
(asyncio.run()) на каждый job (services/work_handlers.py:
_run_node_deep_dive/_run_node_deep_dive_stream) — пул соединений asyncpg,
открытый в ОДНОМ event loop, нельзя использовать из другого. Открытие/
закрытие пула на каждый ход стоит overhead (десятки мс), но безопасно в
текущей архитектуре воркера; переход на persistent event loop — отдельная,
более крупная задача (не в этом Phase, см. отчёт).

Что даёт:
1. Корректное открытие/закрытие AsyncPostgresSaver без дедлоков внутри
   ОДНОГО event loop (AsyncExitStack).
2. await saver.setup() — идемпотентно создаёт служебные таблицы
   checkpointer'а (не пересоздаёт при повторных вызовах).
3. run_or_resume() — паттерн, которого в проекте раньше не было нигде (см.
   аудит tutor Eval): проверка незавершённого checkpoint'а по thread_id
   перед тем, как гнать граф с нуля — не переигрывает уже оплаченные
   LLM-узлы после падения посреди хода.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge_engine.config import POSTGRES_DSN


class TutorGraphService:
    def __init__(self, dsn: str = POSTGRES_DSN) -> None:
        self._dsn = dsn
        self._stack: AsyncExitStack | None = None
        self._saver: AsyncPostgresSaver | None = None

    async def start(self) -> None:
        """Идемпотентно: повторный вызов не переоткрывает пул соединений."""
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

    async def __aenter__(self) -> "TutorGraphService":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def compile(self, builder: StateGraph) -> CompiledStateGraph:
        if self._saver is None:
            raise RuntimeError(
                "TutorGraphService.start() не вызван — нет активного AsyncPostgresSaver"
            )
        return builder.compile(checkpointer=self._saver)

    async def run_or_resume(
        self,
        graph: CompiledStateGraph,
        config: dict[str, Any],
        initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Если для thread_id (внутри config["configurable"]) уже есть
        незавершённый checkpoint — продолжить с него (ainvoke(None, ...)),
        не пересчитывая уже пройденные узлы и не повторяя уже оплаченные
        LLM-вызовы. Иначе — обычный старт с нуля.

        Принимает ПОЛНЫЙ config (не голый thread_id) сознательно: реальный
        вызов графа тьютора кладёт в config["configurable"] ещё и
        stream_callback (см. engine.py) — при resume ЭТОТ ЖЕ config должен
        уйти в ainvoke(None, config=config), иначе стриминг тронется в
        никуда на резюмированном ходу.
        """
        existing = await graph.aget_state(config)
        if existing and existing.next:
            # existing.next непусто → граф остановился НЕ на терминальном узле
            # (крах/рестарт посреди хода) — продолжаем с последнего checkpoint'а.
            return await graph.ainvoke(None, config=config)
        return await graph.ainvoke(initial_state, config=config)
