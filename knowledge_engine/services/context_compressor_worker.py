"""Фоновая суммаризация (сжатие) истории диалога тьютора.

Раньше LLM-экстракция fact_manifest (structured summary из вытесненного из
active_window сообщения) выполнялась синхронно прямо на hot path — внутри
``rotate_window_after_message`` (см. ``src/node_deep_dive/step_pipeline.py``),
т.е. блокировала ответ пользователю дополнительным вызовом Gemini и сбросом
кэша префикса. Здесь та же экстракция вынесена в пост-обработку: используем
уже существующую очередь ``work_job_store`` (Redis key-per-job + pub/sub,
либо локальный JSON-файл + poll-fallback, если Redis выключен) — без
отдельного стека Celery/aioredis, которых в проекте нет нигде (worker
процесс — sync redis-py + один ThreadPoolExecutor, не asyncio).

Producer: ``enqueue_dialog_summarize`` — вызывается из hot path
(``rotate_window_after_message``), кладёт задачу и сразу возвращается.
Consumer: ``run_dialog_summarize_job`` — подключена в
``work_handlers.run_work_job`` по ``WorkJobKind.DIALOG_SUMMARIZE``, выполняется
в том же worker-процессе, что и остальные work jobs (сериализовано,
``ThreadPoolExecutor(max_workers=1)`` в ``worker/__main__.py``).

CAS / optimistic locking: см. ``SessionMemory.manifest_version`` и
``session_store.apply_fact_manifest_patch``. Изначально при несовпадении
``expected_manifest_version`` запись абортилась целиком ("защита от гонки"
из исходного тикета) — но живой прогон показал ложные срабатывания: один
ход пользователя может вызвать ``rotate_window_after_message`` дважды
(эвикция user- и tutor-сообщения ОДНОГО хода), оба enqueue стартуют с
одинаковой ``expected_manifest_version``, и второй job всегда получал
mismatch сразу после того, как первый уже применился — хотя реальной
гонки с пользователем тут не было. ``merge_manifest`` аддитивна (union
списков с dedup), поэтому теперь ``apply_fact_manifest_patch`` всегда
мёржит результат поверх СВЕЖЕГО текущего ``fact_manifest`` (не enqueue-time
снимка) и растит версию монотонно — несовпадение версии только логируется,
данные не теряются.

Retry/DLQ: в проекте нет прецедента автоматического ретрая FAILED work job
(есть только реквью зависших RUNNING после рестарта воркера — см.
``work_job_store.requeue_running_work_jobs_on_startup``). Здесь сохранён тот
же принцип, что был в исходном hot-path коде: ошибки/таймауты Gemini не
ретраятся и не заводят job в FAILED — ``run_fact_manifest_extraction`` уже
ловит исключение и делает эвристический fallback (см. fact_manifest.py), так
что job почти всегда завершается успешно (COMPLETED), просто с менее точным
summary при сбое Gemini. Это осознанно не DLQ: цель — не зациклить воркер и
не забивать Redis job'ами, которые будут вечно повторяться. Настоящие баги
(не Gemini, а, например, ошибка записи в session_store) не перехватываются
здесь и утекают в généric try/except в ``redis_worker_dispatch._handle_work_job``,
который помечает job FAILED — тем же способом, что и любой другой work job
в этом коде.
"""

from __future__ import annotations

import time
from typing import Any

from knowledge_engine.services.work_job_store import WorkJobKind, work_job_store
from knowledge_engine.ui.run_log import trace


def enqueue_dialog_summarize(
    curriculum_id: str,
    node_id: str,
    extraction_payload: dict[str, Any],
) -> None:
    """Producer: поставить в очередь фоновую fact_manifest-экстракцию.

    ``extraction_payload`` — результат
    ``fact_manifest.prepare_evicted_for_manifest_extraction()``, уже
    содержащий ``prev_manifest``/``expected_manifest_version``, снятые с live
    memory ДО эвикции (не пересчитывать их здесь — на момент выполнения job'а
    live memory может быть уже другой).
    """
    cid = str(curriculum_id or "").strip()
    nid = str(node_id or "").strip()
    if not cid or not nid:
        trace("WORKER dialog_summarize enqueue ⊘ | missing curriculum_id/node_id")
        return
    job = work_job_store.create(
        WorkJobKind.DIALOG_SUMMARIZE,
        {"curriculum_id": cid, "node_id": nid, **extraction_payload},
    )
    trace(f"WORKER ▶ dialog_summarize enqueue | {cid}/{nid} job={job.id}")


def run_dialog_summarize_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Consumer: вызывается из ``work_handlers.run_work_job`` для
    ``WorkJobKind.DIALOG_SUMMARIZE``. Здесь (не на hot path) уже можно
    спокойно ждать секунды на вызов Gemini.
    """
    from knowledge_engine.src.node_deep_dive.fact_manifest import (
        run_fact_manifest_extraction,
    )
    from knowledge_engine.src.node_deep_dive.session_store import (
        apply_fact_manifest_patch,
    )

    cid = str(payload.get("curriculum_id") or "").strip()
    nid = str(payload.get("node_id") or "").strip()
    expected_version = int(payload.get("expected_manifest_version") or 0)

    t0 = time.perf_counter()
    new_manifest = run_fact_manifest_extraction(payload)
    applied = apply_fact_manifest_patch(cid, nid, expected_version, new_manifest)
    elapsed = time.perf_counter() - t0

    if applied:
        trace(
            f"WORKER ✓ dialog_summarize | {cid}/{nid} merged "
            f"(base expected_version={expected_version}) | {elapsed:.1f}s"
        )
    else:
        trace(
            f"WORKER ⊘ dialog_summarize | {cid}/{nid} | session/memory not found "
            f"(session cleared/reset between enqueue and run) | {elapsed:.1f}s"
        )
    return {"applied": applied, "expected_version": expected_version}
