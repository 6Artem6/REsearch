"""Штурвал (Steering) — изолированные REST-эндпоинты поверх SteeringGraphService.

Этап 5 мастер-плана (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md), финальный:
add-only роутер, отдельный префикс ``/api/v1/curriculum/steering`` —
существующий ``api/routes/curriculum.py`` (Autopilot) не импортируется и
не меняется.

РАЗДЕЛЕНИЕ НА MODE 1/MODE 2 (см. roadmap, "Разделение на Mode 1/Mode 2"):
``POST /generate`` ветвится по ``body.steering_mode`` на два независимых
сценария, которые дальше не пересекаются:

- ``per_node`` (по умолчанию) — курс строится РОВНО как у Autopilot
  (``generate_curriculum_graph`` → Model-First DAG без единого источника,
  DEEP-ноды получают ``pending_grounding``), никакого Gate 1/2 на уровне
  курса нет вообще — точечное заземление нод идёт позже, при их открытии,
  через уже существующий Node Grounding Gate (``node/grounding-discover``
  и т.д.), который сам даёт до ``NODE_GATE1_MAX_APPROVED`` источников на
  ноду через настоящий Map-Reduce. Джоба здесь — обычный
  ``WorkJobKind.CURRICULUM_GENERATE`` (тот же worker-путь, что
  ``/curriculum/generate``), НЕ steering-граф — ``/generate`` тут работает
  как тонкий алиас поверх ``enqueue_curriculum_generate``.
- ``standalone_digest`` — TaxonomyService + Light Discovery прямо внутри
  запроса (быстро — секунды), Gate 1 → Batch Digest → Gate 2, затем ПОСЛЕ
  Gate 2 — РЕАЛЬНЫЙ тяжёлый Map-Reduce ПО ВСЕМ approved URL сразу (тот же
  движок, что Autopilot/Node Grounding Gate — полный текст, MAP+REDUCE, НЕ
  пересборка уже тонких Gate-2-дайджестов), итог — валидный
  ``CurriculumGraph`` из РОВНО ОДНОЙ сфокусированной ноды
  (``node_kind=graph_kind="steering_standalone"`` — отдельный тип со снятым
  капом ``mapped_source_ids``/``nodes>=3``, см. условную валидацию в
  schemas.py; Autopilot продолжает валидироваться как раньше без
  исключений) — см. ``services/steering_topic_node_service.py``. Прежний
  путь через search_first (``steering_generator_bridge.py``,
  ``WorkJobKind.STEERING_MAP_REDUCE``) принудительно превращал 1
  approved-статью в 1 ноду и больше не используется новым кодом; более
  ранняя версия этого же файла (искусственное деление на ≥3 ноды-бакета)
  тоже убрана по прямому запросу — нода ровно одна, с явной структурой
  подтем (``NodeCurriculumBreakdown.subtopics``), контент дальше открывается
  через существующий ``node_deep_dive`` (Лекция/Topic Q&A), отдельной
  генерации лекции здесь нет.

Для ``standalone_digest`` Taxonomy+Discovery/Batch-Digest достаточно
быстрые, чтобы гнать их прямо внутри FastAPI async-обработчика — тот же
постоянный event loop, что и весь остальной ASGI-процесс, НЕ per-job
``asyncio.run()``, как у отдельного worker'а, поэтому здесь безопасно
открывать ``AsyncPostgresSaver`` прямо в хендлере (см.
``SteeringGraphService``/``steering_graph_session``, ``steering_graph/
builder.py``). Только сам финальный Map-Reduce (шаг после Gate 2) уходит в
обычный WorkJob + существующий worker (``WorkJobKind.STEERING_TOPIC_DIGEST``
— имя сохранено для минимальных изменений истории джоб, по сути строит граф,
не документ) — это единственная по-настоящему долгая операция в этом
сценарии.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.steering.graph import steering_graph_session
from knowledge_engine.src.domains.steering.graph.state import SteeringState
from knowledge_engine.src.domains.steering.steering_contracts import (
    Gate1ApprovePayload,
    Gate2ApprovePayload,
    SteeringCurriculumGenerateInput,
    SurfaceDigestResponse,
    TaxonomyDiscoveryResponse,
)
from knowledge_engine.src.entrypoints.api.helpers.work_enqueue import (
    enqueue_curriculum_generate,
)
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
)

router = APIRouter(prefix="/curriculum/steering", tags=["curriculum-steering"])


def _graph_config(work_job_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": work_job_id}}


def _coerce_taxonomy(value: Any) -> TaxonomyDiscoveryResponse:
    """Checkpointer может вернуть уже готовый Pydantic-инстанс или сырой

    dict (в зависимости от сериализации ``AsyncPostgresSaver``) — не
    полагаемся на конкретную форму."""
    if isinstance(value, TaxonomyDiscoveryResponse):
        return value
    return TaxonomyDiscoveryResponse.model_validate(value or {})


def _coerce_digests(value: Any) -> SurfaceDigestResponse:
    if isinstance(value, SurfaceDigestResponse):
        return value
    return SurfaceDigestResponse.model_validate(value or {})


def _require_job(work_job_id: str) -> Any:
    job = work_job_store.get(work_job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"work job {work_job_id} not found")
    return job


def _require_status(job: Any, expected: WorkJobStatus) -> None:
    if job.status != expected:
        raise HTTPException(
            status_code=409,
            detail=(
                f"work job {job.id} is {job.status.value}, expected "
                f"{expected.value}"
            ),
        )


@router.get("/{work_job_id}/status")
async def get_steering_status(work_job_id: str) -> dict[str, Any]:
    """Read-only восстановление состояния Gate 1/2 после обновления

    страницы. НЕ двигает граф вперёд (в отличие от approve-gate1/2) — та же
    ``graph.aget_state(config)``, что approve-эндпоинты уже вызывают
    ПОСЛЕ ``run_or_resume`` для построения ответа, просто без самого
    ``run_or_resume``. Нужен, потому что виртуальный
    ``curriculum_id = steering-{work_job_id}`` никогда не попадает в
    обычный ``skill_tree_store`` — при F5 фронтенд раньше не мог
    восстановить состояние иначе и падал на 404 "Маршрут не найден"
    (обычный ``GET /skill-tree/curricula/{id}/workspace``, который ищет
    этот ID там, где его никогда не было и не будет, пока Gate 2 не
    завершится Map-Reduce'ом)."""
    job = _require_job(work_job_id)
    config = _graph_config(work_job_id)
    async with steering_graph_session() as (_svc, graph):
        snapshot = await graph.aget_state(config)

    taxonomy = _coerce_taxonomy(snapshot.values.get("taxonomy"))
    digests = _coerce_digests(snapshot.values.get("surface_digests"))
    payload = job.payload or {}
    return {
        "work_job_id": job.id,
        "status": job.status.value,
        "target_goal": payload.get("target_goal", ""),
        "taxonomy_discovery": taxonomy.model_dump(),
        "surface_digests": digests.model_dump(),
        "approved_gate1_urls": list(snapshot.values.get("approved_gate1_urls") or []),
        "result": job.result,
        "error": job.error,
    }


@router.post("/generate")
async def post_steering_generate(
    body: SteeringCurriculumGenerateInput,
) -> dict[str, Any]:
    """Разводит по ``steering_mode`` — см. докстринг модуля."""
    if body.steering_mode == "per_node":
        return await _post_steering_generate_per_node(body)
    return await _post_steering_generate_standalone_digest(body)


async def _post_steering_generate_per_node(
    body: SteeringCurriculumGenerateInput,
) -> dict[str, Any]:
    """Mode 1 (per_node): без Gate 1/2 на уровне курса — тонкий алиас поверх

    ``enqueue_curriculum_generate`` (тот же ``WorkJobKind.CURRICULUM_GENERATE``,
    что ``/curriculum/generate``). Заземление DEEP-нод остаётся точечным/
    ленивым — Node Grounding Gate перехватывает его при открытии ноды,
    независимо от того, как курс был создан."""
    payload = {
        "target_goal": body.target_goal.strip(),
        "user_level": body.user_level,
        "depth_level": body.depth_level,
        "generation_mode": body.generation_mode,
        "source_policy": body.source_policy,
    }
    trace(
        f"API ▶ POST /curriculum/steering/generate | mode=per_node "
        f"{body.target_goal[:60]}…"
    )
    job_id = enqueue_curriculum_generate(payload)
    job = work_job_store.get(job_id)
    status_value = job.status.value if job else WorkJobStatus.PENDING.value
    result: dict[str, Any] = {
        "work_job_id": job_id,
        "status": status_value,
        "steering_mode": "per_node",
        "message": f"GET /api/v1/work-jobs/{job_id}/wait",
    }
    if job and job.status == WorkJobStatus.COMPLETED and job.result:
        result["graph"] = job.result
    elif job and job.status == WorkJobStatus.FAILED:
        raise HTTPException(status_code=502, detail=job.error or "generate failed")
    return result


async def _post_steering_generate_standalone_digest(
    body: SteeringCurriculumGenerateInput,
) -> dict[str, Any]:
    """Mode 2 (standalone_digest): TaxonomyService → Light Discovery → Gate 1.

    Ждёт первую паузу (Gate 1) прямо внутри запроса (Taxonomy+Discovery —
    секунды, не минуты) и возвращает уже готовые ``candidate_articles``, а
    не голый "pending", как Autopilot ``/curriculum/generate``."""
    job = work_job_store.create(
        WorkJobKind.STEERING_FUNNEL, body.model_dump(), publish=False
    )
    # Немедленно уводим из PENDING — see create(publish=False) docstring:
    # никакой worker не должен пытаться claim'ить эту джобу вообще никогда.
    work_job_store.set_status(job.id, WorkJobStatus.RUNNING)
    trace(
        f"API ▶ POST /curriculum/steering/generate | mode=standalone_digest "
        f"job={job.id} {body.target_goal[:60]}…"
    )

    initial: SteeringState = {
        "target_goal": body.target_goal,
        "work_job_id": job.id,
        # Отладка "утечка Consensus/arXiv/S2 в Practical": раньше source_policy
        # доходил только до payload джобы (для approve-gate2 → Map-Reduce), но
        # НЕ до самого графа — TaxonomyService/Light Discovery всегда искали
        # без учёта policy. Теперь это поле состояния читает
        # node_taxonomy_and_discovery (builder.py).
        "source_policy": body.source_policy,
        "taxonomy": None,
        "approved_gate1_urls": [],
        "surface_digests": None,
        "final_approved_urls": [],
    }
    config = _graph_config(job.id)
    try:
        async with steering_graph_session() as (svc, graph):
            await svc.run_or_resume(graph, config, initial)
            snapshot = await graph.aget_state(config)
    except Exception as exc:
        work_job_store.fail(job.id, str(exc))
        raise HTTPException(
            status_code=502, detail=f"steering generate failed: {exc}"
        ) from exc

    taxonomy = _coerce_taxonomy(snapshot.values.get("taxonomy"))
    job = work_job_store.get(job.id)
    return {
        "work_job_id": job.id,
        "status": job.status.value,
        "taxonomy_discovery": taxonomy.model_dump(),
    }


@router.post("/approve-gate1")
async def post_steering_approve_gate1(body: Gate1ApprovePayload) -> dict[str, Any]:
    """Gate 1 approve → ``Command(resume=...)`` → Batch Digest Generator → Gate 2.

    409, если джоба не в ``AWAITING_GATE_1`` (например, уже прошла Gate 1
    или ещё не дошла до него)."""
    job = _require_job(body.work_job_id)
    _require_status(job, WorkJobStatus.AWAITING_GATE_1)
    trace(
        f"API ▶ POST /curriculum/steering/approve-gate1 | job={job.id} "
        f"approved={len(body.approved_urls)}"
    )

    config = _graph_config(job.id)
    try:
        async with steering_graph_session() as (svc, graph):
            await svc.run_or_resume(graph, config, resume=body.model_dump())
            snapshot = await graph.aget_state(config)
    except Exception as exc:
        work_job_store.fail(job.id, str(exc))
        raise HTTPException(
            status_code=502, detail=f"approve-gate1 failed: {exc}"
        ) from exc

    digests = _coerce_digests(snapshot.values.get("surface_digests"))
    job = work_job_store.get(job.id)
    return {
        "work_job_id": job.id,
        "status": job.status.value,
        "surface_digests": digests.model_dump(),
    }


@router.post("/approve-gate2")
async def post_steering_approve_gate2(body: Gate2ApprovePayload) -> dict[str, Any]:
    """Gate 2 approve → ``Command(resume=...)`` → ``node_finalize_steering``

    (COMPLETED), затем СРАЗУ ставит в очередь реальный Map-Reduce по
    approved-статьям → ``CurriculumGraph`` (``WorkJobKind.STEERING_TOPIC_
    DIGEST``, обычный worker) СТРОГО с ``final_approved_urls`` — Discovery
    Phase не вызывается ни разу, см.
    ``src/curriculum/services/steering_topic_node_service.py``. Этот
    эндпоинт достижим только в Mode 2 (standalone_digest) — Mode 1 (per_node)
    не проходит через Gate 1/2 вообще (см. докстринг модуля). 409, если джоба
    не в ``AWAITING_GATE_2``."""
    job = _require_job(body.work_job_id)
    _require_status(job, WorkJobStatus.AWAITING_GATE_2)
    trace(
        f"API ▶ POST /curriculum/steering/approve-gate2 | job={job.id} "
        f"final_approved={len(body.final_approved_urls)}"
    )

    config = _graph_config(job.id)
    try:
        async with steering_graph_session() as (svc, graph):
            await svc.run_or_resume(graph, config, resume=body.model_dump())
            snapshot = await graph.aget_state(config)
    except Exception as exc:
        work_job_store.fail(job.id, str(exc))
        raise HTTPException(
            status_code=502, detail=f"approve-gate2 failed: {exc}"
        ) from exc

    digests = _coerce_digests(snapshot.values.get("surface_digests"))
    original = job.payload or {}
    gen_payload = {
        "target_goal": original.get("target_goal", ""),
        "surface_digests": digests.model_dump(),
        "final_approved_urls": list(body.final_approved_urls),
        "funnel_job_id": job.id,
    }
    gen_job = work_job_store.create(WorkJobKind.STEERING_TOPIC_DIGEST, gen_payload)
    # Дописываем generation_job_id в РЕЗУЛЬТАТ уже COMPLETED funnel-джобы
    # (node_finalize_steering сохранил туда только final_approved_urls) —
    # иначе после обновления страницы GET .../status не сможет сказать,
    # какую джобу синтеза документа ждать (см. её докстринг).
    work_job_store.complete(
        job.id,
        {
            "final_approved_urls": list(body.final_approved_urls),
            "generation_job_id": gen_job.id,
        },
    )

    job = work_job_store.get(job.id)
    return {
        "work_job_id": job.id,
        "status": job.status.value,
        "final_approved_urls": list(body.final_approved_urls),
        "generation_job_id": gen_job.id,
        "message": f"GET /api/v1/work-jobs/{gen_job.id}/wait",
    }


__all__ = ["router"]
