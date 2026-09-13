"""GET /curriculum/steering/{work_job_id}/status — read-only восстановление

Gate 1/2 после обновления страницы (не двигает граф, в отличие от
approve-gate1/2). Добавлено после отчёта пользователя: обновление
страницы во время Штурвала показывало "Маршрут не найден", потому что
виртуальный ``curriculum_id = steering-{job_id}`` никогда не попадает в
обычный ``skill_tree_store`` — см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.

``work_job_store`` — реальный (не Redis в тестовом окружении), джобы
создаются и сразу подчищаются через ``work_job_store.fail(...)``, чтобы не
замусоривать локальный ``.runs/work_jobs.json`` (см. прошлый инцидент,
задокументированный в остальных тестах этой сессии)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from knowledge_engine.src.domains.steering import routes as steering_routes
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobStatus,
    work_job_store,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSnapshot:
    def __init__(self, values: dict):
        self.values = values


class _FakeGraph:
    def __init__(self, values: dict):
        self._values = values

    async def aget_state(self, config):
        return _FakeSnapshot(self._values)


class _FakeSvc:
    async def run_or_resume(self, graph, config, *args, **kwargs):
        return None


class _FakeSteeringSession:
    def __init__(self, values: dict):
        self._graph = _FakeGraph(values)

    async def __aenter__(self):
        return _FakeSvc(), self._graph

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def cleanup_jobs():
    created: list[str] = []
    yield created
    for job_id in created:
        try:
            work_job_store.fail(job_id, "test cleanup")
        except Exception:
            pass


@pytest.mark.anyio
async def test_status_reconstructs_awaiting_gate_1(cleanup_jobs) -> None:
    job = work_job_store.create(
        steering_routes.WorkJobKind.STEERING_FUNNEL,
        {"target_goal": "распределённые системы"},
        publish=False,
    )
    cleanup_jobs.append(job.id)
    work_job_store.set_status(job.id, WorkJobStatus.AWAITING_GATE_1)

    fake_taxonomy = {
        "tags": ["базы данных"],
        "company_hubs": ["yandex"],
        "keywords": ["k"],
        "candidate_articles": [
            {
                "url": "https://a",
                "title": "A",
                "source_hub": "yandex",
                "lead_paragraph": "lead",
            }
        ],
    }
    with patch.object(
        steering_routes,
        "steering_graph_session",
        lambda: _FakeSteeringSession({"taxonomy": fake_taxonomy}),
    ):
        result = await steering_routes.get_steering_status(job.id)

    assert result["status"] == "awaiting_gate_1"
    assert result["target_goal"] == "распределённые системы"
    assert len(result["taxonomy_discovery"]["candidate_articles"]) == 1
    assert result["surface_digests"]["digests"] == []


@pytest.mark.anyio
async def test_status_reconstructs_awaiting_gate_2(cleanup_jobs) -> None:
    job = work_job_store.create(
        steering_routes.WorkJobKind.STEERING_FUNNEL, {}, publish=False
    )
    cleanup_jobs.append(job.id)
    work_job_store.set_status(job.id, WorkJobStatus.AWAITING_GATE_2)

    fake_digests = {
        "digests": [
            {
                "url": "https://a",
                "title": "A",
                "company_or_author": "Yandex",
                "problem_solved": "p",
                "main_tech_stack": [],
                "two_sentence_summary": "s",
            }
        ]
    }
    with patch.object(
        steering_routes,
        "steering_graph_session",
        lambda: _FakeSteeringSession(
            {
                "surface_digests": fake_digests,
                "approved_gate1_urls": ["https://a", "https://b"],
            }
        ),
    ):
        result = await steering_routes.get_steering_status(job.id)

    assert result["status"] == "awaiting_gate_2"
    assert len(result["surface_digests"]["digests"]) == 1
    # 2 утверждено на Gate 1, но только 1 дайджест построился — фронтенд
    # использует это расхождение для hint'а на Gate 2 (см. SteeringGatePanel).
    assert result["approved_gate1_urls"] == ["https://a", "https://b"]


@pytest.mark.anyio
async def test_status_completed_exposes_generation_job_id(cleanup_jobs) -> None:
    job = work_job_store.create(
        steering_routes.WorkJobKind.STEERING_FUNNEL, {}, publish=False
    )
    cleanup_jobs.append(job.id)
    work_job_store.complete(
        job.id, {"final_approved_urls": ["https://a"], "generation_job_id": "gen-1"}
    )

    with patch.object(
        steering_routes, "steering_graph_session", lambda: _FakeSteeringSession({})
    ):
        result = await steering_routes.get_steering_status(job.id)

    assert result["status"] == "completed"
    assert result["result"]["generation_job_id"] == "gen-1"


@pytest.mark.anyio
async def test_status_404_on_unknown_job() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await steering_routes.get_steering_status("does-not-exist")
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_approve_gate2_persists_generation_job_id_on_funnel_job(
    cleanup_jobs,
) -> None:
    """Регрессия для самого фикса: до него ``generation_job_id`` уходил

    только в HTTP-ответ approve-gate2 и терялся при обновлении страницы —
    ``GET .../status`` не мог сказать, какую Map-Reduce джобу ждать."""
    from knowledge_engine.src.domains.steering.steering_contracts import (
        Gate2ApprovePayload,
    )

    job = work_job_store.create(
        steering_routes.WorkJobKind.STEERING_FUNNEL,
        {"target_goal": "g", "generation_mode": "fast", "source_policy": "hybrid"},
        publish=False,
    )
    cleanup_jobs.append(job.id)
    work_job_store.set_status(job.id, WorkJobStatus.AWAITING_GATE_2)

    with patch.object(
        steering_routes,
        "steering_graph_session",
        lambda: _FakeSteeringSession({"surface_digests": {"digests": []}}),
    ):
        body = Gate2ApprovePayload(
            work_job_id=job.id, final_approved_urls=["https://a"]
        )
        res = await steering_routes.post_steering_approve_gate2(body)
        cleanup_jobs.append(res["generation_job_id"])

    stored = work_job_store.get(job.id)
    assert stored.result["generation_job_id"] == res["generation_job_id"]
