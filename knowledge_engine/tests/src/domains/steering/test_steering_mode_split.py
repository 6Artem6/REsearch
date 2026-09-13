"""Разделение Штурвала на Mode 1 (per_node) / Mode 2 (standalone_digest) —

см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "Разделение на Mode 1/Mode 2".

До этой правки ``POST /curriculum/steering/generate`` имел только один
сценарий (Taxonomy+Discovery+Gate1+Gate2 → CurriculumGraph через
search-first), из-за чего 1 approved-статья принудительно становилась
1 нодой графа. Тесты здесь проверяют, что:
- ``steering_mode="per_node"`` вообще не трогает steering_graph/Gate 1/2 —
  это тонкий алиас поверх обычной ``CURRICULUM_GENERATE`` джобы (тот же
  путь, что Autopilot);
- ``steering_mode="standalone_digest"`` (после Gate 2) ставит
  ``STEERING_TOPIC_DIGEST`` джобу с урезанным payload'ом (без user_level/
  depth_level/generation_mode/source_policy — реальный Map-Reduce
  (``services/steering_topic_node_service.py``) их не использует, только
  ``target_goal``/``surface_digests``/``final_approved_urls``/
  ``funnel_job_id``).

``generate_standalone_topic_graph`` (реальный Map-Reduce ПО ВСЕМ статьям →
CurriculumGraph из ОДНОЙ ноды node_kind="steering_standalone", fail-open
только на уровне content-generation LLM-вызова) — отдельно в
test_steering_topic_node_service.py, здесь не дублируется."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from knowledge_engine.src.domains.steering import routes as steering_routes
from knowledge_engine.src.domains.steering.steering_contracts import (
    Gate2ApprovePayload,
    SteeringCurriculumGenerateInput,
)
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cleanup_jobs():
    created: list[str] = []
    yield created
    for job_id in created:
        try:
            work_job_store.fail(job_id, "test cleanup")
        except Exception:
            pass


# --- POST /generate — ветвление по steering_mode ---------------------------


@pytest.mark.anyio
async def test_generate_per_node_never_touches_steering_graph(cleanup_jobs) -> None:
    """Mode 1: никакого Gate 1/2 на уровне курса — steering_graph_session

    не должен вызываться вообще (падение теста при вызове означает
    архитектурную регрессию — Mode 1 снова потащил бы за собой воронку).
    Джоба реальная (не мок) — только сам enqueue замокан, чтобы не гонять
    настоящий Curriculum Generator/Gemini."""

    def _boom():
        raise AssertionError("Mode 1 (per_node) не должен открывать steering_graph")

    real_job = work_job_store.create(WorkJobKind.CURRICULUM_GENERATE, {}, publish=False)
    cleanup_jobs.append(real_job.id)

    with (
        patch.object(steering_routes, "steering_graph_session", _boom),
        patch.object(
            steering_routes, "enqueue_curriculum_generate", return_value=real_job.id
        ) as mock_enqueue,
    ):
        body = SteeringCurriculumGenerateInput(
            target_goal="Изучить распределённые системы",
            steering_mode="per_node",
        )
        result = await steering_routes.post_steering_generate(body)

    mock_enqueue.assert_called_once()
    assert result["work_job_id"] == real_job.id
    assert result["steering_mode"] == "per_node"
    assert result["status"] == "pending"


@pytest.mark.anyio
async def test_generate_per_node_forwards_only_curriculum_fields(cleanup_jobs) -> None:
    """payload, переданный в enqueue_curriculum_generate, не должен тащить

    control_axis/steering_mode — worker (_run_curriculum_generate) их не
    ждёт, это лишний, потенциально запутывающий шум в .runs/work_jobs.json."""
    captured: dict = {}

    def _capture(payload):
        captured.update(payload)
        return "job-x"

    with (
        patch.object(steering_routes, "enqueue_curriculum_generate", _capture),
        patch.object(work_job_store, "get", lambda jid: None),
    ):
        body = SteeringCurriculumGenerateInput(
            target_goal="Изучить распределённые системы",
            steering_mode="per_node",
            source_policy="hybrid",
        )
        await steering_routes.post_steering_generate(body)

    assert captured.get("target_goal") == "Изучить распределённые системы"
    assert captured.get("source_policy") == "hybrid"
    assert "control_axis" not in captured
    assert "steering_mode" not in captured


@pytest.mark.anyio
async def test_generate_standalone_digest_still_runs_funnel(cleanup_jobs) -> None:
    """Mode 2 — прежнее поведение не тронуто: steering_graph_session

    вызывается, Taxonomy возвращается в ответе."""

    class _FakeSnapshot:
        def __init__(self, values):
            self.values = values

    class _FakeGraph:
        async def aget_state(self, config):
            return _FakeSnapshot({"taxonomy": {"candidate_articles": []}})

    class _FakeSvc:
        async def run_or_resume(self, graph, config, *a, **kw):
            return None

    class _FakeSession:
        async def __aenter__(self):
            return _FakeSvc(), _FakeGraph()

        async def __aexit__(self, *exc):
            return False

    with patch.object(
        steering_routes, "steering_graph_session", lambda: _FakeSession()
    ):
        body = SteeringCurriculumGenerateInput(
            target_goal="Изучить распределённые системы",
            steering_mode="standalone_digest",
        )
        result = await steering_routes.post_steering_generate(body)
        cleanup_jobs.append(result["work_job_id"])

    assert result["status"] == "awaiting_gate_1" or "taxonomy_discovery" in result
    assert "taxonomy_discovery" in result


# --- POST /approve-gate2 — Mode 2 создаёт STEERING_TOPIC_DIGEST -----------


@pytest.mark.anyio
async def test_approve_gate2_creates_topic_digest_not_map_reduce(cleanup_jobs) -> None:
    class _FakeSnapshot:
        def __init__(self, values):
            self.values = values

    class _FakeGraph:
        async def aget_state(self, config):
            return _FakeSnapshot({"surface_digests": {"digests": []}})

    class _FakeSvc:
        async def run_or_resume(self, graph, config, *a, **kw):
            return None

    class _FakeSession:
        async def __aenter__(self):
            return _FakeSvc(), _FakeGraph()

        async def __aexit__(self, *exc):
            return False

    job = work_job_store.create(
        WorkJobKind.STEERING_FUNNEL,
        {
            "target_goal": "g",
            "user_level": "Beginner",
            "depth_level": "Overview",
            "generation_mode": "fast",
            "source_policy": "hybrid",
        },
        publish=False,
    )
    cleanup_jobs.append(job.id)
    work_job_store.set_status(job.id, WorkJobStatus.AWAITING_GATE_2)

    with patch.object(
        steering_routes, "steering_graph_session", lambda: _FakeSession()
    ):
        body = Gate2ApprovePayload(
            work_job_id=job.id, final_approved_urls=["https://a"]
        )
        res = await steering_routes.post_steering_approve_gate2(body)
        cleanup_jobs.append(res["generation_job_id"])

    gen_job = work_job_store.get(res["generation_job_id"])
    assert gen_job.kind == WorkJobKind.STEERING_TOPIC_DIGEST
    assert gen_job.payload["target_goal"] == "g"
    assert gen_job.payload["final_approved_urls"] == ["https://a"]
    assert gen_job.payload["funnel_job_id"] == job.id
    # Урезанный payload — реальный Map-Reduce (steering_topic_node_service.py)
    # эти поля не использует, только target_goal/surface_digests/
    # final_approved_urls/funnel_job_id (для уникальности curriculum_id).
    assert "user_level" not in gen_job.payload
    assert "depth_level" not in gen_job.payload
    assert "generation_mode" not in gen_job.payload
    assert "source_policy" not in gen_job.payload
