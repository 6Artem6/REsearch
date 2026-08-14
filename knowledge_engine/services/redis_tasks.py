"""Pub/Sub: распределение задач между API и worker."""

from __future__ import annotations

import json
from typing import Any

import knowledge_engine.config as cfg
from knowledge_engine.services.redis_client import get_redis, redis_enabled


def publish_task(message: dict[str, Any]) -> None:
    if not redis_enabled():
        return
    body = json.dumps(message, ensure_ascii=False)
    get_redis().publish(cfg.KE_TASKS_CHANNEL, body)


def publish_work_job(job_id: str) -> None:
    publish_task({"type": "work_job", "id": job_id})


def publish_analysis_job(job_id: str, clarify_answer: str | None = None) -> None:
    msg: dict[str, Any] = {"type": "analysis", "id": job_id}
    if clarify_answer is not None:
        msg["clarify_answer"] = clarify_answer
    publish_task(msg)


def publish_analysis_unravel(job_id: str, option_id: int) -> None:
    publish_task({"type": "analysis_unravel", "id": job_id, "option_id": option_id})


def publish_v07_run(run_id: str) -> None:
    publish_task({"type": "v07", "id": run_id})
