"""FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from knowledge_engine.src.shared.job_queue.job_store import JobStore, job_store
from knowledge_engine.src.shared.job_queue.work_job_store import (
    WorkJobStore,
    work_job_store,
)


def get_job_store() -> JobStore:
    return job_store


def get_work_job_store() -> WorkJobStore:
    return work_job_store


@lru_cache
def get_executor_workers() -> int:
    return 2
