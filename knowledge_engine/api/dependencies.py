"""FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from knowledge_engine.services.job_store import JobStore, job_store


def get_job_store() -> JobStore:
    return job_store


@lru_cache
def get_executor_workers() -> int:
    return 2
