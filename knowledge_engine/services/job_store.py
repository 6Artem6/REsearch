"""In-memory хранилище задач анализа (API) + персист в .runs/job_store.json."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from knowledge_engine.config import GRAPH_THREAD_ID, PACKAGE_ROOT

_JOB_STORE_PATH: Path = (PACKAGE_ROOT / ".runs" / "job_store.json").resolve()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    MATRIX_READY = "matrix_ready"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    id: str
    problem: str
    constraints: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    thread_id: str = ""
    matrix_only: bool = False
    discovery_cache_first: bool = False
    report: Optional[dict[str, Any]] = None
    unraveled_details: Optional[str] = None
    selected_option_id: Optional[int] = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    clarify_question: Optional[str] = None


def _dt_parse(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _job_to_dict(job: AnalysisJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "problem": job.problem,
        "constraints": job.constraints,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "thread_id": job.thread_id,
        "matrix_only": job.matrix_only,
        "discovery_cache_first": job.discovery_cache_first,
        "report": job.report,
        "unraveled_details": job.unraveled_details,
        "selected_option_id": job.selected_option_id,
        "error": job.error,
        "log_path": job.log_path,
        "clarify_question": job.clarify_question,
    }


def _job_from_dict(data: dict[str, Any]) -> AnalysisJob:
    return AnalysisJob(
        id=data["id"],
        problem=data["problem"],
        constraints=data.get("constraints", ""),
        status=JobStatus(data.get("status", JobStatus.PENDING.value)),
        created_at=_dt_parse(data["created_at"]),
        updated_at=_dt_parse(data["updated_at"]),
        thread_id=data.get("thread_id", ""),
        matrix_only=bool(data.get("matrix_only", False)),
        discovery_cache_first=bool(data.get("discovery_cache_first", False)),
        report=data.get("report"),
        unraveled_details=data.get("unraveled_details"),
        selected_option_id=data.get("selected_option_id"),
        error=data.get("error"),
        log_path=data.get("log_path"),
        clarify_question=data.get("clarify_question"),
    )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not _JOB_STORE_PATH.is_file():
            return
        try:
            raw = json.loads(_JOB_STORE_PATH.read_text(encoding="utf-8"))
            items = raw.get("jobs") if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                return
            for item in items:
                job = _job_from_dict(item)
                self._jobs[job.id] = job
        except Exception:
            # Повреждённый файл — не блокируем API
            return

    def _persist(self) -> None:
        try:
            _JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            jobs = [_job_to_dict(j) for j in self._jobs.values()]
            tmp = _JOB_STORE_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(_JOB_STORE_PATH)
        except Exception:
            pass

    def create(
        self,
        problem: str,
        constraints: str,
        matrix_only: bool = False,
        discovery_cache_first: bool = False,
    ) -> AnalysisJob:
        job_id = uuid.uuid4().hex[:12]
        thread_id = f"{GRAPH_THREAD_ID}-{job_id}"
        job = AnalysisJob(
            id=job_id,
            problem=problem,
            constraints=constraints,
            thread_id=thread_id,
            matrix_only=matrix_only,
            discovery_cache_first=discovery_cache_first,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist()
        return job

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[AnalysisJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, val in fields.items():
                if hasattr(job, key):
                    setattr(job, key, val)
            job.updated_at = datetime.now(timezone.utc)
            self._persist()
            return job

    def list_recent(self, limit: int = 20) -> list[AnalysisJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


job_store = JobStore()
