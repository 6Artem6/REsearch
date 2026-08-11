"""Очередь тяжёлых задач (Gemini, граф, Skill Tree)."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from knowledge_engine.config import KE_WORKER_STALE_RUNNING_SEC, PACKAGE_ROOT
from knowledge_engine.services.redis_client import get_redis, redis_enabled
from knowledge_engine.services.redis_tasks import publish_work_job

_STORE_PATH = (PACKAGE_ROOT / ".runs" / "work_jobs.json").resolve()
_HEARTBEAT_PATH = (PACKAGE_ROOT / ".runs" / "worker_heartbeat.json").resolve()
_RKEY_JOB = "ke:work:job:"


class WorkJobKind(str, Enum):
    CURRICULUM_GENERATE = "curriculum_generate"
    CURRICULUM_EXPAND = "curriculum_expand"
    NODE_DEEP_DIVE = "node_deep_dive"


class WorkJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkJob:
    id: str
    kind: WorkJobKind
    payload: dict[str, Any]
    status: WorkJobStatus = WorkJobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _dt_parse(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _to_dict(job: WorkJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind.value,
        "payload": job.payload,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "result": job.result,
        "error": job.error,
    }


def _from_dict(data: dict[str, Any]) -> WorkJob:
    return WorkJob(
        id=data["id"],
        kind=WorkJobKind(data.get("kind")),
        payload=dict(data.get("payload") or {}),
        status=WorkJobStatus(data.get("status", WorkJobStatus.PENDING.value)),
        created_at=_dt_parse(data["created_at"]),
        updated_at=_dt_parse(data["updated_at"]),
        result=data.get("result"),
        error=data.get("error"),
    )


def _redis_job_key(job_id: str) -> str:
    return f"{_RKEY_JOB}{job_id}"


class WorkJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, WorkJob] = {}
        self._lock = threading.Lock()
        if not redis_enabled():
            self._load()

    def _load(self) -> None:
        if not _STORE_PATH.is_file():
            return
        try:
            raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            items = raw.get("jobs") if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                return
            loaded: dict[str, WorkJob] = {}
            for item in items:
                job = _from_dict(item)
                loaded[job.id] = job
            self._jobs = loaded
        except Exception:
            return

    def _reload_from_disk(self) -> None:
        if redis_enabled():
            return
        with self._lock:
            self._load()

    def _persist(self) -> None:
        if redis_enabled():
            return
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            jobs = [_to_dict(j) for j in self._jobs.values()]
            tmp = _STORE_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(_STORE_PATH)
        except Exception:
            pass

    def _redis_save(self, job: WorkJob) -> None:
        get_redis().set(
            _redis_job_key(job.id),
            json.dumps(_to_dict(job), ensure_ascii=False),
        )

    def _redis_get(self, job_id: str) -> Optional[WorkJob]:
        import time

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                raw = get_redis().get(_redis_job_key(job_id))
                if not raw:
                    return None
                return _from_dict(json.loads(raw))
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        if last_exc:
            raise last_exc
        return None

    def create(self, kind: WorkJobKind, payload: dict[str, Any]) -> WorkJob:
        job_id = uuid.uuid4().hex[:12]
        job = WorkJob(id=job_id, kind=kind, payload=dict(payload))
        if redis_enabled():
            self._redis_save(job)
            publish_work_job(job_id)
            return job
        with self._lock:
            self._jobs[job_id] = job
            self._persist()
        return job

    def get(self, job_id: str) -> Optional[WorkJob]:
        if redis_enabled():
            return self._redis_get(job_id)
        self._reload_from_disk()
        with self._lock:
            return self._jobs.get(job_id)

    def find_active_node_deep_dive(
        self,
        curriculum_id: str,
        node_id: str,
        *,
        user_action: str = "init",
        exclude_job_id: str | None = None,
    ) -> Optional[WorkJob]:
        """PENDING/RUNNING node_deep_dive для той же ноды (коалесинг повторного init)."""
        cid = str(curriculum_id or "").strip()
        nid = str(node_id or "").strip()
        action = str(user_action or "init").strip().lower()
        if not cid or not nid:
            return None
        stale_sec = KE_WORKER_STALE_RUNNING_SEC
        now = datetime.now(timezone.utc)
        active_status = {WorkJobStatus.PENDING, WorkJobStatus.RUNNING}

        def _matches(job: WorkJob) -> bool:
            if job.kind != WorkJobKind.NODE_DEEP_DIVE:
                return False
            if job.status not in active_status:
                return False
            if exclude_job_id and job.id == exclude_job_id:
                return False
            if job.status == WorkJobStatus.RUNNING:
                age = (now - job.updated_at).total_seconds()
                if age > stale_sec:
                    return False
            p = job.payload or {}
            if str(p.get("user_action") or "init").strip().lower() != action:
                return False
            pc = str(p.get("curriculum_id") or "").strip()
            pn = str((p.get("node_data") or {}).get("node_id") or "").strip()
            return pc == cid and pn == nid

        if redis_enabled():
            r = get_redis()
            candidates: list[WorkJob] = []
            for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
                try:
                    raw = r.get(key)
                    if not raw:
                        continue
                    job = _from_dict(json.loads(raw))
                    if _matches(job):
                        candidates.append(job)
                except Exception:
                    continue
            if not candidates:
                return None
            return min(candidates, key=lambda j: j.created_at)

        self._reload_from_disk()
        with self._lock:
            candidates = [j for j in self._jobs.values() if _matches(j)]
            if not candidates:
                return None
            return min(candidates, key=lambda j: j.created_at)

    def find_latest_completed_node_deep_dive(
        self,
        curriculum_id: str,
        node_id: str,
        *,
        user_action: str = "init",
    ) -> Optional[WorkJob]:
        """Latest COMPLETED node_deep_dive with a non-empty result for the node."""
        cid = str(curriculum_id or "").strip()
        nid = str(node_id or "").strip()
        action = str(user_action or "init").strip().lower()
        if not cid or not nid:
            return None

        def _matches(job: WorkJob) -> bool:
            if job.kind != WorkJobKind.NODE_DEEP_DIVE:
                return False
            if job.status != WorkJobStatus.COMPLETED:
                return False
            if not isinstance(job.result, dict) or not job.result:
                return False
            p = job.payload or {}
            if str(p.get("user_action") or "init").strip().lower() != action:
                return False
            pc = str(p.get("curriculum_id") or "").strip()
            pn = str((p.get("node_data") or {}).get("node_id") or "").strip()
            return pc == cid and pn == nid

        if redis_enabled():
            r = get_redis()
            candidates: list[WorkJob] = []
            for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
                try:
                    raw = r.get(key)
                    if not raw:
                        continue
                    job = _from_dict(json.loads(raw))
                    if _matches(job):
                        candidates.append(job)
                except Exception:
                    continue
            if not candidates:
                return None
            return max(candidates, key=lambda j: j.updated_at)

        self._reload_from_disk()
        with self._lock:
            candidates = [j for j in self._jobs.values() if _matches(j)]
            if not candidates:
                return None
            return max(candidates, key=lambda j: j.updated_at)

    def try_claim(self, job_id: str) -> Optional[WorkJob]:
        if not redis_enabled():
            return None
        stale_sec = KE_WORKER_STALE_RUNNING_SEC
        now = datetime.now(timezone.utc)
        r = get_redis()
        lock = r.lock(f"ke:lock:work:{job_id}", timeout=10, blocking_timeout=5)
        if not lock.acquire(blocking=False):
            return None
        try:
            job = self._redis_get(job_id)
            if not job:
                return None
            if job.status == WorkJobStatus.RUNNING:
                age = (now - job.updated_at).total_seconds()
                if age <= stale_sec:
                    return None
                try:
                    from knowledge_engine.services.worker_busy import (
                        worker_busy_for_reload,
                    )

                    if worker_busy_for_reload():
                        return None
                except Exception:
                    pass
                job.status = WorkJobStatus.PENDING
            if job.status != WorkJobStatus.PENDING:
                return None
            job.status = WorkJobStatus.RUNNING
            job.updated_at = now
            self._redis_save(job)
            return job
        finally:
            try:
                lock.release()
            except Exception:
                pass

    def claim_next_pending(self) -> Optional[WorkJob]:
        if redis_enabled():
            return None
        self._reload_from_disk()
        with self._lock:
            stale_sec = KE_WORKER_STALE_RUNNING_SEC
            now = datetime.now(timezone.utc)
            for j in self._jobs.values():
                if j.status != WorkJobStatus.RUNNING:
                    continue
                age = (now - j.updated_at).total_seconds()
                if age > stale_sec:
                    try:
                        from knowledge_engine.services.worker_busy import (
                            worker_busy_for_reload,
                        )

                        if worker_busy_for_reload():
                            continue
                    except Exception:
                        pass
                    j.status = WorkJobStatus.PENDING
                    j.updated_at = now
            pending = [
                j for j in self._jobs.values() if j.status == WorkJobStatus.PENDING
            ]
            if not pending:
                return None
            job = min(pending, key=lambda j: j.created_at)
            job.status = WorkJobStatus.RUNNING
            job.updated_at = datetime.now(timezone.utc)
            self._persist()
            return job

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        if redis_enabled():
            job = self._redis_get(job_id)
            if not job:
                return
            job.status = WorkJobStatus.COMPLETED
            job.result = result
            job.error = None
            job.updated_at = datetime.now(timezone.utc)
            self._redis_save(job)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = WorkJobStatus.COMPLETED
            job.result = result
            job.error = None
            job.updated_at = datetime.now(timezone.utc)
            self._persist()

    def fail(self, job_id: str, error: str) -> None:
        if redis_enabled():
            job = self._redis_get(job_id)
            if not job:
                return
            job.status = WorkJobStatus.FAILED
            job.error = error
            job.updated_at = datetime.now(timezone.utc)
            self._redis_save(job)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = WorkJobStatus.FAILED
            job.error = error
            job.updated_at = datetime.now(timezone.utc)
            self._persist()


work_job_store = WorkJobStore()


def count_running_work_jobs(max_age_sec: float | None = None) -> int:
    """Число work jobs в RUNNING (не stale) — для dev reload defer."""
    stale_sec = max_age_sec
    if stale_sec is None:
        stale_sec = KE_WORKER_STALE_RUNNING_SEC
    now = datetime.now(timezone.utc)
    n = 0
    if redis_enabled():
        r = get_redis()
        for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
            try:
                raw = r.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                if data.get("status") != WorkJobStatus.RUNNING.value:
                    continue
                updated = data.get("updated_at")
                if updated:
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    if (now - dt).total_seconds() > stale_sec:
                        continue
                n += 1
            except Exception:
                continue
        return n
    work_job_store._reload_from_disk()
    with work_job_store._lock:
        for j in work_job_store._jobs.values():
            if j.status != WorkJobStatus.RUNNING:
                continue
            age = (now - j.updated_at).total_seconds()
            if age <= stale_sec:
                n += 1
    return n


def list_pending_work_job_ids() -> list[str]:
    """All PENDING work job ids (Redis). Empty when Redis is off."""
    if not redis_enabled():
        return []
    r = get_redis()
    out: list[str] = []
    for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
        try:
            raw = r.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("status") != WorkJobStatus.PENDING.value:
                continue
            jid = str(data.get("id") or "").strip()
            if jid:
                out.append(jid)
        except Exception:
            continue
    return out


def republish_pending_work_jobs() -> int:
    """
    Re-notify worker for PENDING jobs (lost pub/sub after restart / duplicate coalesce).
    Returns number of publish attempts.
    """
    if not redis_enabled():
        return 0
    ids = list_pending_work_job_ids()
    for jid in ids:
        publish_work_job(jid)
    return len(ids)


def recover_stale_running_work_jobs() -> int:
    """Пометить зависшие running jobs как failed (после краша/kill worker)."""
    if not redis_enabled():
        return 0
    stale_sec = KE_WORKER_STALE_RUNNING_SEC
    now = datetime.now(timezone.utc)
    recovered = 0
    r = get_redis()
    for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
        try:
            raw = r.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("status") != WorkJobStatus.RUNNING.value:
                continue
            updated = _dt_parse(str(data.get("updated_at") or now.isoformat()))
            age = (now - updated).total_seconds()
            if age <= stale_sec:
                continue
            try:
                from knowledge_engine.services.worker_busy import worker_busy_for_reload

                if worker_busy_for_reload():
                    continue
            except Exception:
                pass
            data["status"] = WorkJobStatus.FAILED.value
            data["error"] = (
                f"Work job stale (running {int(age)}s > {int(stale_sec)}s). "
                "Worker перезапустите: make dev."
            )
            data["updated_at"] = now.isoformat()
            r.set(key, json.dumps(data, ensure_ascii=False))
            recovered += 1
        except Exception:
            continue
    return recovered


def requeue_running_work_jobs_on_startup() -> int:
    """
    Worker boot: any RUNNING job is orphaned (previous process is gone).
    Requeue as PENDING, drop node grounding locks, republish.
    Age-based fail (``recover_stale_running_work_jobs``) alone is too slow
    (default 2h) and leaves the UI blocked on a dead ``running`` job.
    """
    if not redis_enabled():
        return 0
    now = datetime.now(timezone.utc)
    r = get_redis()
    requeued: list[dict] = []
    for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
        try:
            raw = r.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("status") != WorkJobStatus.RUNNING.value:
                continue
            data["status"] = WorkJobStatus.PENDING.value
            data["error"] = None
            data["updated_at"] = now.isoformat()
            r.set(key, json.dumps(data, ensure_ascii=False))
            requeued.append(data)
        except Exception:
            continue

    if not requeued:
        return 0

    try:
        from knowledge_engine.services.node_grounding_lock import (
            force_release_node_grounding_lock,
        )
        from knowledge_engine.services.worker_busy import clear_worker_busy_file

        clear_worker_busy_file()
        for data in requeued:
            jid = str(data.get("id") or "").strip()
            if jid:
                try:
                    r.delete(f"ke:lock:work:{jid}")
                except Exception:
                    pass
            payload = data.get("payload") or {}
            if str(data.get("kind") or "") != WorkJobKind.NODE_DEEP_DIVE.value:
                continue
            cid = str(payload.get("curriculum_id") or "").strip()
            nid = str((payload.get("node_data") or {}).get("node_id") or "").strip()
            if cid and nid:
                force_release_node_grounding_lock(cid, nid)
    except Exception:
        pass

    for data in requeued:
        jid = str(data.get("id") or "").strip()
        if jid:
            publish_work_job(jid)
    return len(requeued)


def force_release_work_claim_lock(job_id: str) -> bool:
    """Delete Redis claim lock ``ke:lock:work:{id}`` left after kill/cancel."""
    jid = str(job_id or "").strip()
    if not jid or not redis_enabled():
        return False
    try:
        return bool(get_redis().delete(f"ke:lock:work:{jid}"))
    except Exception:
        return False


def write_worker_heartbeat(pid: int) -> None:
    if redis_enabled():
        from knowledge_engine.services.redis_run_log import write_heartbeat

        write_heartbeat(pid)
        return
    try:
        _HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_PATH.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def worker_is_alive(max_age_sec: float = 45.0) -> bool:
    if redis_enabled():
        from knowledge_engine.services.redis_run_log import worker_heartbeat_alive

        return worker_heartbeat_alive(max_age_sec)
    if not _HEARTBEAT_PATH.is_file():
        return False
    try:
        raw = json.loads(_HEARTBEAT_PATH.read_text(encoding="utf-8"))
        ts = _dt_parse(str(raw.get("ts") or ""))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age <= max_age_sec
    except Exception:
        return False
