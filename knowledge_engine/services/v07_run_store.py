"""In-memory store for v0.7 web/API runs."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.redis_client import get_redis, redis_enabled

_V07_STORE_PATH = (PACKAGE_ROOT / ".runs" / "v07_runs.json").resolve()
_RKEY_V07 = "ke:v07:run:"


class V07RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class V07Run:
    id: str
    query: str
    thread_id: str
    status: V07RunStatus = V07RunStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_step: str = "init"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    questions_log: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str = "fast"


def _run_to_dict(run: V07Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "query": run.query,
        "thread_id": run.thread_id,
        "status": run.status.value,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "current_step": run.current_step,
        "result": run.result,
        "error": run.error,
        "log_path": run.log_path,
        "questions_log": run.questions_log,
        "retrieval_mode": run.retrieval_mode,
    }


def _run_from_dict(item: dict[str, Any]) -> V07Run:
    return V07Run(
        id=item["id"],
        query=item["query"],
        thread_id=item.get("thread_id", ""),
        status=V07RunStatus(item.get("status", "pending")),
        created_at=datetime.fromisoformat(item["created_at"]),
        updated_at=datetime.fromisoformat(item["updated_at"]),
        current_step=item.get("current_step", "init"),
        result=item.get("result"),
        error=item.get("error"),
        log_path=item.get("log_path"),
        questions_log=list(item.get("questions_log") or []),
        retrieval_mode=str(item.get("retrieval_mode") or "fast"),
    )


class V07RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, V07Run] = {}
        if not redis_enabled():
            self._load()

    def _load(self) -> None:
        if not _V07_STORE_PATH.is_file():
            return
        try:
            data = json.loads(_V07_STORE_PATH.read_text(encoding="utf-8"))
            loaded: dict[str, V07Run] = {}
            for item in data.get("runs", []):
                run = V07Run(
                    id=item["id"],
                    query=item["query"],
                    thread_id=item.get("thread_id", ""),
                    status=V07RunStatus(item.get("status", "pending")),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    current_step=item.get("current_step", "init"),
                    result=item.get("result"),
                    error=item.get("error"),
                    log_path=item.get("log_path"),
                    questions_log=list(item.get("questions_log") or []),
                    retrieval_mode=str(item.get("retrieval_mode") or "fast"),
                )
                loaded[run.id] = run
            self._runs = loaded
        except Exception:
            pass

    def _reload_from_disk(self) -> None:
        if redis_enabled():
            return
        with self._lock:
            self._load()

    def _redis_save(self, run: V07Run) -> None:
        get_redis().set(
            f"{_RKEY_V07}{run.id}",
            json.dumps(_run_to_dict(run), ensure_ascii=False),
        )

    def _redis_get(self, run_id: str) -> Optional[V07Run]:
        raw = get_redis().get(f"{_RKEY_V07}{run_id}")
        if not raw:
            return None
        return _run_from_dict(json.loads(raw))

    def _persist(self) -> None:
        if redis_enabled():
            return
        _V07_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "runs": [
                    {
                        "id": r.id,
                        "query": r.query,
                        "thread_id": r.thread_id,
                        "status": r.status.value,
                        "created_at": r.created_at.isoformat(),
                        "updated_at": r.updated_at.isoformat(),
                        "current_step": r.current_step,
                        "result": r.result,
                        "error": r.error,
                        "log_path": r.log_path,
                        "questions_log": r.questions_log,
                        "retrieval_mode": r.retrieval_mode,
                    }
                    for r in sorted(
                        self._runs.values(),
                        key=lambda x: x.created_at,
                        reverse=True,
                    )[:80]
                ]
            }
        try:
            _V07_STORE_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def create(
        self,
        query: str,
        thread_id: str,
        *,
        retrieval_mode: str = "fast",
    ) -> V07Run:
        run_id = uuid.uuid4().hex[:12]
        mode = (retrieval_mode or "fast").strip().lower()
        if mode not in ("fast", "consensus"):
            mode = "fast"
        run = V07Run(
            id=run_id,
            query=query.strip(),
            thread_id=thread_id,
            retrieval_mode=mode,
        )
        with self._lock:
            self._runs[run_id] = run
        if redis_enabled():
            self._redis_save(run)
        else:
            self._persist()
        return run

    def get(self, run_id: str) -> Optional[V07Run]:
        if redis_enabled():
            return self._redis_get(run_id)
        self._reload_from_disk()
        with self._lock:
            return self._runs.get(run_id)

    def update(
        self,
        run_id: str,
        *,
        status: Optional[V07RunStatus] = None,
        current_step: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> Optional[V07Run]:
        if redis_enabled():
            run = self._redis_get(run_id)
            if not run:
                return None
            if status is not None:
                run.status = status
            if current_step is not None:
                run.current_step = current_step
            if result is not None:
                run.result = result
            if error is not None:
                run.error = error
            if log_path is not None:
                run.log_path = log_path
            run.updated_at = datetime.now(timezone.utc)
            self._redis_save(run)
            return run
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            if status is not None:
                run.status = status
            if current_step is not None:
                run.current_step = current_step
            if result is not None:
                run.result = result
            if error is not None:
                run.error = error
            if log_path is not None:
                run.log_path = log_path
            run.updated_at = datetime.now(timezone.utc)
        if redis_enabled():
            self._redis_save(run)
        else:
            self._persist()
        return run

    def merge_result(
        self,
        run_id: str,
        patch: dict[str, Any],
        *,
        current_step: Optional[str] = None,
    ) -> Optional[V07Run]:
        if redis_enabled():
            run = self._redis_get(run_id)
            if not run:
                return None
            base = dict(run.result or {})
            for key, val in patch.items():
                base[key] = val
            run.result = base
            if current_step is not None:
                run.current_step = current_step
            run.updated_at = datetime.now(timezone.utc)
            self._redis_save(run)
            return run
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            base = dict(run.result or {})
            for key, val in patch.items():
                base[key] = val
            run.result = base
            if current_step is not None:
                run.current_step = current_step
            run.updated_at = datetime.now(timezone.utc)
        if redis_enabled():
            self._redis_save(run)
        else:
            self._persist()
        return run

    def append_question_log(
        self,
        run_id: str,
        entry: dict[str, Any],
        *,
        max_entries: int = 80,
    ) -> Optional[V07Run]:
        if redis_enabled():
            run = self._redis_get(run_id)
            if not run:
                return None
            log = list(run.questions_log)
            norm_text = str(entry.get("text") or "").strip()
            norm_snip = str(entry.get("snippet") or "").strip()
            norm_type = str(entry.get("type") or "other").strip()
            for ex in log:
                if (
                    str(ex.get("type") or "").strip() == norm_type
                    and str(ex.get("text") or "").strip() == norm_text
                    and str(ex.get("snippet") or "").strip() == norm_snip
                ):
                    return run
            log.append(entry)
            if len(log) > max_entries:
                log = log[-max_entries:]
            run.questions_log = log
            run.updated_at = datetime.now(timezone.utc)
            self._redis_save(run)
            return run
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            log = list(run.questions_log)
            norm_text = str(entry.get("text") or "").strip()
            norm_snip = str(entry.get("snippet") or "").strip()
            norm_type = str(entry.get("type") or "other").strip()
            for ex in log:
                if (
                    str(ex.get("type") or "").strip() == norm_type
                    and str(ex.get("text") or "").strip() == norm_text
                    and str(ex.get("snippet") or "").strip() == norm_snip
                ):
                    return run
            log.append(entry)
            if len(log) > max_entries:
                log = log[-max_entries:]
            run.questions_log = log
            run.updated_at = datetime.now(timezone.utc)
        if redis_enabled():
            self._redis_save(run)
        else:
            self._persist()
        return run

    def list_recent(self, limit: int = 20) -> list[V07Run]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.updated_at, reverse=True)
        return runs[:limit]

    def claim_next_pending(self) -> Optional[V07Run]:
        if redis_enabled():
            return None
        self._reload_from_disk()
        with self._lock:
            pending = [
                r for r in self._runs.values() if r.status == V07RunStatus.PENDING
            ]
            if not pending:
                return None
            run = min(pending, key=lambda r: r.created_at)
            run.status = V07RunStatus.RUNNING
            run.updated_at = datetime.now(timezone.utc)
        if redis_enabled():
            self._redis_save(run)
        else:
            self._persist()
        return run

    def try_claim(self, run_id: str) -> Optional[V07Run]:
        if not redis_enabled():
            return None
        r = get_redis()
        lock = r.lock(f"ke:lock:v07:{run_id}", timeout=30, blocking_timeout=5)
        if not lock.acquire(blocking=False):
            return None
        try:
            run = self._redis_get(run_id)
            if not run or run.status != V07RunStatus.PENDING:
                return None
            run.status = V07RunStatus.RUNNING
            run.updated_at = datetime.now(timezone.utc)
            self._redis_save(run)
            return run
        finally:
            try:
                lock.release()
            except Exception:
                pass


v07_run_store = V07RunStore()
