"""Close stuck work jobs that block duplicate node init.

Default: fail (cancel). Prefer ``--complete`` when the pipeline finished but the
job never flipped to completed (orphan ``running``).

Also releases task-related locks / busy flags so a follow-up init does not hang:
  - Redis ``ke:lock:work:{job_id}`` (claim)
  - Redis ``ke:lock:node_ground:{curriculum}/{node}`` (lazy grounding)
  - ``.runs/worker_dev_busy.json`` (dev watch defer)

Examples:
  # complete orphan running job (preferred when work is done)
  python knowledge_engine/scripts/cancel_work_job.py --id 007588729e86 --complete

  # fail / cancel by job id
  python knowledge_engine/scripts/cancel_work_job.py --id 4c1eb765df15

  # active init for a node (+ always drop grounding lock for that node)
  python knowledge_engine/scripts/cancel_work_job.py \\
    --curriculum agentic_systems_architecture \\
    --node governed_agent_pipelines \\
    --complete

  # all pending+running
  python knowledge_engine/scripts/cancel_work_job.py --all-active

  # only clear locks/busy (no job status change)
  python knowledge_engine/scripts/cancel_work_job.py --release-locks-only \\
    --curriculum agentic_systems_architecture --node governed_agent_pipelines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.services.redis_client import get_redis, redis_enabled
from knowledge_engine.services.work_job_store import (
    _RKEY_JOB,
    WorkJobKind,
    WorkJobStatus,
    force_release_work_claim_lock,
    work_job_store,
)


def _result_from_session(job) -> dict[str, Any]:
    """Build a NodeDeepDiveResponse from the persisted session (best-effort)."""
    from knowledge_engine.src.node_deep_dive.schemas import (
        NodeContentBlock,
        NodeDeepDiveResponse,
    )
    from knowledge_engine.src.node_deep_dive.session_store import (
        _session_key,
        get_session,
    )

    payload = job.payload or {}
    cid = str(payload.get("curriculum_id") or "").strip()
    nid = str((payload.get("node_data") or {}).get("node_id") or "").strip()
    if not cid or not nid:
        return {
            "node_id": nid or "unknown",
            "node_status": "unexplored",
            "content": NodeContentBlock().model_dump(),
            "tutor_message": "",
            "session_key": "",
            "closed_orphan_job": True,
        }

    sess = get_session(cid, nid)
    content = NodeContentBlock()
    node_status = "unexplored"
    history: list[dict[str, str]] = []
    if sess is not None:
        content = getattr(sess, "content", None) or content
        if hasattr(content, "model_dump"):
            pass
        else:
            content = NodeContentBlock.model_validate(content)
        node_status = str(getattr(sess, "node_status", None) or "unexplored")
        history = list(getattr(sess, "history", None) or [])

    resp = NodeDeepDiveResponse(
        node_id=nid,
        node_status=node_status,  # type: ignore[arg-type]
        content=content if hasattr(content, "model_dump") else NodeContentBlock(),
        tutor_message="",
        history=history,
        session_key=_session_key(cid, nid),
    )
    out = resp.model_dump()
    out["closed_orphan_job"] = True
    return out


def _fail(job_id: str, reason: str) -> bool:
    job = work_job_store.get(job_id)
    if job is None:
        print(f"not found: {job_id}")
        return False
    if job.status in (WorkJobStatus.COMPLETED, WorkJobStatus.FAILED):
        print(f"already terminal: {job_id} status={job.status.value}")
        return False
    work_job_store.fail(job_id, reason)
    print(f"failed: {job_id} was={job.status.value} → failed | {reason}")
    return True


def _complete(job_id: str) -> bool:
    job = work_job_store.get(job_id)
    if job is None:
        print(f"not found: {job_id}")
        return False
    if job.status == WorkJobStatus.COMPLETED and job.result:
        print(f"already completed: {job_id}")
        return False
    if job.status == WorkJobStatus.FAILED:
        print(f"already failed: {job_id} | {job.error}")
        return False
    result = job.result if isinstance(job.result, dict) and job.result else None
    if not result:
        result = _result_from_session(job)
    work_job_store.complete(job_id, result)
    print(
        f"completed: {job_id} was={job.status.value} → completed | "
        f"result_keys={list(result.keys())[:12]}"
    )
    return True


def _kind_value(job) -> str:
    kind = getattr(job, "kind", None)
    if kind is None:
        return ""
    return kind.value if hasattr(kind, "value") else str(kind)


def _payload_node(job) -> tuple[str, str]:
    payload = getattr(job, "payload", None) or {}
    cid = str(payload.get("curriculum_id") or "").strip()
    nid = str((payload.get("node_data") or {}).get("node_id") or "").strip()
    return cid, nid


def _release_node_grounding_lock(cid: str, nid: str) -> bool:
    if not cid or not nid:
        return False
    try:
        from knowledge_engine.services.node_grounding_lock import (
            force_release_node_grounding_lock,
        )

        if force_release_node_grounding_lock(cid, nid):
            print(f"released grounding lock | {cid}/{nid}")
            return True
    except Exception as exc:
        print(f"grounding lock release skip | {exc}")
    return False


def _release_work_claim_lock(job_id: str) -> bool:
    jid = str(job_id or "").strip()
    if not jid:
        return False
    try:
        if force_release_work_claim_lock(jid):
            print(f"released work claim lock | job={jid}")
            return True
    except Exception as exc:
        print(f"work claim lock release skip | {exc}")
    return False


def _clear_worker_busy() -> bool:
    try:
        from knowledge_engine.services.worker_busy import clear_worker_busy_file

        if clear_worker_busy_file():
            print("cleared worker_dev_busy.json")
            return True
    except Exception as exc:
        print(f"worker busy clear skip | {exc}")
    return False


def release_locks_for_job(job) -> int:
    """Release all known locks tied to one work job. Returns count released."""
    n = 0
    jid = str(getattr(job, "id", "") or "").strip()
    if _release_work_claim_lock(jid):
        n += 1
    if _kind_value(job) == WorkJobKind.NODE_DEEP_DIVE.value:
        cid, nid = _payload_node(job)
        if _release_node_grounding_lock(cid, nid):
            n += 1
    return n


def release_locks_for_node(curriculum_id: str, node_id: str) -> int:
    """Drop grounding lock for a node (even if no active job remains)."""
    return 1 if _release_node_grounding_lock(curriculum_id, node_id) else 0


def release_orphan_work_claim_locks() -> int:
    """Delete leftover ``ke:lock:work:*`` keys (safe after cancel/kill)."""
    if not redis_enabled():
        return 0
    n = 0
    try:
        r = get_redis()
        for key in r.scan_iter(match="ke:lock:work:*"):
            try:
                if r.delete(key):
                    k = key.decode() if isinstance(key, bytes) else str(key)
                    print(f"released orphan work claim lock | {k}")
                    n += 1
            except Exception:
                continue
    except Exception as exc:
        print(f"orphan work claim scan skip | {exc}")
    return n


def _close(job_id: str, *, complete: bool, reason: str) -> tuple[bool, int]:
    job = work_job_store.get(job_id)
    ok = _complete(job_id) if complete else _fail(job_id, reason)
    n_locks = 0
    if job is not None:
        # Always try locks: even already-terminal jobs may leave Redis locks.
        n_locks = release_locks_for_job(job)
    return ok, n_locks


def _iter_active_ids() -> list[str]:
    out: list[str] = []
    if redis_enabled():
        r = get_redis()
        for key in r.scan_iter(match=f"{_RKEY_JOB}*"):
            raw = r.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if data.get("status") in (
                WorkJobStatus.PENDING.value,
                WorkJobStatus.RUNNING.value,
            ):
                jid = str(data.get("id") or "").strip()
                if jid:
                    out.append(jid)
        return sorted(set(out))
    work_job_store._reload_from_disk()
    with work_job_store._lock:
        for j in work_job_store._jobs.values():
            if j.status in (WorkJobStatus.PENDING, WorkJobStatus.RUNNING):
                out.append(j.id)
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Close stuck KE work jobs (complete or fail) and release "
            "related Redis locks / worker busy flag"
        )
    )
    p.add_argument("--id", help="Work job id (prefix ok if unique)")
    p.add_argument("--curriculum", help="curriculum_id for active node_deep_dive")
    p.add_argument("--node", help="node_id for active node_deep_dive")
    p.add_argument(
        "--all-active",
        action="store_true",
        help="Close all pending+running work jobs",
    )
    p.add_argument(
        "--complete",
        action="store_true",
        help="Mark job completed (from session result if empty), not failed",
    )
    p.add_argument(
        "--release-locks-only",
        action="store_true",
        help=(
            "Do not change job status; only release claim/grounding locks "
            "and clear worker busy (scoped by --id / --curriculum+--node / "
            "--all-active)"
        ),
    )
    p.add_argument(
        "--reason",
        default="Cancelled by cancel_work_job.py (manual)",
        help="Error message stored on the job (fail mode only)",
    )
    args = p.parse_args()

    reason = (args.reason or "").strip() or "Cancelled by cancel_work_job.py"
    complete = bool(args.complete)
    locks_only = bool(args.release_locks_only)
    done = 0
    locks = 0
    touched = False

    if args.id:
        jid = args.id.strip()
        job = work_job_store.get(jid)
        if job is None and redis_enabled() and len(jid) >= 8:
            # prefix match
            matches = []
            r = get_redis()
            for key in r.scan_iter(match=f"{_RKEY_JOB}{jid}*"):
                raw = r.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                matches.append(str(data.get("id") or ""))
            matches = [m for m in matches if m]
            if len(matches) == 1:
                jid = matches[0]
                job = work_job_store.get(jid)
            elif len(matches) > 1:
                print(f"ambiguous id prefix {args.id!r}: {matches[:8]}")
                return 2
        touched = True
        if locks_only:
            if job is None:
                print(f"not found: {jid}")
                # still drop claim lock key if present
                if _release_work_claim_lock(jid):
                    locks += 1
            else:
                locks += release_locks_for_job(job)
        else:
            ok, n = _close(jid, complete=complete, reason=reason)
            locks += n
            if ok:
                done += 1

    if args.curriculum and args.node:
        cid = args.curriculum.strip()
        nid = args.node.strip()
        touched = True
        active = work_job_store.find_active_node_deep_dive(cid, nid, user_action="init")
        if locks_only:
            if active is not None:
                locks += release_locks_for_job(active)
            else:
                locks += release_locks_for_node(cid, nid)
        else:
            if active is None:
                print(f"no active init job for {cid}/{nid}")
            else:
                ok, n = _close(active.id, complete=complete, reason=reason)
                locks += n
                if ok:
                    done += 1
            # Always clear grounding for this node (covers already-terminal / orphan lock).
            locks += release_locks_for_node(cid, nid)

    if args.all_active:
        touched = True
        for jid in _iter_active_ids():
            job = work_job_store.get(jid)
            if locks_only:
                if job is not None:
                    locks += release_locks_for_job(job)
                else:
                    if _release_work_claim_lock(jid):
                        locks += 1
            else:
                ok, n = _close(jid, complete=complete, reason=reason)
                locks += n
                if ok:
                    done += 1
        locks += release_orphan_work_claim_locks()

    if not touched:
        p.print_help()
        print("\nActive jobs:")
        for jid in _iter_active_ids():
            job = work_job_store.get(jid)
            if not job:
                continue
            nid = ((job.payload or {}).get("node_data") or {}).get("node_id")
            cid = (job.payload or {}).get("curriculum_id")
            print(f"  {job.id}  {job.status.value}  {job.kind.value}  {cid}/{nid}")
        return 1

    # Cancelled/killed workers often leave busy>0; always clear after ops.
    if _clear_worker_busy():
        locks += 1

    print(f"done={done} locks_released≈{locks}")
    if locks_only:
        return 0
    # Locks-only side effects (orphan grounding / busy) still count as success.
    return 0 if done or locks else 1


if __name__ == "__main__":
    raise SystemExit(main())
