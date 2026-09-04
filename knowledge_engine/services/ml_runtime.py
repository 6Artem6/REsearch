"""Process role: local ML weights (BGE-M3 / Cross-Encoder) load only in the worker.

The HTTP API is a stateless queue/proxy: it must not hold SentenceTransformer
or CrossEncoder in RAM. Tests and CLI leave ``KE_PROCESS_ROLE`` unset and may
load models. Entry points set the role (do not put this key in ``.env``).
"""

from __future__ import annotations

import os

_ROLE_ENV = "KE_PROCESS_ROLE"
_ROLE_API = "api"
_ROLE_WORKER = "worker"


def get_process_role() -> str:
    return (os.environ.get(_ROLE_ENV) or "").strip().lower()


def is_api_process() -> bool:
    return get_process_role() == _ROLE_API


def is_worker_process() -> bool:
    return get_process_role() == _ROLE_WORKER


def mark_api_process() -> None:
    os.environ[_ROLE_ENV] = _ROLE_API


def mark_worker_process() -> None:
    os.environ[_ROLE_ENV] = _ROLE_WORKER


def ml_weights_allowed() -> bool:
    """False only when this process is the HTTP API."""
    return not is_api_process()


def assert_ml_weights_allowed(what: str) -> None:
    if ml_weights_allowed():
        return
    raise RuntimeError(
        f"{what} must run in the KE worker, not the API process. "
        "Start the worker (`python -m knowledge_engine.worker` or `make dev`)."
    )
