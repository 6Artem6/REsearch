"""Контекст быстрого академического fetch (node/init lazy grounding)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_fast_academic: ContextVar[bool] = ContextVar("ke_fast_academic_fetch", default=False)


def fast_academic_fetch_enabled() -> bool:
    return bool(_fast_academic.get())


@contextmanager
def fast_academic_fetch_scope(enabled: bool = True):
    token = _fast_academic.set(bool(enabled))
    try:
        yield
    finally:
        _fast_academic.reset(token)
