"""Centralized stdlib `logging` setup.

Separate from `knowledge_engine/ui/run_log.py`'s `trace()` — that one feeds the
Redis SSE run-log stream, the Rich Live terminal dashboard, and `.runs/*.log`
files, and stays untouched. This module configures a single "knowledge_engine"
logger (level from LOG_LEVEL, optional rotating file via LOG_TO_FILE/
LOG_FILE_PATH); regular `logging.getLogger("knowledge_engine.<module>")`
loggers propagate up to it automatically.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

import knowledge_engine.config as _config

_LOGGER_NAME = "knowledge_engine"
_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s] "
    "[%(filename)s:%(funcName)s:%(lineno)d]: %(message)s"
)
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_configured = False


def configure_logging(
    *,
    level: str | None = None,
    to_file: bool | None = None,
    file_path: Any | None = None,
    force: bool = False,
) -> logging.Logger:
    """Idempotent — safe to call from many entry points (API, worker, scripts,
    tests). Reads knowledge_engine.config dynamically so overrides/monkeypatches
    made before the first call (or with force=True) take effect."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured and not force:
        return logger

    resolved_level = (level or _config.LOG_LEVEL or "INFO").strip().upper()
    resolved_to_file = _config.LOG_TO_FILE if to_file is None else to_file
    resolved_path = Path(file_path) if file_path is not None else _config.LOG_FILE_PATH

    for h in logger.handlers:
        h.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if resolved_to_file:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            resolved_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """`logging.getLogger(name)`, ensuring configure_logging() has run at least once."""
    configure_logging()
    return logging.getLogger(name)


_trace_mirror_logger: logging.Logger | None = None
_trace_mirror_checked = False


def trace_mirror_logger() -> logging.Logger | None:
    """File-only logger for ui/run_log.py's trace() to duplicate into, when
    LOG_TO_FILE is on. No console handler and no propagation — trace()'s own
    stdout behavior (KE_TRACE_STDOUT) must stay the only console path, or every
    trace() call would start printing by default (LOG_LEVEL defaults to INFO).
    Returns None when LOG_TO_FILE is off, so trace() skips the dual-write
    entirely (unchanged behavior, zero added cost) rather than logging nowhere."""
    global _trace_mirror_logger, _trace_mirror_checked
    if _trace_mirror_checked:
        return _trace_mirror_logger
    _trace_mirror_checked = True
    if not _config.LOG_TO_FILE:
        return None

    logger = logging.getLogger(f"{_LOGGER_NAME}.trace")
    for h in logger.handlers:
        h.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, (_config.LOG_LEVEL or "INFO").strip().upper(), logging.INFO))

    _config.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        _config.LOG_FILE_PATH,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(file_handler)

    _trace_mirror_logger = logger
    return logger
