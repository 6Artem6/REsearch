"""Temporary fetch→MAP volume tracer. Keep until source-loss audit is closed."""

from __future__ import annotations

from knowledge_engine.ui.run_log import trace


def pipeline_audit(phase: str, url: str, text: str, *, extra: str = "") -> None:
    body = text or ""
    line = (
        f"[Pipeline Audit] Phase: {phase} | Target: {url} | "
        f"Chars: {len(body)} | Words: {len(body.split())}"
    )
    if extra:
        line = f"{line} | {extra}"
    trace(line)
