"""Консольная цепочка v0.4 pipeline."""

from __future__ import annotations

from rich.console import Console

from knowledge_engine.src.core.run_log import trace

_console = Console()


def pipeline_phase(label: str) -> None:
    trace(f"PIPELINE ▶ {label}")
    from knowledge_engine.src.config.settings import KE_TRACE_STDOUT

    if KE_TRACE_STDOUT:
        print(f"▶ {label}", flush=True)
    else:
        _console.print(f"[bold cyan]▶ {label}[/bold cyan]")
