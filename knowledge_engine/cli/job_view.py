"""Просмотр API job: Rich-матрица, JSON, интерактивный unravel."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click
import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from knowledge_engine.config import KE_API_BASE, PACKAGE_ROOT
from knowledge_engine.schemas import AnalysisReport
from knowledge_engine.ui.errors import trace_exception
from knowledge_engine.ui.job_report import (
    pick_option_id,
    render_abstractions,
    render_report_table,
)
from knowledge_engine.ui.markdown_terminal import unravel_panel

app = typer.Typer(add_completion=False, invoke_without_command=True)
console = Console()

_LAST_WAIT = PACKAGE_ROOT / ".runs" / "last-wait-response.json"


def _base_url() -> str:
    return KE_API_BASE


def _load_payload(path: Optional[Path]) -> dict[str, Any]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise typer.BadParameter("Нет JSON (файл или stdin)")
    return json.loads(text)


def _job_from_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "job" in data:
        return data["job"]
    return data


def _print_job_header(job: dict[str, Any]) -> None:
    console.print(
        Panel(
            f"[bold]id[/bold] {job.get('id')}\n"
            f"[bold]status[/bold] {job.get('status')}\n"
            f"[bold]problem[/bold] {job.get('problem')}\n"
            f"[bold]constraints[/bold] {job.get('constraints') or '(нет)'}\n"
            f"[bold]log[/bold] {job.get('log_path') or '—'}",
            title="Analysis job",
            border_style="blue",
        )
    )
    if job.get("error"):
        console.print(f"[bold red]error:[/bold red] {job['error']}")
    if job.get("clarify_question"):
        console.print(
            Panel(str(job["clarify_question"]), title="Clarify", border_style="yellow")
        )


def _fallback_job_from_disk(job_id: str) -> Optional[dict[str, Any]]:
    if not _LAST_WAIT.is_file():
        return None
    try:
        data = json.loads(_LAST_WAIT.read_text(encoding="utf-8"))
        job = _job_from_payload(data)
        if job.get("id") == job_id and job.get("report"):
            return job
    except Exception:
        return None
    return None


def _fetch_job_from_api(client: httpx.Client, job_id: str) -> dict[str, Any]:
    try:
        r = client.get(f"/api/v1/analyses/{job_id}")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            fb = _fallback_job_from_disk(job_id)
            if fb is not None:
                console.print(
                    f"[yellow]fallback[/yellow]: job {job_id} из {_LAST_WAIT.relative_to(PACKAGE_ROOT.parent)} "
                    "(API перезапущен, unravel через API недоступен без job в store)"
                )
                return fb
        raise


def _print_unravel_hint(job: dict[str, Any], api: str) -> None:
    jid = job.get("id")
    if not jid:
        return
    base = api.rstrip("/")
    console.print(
        Panel(
            f"Unravel не перезапускает analyze — POST /unravel для job {jid}:\n\n"
            f"  ./knowledge_engine/scripts/unravel-analysis.sh {jid} 2\n\n"
            f"  curl -s -X POST '{base}/api/v1/analyses/{jid}/unravel' "
            f"-H 'Content-Type: application/json' "
            f'-d \'{{"option_id": 2, "async_mode": true}}\'\n\n'
            f"  ./knowledge_engine/scripts/view-job.sh --id {jid} --no-interactive\n"
            f"  ./knowledge_engine/scripts/view-job.sh -f "
            f"{_LAST_WAIT.relative_to(PACKAGE_ROOT.parent)} --no-interactive\n"
            f"  ./knowledge_engine/scripts/view-job.sh --no-interactive",
            title="Продолжить unravel",
            border_style="cyan",
        )
    )


def _confirm_or_hint(message: str, job: dict[str, Any], api: str) -> bool:
    """True = продолжить; False = отказ или прерывание; не бросает Abort наружу."""
    try:
        return typer.confirm(message, default=True)
    except click.Abort:
        console.print(
            "[yellow]Подтверждение прервано (Ctrl+C или stdin был pipe, не терминал).[/yellow]"
        )
        _print_unravel_hint(job, api)
        return False


def _wait_job(
    client: httpx.Client,
    job_id: str,
    target: str = "completed",
    timeout_sec: float = 600,
) -> dict[str, Any]:
    url = f"/api/v1/analyses/{job_id}/wait"
    params = {"timeout_sec": timeout_sec, "interval_sec": 2, "target": target}
    resp = client.get(url, params=params, timeout=timeout_sec + 30)
    resp.raise_for_status()
    return resp.json()


def _render_job(
    job: dict[str, Any],
    full_json: bool,
    interactive: bool,
    api: str,
) -> None:
    _print_job_header(job)

    if full_json:
        text = json.dumps(job, indent=2, ensure_ascii=False, default=str)
        console.print(Syntax(text, "json", theme="monokai", word_wrap=True))
        return

    report_raw = job.get("report")
    if report_raw:
        report = AnalysisReport.model_validate(report_raw)
        render_abstractions(report)
        render_report_table(report)

    if job.get("unraveled_details"):
        console.print(
            unravel_panel(
                str(job["unraveled_details"]),
                f"Unravel (option {job.get('selected_option_id')})",
            )
        )

    status = job.get("status")
    if not interactive:
        return
    if status != "matrix_ready" or job.get("matrix_only"):
        return
    if not report_raw:
        return

    report = AnalysisReport.model_validate(report_raw)
    if not _confirm_or_hint("Запустить unravel для выбранного варианта?", job, api):
        return

    option_id = pick_option_id(report)
    jid = job.get("id")
    if not jid:
        console.print("[red]Нет job id для unravel[/red]")
        return

    try:
        with httpx.Client(base_url=api, timeout=30) as client:
            console.print(f"[cyan]POST unravel option_id={option_id}…[/cyan]")
            r = client.post(
                f"/api/v1/analyses/{jid}/unravel",
                json={"option_id": option_id, "async_mode": True},
            )
            r.raise_for_status()
            console.print("[dim]Ждём completed (long poll)…[/dim]")
            wait = _wait_job(client, jid, target="completed", timeout_sec=600)
            final_job = wait.get("job") or wait
            if final_job.get("unraveled_details"):
                console.print(
                    unravel_panel(
                        str(final_job["unraveled_details"]),
                        f"Unravel ✓ option {option_id}",
                    )
                )
            elif final_job.get("error"):
                console.print(f"[red]{final_job['error']}[/red]")
            if typer.confirm("Показать полный JSON job?", default=False):
                text = json.dumps(final_job, indent=2, ensure_ascii=False, default=str)
                console.print(Syntax(text, "json", theme="monokai", word_wrap=True))
    except click.Abort:
        console.print("[yellow]Прервано.[/yellow]")
        _print_unravel_hint(job, api)
    except Exception as exc:
        detail = trace_exception(exc, "job_view unravel")
        console.print(f"[bold red]{detail}[/bold red]")
        console.print(
            "[dim]Подсказка: перезапустите analyze или POST unravel вручную через curl[/dim]"
        )


@app.callback()
def main(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Option(
        None,
        "--id",
        "-j",
        help="ID job (альтернатива --file / stdin)",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="JSON: ответ GET job или wait",
    ),
    full_json: bool = typer.Option(
        False,
        "--json",
        help="Полный JSON (indent, UTF-8)",
    ),
    no_interactive: bool = typer.Option(
        False,
        "--no-interactive",
        help="Не предлагать unravel после матрицы",
    ),
    base_url: Optional[str] = typer.Option(None, envvar="KE_API_BASE"),
) -> None:
    """Красивый вывод job; из файла, stdin, --id или без аргументов (last-wait)."""
    if ctx.invoked_subcommand is not None:
        return

    interactive = not no_interactive
    api = base_url or _base_url()

    try:
        if file is not None:
            payload = _load_payload(file)
            job = _job_from_payload(payload)
        elif job_id is not None:
            with httpx.Client(base_url=api, timeout=30) as client:
                job = _job_from_payload(_fetch_job_from_api(client, job_id))
        elif _LAST_WAIT.is_file():
            payload = _load_payload(_LAST_WAIT)
            job = _job_from_payload(payload)
            console.print(
                f"[dim]job из {_LAST_WAIT.relative_to(PACKAGE_ROOT.parent)}[/dim]"
            )
        elif not sys.stdin.isatty():
            payload = _load_payload(None)
            job = _job_from_payload(payload)
        else:
            console.print(
                "[red]Укажите --id JOB, --file JSON, pipe в stdin или сохраните last-wait[/red]"
            )
            raise typer.Exit(1)

        _render_job(job, full_json, interactive, api)
    except (typer.Exit, click.Abort):
        raise
    except Exception as exc:
        detail = trace_exception(exc, "job_view")
        console.print(f"[bold red]{detail}[/bold red]")
        if _LAST_WAIT.is_file():
            console.print(
                f"[dim]fallback: ./knowledge_engine/scripts/view-job.sh -f {_LAST_WAIT}[/dim]"
            )
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
