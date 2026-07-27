"""Rich-отображение Trade-off матрицы (CLI и API job viewer)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_engine.schemas import AnalysisReport

console = Console()


def render_abstractions(report: AnalysisReport) -> None:
    if not report.abstractions:
        return
    table = Table(
        title="CS-абстракции (L0)",
        header_style="bold magenta",
        expand=True,
        show_lines=True,
    )
    table.add_column("Слой", style="cyan", width=12)
    table.add_column("Концепт", min_width=20)
    table.add_column("Описание", min_width=30)
    for ab in report.abstractions:
        table.add_row(ab.title, ab.cs_concept[:80], ab.description[:200])
    console.print()
    console.print(table)
    console.print()


def render_report_table(report: AnalysisReport) -> None:
    table = Table(
        title="Trade-off матрица (выберите ID для детальной раскрутки)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
        pad_edge=True,
        padding=(0, 1),
        show_lines=True,
    )
    table.add_column("ID", style="bold green", width=4)
    table.add_column("Паттерн", min_width=16)
    table.add_column("Категория", width=14)
    table.add_column("Суть", min_width=24)
    table.add_column("Риски (кратко)", min_width=20)

    options = sorted(report.options, key=lambda o: o.id)
    for idx, opt in enumerate(options):
        if idx > 0:
            table.add_section()
        risks = "; ".join(opt.cons_and_risks[:2])
        if len(opt.cons_and_risks) > 2:
            risks += " …"
        table.add_row(
            str(opt.id),
            opt.pattern_name,
            opt.category,
            opt.fundamental_idea[:120]
            + ("…" if len(opt.fundamental_idea) > 120 else ""),
            risks[:100] + ("…" if len(risks) > 100 else ""),
        )

    console.print()
    console.print(table)
    console.print()

    for idx, opt in enumerate(options):
        if idx > 0:
            console.print()
        console.print(
            Panel(
                f"[bold]Плюсы:[/bold] {', '.join(opt.pros)}\n"
                f"[bold]Operational cost:[/bold] {opt.operational_cost}",
                title=f"Вариант {opt.id}: {opt.pattern_name}",
                border_style="dim",
                padding=(1, 2),
            )
        )
    console.print()


def pick_option_id(report: AnalysisReport) -> int:
    valid_ids = {o.id for o in report.options}
    while True:
        raw = typer.prompt("Введите ID варианта для unraveling (1–3)")
        try:
            choice = int(raw.strip())
        except ValueError:
            console.print("[red]Нужно целое число.[/red]")
            continue
        if choice in valid_ids:
            return choice
        console.print(
            f"[red]Нет варианта с id={choice}. Доступны: {sorted(valid_ids)}[/red]"
        )
