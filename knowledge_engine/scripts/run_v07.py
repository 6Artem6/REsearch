"""CLI: полный прогон Knowledge Engine v0.7 + развернутый вывод + REPL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Plain line trace в терминале (без Rich Live-панели)
os.environ.setdefault("KE_TRACE_STDOUT", "1")
os.environ.setdefault("KE_LOG_PLAIN", "1")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from knowledge_engine.config import PACKAGE_ROOT, get_graph_version
from knowledge_engine.src.analytics.repl import answer_follow_up, build_repl_context

console = Console()
_DEFAULT_PROFILE = PACKAGE_ROOT / "user_profile.md"

_REPL_EXIT_COMMANDS = frozenset(
    {
        "/quit",
        "/q",
        "/exit",
        "/выход",
        "quit",
        "exit",
        "q",
        "выход",
        "стоп",
        ":q",
    }
)


def _normalize_repl_line(line: str) -> str:
    s = line.strip()
    for ch in ("／", "⁄", "∕"):
        s = s.replace(ch, "/")
    return s


def _repl_should_exit(line: str) -> bool:
    s = _normalize_repl_line(line)
    if not s:
        return True
    first = s.split()[0].lower()
    return first in _REPL_EXIT_COMMANDS


def _bullet_list(title: str, items: List[str], border: str = "cyan") -> None:
    if not items:
        return
    body = "\n".join(f"- {x}" for x in items)
    console.print(Panel(body, title=title, border_style=border))


def _text_panel(title: str, text: str, border: str = "blue") -> None:
    if not (text or "").strip():
        return
    console.print(Panel(text.strip(), title=title, border_style=border))


def _global_anchor_from_result(result: Dict[str, Any]) -> str:
    spec = result.get("query_spec")
    formal = ""
    if isinstance(spec, dict):
        formal = str(spec.get("cs_formal_query") or "")
    if not formal:
        formal = str(result.get("user_query") or "")
    profile = (result.get("user_profile_md") or "")[:1500]
    return f"Задача: {formal}\nПрофиль:\n{profile}"


def _print_concept_graph(cg: Dict[str, Any]) -> None:
    console.print(Rule("[bold]L2a — Синтез источников и концептов[/bold]"))
    _text_panel(
        "Исследовательский синтез",
        cg.get("research_synthesis") or cg.get("task_summary", ""),
    )
    if cg.get("task_summary") and cg.get("research_synthesis"):
        _text_panel("Сводка задачи", cg.get("task_summary", ""), border="dim")

    contrasts = cg.get("cross_source_contrasts") or []
    if contrasts:
        lines: List[str] = []
        for c in contrasts:
            if not isinstance(c, dict):
                continue
            lines.append(
                f"**{c.get('topic', 'Тема')}**\n"
                f"A: {c.get('approach_a', '')}\n"
                f"B: {c.get('approach_b', '')}\n"
                f"Различие: {c.get('principal_difference', '')}\n"
                f"Подводный камень: {c.get('pitfall', '')}"
            )
        console.print(
            Panel(
                Markdown("\n\n---\n\n".join(lines)),
                title="Сравнение подходов из источников",
            )
        )

    _bullet_list(
        "Инженерные подводные камни (из авторов)",
        list(cg.get("engineering_pitfalls") or []),
        "yellow",
    )
    _bullet_list(
        "Теория ↔ практика", list(cg.get("theory_practice_bridges") or []), "green"
    )
    _bullet_list("Инварианты", list(cg.get("invariants") or []), "dim")

    nodes = cg.get("nodes") or []
    if nodes:
        table = Table(
            title="Концепты (nodes)", show_header=True, expand=True, show_lines=True
        )
        table.add_column("ID", width=8)
        table.add_column("Label", min_width=14)
        table.add_column("Kind", width=12)
        table.add_column("Detail", min_width=32)
        for n in nodes[:24]:
            if not isinstance(n, dict):
                continue
            table.add_row(
                str(n.get("id", "")),
                str(n.get("label", "")),
                str(n.get("kind", "")),
                str(n.get("detail", "")),
            )
        console.print(table)

    console.print()


def _print_profile_gap_map(gap: Dict[str, Any]) -> None:
    console.print(Rule("[bold]L2b — Условия, допущения и контекст[/bold]"))
    _text_panel("Синтез контекста", gap.get("context_synthesis", ""))
    _bullet_list(
        "Столкновения допущений с задачей", list(gap.get("assumption_clashes") or [])
    )
    _bullet_list(
        "Контекстные флаги (не фильтр)", list(gap.get("context_flags") or []), "dim"
    )

    gaps = gap.get("gaps") or []
    if gaps:
        table = Table(
            title="Gaps / риски", show_header=True, expand=True, show_lines=True
        )
        table.add_column("Область", min_width=12)
        table.add_column("Риск", min_width=20)
        table.add_column("Severity", width=10)
        table.add_column("Mitigation", min_width=16)
        table.add_column("Базис в источниках", min_width=16)
        for g in gaps:
            if not isinstance(g, dict):
                continue
            table.add_row(
                str(g.get("area", "")),
                str(g.get("risk", "")),
                str(g.get("severity", "")),
                str(g.get("mitigation_hint", "")),
                str(g.get("source_basis", "")),
            )
        console.print(table)

    optional = []
    for key, label in (
        ("uma_risks", "UMA (если релевантно)"),
        ("latency_risks", "Latency"),
        ("sla_risks", "SLA"),
        ("stack_incompatibilities", "Стек"),
    ):
        items = list(gap.get(key) or [])
        if items:
            optional.append(f"{label}: " + "; ".join(items))
    if optional:
        _bullet_list("Опциональные контекстные риски", optional, "dim")
    console.print()


def _print_tradeoff_matrix(rows: list[dict]) -> None:
    console.print(Rule("[bold]L2c — Сравнительный архитектурный разбор[/bold]"))
    if not rows:
        console.print("[yellow]tradeoff_matrix пуст[/yellow]")
        return

    for row in rows:
        col = row.get("column", row.get("category", ""))
        title = f"#{row.get('id', '?')} · {col} · {row.get('pattern_name', '')}"
        parts: List[str] = []
        if row.get("fundamental_idea"):
            parts.append(f"### Идея\n{row['fundamental_idea']}")
        if row.get("mechanics_detail"):
            parts.append(f"### Механика\n{row['mechanics_detail']}")
        if row.get("data_structure_notes"):
            parts.append(f"### Структуры данных\n{row['data_structure_notes']}")
        impl = row.get("implementation_details") or []
        if impl:
            parts.append("### Реализация\n" + "\n".join(f"- {x}" for x in impl))
        pros = row.get("pros") or []
        if pros:
            parts.append("### Плюсы\n" + "\n".join(f"- {x}" for x in pros))
        cons = row.get("cons_and_risks") or []
        if cons:
            parts.append("### Минусы и риски\n" + "\n".join(f"- {x}" for x in cons))
        limits = row.get("fundamental_limits") or []
        if limits:
            parts.append(
                "### Фундаментальные ограничения\n"
                + "\n".join(f"- {x}" for x in limits)
            )
        if row.get("applicability"):
            parts.append(f"### Применимость\n{row['applicability']}")
        if row.get("operational_cost"):
            parts.append(f"### Операционная стоимость\n{row['operational_cost']}")
        align = row.get("aligning_sources") or []
        if align:
            parts.append(
                "### Связь с источниками\n" + "\n".join(f"- {x}" for x in align)
            )

        console.print(
            Panel(Markdown("\n\n".join(parts)), title=title, border_style="cyan")
        )
    console.print()


def _print_run_banner(query: str, thread_id: str, log_path: Path | None) -> None:
    pid = os.getpid()
    gv = get_graph_version()
    log_line = str(log_path) if log_path else "(нет файла — init_run_log не вызван)"
    if gv == "0.8":
        orch = "v0.8 Consensus + Light RAG + Gemini Lite/Reasoner (без scholar_fetch)"
        stages = (
            "Consensus Playwright → Gemini Lite validate → fetch papers → "
            "L2a + Gemini Reasoner"
        )
    else:
        orch = f"LangGraph v{gv}, checkpoint MemorySaver"
        stages = (
            "Ollama personal context → Semantic Scholar/arXiv → dedup → "
            "Gemini Lite chunking → Gemini Flash L2a–L2c"
        )
    body = (
        f"[bold]GRAPH_VERSION={gv}[/bold] (из .env; перекрывает export в shell)\n"
        f"[bold]Где запрос:[/bold] этот процесс Python (pid={pid})\n"
        f"[bold]Оркестратор:[/bold] {orch}\n"
        f"[bold]thread_id:[/bold] {thread_id}\n"
        f"[bold]Лог:[/bold] {log_line}\n\n"
        f"[dim]{stages}. Строки ▶/✓ ниже — прогресс и время.[/dim]"
    )
    console.print(
        Panel(body, title="Прогресс выполнения", border_style="yellow"),
    )
    console.print()


def _run_repl(result: Dict[str, Any]) -> None:
    repl_ctx = build_repl_context(result)
    anchor = _global_anchor_from_result(result)
    console.print(Rule("[bold green]REPL — уточняющие вопросы[/bold green]"))
    console.print(
        "[dim]Уточнение по загруженным источникам и графу. "
        "Выход: [bold]/quit[/bold], quit, пустая строка, Ctrl+C[/dim]\n"
    )
    while True:
        try:
            sys.stdout.write("REPL ? ")
            sys.stdout.flush()
            question = _normalize_repl_line(input())
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Выход из REPL (прерывание).[/dim]")
            break
        if _repl_should_exit(question):
            console.print("[dim]Выход из REPL.[/dim]")
            break
        try:
            from knowledge_engine.ui.run_log import trace

            trace(f"REPL ▶ вопрос | {question[:120]}")
            answer = answer_follow_up(question, repl_ctx, anchor)
            trace(f"REPL ✓ ответ ({len(answer)} символов)")
            console.print(Panel(Markdown(answer), title="Ответ", border_style="green"))
            console.print()
        except Exception as exc:
            console.print(f"[red]REPL ошибка: {exc}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный прогон LangGraph v0.7")
    parser.add_argument("query", help="IT/CS вопрос")
    parser.add_argument(
        "--profile",
        type=Path,
        default=_DEFAULT_PROFILE,
        help="Путь к user_profile.md",
    )
    parser.add_argument("--thread-id", default="v07-run")
    parser.add_argument("--json", action="store_true", help="Сырой JSON state")
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Не запускать интерактивный REPL после вывода",
    )
    args = parser.parse_args()

    profile_md = ""
    if args.profile.is_file():
        profile_md = args.profile.read_text(encoding="utf-8")

    from knowledge_engine.ui.logger import (
        live_session,
        print_timing_summary,
        set_status,
    )
    from knowledge_engine.ui.run_log import init_run_log

    log_path = init_run_log(args.query)
    _print_run_banner(args.query[:200], args.thread_id, log_path)

    from knowledge_engine.ui.run_log import trace

    gv = get_graph_version()
    trace(f"CLI ▶ pipeline | GRAPH_VERSION={gv}")
    v08 = gv == "0.8"

    if v08:
        from knowledge_engine.src.agent.local_orchestrator import (
            run_knowledge_engine_v08,
        )

        runner = run_knowledge_engine_v08
    else:
        from knowledge_engine.src.graph import run_knowledge_engine_v07

        runner = run_knowledge_engine_v07

    pipeline_label = "v0.8 consensus" if v08 else f"v{gv} pipeline"

    console.print(
        Panel(
            f"[bold]{pipeline_label}[/bold]\n{args.query[:200]}",
            title="Knowledge Engine",
            border_style="blue",
        )
    )

    with live_session():
        set_status(f"{pipeline_label} — выполняется…")
        result = asyncio.run(runner(args.query, profile_md, args.thread_id))
        print_timing_summary()

    console.print(Rule("[bold]Результат исследования[/bold]"))

    if args.json:
        console.print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not args.no_repl and sys.stdin.isatty():
            _run_repl(result)
        return

    console.print(
        f"[dim]step={result.get('current_step')} "
        f"depth={result.get('search_depth')} "
        f"delta={result.get('density_delta')} "
        f"docs={len(result.get('documents') or [])} "
        f"chunks={len(result.get('structured_chunks') or [])}[/dim]\n"
    )

    final = (result.get("user_final_answer") or "").strip()
    if final:
        _text_panel("Финальный ответ (Gemini Reasoner)", final, border="green")

    spec = result.get("query_spec")
    if spec and isinstance(spec, dict) and spec.get("cs_formal_query"):
        _text_panel("CS formal query", spec["cs_formal_query"], border="dim")

    cg = result.get("concept_graph")
    if isinstance(cg, dict):
        _print_concept_graph(cg)

    gap = result.get("profile_gap_map")
    if isinstance(gap, dict):
        _print_profile_gap_map(gap)

    _print_tradeoff_matrix(list(result.get("tradeoff_matrix") or []))

    if not args.no_repl and sys.stdin.isatty():
        _run_repl(result)


if __name__ == "__main__":
    main()
