"""CLI entry point: Typer + Rich, graph run and resume after option selection."""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_engine.config import (
    GRAPH_RECURSION_LIMIT,
    GRAPH_THREAD_ID,
    MIN_VALIDATED_SOURCES,
)
from knowledge_engine.graph.runtime import get_compiled_graph
from knowledge_engine.schemas import EngineState
from knowledge_engine.ui.errors import format_error_with_cause, trace_exception
from knowledge_engine.ui.job_report import pick_option_id, render_report_table
from knowledge_engine.ui.logger import (
    live_session,
    print_timing_summary,
    set_phase,
    set_status,
)
from knowledge_engine.ui.markdown_terminal import unravel_panel
from knowledge_engine.ui.run_log import get_run_log_path, init_run_log, trace

app = typer.Typer(
    name="knowledge-engine",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _apply_gemini_research_mode() -> None:
    """CLI --gemini-research: Gemini + API делают bulk, локальный 1.5B Re-Act."""
    import knowledge_engine.config as cfg

    os.environ["SKIP_GEMINI"] = "false"
    os.environ["REQUIRE_GEMINI"] = "true"
    os.environ["GEMINI_PRIMARY"] = "true"
    os.environ["GEMINI_PRIMARY"] = "true"
    cfg.SKIP_GEMINI = False
    cfg.REQUIRE_GEMINI = True
    cfg.GEMINI_PRIMARY = True
    if "SEARXNG_ENABLED" not in os.environ:
        os.environ["SEARXNG_ENABLED"] = "false"
        cfg.SEARXNG_ENABLED = False


def _run_config() -> dict:
    return {
        "configurable": {"thread_id": GRAPH_THREAD_ID},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


def _clarification_question_from_snapshot(snapshot) -> str | None:
    for task in snapshot.tasks or []:
        for intr in task.interrupts or ():
            val = intr.value
            if isinstance(val, dict):
                return str(val.get("question") or val)
            return str(val)
    return None


def _invoke_with_clarification_loops(
    graph,
    initial: dict,
    config: dict,
) -> dict:
    """HITL: interrupt() в intent_and_clarify + interrupt_before unraveling."""
    from langgraph.types import Command

    current: dict | Command = initial
    result: dict = {}
    while True:
        with live_session():
            set_status("[Graph] SLM router → RAG → search → Gemini heavy…")
            result = graph.invoke(current, config=config)
        snapshot = graph.get_state(config)
        clarify_q = _clarification_question_from_snapshot(snapshot)
        if clarify_q:
            console.print(
                Panel(clarify_q, title="Уточнение (1.5B router)", border_style="yellow")
            )
            answer = typer.prompt("Ответ для ТТЖ")
            current = Command(resume=answer)
            continue
        return result


def _initial_state(problem: str, constraints: str) -> dict:
    from knowledge_engine.graph.initial_state import build_initial_state

    return build_initial_state(problem, constraints)


@app.callback()
def cli() -> None:
    """Локальный движок архитектурного анализа (LangGraph + Ollama)."""


@app.command("test-search")
def test_search(
    query: str = typer.Argument(
        "cache invalidation RAG", help="Инженерная задача / тема"
    ),
    constraints: str = typer.Option(
        "",
        "--constraints",
        "-c",
        help="Ограничения (железо, стек)",
    ),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Один запрос на все провайдеры (без SOTA/Infra/Prod)",
    ),
) -> None:
    """Проверить SearchRegistry: по умолчанию три горизонта с разными запросами."""
    from knowledge_engine.config import OLLAMA_BASE_URL, SEARXNG_BASE_URL
    from knowledge_engine.services.search.horizons import (
        HORIZON_LABELS,
        HORIZON_PROVIDERS,
        SearchHorizon,
        build_horizon_queries,
    )
    from knowledge_engine.services.search.registry import default_registry
    from knowledge_engine.services.search.searxng_health import check_searxng

    ok, msg = check_searxng()
    console.print(f"SearXNG: [{'green' if ok else 'red'}]{msg}[/]")
    if not ok:
        console.print(
            "[dim]Подсказка: docker compose up -d searxng "
            "(из корня REsearch, с knowledge_engine/docker/searxng/settings.yml)[/dim]"
        )

    console.print(
        f"SEARXNG_BASE_URL={SEARXNG_BASE_URL}  OLLAMA_BASE_URL={OLLAMA_BASE_URL}"
    )
    registry = default_registry()

    if flat:
        hits = registry.multi_search_sync(query, limit_per_provider=2)
        if not hits:
            console.print("[red]SearchRegistry: 0 результатов.[/red]")
            raise typer.Exit(code=1)
        table = Table(title=f"[flat] все провайдеры: {query}")
        table.add_column("source")
        table.add_column("title")
        table.add_column("url")
        for h in hits[:12]:
            table.add_row(h.source, (h.title or "")[:50], h.url[:70])
        console.print(table)
        return

    fake_abs = [
        {
            "title": query[:80],
            "cs_concept": query,
            "description": "test-search CLI",
        }
    ]
    horizon_queries = build_horizon_queries(query, constraints, fake_abs)
    hits, _ = registry.multi_search_horizons_sync(
        query,
        constraints,
        fake_abs,
        limit_per_provider=3,
    )

    console.print()
    console.print("[bold]Три горизонта — разные запросы и провайдеры[/bold]")
    for horizon in SearchHorizon:
        q = horizon_queries[horizon]
        providers = ", ".join(HORIZON_PROVIDERS[horizon])
        console.print(
            Panel(
                f"[dim]провайдеры:[/dim] {providers}\n\n[bold]запрос:[/bold]\n{q}",
                title=HORIZON_LABELS[horizon],
                border_style="cyan",
            )
        )

    if not hits:
        console.print("[red]0 URL по всем горизонтам.[/red]")
        raise typer.Exit(code=1)

    for horizon in SearchHorizon:
        bucket = [h for h in hits if h.horizon == horizon.value]
        table = Table(title=f"{horizon.value.upper()} — {len(bucket)} URL")
        table.add_column("source", width=14)
        table.add_column("title", min_width=24)
        table.add_column("url", min_width=30)
        for h in bucket[:8]:
            table.add_row(h.source, (h.title or "")[:55], h.url[:75])
        console.print(table)

    console.print(
        f"[dim]Уникальных URL после дедупа: {len(hits)}. "
        "SOTA = arxiv/scholar; Infra = Bing; Prod = Habr + Bing.[/dim]"
    )


@app.command("browser-login")
def browser_login() -> None:
    """Один раз открыть Gemini в Playwright — сохранить persistent сессию (гость или Google)."""
    from knowledge_engine.config import AI_CHAT_START_URL, PLAYWRIGHT_BROWSER
    from knowledge_engine.services.search.browser_search import (
        persistent_browser,
        wait_for_terminal_enter,
    )
    from knowledge_engine.services.search.playwright_launch import (
        playwright_browser_label,
    )

    console.print(
        f"[yellow]Откроется {playwright_browser_label()} (Playwright). "
        "Дойдите до экрана чата Gemini.[/yellow]"
    )
    console.print(
        "[dim]Логин Google не обязателен, если Gemini даёт чат без аккаунта. "
        "Куки сохраняются в профиле — дальше analyze без browser-login.[/dim]"
    )
    console.print(
        f"[dim]Профиль: knowledge_engine/.browser_state/{PLAYWRIGHT_BROWSER}[/dim]"
    )
    console.print(
        "[bold cyan]Enter нажимайте в этом терминале (Cursor), не в окне Firefox.[/bold cyan]"
    )
    with persistent_browser(headless=False) as (_, context):
        page = context.new_page()
        try:
            page.goto(AI_CHAT_START_URL, wait_until="domcontentloaded", timeout=90000)
        except Exception as exc:
            console.print(
                f"[red]Не открылся Gemini: {format_error_with_cause(exc)}[/red]"
            )
            trace_exception(exc, "browser-login")
            raise typer.Exit(code=1) from exc
        wait_for_terminal_enter("Чат готов → Enter в терминале… ")
        try:
            page.close()
        except Exception:
            pass


@app.command("consensus-login")
def consensus_login() -> None:
    """Один раз войти в Consensus.app (Google/email) — cookies в Playwright profile."""
    from knowledge_engine.config import (
        BROWSER_PROFILE_PATH,
        CONSENSUS_START_URL,
        PLAYWRIGHT_BROWSER,
    )
    from knowledge_engine.services.search.browser_search import (
        persistent_browser,
        wait_for_terminal_enter,
    )
    from knowledge_engine.services.search.playwright_launch import (
        playwright_browser_label,
    )

    console.print(
        f"[yellow]Откроется {playwright_browser_label()} на consensus.app. "
        "Войдите через Google или email.[/yellow]"
    )
    console.print(
        "[dim]Это не browser-login (Gemini). Остановите dev-native/API, если профиль уже "
        "занят открытым Chromium.[/dim]"
    )
    console.print(f"[dim]Профиль: {BROWSER_PROFILE_PATH}[/dim]")
    console.print(
        "[bold cyan]Enter — в этом терминале после успешного входа и поля поиска.[/bold cyan]"
    )
    with persistent_browser(headless=False) as (_, context):
        page = context.new_page()
        try:
            page.goto(CONSENSUS_START_URL, wait_until="domcontentloaded", timeout=90000)
        except Exception as exc:
            console.print(
                f"[red]Не открылся Consensus: {format_error_with_cause(exc)}[/red]"
            )
            trace_exception(exc, "consensus-login")
            raise typer.Exit(code=1) from exc
        wait_for_terminal_enter(
            "Consensus: поиск доступен → Enter в терминале (сохранение cookies)… "
        )
        try:
            page.close()
        except Exception:
            pass
    console.print(
        f"[green]Profile сохранён ({PLAYWRIGHT_BROWSER}). "
        "Запустите API и анализ — CONSENSUS_REUSE_BROWSER_SESSION=true.[/green]"
    )


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Запустить FastAPI (uvicorn): docs на /docs."""
    import os

    os.environ["KE_API_HOST"] = host
    os.environ["KE_API_PORT"] = str(port)
    os.environ["KE_API_RELOAD"] = "true" if reload else "false"
    from knowledge_engine.api.__main__ import main as api_main

    console.print(f"[green]API:[/green] http://{host}:{port}/docs")
    api_main()


@app.command()
def analyze(
    problem: str = typer.Argument(..., help="Инженерная задача для анализа"),
    constraints: str = typer.Option(
        "",
        "--constraints",
        "-c",
        help="Контекст и ограничения (стек, железо, latency и т.д.)",
    ),
    option_id: int | None = typer.Option(
        None,
        "--option-id",
        "-o",
        min=1,
        max=3,
        help="ID варианта матрицы (1–3) без интерактивного prompt",
    ),
    matrix_only: bool = typer.Option(
        False,
        "--matrix-only",
        help="Остановиться после Trade-off матрицы (без unraveling)",
    ),
    gemini_research: bool = typer.Option(
        False,
        "--gemini-research",
        help="Gemini (Playwright) + поисковые API — основная работа; 1.5B Re-Act; без 7B×URL",
    ),
) -> None:
    """Запустить граф анализа, выбрать вариант и получить детальную раскрутку."""
    if gemini_research:
        _apply_gemini_research_mode()
    run_analysis(problem, constraints, option_id=option_id, matrix_only=matrix_only)


def run_analysis(
    problem: str,
    constraints: str,
    headless_browser: bool = True,
    option_id: int | None = None,
    matrix_only: bool = False,
) -> None:
    """Run graph, show matrix, prompt for option id, unravel."""
    graph = get_compiled_graph()
    config = _run_config()
    initial = _initial_state(problem, constraints)
    log_path = init_run_log(problem)
    trace(f"GRAPH ▶ analyze | constraints={constraints or '(нет)'}")

    import knowledge_engine.config as cfg

    version = (cfg.GRAPH_VERSION or "0.3").strip()
    console.print(
        Panel(
            f"[bold]{problem}[/bold]\n\n[dim]{constraints or 'Без дополнительных ограничений'}[/dim]",
            title=f"Knowledge Engine {version}",
            border_style="blue",
        )
    )
    console.print(f"[dim]Trace log (tail -f):[/dim] [cyan]{log_path}[/cyan]")
    console.print(
        "[dim]В панели: [MM:SS] и фаза в заголовке; ▶/✓ NODE и OLLAMA с секундами.[/dim]\n"
    )
    import knowledge_engine.config as cfg

    if version in ("0.3", "0.4"):
        console.print(
            f"[cyan]Путь:[/cyan] v{version} (GRAPH_VERSION={version}, GEMINI_API_KEY)"
        )
    elif not cfg.SKIP_GEMINI:
        console.print(
            "[cyan]Путь:[/cyan] Deep Researcher (Gemini find→extract→1.5B validate→matrix) "
            f"| MIN_VALIDATED={MIN_VALIDATED_SOURCES}"
        )
    else:
        console.print(
            "[yellow]SKIP_GEMINI=true — без диалога Gemini, только multi_search.[/yellow]"
        )
    if not cfg.SEARXNG_ENABLED:
        console.print(
            "[dim]SEARXNG_ENABLED=false — без Bing/Google; arxiv/scholar/habr API.[/dim]"
        )

    try:
        result = _invoke_with_clarification_loops(graph, initial, config)
        trace("GRAPH ✓ matrix ready (interrupt_before unraveling)")
    except Exception as exc:
        detail = trace_exception(exc, "GRAPH")
        console.print(f"[bold red]Ошибка выполнения графа:[/bold red] {detail}")
        raise typer.Exit(code=1) from exc

    state = EngineState.model_validate(result)
    if state.report is None:
        console.print("[red]Матрица не сформирована — проверьте логи Ollama.[/red]")
        print_timing_summary()
        raise typer.Exit(code=1)

    render_report_table(state.report)
    print_timing_summary()

    if matrix_only:
        console.print("[dim]--matrix-only: unraveling пропущен.[/dim]")
        return
    set_phase("human_review — выбор варианта")
    if option_id is not None:
        valid_ids = {o.id for o in state.report.options}
        if option_id not in valid_ids:
            console.print(
                f"[red]--option-id={option_id} недоступен. Варианты: {sorted(valid_ids)}[/red]"
            )
            raise typer.Exit(code=1)
        selected_id = option_id
        console.print(f"[dim]Вариант задан через CLI: id={selected_id}[/dim]")
    else:
        console.print(
            "[dim]Пауза графа: выберите ID варианта для unraveling (1–3).[/dim]"
        )
        selected_id = pick_option_id(state.report)

    try:
        graph.update_state(config, {"selected_option_id": selected_id})
        with live_session():
            set_status(f"[Graph] unraveling варианта {selected_id}…")
            final = graph.invoke(None, config=config)
        trace("GRAPH ✓ unraveling complete")
    except Exception as exc:
        detail = trace_exception(exc, "unraveling")
        console.print(f"[bold red]Ошибка unraveling:[/bold red] {detail}")
        raise typer.Exit(code=1) from exc

    final_state = EngineState.model_validate(final)
    if not final_state.unraveled_details:
        console.print("[red]Пустой результат unraveling.[/red]")
        raise typer.Exit(code=1)

    console.print()
    console.print(
        unravel_panel(
            final_state.unraveled_details,
            f"Unraveling — вариант {selected_id}",
        )
    )
    console.print(f"[dim]Полный trace:[/dim] {get_run_log_path() or log_path}")


def _normalize_argv() -> None:
    """Insert missing `analyze` subcommand for legacy invocations."""
    if len(sys.argv) <= 1:
        return
    first = sys.argv[1]
    if first in {
        "analyze",
        "browser-login",
        "consensus-login",
        "test-search",
        "serve-api",
        "--help",
        "-h",
    }:
        return
    sys.argv.insert(1, "analyze")


def main() -> None:
    _normalize_argv()
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Прервано пользователем.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
