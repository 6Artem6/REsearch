r"""Isolated MAP+REDUCE pipeline benchmark: baseline timing + knowledge_atoms recall.

Runs the REAL production MAP+REDUCE code path — the same
``map_reduce_summarize_blog_outcome_async`` that ``blog_spatial_pipeline.py``
calls — against a single test article. No LanceDB/Qdrant persistence, no HTML
annotation/ingest-gate: isolates chunking + MAP + REDUCE timing (align_sleep,
gemma failover, two_phase dedup, GEMMA HTTP calls) from everything else, to
get a clean baseline for the ~5min→~10min pipeline regression.

Metrics are collected two ways:
- Directly from the returned ``MapReduceJobOutcome`` (atom counts, wall time).
- By re-reading the lines this run appended to the log file (align_sleep sum,
  GEMMA HTTP call count/latency, schema parse failures/retries), same
  technique used in the perf_debug.log drill-down audits.

Usage (repo root, ``.env`` with Gemma key). Prefer ``--log-file`` over setting
LOG_TO_FILE/LOG_FILE_PATH env vars — it configures logging directly inside the
script, so it can't silently fall back to whatever the environment/.env
already points at (e.g. perf_debug.log, shared with a live worker):

  PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.benchmark_pipeline_recall \
    --file /path/to/test_article.txt --log-file perf_benchmark.log

  # raw.githubusercontent.com, not github.com/.../blob/... — the blob HTML
  # viewer page produces garbled/truncated text after HTML-stripping.
  PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.benchmark_pipeline_recall \
    --url https://raw.githubusercontent.com/python/cpython/main/Python/pystate.c \
    --log-file perf_benchmark.log
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.scripts.benchmark_gemma_chunk_sizes import (  # noqa: E402
    _estimate_tokens,
    _load_text,
)

_ALIGN_SLEEP_RE = re.compile(r"align_sleep=([\d.]+)s")
# rate_limiter.py::AsyncRateLimiter.try_acquire — a THIRD, independent
# rate-limit wait (per-model slot RPM/TPM), distinct from _fire_batch's
# align_sleep and from GemmaTokenBudgetManager.acquire_budget. Format:
# "BLOG_SPATIAL gemma slot wait ▶ | model=<name> <sleep_for:.1f>s (max_wait=...".
_SLOT_WAIT_RE = re.compile(r"gemma slot wait ▶ \| model=\S+ ([\d.]+)s")
_GEMMA_HTTP_OK_RE = re.compile(r"GEMMA HTTP ✓ (\S+) \| ([\d.]+)s")
_GEMMA_HTTP_ERR_RE = re.compile(r"GEMMA HTTP ✗ (\S+) \|")
_GEMMA_FAILOVER_RE = re.compile(r"gemma failover")


@dataclass
class LogMetrics:
    align_sleep_total_sec: float = 0.0
    align_sleep_count: int = 0
    slot_wait_total_sec: float = 0.0
    slot_wait_count: int = 0
    gemma_http_ok_count: int = 0
    gemma_http_ok_sec: float = 0.0
    gemma_http_err_count: int = 0
    gemma_failover_count: int = 0
    schema_parse_failures: int = 0
    schema_parse_retries: int = 0
    filtered_lines: list[str] = field(default_factory=list)

    @property
    def rate_limiter_total_sec(self) -> float:
        return self.align_sleep_total_sec + self.slot_wait_total_sec


def _scan_new_log_lines(log_path: Path, *, since_line: int) -> LogMetrics:
    """Parse only the lines this run appended (offset from a pre-run line count)."""
    m = LogMetrics()
    if not log_path.exists():
        return m
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[since_line:]:
        align = _ALIGN_SLEEP_RE.search(line)
        if align:
            m.align_sleep_total_sec += float(align.group(1))
            m.align_sleep_count += 1
            m.filtered_lines.append(line)
        slot_wait = _SLOT_WAIT_RE.search(line)
        if slot_wait:
            m.slot_wait_total_sec += float(slot_wait.group(1))
            m.slot_wait_count += 1
            m.filtered_lines.append(line)
        ok = _GEMMA_HTTP_OK_RE.search(line)
        if ok:
            m.gemma_http_ok_count += 1
            m.gemma_http_ok_sec += float(ok.group(2))
            m.filtered_lines.append(line)
        elif _GEMMA_HTTP_ERR_RE.search(line):
            m.gemma_http_err_count += 1
            m.filtered_lines.append(line)
        elif "GEMMA HTTP ▶" in line:
            m.filtered_lines.append(line)
        if _GEMMA_FAILOVER_RE.search(line):
            m.gemma_failover_count += 1
        if "schema parse retry" in line:
            m.schema_parse_retries += 1
        if "schema parse failed" in line or "Gemma schema parse failed" in line:
            m.schema_parse_failures += 1
    return m


@dataclass
class BenchmarkResult:
    source: str
    title: str
    est_input_tokens: int
    windows: int
    map_windows_ok: int
    map_atoms_raw: int
    final_atoms: int
    atoms_merged_or_dropped: int
    total_wall_sec: float
    key_takeaways: int
    executive_summary_chars: int
    sample_statements: list[str]
    log: LogMetrics


def print_result_table(r: BenchmarkResult) -> None:
    rows = [
        ("Источник", r.source),
        ("Заголовок", r.title[:60]),
        ("Вход, оценка токенов", str(r.est_input_tokens)),
        ("MAP-окон (чанков)", str(r.windows)),
        ("MAP-окон успешно распознано", f"{r.map_windows_ok}/{r.windows}"),
        ("Общее время выполнения (MAP+REDUCE)", f"{r.total_wall_sec:.1f} сек"),
        (
            "align_sleep (batch pacing)",
            f"{r.log.align_sleep_total_sec:.2f} сек ({r.log.align_sleep_count} батчей)",
        ),
        (
            "gemma slot wait (slot rate-limit)",
            f"{r.log.slot_wait_total_sec:.2f} сек ({r.log.slot_wait_count} вызовов)",
        ),
        (
            "Общее время пауз Rate Limiter",
            f"{r.log.rate_limiter_total_sec:.2f} сек",
        ),
        (
            "GEMMA HTTP — успешных вызовов / сумма latency",
            f"{r.log.gemma_http_ok_count} / {r.log.gemma_http_ok_sec:.2f} сек",
        ),
        ("GEMMA HTTP — ошибок (429/5xx/timeout)", str(r.log.gemma_http_err_count)),
        ("Переключений на fallback-модель (gemma failover)", str(r.log.gemma_failover_count)),
        ("Извлечено атомов на фазе MAP (raw, до дедупа)", str(r.map_atoms_raw)),
        ("Финальный объём knowledge_atoms (после REDUCE)", str(r.final_atoms)),
        ("Объединено/отброшено при дедупе (MAP−final)", str(r.atoms_merged_or_dropped)),
        (
            "Recall knowledge_atoms (final/raw)",
            f"{(100.0 * r.final_atoms / r.map_atoms_raw) if r.map_atoms_raw else 0.0:.1f}%",
        ),
        ("key_takeaways в финальном REDUCE", str(r.key_takeaways)),
        ("executive_summary, символов", str(r.executive_summary_chars)),
        ("Ошибки/ретраи Pydantic (schema failures)", str(r.log.schema_parse_failures)),
        ("Ретраи парсинга схемы (Gemma attempt 2/2)", str(r.log.schema_parse_retries)),
    ]
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="MAP+REDUCE pipeline benchmark")
        table.add_column("Метрика", justify="left")
        table.add_column("Значение", justify="right")
        for k, v in rows:
            table.add_row(k, v)
        Console().print(table)
    except Exception:
        print()
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            print(f"| {k.ljust(width)} | {v} |")

    if r.sample_statements:
        print("\nВыборка knowledge_atoms (recall sanity-check, первые 5):")
        for s in r.sample_statements[:5]:
            print(f"  - {s[:160]}")

    if r.log.filtered_lines:
        print(f"\nОтфильтрованные строки лога (align_sleep / GEMMA HTTP), {len(r.log.filtered_lines)} шт:")
        for line in r.log.filtered_lines:
            print(f"  {line}")


async def run_benchmark(
    *,
    text: str,
    source: str,
    title: str,
    log_file_override: str = "",
) -> BenchmarkResult:
    from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
        map_reduce_summarize_blog_outcome_async,
    )
    from knowledge_engine.services.article_ingestion.raw_source import (
        is_code_or_raw_source,
        wrap_raw_source_as_annotated,
    )
    from knowledge_engine.config import LOG_FILE_PATH, LOG_TO_FILE

    annotated = wrap_raw_source_as_annotated(text, source)
    body = (annotated.annotated_markdown or "").strip()
    if len(body) < 80:
        raise SystemExit(f"annotated body too short ({len(body)} chars) — bad input?")
    source_kind = "source_code" if is_code_or_raw_source(source, text) else "article"
    est_tokens = _estimate_tokens(body)

    # --log-file (via configure_logging, see _amain) wins over whatever
    # LOG_FILE_PATH/LOG_TO_FILE happened to resolve to from the environment —
    # config.LOG_FILE_PATH itself is a module-level constant set once at
    # import time and configure_logging() doesn't mutate it.
    log_to_file = bool(log_file_override) or LOG_TO_FILE
    log_path = Path(log_file_override) if log_file_override else Path(LOG_FILE_PATH)
    since_line = 0
    if log_to_file and log_path.exists():
        since_line = len(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
    if not log_to_file:
        print(
            "WARNING: логирование в файл не включено — align_sleep / GEMMA HTTP / "
            "schema-failure метрики будут нулевыми. Запускайте с --log-file perf_benchmark.log"
        )

    t0 = time.perf_counter()
    outcome, windows = await map_reduce_summarize_blog_outcome_async(
        body,
        title=title,
        url=source,
        all_figure_ids=[],
        figure_registry=None,
        source_kind=source_kind,
    )
    # trace() worker is a background queue — give it a moment to drain to file
    # before reading it back, otherwise the tail of this run gets missed.
    await asyncio.sleep(0.5)
    total_wall = time.perf_counter() - t0

    map_results = list(outcome.map_results) if outcome else []
    map_windows_ok = sum(1 for m in map_results if m is not None)
    map_atoms_raw = sum(len(m.knowledge_atoms or []) for m in map_results if m is not None)
    final = outcome.final if outcome else None
    final_atoms = len(final.knowledge_atoms or []) if final else 0
    sample_statements = (
        [a.statement for a in (final.knowledge_atoms or [])] if final else []
    )

    log_metrics = _scan_new_log_lines(log_path, since_line=since_line) if log_to_file else LogMetrics()

    return BenchmarkResult(
        source=source,
        title=title,
        est_input_tokens=est_tokens,
        windows=len(windows),
        map_windows_ok=map_windows_ok,
        map_atoms_raw=map_atoms_raw,
        final_atoms=final_atoms,
        atoms_merged_or_dropped=max(0, map_atoms_raw - final_atoms),
        total_wall_sec=total_wall,
        key_takeaways=len(final.key_takeaways or []) if final else 0,
        executive_summary_chars=len((final.executive_summary or "")) if final else 0,
        sample_statements=sample_statements,
        log=log_metrics,
    )


async def _amain(args: argparse.Namespace) -> int:
    if args.log_file:
        # trace() — which emits almost everything interesting here (MAP/REDUCE,
        # align_sleep, GEMMA HTTP) — dual-writes to file via
        # logging_setup.trace_mirror_logger(), which reads
        # knowledge_engine.config.LOG_TO_FILE/LOG_FILE_PATH directly and
        # caches on first use; configure_logging(file_path=...) does NOT
        # touch those module attributes, so it alone would only redirect the
        # small minority of output that goes through plain `logging` (the
        # PERF DEBUG lines), silently leaving trace() writing to whatever the
        # environment/.env already pointed at. Must set both, and before any
        # trace() call happens (module import order here guarantees that).
        import knowledge_engine.config as _ke_config
        from knowledge_engine.logging_setup import configure_logging

        log_path = Path(args.log_file)
        _ke_config.LOG_TO_FILE = True
        _ke_config.LOG_FILE_PATH = log_path
        configure_logging(level=args.log_level, to_file=True, file_path=log_path, force=True)
        print(f"log_file={log_path.resolve()} log_level={args.log_level}")

    text, source = _load_text(file=args.file, url=args.url, pdf=None)
    title = (
        args.title or (Path(source).stem if not source.startswith("http") else source)
    )[:200]
    print(f"source={source}")
    print(f"chars={len(text)} est_tokens≈{_estimate_tokens(text)}")

    from knowledge_engine.config import GEMMA_API_KEY

    if not (GEMMA_API_KEY or "").strip():
        raise SystemExit("GEMMA_API_KEY / GEMINI_API_KEY empty — set it in .env first")

    result = await run_benchmark(
        text=text, source=source, title=str(title), log_file_override=args.log_file
    )
    print_result_table(result)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Isolated MAP+REDUCE pipeline benchmark (timing + atoms recall)"
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", help="Path to article .txt / .md / .pdf")
    src.add_argument("--url", help="HTTP(S) URL to text/HTML/PDF")
    p.add_argument("--title", default="", help="ARTICLE_TITLE override")
    p.add_argument(
        "--log-file",
        default="",
        help=(
            "Write DEBUG-level logs here (configures logging directly — no need "
            "to set LOG_TO_FILE/LOG_FILE_PATH env vars). Default: off (falls back "
            "to whatever LOG_TO_FILE/LOG_FILE_PATH the environment already has)."
        ),
    )
    p.add_argument("--log-level", default="DEBUG", help="Level for --log-file (default DEBUG)")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
