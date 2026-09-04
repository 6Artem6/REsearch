"""Log Profiler — timing breakdown for a Knowledge Engine node/worker run log.

Parses the log formats actually produced by this codebase:
  1. trace() (knowledge_engine/ui/run_log.py): "HH:MM:SS | LABEL ..."
  2. Python `logging` module output (e.g. bge_m3_embed PERF lines):
     "[YYYY-MM-DD HH:MM:SS] [LEVEL] [logger.name] [file:line:func]: message"
  3. pytest caplog capture: "LEVEL    logger.name:file.py:line message"

Classifies each line into one of 6 pipeline stages by keyword match (see
STAGE_KEYWORDS below), strips noise (httpx/urllib3 internals, repeated
warnings, long body dumps), and reports:
  - total wall time attributed to each stage (self-reported "N.NNs" in the
    message when present — most of this project's own instrumentation
    reports its own elapsed time directly; falls back to a delta-to-next-
    timestamped-line estimate otherwise),
  - event counts and average duration per stage,
  - an explicit concurrency check for GitHub API calls: do different
    candidates' github_tree/github_readme fetch spans overlap in time
    (parallel) or never overlap (sequential/blocking)?

Usage:
    python knowledge_engine/scripts/log_profiler.py <log_file> [--top N]
    python knowledge_engine/scripts/log_profiler.py <log_file> --since "22:51:59" \
        --llm-audit [--top N]
    python knowledge_engine/scripts/log_profiler.py <log_file> --find-run "396.9s"

--since HH:MM:SS (or a full "YYYY-MM-DD HH:MM:SS") slices the log to that
point onward before any analysis; --until bounds the other end the same way
— together they isolate one run inside a long-lived, continuously-appended
log file. Both are ignored (a warning is printed) when --find-run is given.

--find-run TEXT locates one run automatically instead of hand-picking
--since/--until: scans for "WORKER node_deep_dive ✓/✗ init | <label> | N.Ns"
lines containing TEXT (e.g. a duration like "396.9s", or a node label like
"gil_internals" — the LAST/most-recent match wins, so a bare label finds
"the latest run"), then pairs it with the nearest preceding "▶ init | <label>"
line for the SAME label to get the start timestamp. The report then covers
exactly that run's window. Prints which run it found (label, start→end,
duration) before the report.

--llm-audit adds a fine-grained LLM-call registry on top of the stage
summary: a chronological table of every real Gemini/Gemma HTTP call (start/
end timestamp, initiating label, latency, prompt/completion tokens where the
trace line reports them), RPM-spacing/quota/retry/429 events, a concurrency
check (do calls overlap in time or queue up strictly sequentially, and what
was the peak concurrent-call count), and prompt-size stats (min/avg/max
chars+tokens) broken out per known stage (Triage, Step 3a Bulk Gate, Step 3b
Code Dedup).

--errors adds a chronological error/failure report: schema validation
failures (Pydantic `validation_errors=`, with the response's `raw_len` —
useful for correlating parse failures with oversized payloads/output), HTTP
read timeouts (`ReadTimeout`, with the elapsed seconds actually hit),
unhandled Python tracebacks (e.g. a Redis timeout surfacing as an API 500),
and any other `[ERROR]`-level logging-module line not otherwise categorized.
Counts a de-duplicated summary per category plus every occurrence in order.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Шумоподавление: строки, которые всегда выбрасываются перед классификацией.
# ---------------------------------------------------------------------------
_NOISE_LINE_RE = re.compile(
    r"urllib3\.connectionpool|httpx\._client|DEBUG.*Starting new HTTPS? connection"
    r"|charset_normalizer|asyncio - DEBUG|Using selector|filelock|"
    r"huggingface_hub\.file_download|"
    # WORKER redis command <label> — reconnect/failed after reconnect | ... —
    # цикл переподключения worker'а к Redis (см. _safe_redis_command в
    # worker/__main__.py); не относится к тайминг-анализу пайплайна ноды,
    # засоряет отчёт при недоступном/переподключающемся Redis.
    r"WORKER redis (?:command|pubsub) .+ — (?:reconnect|failed after reconnect|retry)",
    re.I,
)
_MAX_LINE_CHARS = 400  # длинные дампы тел функций/JSON обрезаются для отображения

# ---------------------------------------------------------------------------
# Классификация по этапам пайплайна ноды. Порядок важен — первое совпадение
# по ключевому слову побеждает, так что более специфичные метки идут раньше
# более общих (например, "CODE_DEDUP llm" раньше общего "GEMINI").
# ---------------------------------------------------------------------------
STAGE_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "2) GitHub API (Trees + README)",
        ["CODE_DEDUP github_tree", "CODE_DEDUP github_readme"],
    ),
    (
        "4) Tree-sitter AST & Head-Tail slicing",
        [
            "PRE_FLIGHT ast_collapse",
            "CODE_DEDUP ast",
            "CODE_DEDUP bundle",
            "PRE_MAP_DEDUP bulk_gate_code_context",
        ],
    ),
    (
        "5b) Step 3b Code Deduplicator (Code LLM)",
        [
            "PRE_MAP_DEDUP step3b",
            "CODE_DEDUP llm",
            "CODE_DEDUP bundles",
            "code_dedup / bulk_gate",
            "CODE_DEDUP ✓",
            "CODE_DEDUP ✗",
            "CODE_DEDUP skip",
        ],
    ),
    (
        # trailing space отличает от "PRE_MAP_DEDUP bulk_gate_code_context"
        # (та строка про AST, не про сам LLM-вызов Bulk Gate) — см. bucket 4.
        "5a) Step 3a Bulk Gate (Text LLM)",
        [
            "PRE_MAP_DEDUP step3a",
            "PRE_MAP_DEDUP bulk_gate ",
            "pre_map_dedup / bulk_gate",
        ],
    ),
    (
        "1) Ingestion / Triage (Pre-flight)",
        [
            "PRE_FLIGHT",
            "PRE_MAP_DEDUP step1",
            "PRE_MAP_DEDUP triage",
            "PRE_MAP_DEDUP context_extract",
            "DOC_TRIAGE",
            "CURRICULUM pre_map_dedup",
            "paper structure",
            "PAPER_STRUCTURE",
        ],
    ),
    (
        "3) BGE-поиск & README sanitization",
        [
            "PERF bge_m3",
            "CODE_DEDUP readme_anchor",
            "CODE_DEDUP local_context",
            "PRE_MAP_DEDUP mmr_fingerprint",
            "PRE_MAP_DEDUP pool_vector",
            "PRE_MAP_DEDUP cluster",
        ],
    ),
    (
        "6) Graph construction & DB writes (LanceDB)",
        [
            "LanceDB",
            "lancedb",
            "vector_store",
            "upsert_rag_chunks",
            "VECTOR_PDF",
            "doc_id_for_url",
        ],
    ),
    # --- DEEP article MAP/REDUCE pipeline (blog_spatial_summarizer.py +
    # entity_consensus_engine.py) — curriculum node lazy-grounding ingest.
    # Not part of the Pre-MAP Dedup / Code Dedup buckets above (those dedup
    # whole SOURCES before MAP even runs; this is the actual MAP/REDUCE of
    # one node's article batch).
    (
        "7) CURRICULUM search/quota/replenish",
        [
            "CURRICULUM stream pipeline",
            "CURRICULUM stream ingest defer",
            "CURRICULUM quota",
            "CURRICULUM replenish",
            "CURRICULUM targeted search",
            "CURRICULUM targeted policy",
            "CURRICULUM hybrid",
            "[SEARCH] pre_replenish",
        ],
    ),
    (
        "8) BLOG_SPATIAL MAP (Gemma, per-window)",
        [
            "BLOG_SPATIAL map ",
            "BLOG_SPATIAL map-reduce ▶",
            "BLOG_SPATIAL map fact-budget",
            "HTTP ▶ map/",
            "HTTP ✓ map/",
        ],
    ),
    (
        "9) BLOG_SPATIAL REDUCE — entity_consensus dedup (BGE-M3/CE + consensus_batch)",
        [
            "BLOG_SPATIAL reduce ✓ entity_consensus",
            "BLOG_SPATIAL reduce ⚠ entity_consensus",
            "BLOG_SPATIAL reduce ▶ two_phase/dedup",
            "BLOG_SPATIAL reduce ✓ two_phase/dedup",
            "BLOG_SPATIAL reduce ⚠ two_phase/dedup",
            "consensus_batch",
            "CONSENSUS batch",
        ],
    ),
    (
        "10) BLOG_SPATIAL REDUCE — synthesis (Gemma)",
        [
            "BLOG_SPATIAL reduce ▶ two_phase/synthesis",
            "reduce_synth",
            "[REDUCE_START]",
            "[REDUCE_DONE]",
            "BLOG_SPATIAL map-reduce ✓",
        ],
    ),
    (
        "11) NODE_DIVE grounding finalize (registry + diagrams)",
        [
            "NODE_DIVE lazy grounding",
            "NODE_DIVE hydrate",
            "NODE_DIVE concept_map",
            "MERMAID_SPLIT",
        ],
    ),
    (
        "12) Gemma rate-limiter queue wait (not compute)",
        ["BLOG_SPATIAL gemma slot wait", "BLOG_SPATIAL gemma wave reserve"],
    ),
    (
        "13) WORKER run marker (▶/✓ init — total duration, not a stage)",
        ["WORKER node_deep_dive"],
    ),
]
_OTHER_STAGE = "0) Прочее / не классифицировано"

# self-reported elapsed time this project's own trace() calls embed, e.g.
# "... | 1.23s | ..." or "... | 1.23s"
_ELAPSED_RE = re.compile(r"\|\s*(\d+(?:\.\d+)?)s\b")
_MS_ELAPSED_RE = re.compile(r"\|\s*(\d+(?:\.\d+)?)ms\b")

_TIMESTAMP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(\d{2}:\d{2}:\d{2})\s*\|"), "%H:%M:%S"),
    (re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"), "%Y-%m-%d %H:%M:%S"),
]


@dataclass
class LogEvent:
    ts: datetime
    stage: str
    raw: str
    self_elapsed: float | None  # секунды, если строка сама сообщила длительность
    request_key: str | None = None  # для github span-анализа: owner/repo/path


# ---------------------------------------------------------------------------
# --llm-audit: точный реестр реальных LLM-вызовов (Gemini/Gemma), с токенами,
# латентностью и RPM/quota-событиями — для микро-аудита конкретного прогона.
# ---------------------------------------------------------------------------
_HTTP_START_RE = re.compile(
    r"GEMINI HTTP ▶ (.+?) \| model=([\w.\-]+) \| .*?payload≈(\d+) sym"
)
_IO_RE = re.compile(
    r"GEMINI IO \| (.+?) \| input_prompt_len=(\d+) sym \(system=(\d+) user=(\d+)\) "
    r"est_in_tokens≈(\d+) \| output_len=(\d+) est_out_tokens≈(\d+)"
)
_HTTP_END_RE = re.compile(
    r"GEMINI HTTP ✓ (.+?) \| model=([\w.\-]+) \| ([\d.]+)s \| ответ (\d+) sym"
)
_GEMMA_END_RE = re.compile(r"GEMMA HTTP ✓ (.+?) \|.*?([\d.]+)s")
_RPM_SPACING_RE = re.compile(r"GEMINI RPM spacing ([\d.]+)s \| model=([\w.\-]+)")
_QUOTA_EVENT_RE = re.compile(
    r"429|RPM hard_cap|GEMINI wait|GEMINI fallback|GEMINI overload|"
    r"GEMINI timeout|daily quota|GEMINI quota"
)


@dataclass
class LLMCall:
    label: str
    model: str
    provider: str
    start_ts: datetime | None
    end_ts: datetime
    latency: float
    payload_chars: int | None = None
    input_chars: int | None = None
    in_tokens: int | None = None
    output_chars: int | None = None
    out_tokens: int | None = None


def _parse_timestamp(line: str, *, fallback_date: datetime) -> datetime | None:
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        m = pattern.match(line)
        if not m:
            continue
        try:
            parsed = datetime.strptime(m.group(1), fmt)
        except ValueError:
            continue
        if fmt == "%H:%M:%S":
            parsed = parsed.replace(
                year=fallback_date.year,
                month=fallback_date.month,
                day=fallback_date.day,
            )
        return parsed
    return None


def _classify(line: str) -> str:
    for stage, keywords in STAGE_KEYWORDS:
        for kw in keywords:
            if kw in line:
                return stage
    return _OTHER_STAGE


def _self_reported_elapsed(line: str) -> float | None:
    m = _ELAPSED_RE.search(line)
    if m:
        return float(m.group(1))
    m = _MS_ELAPSED_RE.search(line)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _github_request_key(line: str) -> str | None:
    """owner/repo@ref or a readme path — used to pair ▶/✓/✗ spans per request
    for the parallel-vs-sequential check."""
    m = re.search(r"github_(?:tree|readme)\s+[▶✓✗]\s*\|\s*([^\s|]+)", line)
    return m.group(1) if m else None


def _parse_since(value: str, *, fallback_date: datetime) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if fmt == "%H:%M:%S":
            parsed = parsed.replace(
                year=fallback_date.year,
                month=fallback_date.month,
                day=fallback_date.day,
            )
        return parsed
    raise ValueError(f"--since: не удалось разобрать {value!r} (ожидается HH:MM:SS)")


# "WORKER node_deep_dive ✓/✗ init | <curriculum>/<node> | 396.9s"
_RUN_END_RE = re.compile(r"WORKER node_deep_dive [✓✗] init \| (.+?) \| ([\d.]+)s")
# "WORKER node_deep_dive ▶ init | <curriculum>/<node>"
_RUN_START_RE = re.compile(r"WORKER node_deep_dive ▶ init \| (.+)$")


def find_run_window(
    path: str, text: str
) -> tuple[datetime, datetime, str, float] | None:
    """Находит один WORKER node_deep_dive-прогон по подстроке TEXT в его
    завершающей строке ("✓/✗ init | <label> | N.Ns") — например, точной
    длительности ("396.9s") или части label'а ("gil_internals", тогда
    побеждает последнее/самое свежее совпадение — "последний прогон").
    Возвращает (start_ts, end_ts, label, duration_s) для найденного окна, или
    None, если совпадений нет."""
    fallback_date = datetime.now()
    starts: list[tuple[datetime, str]] = []  # (ts, label), в порядке файла
    matches: list[tuple[datetime, str, str, float]] = []  # (end_ts, label, raw, dur)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            ts = _parse_timestamp(line, fallback_date=fallback_date)
            if ts is None:
                continue
            m_start = _RUN_START_RE.search(line)
            if m_start:
                starts.append((ts, m_start.group(1).strip()))
                continue
            m_end = _RUN_END_RE.search(line)
            if m_end and text in line:
                matches.append(
                    (ts, m_end.group(1).strip(), line, float(m_end.group(2)))
                )

    if not matches:
        return None
    end_ts, label, _raw, duration = matches[-1]  # последнее (самое свежее) совпадение
    start_ts = None
    for s_ts, s_label in reversed(starts):
        if s_label == label and s_ts <= end_ts:
            start_ts = s_ts
            break
    if start_ts is None:
        # Не нашли парный ▶ (например, файл начинается посреди прогона) —
        # берём небольшой запас назад от конца, чтобы не потерять весь отчёт.
        start_ts = end_ts - timedelta(seconds=duration + 5)
    return start_ts, end_ts, label, duration


def parse_log(
    path: str, *, since: datetime | None = None, until: datetime | None = None
) -> list[LogEvent]:
    fallback_date = datetime.now()
    events: list[LogEvent] = []
    dedup_seen: dict[str, int] = defaultdict(int)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if _NOISE_LINE_RE.search(line):
                continue
            # схлопываем повторяющиеся однотипные предупреждения
            key = line[:80]
            dedup_seen[key] += 1
            if dedup_seen[key] > 3 and "WARNING" in line:
                continue
            ts = _parse_timestamp(line, fallback_date=fallback_date)
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            display = (
                line
                if len(line) <= _MAX_LINE_CHARS
                else line[:_MAX_LINE_CHARS] + " …[обрезано]"
            )
            events.append(
                LogEvent(
                    ts=ts,
                    stage=_classify(line),
                    raw=display,
                    self_elapsed=_self_reported_elapsed(line),
                    request_key=_github_request_key(line),
                )
            )
    events.sort(key=lambda e: e.ts)
    return events


def parse_llm_calls(
    path: str, *, since: datetime | None = None, until: datetime | None = None
) -> tuple[
    list[LLMCall], list[tuple[datetime, float, str]], list[tuple[datetime, str]]
]:
    """Реестр реальных LLM-вызовов: пары GEMINI HTTP ▶ / GEMINI IO / GEMINI
    HTTP ✓ (плюс GEMMA HTTP ✓, латентность-only) собираются в LLMCall по
    общей label. Возвращает (calls, rpm_waits, quota_events)."""
    fallback_date = datetime.now()
    calls: list[LLMCall] = []
    open_starts: dict[str, tuple[datetime, str, int]] = {}
    io_data: dict[str, dict] = {}
    rpm_waits: list[tuple[datetime, float, str]] = []
    quota_events: list[tuple[datetime, str]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            ts = _parse_timestamp(line, fallback_date=fallback_date)
            if ts is None or (since is not None and ts < since):
                continue
            if until is not None and ts > until:
                continue

            m = _HTTP_START_RE.search(line)
            if m:
                open_starts[m.group(1)] = (ts, m.group(2), int(m.group(3)))
                continue

            m = _IO_RE.search(line)
            if m:
                io_data[m.group(1)] = {
                    "input_chars": int(m.group(2)),
                    "in_tokens": int(m.group(5)),
                    "output_chars": int(m.group(6)),
                    "out_tokens": int(m.group(7)),
                }
                continue

            m = _HTTP_END_RE.search(line)
            if m:
                label, model, latency = m.group(1), m.group(2), float(m.group(3))
                start = open_starts.pop(label, None)
                io = io_data.pop(label, {})
                calls.append(
                    LLMCall(
                        label=label,
                        model=model,
                        provider="gemini",
                        start_ts=start[0] if start else None,
                        end_ts=ts,
                        latency=latency,
                        payload_chars=start[2] if start else None,
                        input_chars=io.get("input_chars"),
                        in_tokens=io.get("in_tokens"),
                        output_chars=io.get("output_chars"),
                        out_tokens=io.get("out_tokens"),
                    )
                )
                continue

            m = _GEMMA_END_RE.search(line)
            if m:
                gemma_latency = float(m.group(2))
                calls.append(
                    LLMCall(
                        label=m.group(1),
                        model="gemma",
                        provider="gemma",
                        # Gemma логирует только завершение (нет "▶" старта, в
                        # отличие от Gemini) — start_ts восстанавливаем из
                        # end_ts - latency, иначе эти вызовы выпадают из
                        # concurrency-статистики (см. llm_concurrency_stats).
                        start_ts=ts - timedelta(seconds=gemma_latency),
                        end_ts=ts,
                        latency=gemma_latency,
                    )
                )
                continue

            m = _RPM_SPACING_RE.search(line)
            if m:
                rpm_waits.append((ts, float(m.group(1)), m.group(2)))
                continue

            if _QUOTA_EVENT_RE.search(line):
                tail = (
                    line.split("]:", 1)[-1] if "]:" in line else line.split("|", 1)[-1]
                )
                quota_events.append((ts, tail.strip()[:160]))

    calls.sort(key=lambda c: c.end_ts)
    return calls, rpm_waits, quota_events


def llm_concurrency_stats(
    calls: list[LLMCall],
) -> tuple[int, int, int, datetime | None]:
    """(known_spans, max_concurrent, overlapping_pairs, peak_time) — только
    для вызовов с известными start_ts И end_ts (GEMMA-только-latency
    вызовы исключены, у них нет start_ts)."""
    intervals = [
        (c.start_ts, c.end_ts, c.label) for c in calls if c.start_ts is not None
    ]
    if not intervals:
        return 0, 0, 0, None
    events: list[tuple[datetime, int]] = []
    for s, e, _lbl in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))
    cur = 0
    peak = 0
    peak_time: datetime | None = None
    for ts, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
            peak_time = ts
    overlaps = 0
    for i in range(len(intervals)):
        s1, e1, _ = intervals[i]
        for j in range(i + 1, len(intervals)):
            s2, e2, _ = intervals[j]
            if s2 < e1 and s1 < e2:
                overlaps += 1
    return len(intervals), peak, overlaps, peak_time


def llm_idle_gaps(
    calls: list[LLMCall], *, min_gap_sec: float = 5.0
) -> tuple[float, float, list[tuple[datetime, datetime, float]]]:
    """(busy_s, window_s, gaps) — busy_s — суммарное время, когда хотя бы один
    LLM-вызов был в полёте (объединение пересекающихся интервалов); gaps —
    промежутки между вызовами длиннее min_gap_sec, где НИ одного LLM-вызова
    не выполнялось (кандидаты на дополнительный parallelism/I-O bottleneck).
    Только вызовы с известным start_ts (Gemma restored via end-latency)."""
    intervals = sorted(
        ((c.start_ts, c.end_ts) for c in calls if c.start_ts is not None),
        key=lambda p: p[0],
    )
    if not intervals:
        return 0.0, 0.0, []
    merged: list[list[datetime]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    busy = sum((e - s).total_seconds() for s, e in merged)
    window = (merged[-1][1] - merged[0][0]).total_seconds()
    gaps = [
        (
            merged[i][1],
            merged[i + 1][0],
            (merged[i + 1][0] - merged[i][1]).total_seconds(),
        )
        for i in range(len(merged) - 1)
        if (merged[i + 1][0] - merged[i][1]).total_seconds() >= min_gap_sec
    ]
    return busy, window, gaps


def print_llm_audit(
    calls: list[LLMCall],
    rpm_waits: list[tuple[datetime, float, str]],
    quota_events: list[tuple[datetime, str]],
) -> None:
    if not calls:
        print(
            "\n--llm-audit: реальных LLM-вызовов (GEMINI HTTP ✓ / GEMMA HTTP ✓) не найдено."
        )
        return

    print("\n" + "=" * 78)
    print(f"LLM CALL REGISTRY — {len(calls)} вызовов")
    print("=" * 78)
    print(
        f"{'#':>3} {'start':>8} {'end':>8} {'lat(s)':>7} {'in_tok':>7} "
        f"{'out_tok':>7} {'model':<20} label"
    )
    print("-" * 100)
    for i, c in enumerate(calls, 1):
        st = c.start_ts.strftime("%H:%M:%S") if c.start_ts else "?"
        en = c.end_ts.strftime("%H:%M:%S")
        in_tok = str(c.in_tokens) if c.in_tokens is not None else "-"
        out_tok = str(c.out_tokens) if c.out_tokens is not None else "-"
        print(
            f"{i:>3} {st:>8} {en:>8} {c.latency:>7.1f} {in_tok:>7} {out_tok:>7} {c.model:<20} {c.label}"
        )

    total_latency = sum(c.latency for c in calls)
    total_in_tok = sum(c.in_tokens or 0 for c in calls)
    total_out_tok = sum(c.out_tokens or 0 for c in calls)
    print(
        f"\nSum(latency) = {total_latency:.1f}s | total in_tokens≈{total_in_tok} "
        f"out_tokens≈{total_out_tok}"
    )

    total_rpm_wait = sum(w[1] for w in rpm_waits)
    print(
        f"RPM-spacing waits: {len(rpm_waits)} событий, суммарно {total_rpm_wait:.1f}s"
    )
    print(f"Quota/retry/429/backoff события: {len(quota_events)}")
    for ts, text in quota_events[:20]:
        print(f"  {ts.strftime('%H:%M:%S')} | {text}")
    if len(quota_events) > 20:
        print(f"  … ещё {len(quota_events) - 20}")

    known, peak, overlaps, peak_time = llm_concurrency_stats(calls)
    total_pairs = known * (known - 1) // 2
    print(f"\nКонкурентность (по {known} вызовам с известными start/end):")
    print(
        f"  Пиковое число одновременных LLM-вызовов = {peak}"
        + (f" (в {peak_time.strftime('%H:%M:%S')})" if peak_time else "")
    )
    if total_pairs:
        print(f"  Пересекающихся пар = {overlaps} из {total_pairs} возможных")
    if peak <= 1:
        print(
            "  Вывод: вызовы НЕ пересекаются — строго последовательный (sequential) конвейер."
        )
    else:
        print(f"  Вывод: до {peak} вызовов одновременно — есть реальный параллелизм.")

    busy, window, gaps = llm_idle_gaps(calls)
    if window > 0:
        util = 100.0 * busy / window
        print(
            f"\nУтилизация LLM-окна: busy={busy:.1f}s / window={window:.1f}s "
            f"({util:.0f}% — хотя бы 1 LLM-вызов в полёте)"
        )
        if gaps:
            print(f"  Простои ≥5s без LLM-вызовов ({len(gaps)}):")
            for g_start, g_end, dur in sorted(gaps, key=lambda g: -g[2])[:10]:
                print(
                    f"    {g_start.strftime('%H:%M:%S')} → {g_end.strftime('%H:%M:%S')} "
                    f"| {dur:.1f}s простой"
                )
            if len(gaps) > 10:
                print(f"    … ещё {len(gaps) - 10}")

    print("\nРазмер payload по стадиям (chars / input tokens):")
    stage_groups = [
        ("Triage (paper structure)", "paper structure"),
        ("Step 3a Bulk Gate (text)", "pre_map_dedup / bulk_gate"),
        ("Step 3b Code Dedup (code)", "code_dedup / bulk_gate"),
    ]
    for tag, keyword in stage_groups:
        sub = [c for c in calls if keyword in c.label]
        if not sub:
            continue
        chars = [c.payload_chars or c.input_chars or 0 for c in sub]
        tokens = [c.in_tokens or 0 for c in sub]
        lat = [c.latency for c in sub]
        print(
            f"  {tag}: n={len(sub)} "
            f"chars(min/avg/max)={min(chars)}/{sum(chars)//len(chars)}/{max(chars)} "
            f"in_tokens(min/avg/max)={min(tokens)}/{sum(tokens)//len(tokens)}/{max(tokens)} "
            f"latency(min/avg/max)={min(lat):.1f}/{sum(lat)/len(lat):.1f}/{max(lat):.1f}s"
        )


def aggregate_by_stage(events: list[LogEvent]) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "self_reported_s": 0.0,
            "self_reported_n": 0,
            "delta_s": 0.0,
            "count": 0,
        }
    )
    for i, ev in enumerate(events):
        s = stats[ev.stage]
        s["count"] += 1
        if ev.self_elapsed is not None:
            s["self_reported_s"] += ev.self_elapsed
            s["self_reported_n"] += 1
        if i + 1 < len(events):
            delta = (events[i + 1].ts - ev.ts).total_seconds()
            if 0 <= delta < 300:  # отбрасываем аномальные скачки (например, полночь)
                s["delta_s"] += delta
    return dict(stats)


def github_concurrency_check(events: list[LogEvent]) -> str:
    """Собирает ▶/✓/✗ спаны github_tree/github_readme по request_key и
    проверяет, пересекаются ли интервалы РАЗНЫХ запросов по времени."""
    intervals: list[tuple[float, float, str]] = []
    open_starts: dict[str, float] = {}
    for ev in events:
        if ev.stage != "2) GitHub API (Trees + README)" or not ev.request_key:
            continue
        epoch = ev.ts.timestamp()
        if "▶" in ev.raw:
            open_starts[ev.request_key] = epoch
        elif ("✓" in ev.raw or "✗" in ev.raw) and ev.request_key in open_starts:
            start = open_starts.pop(ev.request_key)
            intervals.append((start, epoch, ev.request_key))

    if len(intervals) < 2:
        return (
            "GitHub API: недостаточно данных для проверки "
            f"параллельности (найдено спанов: {len(intervals)})."
        )

    intervals.sort()
    overlaps = 0
    for i in range(len(intervals) - 1):
        _s1, e1, k1 = intervals[i]
        s2, _e2, k2 = intervals[i + 1]
        if k1 != k2 and s2 < e1:
            overlaps += 1
    total_pairs = len(intervals) - 1
    if overlaps == 0:
        return (
            f"GitHub API: {len(intervals)} запросов, 0 пересечений по времени "
            f"из {total_pairs} соседних пар — выполняются ПОСЛЕДОВАТЕЛЬНО "
            f"(блокирующе, не asyncio.gather)."
        )
    return (
        f"GitHub API: {len(intervals)} запросов, {overlaps}/{total_pairs} "
        f"соседних пар пересекаются по времени — есть признаки ПАРАЛЛЕЛЬНОГО выполнения."
    )


def print_report(events: list[LogEvent], *, top: int) -> None:
    if not events:
        print("Не найдено ни одной строки с распознанной временной меткой.")
        return

    total_span = (events[-1].ts - events[0].ts).total_seconds()
    print("=" * 78)
    print(
        f"LOG PROFILER — {len(events)} событий, общий охват {total_span:.1f}s "
        f"({events[0].ts.strftime('%H:%M:%S')} → {events[-1].ts.strftime('%H:%M:%S')})"
    )
    print("=" * 78)

    stats = aggregate_by_stage(events)
    rows = sorted(
        stats.items(),
        key=lambda kv: kv[1]["self_reported_s"] or kv[1]["delta_s"],
        reverse=True,
    )
    print(f"\n{'Этап':<45}{'Время (self)':>14}{'Время (delta)':>15}{'Событий':>10}")
    print("-" * 84)
    for stage, s in rows:
        best = s["self_reported_s"] if s["self_reported_n"] > 0 else 0.0
        print(f"{stage:<45}{best:>13.1f}s{s['delta_s']:>14.1f}s{s['count']:>10}")

    print()
    print(github_concurrency_check(events))

    print(f"\nТоп-{top} самых длинных self-reported событий:")
    timed = [e for e in events if e.self_elapsed is not None]
    timed.sort(key=lambda e: e.self_elapsed or 0, reverse=True)
    for ev in timed[:top]:
        print(f"  {ev.self_elapsed:>7.2f}s | {ev.stage:<38} | {ev.raw[:100]}")


@dataclass
class ErrorEvent:
    ts: datetime
    category: str
    detail: str
    raw_len: int | None = None  # response size, when the error line reports one


# Порядок важен — более специфичные категории проверяются раньше общих.
_VALIDATION_ERRORS_RE = re.compile(r"validation_errors=(\[.*?\])\s*(?:raw_len=(\d+))?")
_RAW_LEN_RE = re.compile(r"raw_len=(\d+)")
_SCHEMA_PARSE_FAIL_RE = re.compile(
    r"Gemma schema parse failed|schema parse failed|BLOG_SPATIAL .*parse ✗"
)
_READ_TIMEOUT_RE = re.compile(r"ReadTimeout")
_GEMMA_HTTP_FAIL_RE = re.compile(r"GEMMA HTTP ✗ (.+?) \| ([\d.]+)s")
_API_5XX_RE = re.compile(r"API (5\d\d) \|")
_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):")
_REDIS_TIMEOUT_RE = re.compile(
    r"WORKER redis (?:command|pubsub) (.+?) — (reconnect|failed after reconnect|retry)"
)
_LOG_LEVEL_RE = re.compile(r"^\[[\d\-: ]+\]\s*\[(\w+)\]")


def parse_errors(
    path: str, *, since: datetime | None = None, until: datetime | None = None
) -> list[ErrorEvent]:
    """Сканирует RAW строки (до шумоподавления parse_log — иначе Redis-реконнект
    и другие уже классифицированные проблемы будут не видны) на предмет
    ошибок/сбоев. Строки без собственной метки времени (кадры traceback)
    привязываются к последней увиденной метке — так один traceback считается
    ОДНИМ событием, а не десятком нечитаемых строк стека."""
    fallback_date = datetime.now()
    events: list[ErrorEvent] = []
    last_ts: datetime | None = None
    in_traceback = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            ts = _parse_timestamp(line, fallback_date=fallback_date)
            if ts is not None:
                last_ts = ts
                in_traceback = (
                    False  # новая строка с меткой — предыдущий traceback закрыт
                )
            eff_ts = ts or last_ts
            if eff_ts is None or (since is not None and eff_ts < since):
                continue
            if until is not None and eff_ts > until:
                continue

            m = _VALIDATION_ERRORS_RE.search(line)
            if m:
                raw_len = int(m.group(2)) if m.group(2) else None
                if raw_len is None:
                    m2 = _RAW_LEN_RE.search(line)
                    raw_len = int(m2.group(1)) if m2 else None
                loc_m = re.search(r"'loc':\s*\(([^)]*)\)", line)
                events.append(
                    ErrorEvent(
                        eff_ts,
                        "SCHEMA_VALIDATION_ERROR",
                        f"missing/invalid field {loc_m.group(1) if loc_m else '?'}",
                        raw_len=raw_len,
                    )
                )
                continue

            if _SCHEMA_PARSE_FAIL_RE.search(line):
                m2 = _RAW_LEN_RE.search(line)
                events.append(
                    ErrorEvent(
                        eff_ts,
                        "SCHEMA_PARSE_FAILED",
                        line.strip()[:120],
                        raw_len=int(m2.group(1)) if m2 else None,
                    )
                )
                continue

            m = _GEMMA_HTTP_FAIL_RE.search(line)
            if m:
                events.append(
                    ErrorEvent(
                        eff_ts, "GEMMA_HTTP_FAIL", f"{m.group(1)} | {m.group(2)}s"
                    )
                )
                continue

            if _READ_TIMEOUT_RE.search(line):
                events.append(ErrorEvent(eff_ts, "READ_TIMEOUT", line.strip()[:120]))
                continue

            m = _API_5XX_RE.search(line)
            if m:
                events.append(
                    ErrorEvent(eff_ts, f"API_{m.group(1)}", line.strip()[:160])
                )
                continue

            m = _REDIS_TIMEOUT_RE.search(line)
            if m:
                events.append(
                    ErrorEvent(eff_ts, "REDIS_TIMEOUT", f"{m.group(1)} — {m.group(2)}")
                )
                continue

            if _TRACEBACK_START_RE.match(line):
                if not in_traceback:
                    events.append(
                        ErrorEvent(eff_ts, "UNHANDLED_TRACEBACK", "Traceback…")
                    )
                in_traceback = True
                continue
            if in_traceback:
                continue  # traceback frame — уже посчитан как одно событие выше

            lvl = _LOG_LEVEL_RE.match(line)
            if lvl and lvl.group(1) in ("ERROR", "CRITICAL"):
                events.append(
                    ErrorEvent(eff_ts, f"LOG_{lvl.group(1)}", line.strip()[:160])
                )

    events.sort(key=lambda e: e.ts)
    return events


def print_error_report(events: list[ErrorEvent]) -> None:
    print("\n" + "=" * 78)
    print(f"ERROR & FAILURE REPORT — {len(events)} событий")
    print("=" * 78)
    if not events:
        print("Ошибок/сбоев не найдено.")
        return

    by_cat: dict[str, list[ErrorEvent]] = defaultdict(list)
    for e in events:
        by_cat[e.category].append(e)
    print("\nПо категориям:")
    for cat, evs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cat:<28}{len(evs):>4}")

    schema_fails = [e for e in events if e.raw_len is not None]
    if schema_fails:
        lens = [e.raw_len for e in schema_fails]
        print(
            f"\nРазмер ответа при parse-ошибках (raw_len, символов): "
            f"min={min(lens)} avg={sum(lens) // len(lens)} max={max(lens)} "
            f"(n={len(lens)})"
        )

    print(f"\nХронология (все {len(events)}):")
    for e in events:
        extra = f" | raw_len={e.raw_len}" if e.raw_len is not None else ""
        print(f"  {e.ts.strftime('%H:%M:%S')} | {e.category:<24} | {e.detail}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="Путь к лог-файлу прогона ноды")
    parser.add_argument(
        "--top", type=int, default=15, help="Сколько самых долгих событий показать"
    )
    parser.add_argument(
        "--since",
        default="",
        help='Обрезать лог с этой временной метки (HH:MM:SS или "YYYY-MM-DD HH:MM:SS")',
    )
    parser.add_argument(
        "--until",
        default="",
        help="Обрезать лог по эту временную метку включительно (тот же формат, что --since)",
    )
    parser.add_argument(
        "--find-run",
        default="",
        help=(
            "Найти один прогон по подстроке в его завершающей строке "
            '("396.9s" — точная длительность; "gil_internals" — последний '
            "прогон этой ноды) и ограничить отчёт его окном. Переопределяет "
            "--since/--until."
        ),
    )
    parser.add_argument(
        "--llm-audit",
        action="store_true",
        help="Детальный реестр LLM-вызовов: токены, конкурентность, размер payload по стадиям",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Отчёт по ошибкам/сбоям: schema validation, ReadTimeout, API 5xx, tracebacks, [ERROR]-строки",
    )
    args = parser.parse_args()

    since = (
        _parse_since(args.since, fallback_date=datetime.now()) if args.since else None
    )
    until = (
        _parse_since(args.until, fallback_date=datetime.now()) if args.until else None
    )

    if args.find_run:
        if args.since or args.until:
            print("--find-run задан — --since/--until игнорируются.")
        found = find_run_window(args.log_file, args.find_run)
        if found is None:
            print(f"--find-run: совпадений с {args.find_run!r} не найдено.")
            return 1
        since, until, label, duration = found
        print(
            f"Найден прогон: {label} | {since.strftime('%Y-%m-%d %H:%M:%S')} → "
            f"{until.strftime('%H:%M:%S')} | {duration:.1f}s\n"
        )
        until = until + timedelta(seconds=1)  # включить саму завершающую строку

    events = parse_log(args.log_file, since=since, until=until)
    print_report(events, top=args.top)

    if args.llm_audit:
        calls, rpm_waits, quota_events = parse_llm_calls(
            args.log_file, since=since, until=until
        )
        print_llm_audit(calls, rpm_waits, quota_events)

    if args.errors:
        error_events = parse_errors(args.log_file, since=since, until=until)
        print_error_report(error_events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
