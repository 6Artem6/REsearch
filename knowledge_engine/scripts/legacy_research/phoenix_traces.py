"""Phoenix-источник для log_profiler.py: те же отчёты, но по OTel-трейсам.

Читает локальный Arize Phoenix (docs/OBSERVABILITY.md) только GET-запросами
REST API и печатает компактный текст — рассчитан на чтение и человеком, и
моделью (агентом), без UI и без выгрузки сырых промптов:

  - трейсы окна: корневой спан, длительность, число спанов, ошибки, токены;
  - стадии: агрегат по имени спана (count / total / avg / max) и топ долгих;
  - ``--llm-audit``: LLM-спаны по label/модели — токены, латентность, пик
    одновременных вызовов;
  - ``--errors``: спаны со статусом ERROR и текстом исключения;
  - ``--evals``: оценки (span annotations) — распределение label, средний
    score, худшие вердикты и отклонённые на гейтах статьи.

Только stdlib (как и log_profiler.py). Запуск — через log_profiler.py --phoenix.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Служебные спаны LangGraph (запись в каналы, старт графа) — шум в отчёте
# стадий; на трейс и подсчёт ошибок не влияют.
_NOISE_SPAN_PREFIXES = ("ChannelWrite<", "__start__")
_PAGE_LIMIT = 1000
_ANNOTATION_BATCH = 50
_MAX_PAGES = 50


@dataclass
class SpanRow:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str
    start: datetime
    end: datetime
    status: str
    status_message: str
    attributes: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)


class PhoenixClient:
    """Read-only клиент Phoenix REST (проверено на version-20.25.0)."""

    def __init__(self, base_url: str, project: str, *, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.project = project
        self.timeout = timeout

    def _get(self, path: str, params: list[tuple[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Phoenix HTTP {exc.code} on {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Phoenix недоступен ({self.base_url}): {exc.reason}. "
                "Запуск: docker compose --profile observability up -d phoenix"
            ) from exc

    def _paged(self, path: str, params: list[tuple[str, str]]) -> list[dict]:
        rows: list[dict] = []
        cursor = ""
        for _ in range(_MAX_PAGES):
            page_params = list(params) + [("limit", str(_PAGE_LIMIT))]
            if cursor:
                page_params.append(("cursor", cursor))
            payload = self._get(path, page_params)
            rows.extend(payload.get("data") or [])
            cursor = payload.get("next_cursor") or ""
            if not cursor:
                break
        return rows

    def spans(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        trace_ids: list[str] | None = None,
    ) -> list[SpanRow]:
        params: list[tuple[str, str]] = []
        if start is not None:
            params.append(("start_time", start.isoformat()))
        if end is not None:
            params.append(("end_time", end.isoformat()))
        for trace_id in trace_ids or []:
            params.append(("trace_id", trace_id))
        path = f"/v1/projects/{urllib.parse.quote(self.project)}/spans"
        return [parse_span(raw) for raw in self._paged(path, params)]

    def span_annotations(self, span_ids: list[str]) -> list[dict[str, Any]]:
        path = f"/v1/projects/{urllib.parse.quote(self.project)}/span_annotations"
        rows: list[dict[str, Any]] = []
        for i in range(0, len(span_ids), _ANNOTATION_BATCH):
            chunk = span_ids[i : i + _ANNOTATION_BATCH]
            rows.extend(self._paged(path, [("span_ids", sid) for sid in chunk]))
        return rows


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_span(raw: dict[str, Any]) -> SpanRow:
    ctx = raw.get("context") or {}
    return SpanRow(
        span_id=str(ctx.get("span_id") or ""),
        trace_id=str(ctx.get("trace_id") or ""),
        parent_id=raw.get("parent_id") or None,
        name=str(raw.get("name") or ""),
        kind=str(raw.get("span_kind") or ""),
        start=_parse_time(raw["start_time"]),
        end=_parse_time(raw.get("end_time") or raw["start_time"]),
        status=str(raw.get("status_code") or "UNSET"),
        status_message=str(raw.get("status_message") or ""),
        attributes=dict(raw.get("attributes") or {}),
        events=list(raw.get("events") or []),
    )


def _is_noise(span: SpanRow) -> bool:
    return span.name.startswith(_NOISE_SPAN_PREFIXES)


def _tokens(span: SpanRow, kind: str) -> int:
    value = span.attr(f"llm.token_count.{kind}")
    return int(value) if isinstance(value, (int, float)) else 0


def _fmt_time(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def group_traces(spans: list[SpanRow]) -> dict[str, list[SpanRow]]:
    traces: dict[str, list[SpanRow]] = defaultdict(list)
    for span in spans:
        traces[span.trace_id].append(span)
    return traces


def trace_root(spans: list[SpanRow]) -> SpanRow:
    """Корень трейса; если окно его обрезало — самый ранний спан."""
    ids = {s.span_id for s in spans}
    roots = [s for s in spans if not s.parent_id or s.parent_id not in ids]
    return min(roots or spans, key=lambda s: s.start)


def select_recent_traces(spans: list[SpanRow], limit: int) -> list[SpanRow]:
    """Спаны ``limit`` самых свежих трейсов (по началу корня)."""
    traces = group_traces(spans)
    ordered = sorted(traces, key=lambda tid: trace_root(traces[tid]).start)
    keep = set(ordered[-limit:]) if limit > 0 else set(ordered)
    return [s for s in spans if s.trace_id in keep]


def print_trace_summary(spans: list[SpanRow]) -> None:
    traces = group_traces(spans)
    print(f"=== Phoenix: {len(traces)} трейсов, {len(spans)} спанов ===")
    rows = []
    for trace_id, items in traces.items():
        root = trace_root(items)
        start = min(s.start for s in items)
        end = max(s.end for s in items)
        errors = sum(1 for s in items if s.status == "ERROR")
        llm = [s for s in items if s.kind == "LLM"]
        rows.append(
            (
                start,
                f"{trace_id[:16]}  {_fmt_time(start)}  "
                f"{(end - start).total_seconds():8.2f}s  spans={len(items):<4} "
                f"llm={len(llm):<3} tokens={sum(_tokens(s, 'total') for s in llm):<7} "
                f"errors={errors:<2} root={root.name}",
            )
        )
    for _, line in sorted(rows):
        print(line)
    print()


def print_stage_report(spans: list[SpanRow], *, top: int) -> None:
    useful = [s for s in spans if not _is_noise(s)]
    by_name: dict[str, list[SpanRow]] = defaultdict(list)
    for span in useful:
        by_name[span.name].append(span)
    print("=== Стадии (по имени спана; вложенные спаны входят в total родителя) ===")
    print(f"{'span':<44} {'count':>5} {'total':>9} {'avg':>8} {'max':>8} {'err':>4}")
    for name, items in sorted(
        by_name.items(), key=lambda kv: -sum(s.duration for s in kv[1])
    ):
        durations = [s.duration for s in items]
        errors = sum(1 for s in items if s.status == "ERROR")
        print(
            f"{name[:44]:<44} {len(items):>5} {sum(durations):>8.2f}s "
            f"{sum(durations) / len(durations):>7.2f}s {max(durations):>7.2f}s "
            f"{errors:>4}"
        )
    print(f"\n--- Топ-{top} самых долгих спанов ---")
    for span in sorted(useful, key=lambda s: -s.duration)[:top]:
        extra = span.attr("ke.llm.label") or span.attr("ke.job.kind") or ""
        print(
            f"{span.duration:8.2f}s  {_fmt_time(span.start)}  {span.kind:<6} "
            f"{span.name}" + (f"  [{extra}]" if extra else "")
        )
    print()


def llm_peak_concurrency(spans: list[SpanRow]) -> int:
    edges: list[tuple[datetime, int]] = []
    for span in spans:
        edges.append((span.start, 1))
        edges.append((span.end, -1))
    # При равном времени сначала закрытие — соседние вызовы не считаются параллельными.
    peak = current = 0
    for _, delta in sorted(edges, key=lambda e: (e[0], e[1])):
        current += delta
        peak = max(peak, current)
    return peak


def print_llm_audit(spans: list[SpanRow]) -> None:
    llm = [s for s in spans if s.kind == "LLM"]
    print(f"=== LLM-вызовы: {len(llm)} ===")
    if not llm:
        print("(нет LLM-спанов в окне)\n")
        return
    groups: dict[tuple[str, str], list[SpanRow]] = defaultdict(list)
    for span in llm:
        label = str(span.attr("ke.llm.label") or span.name)
        groups[(label, str(span.attr("llm.model_name") or "?"))].append(span)
    print(
        f"{'label':<36} {'model':<26} {'calls':>5} {'prompt':>8} "
        f"{'compl':>7} {'reason':>7} {'avg':>7} {'max':>7} {'err':>4}"
    )
    for (label, model), items in sorted(
        groups.items(), key=lambda kv: -sum(_tokens(s, "total") for s in kv[1])
    ):
        durations = [s.duration for s in items]
        print(
            f"{label[:36]:<36} {model[:26]:<26} {len(items):>5} "
            f"{sum(_tokens(s, 'prompt') for s in items):>8} "
            f"{sum(_tokens(s, 'completion') for s in items):>7} "
            f"{sum(_tokens(s, 'completion_details.reasoning') for s in items):>7} "
            f"{sum(durations) / len(durations):>6.2f}s {max(durations):>6.2f}s "
            f"{sum(1 for s in items if s.status == 'ERROR'):>4}"
        )
    total = sum(_tokens(s, "total") for s in llm)
    print(
        f"\nИтого токенов: {total} (prompt {sum(_tokens(s, 'prompt') for s in llm)}, "
        f"completion {sum(_tokens(s, 'completion') for s in llm)}); "
        f"пик одновременных вызовов: {llm_peak_concurrency(llm)}\n"
    )


def _exception_message(span: SpanRow) -> str:
    for event in span.events:
        if event.get("name") == "exception":
            attrs = event.get("attributes") or {}
            return (
                f"{attrs.get('exception.type', '')}: "
                f"{attrs.get('exception.message', '')}"
            ).strip(": ")
    return span.status_message


def print_error_report(spans: list[SpanRow]) -> None:
    errors = sorted((s for s in spans if s.status == "ERROR"), key=lambda s: s.start)
    print(f"=== Ошибки (status=ERROR): {len(errors)} ===")
    summary = Counter((s.name, _exception_message(s)[:120]) for s in errors)
    for (name, message), count in summary.most_common():
        print(f"{count:>4}×  {name}  |  {message}")
    if errors:
        print("\n--- По порядку ---")
    for span in errors:
        print(
            f"{_fmt_time(span.start)}  trace={span.trace_id[:16]}  {span.name}  |  "
            f"{_exception_message(span)[:300]}"
        )
    print()


def print_eval_report(annotations: list[dict[str, Any]], *, top: int) -> None:
    print(f"=== Оценки (span annotations): {len(annotations)} ===")
    if not annotations:
        print(
            "(нет оценок; нужен KE_OTEL_ANNOTATIONS_ENABLED=true и вызовы "
            "оценщика/гейтов в окне)\n"
        )
        return
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        by_name[str(item.get("name") or "?")].append(item)
    for name, items in sorted(by_name.items()):
        labels = Counter(str((i.get("result") or {}).get("label")) for i in items)
        scores = [
            float(s)
            for s in ((i.get("result") or {}).get("score") for i in items)
            if isinstance(s, (int, float))
        ]
        kinds = Counter(str(i.get("annotator_kind") or "?") for i in items)
        avg = f"{sum(scores) / len(scores):.2f}" if scores else "—"
        print(
            f"{name}: n={len(items)} avg_score={avg} "
            f"kinds={dict(kinds)} labels={dict(labels.most_common())}"
        )
        scored = [
            i
            for i in items
            if isinstance((i.get("result") or {}).get("score"), (int, float))
        ]
        for item in sorted(scored, key=lambda i: i["result"]["score"])[:top]:
            result = item["result"]
            meta = item.get("metadata") or {}
            subject = meta.get("concept_label") or meta.get("target_goal") or ""
            print(
                f"    score={result['score']:.2f} label={result.get('label')} "
                f"span={item.get('span_id')} {subject[:60]}"
            )
            if result.get("explanation"):
                print(f"      {str(result['explanation'])[:240]}")
        rejected = [
            r for i in items for r in (i.get("metadata") or {}).get("rejected") or []
        ]
        if rejected:
            sources = Counter(str(r.get("source") or "?") for r in rejected)
            print(
                f"    отклонено всего: {len(rejected)}; по источникам: "
                f"{dict(sources.most_common(8))}"
            )
            for row in rejected[:top]:
                print(f"      ✗ {str(row.get('title'))[:80]}  <{row.get('url')}>")
    print()


def run_report(
    *,
    base_url: str,
    project: str,
    since: datetime | None,
    until: datetime | None,
    trace_ids: list[str],
    last_traces: int,
    top: int,
    llm_audit: bool,
    errors: bool,
    evals: bool,
) -> int:
    client = PhoenixClient(base_url, project)
    try:
        spans = client.spans(
            start=since.astimezone(timezone.utc) if since else None,
            end=until.astimezone(timezone.utc) if until else None,
            trace_ids=trace_ids,
        )
    except RuntimeError as exc:
        print(exc)
        return 1
    if not trace_ids:
        spans = select_recent_traces(spans, last_traces)
    if not spans:
        print("Phoenix: в окне нет спанов (проверьте KE_OTEL_ENABLED и --since).")
        return 0
    print_trace_summary(spans)
    print_stage_report(spans, top=top)
    if llm_audit:
        print_llm_audit(spans)
    if errors:
        print_error_report(spans)
    if evals:
        try:
            annotations = client.span_annotations([s.span_id for s in spans])
        except RuntimeError as exc:
            print(exc)
            return 1
        print_eval_report(annotations, top=top)
    return 0


__all__ = [
    "PhoenixClient",
    "SpanRow",
    "llm_peak_concurrency",
    "parse_span",
    "print_error_report",
    "print_eval_report",
    "print_llm_audit",
    "print_stage_report",
    "print_trace_summary",
    "run_report",
    "select_recent_traces",
    "trace_root",
]
