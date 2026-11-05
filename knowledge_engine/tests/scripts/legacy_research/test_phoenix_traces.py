"""phoenix_traces.py (log_profiler.py --phoenix) без сети: ответы Phoenix REST
подменяются, проверяются разбор спанов, выбор трейсов, LLM-аудит, ошибки
и отчёт по оценкам."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from knowledge_engine.scripts.legacy_research import phoenix_traces as pt

_T0 = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)


def _raw_span(
    span_id: str,
    trace_id: str,
    name: str,
    *,
    start: float,
    end: float,
    parent: str | None = None,
    kind: str = "CHAIN",
    status: str = "UNSET",
    attributes: dict | None = None,
    events: list | None = None,
) -> dict:
    return {
        "name": name,
        "context": {"trace_id": trace_id, "span_id": span_id},
        "span_kind": kind,
        "parent_id": parent,
        "start_time": (_T0 + timedelta(seconds=start)).isoformat(),
        "end_time": (_T0 + timedelta(seconds=end)).isoformat(),
        "status_code": status,
        "status_message": "",
        "attributes": attributes or {},
        "events": events or [],
    }


_SPANS = [
    _raw_span("r1", "t1", "work_job.node_grounding", start=0, end=10),
    _raw_span(
        "l1",
        "t1",
        "gemma.chat_completions",
        start=1,
        end=4,
        parent="r1",
        kind="LLM",
        attributes={
            "ke.llm.label": "map",
            "llm.model_name": "gemma",
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 20,
            "llm.token_count.total": 120,
        },
    ),
    _raw_span(
        "l2",
        "t1",
        "gemma.chat_completions",
        start=2,
        end=5,
        parent="r1",
        kind="LLM",
        attributes={"ke.llm.label": "map", "llm.token_count.total": 30},
    ),
    _raw_span("n1", "t1", "ChannelWrite<...>", start=5, end=5, parent="r1"),
    _raw_span(
        "e1",
        "t1",
        "dedup.pre_map",
        start=6,
        end=7,
        parent="r1",
        status="ERROR",
        events=[
            {
                "name": "exception",
                "attributes": {
                    "exception.type": "TimeoutError",
                    "exception.message": "bge slow",
                },
            }
        ],
    ),
    _raw_span("r2", "t2", "steering.gate1", start=20, end=21),
]

_ANNOTATIONS = [
    {
        "span_id": "r2",
        "name": "steering_gate1",
        "annotator_kind": "HUMAN",
        "result": {"label": "partial", "score": 0.25, "explanation": "approved 1 of 4"},
        "metadata": {
            "target_goal": "RAG",
            "rejected": [
                {"url": "https://x/1", "title": "Noise A", "source": "medium"},
                {"url": "https://x/2", "title": "Noise B", "source": "medium"},
            ],
        },
    }
]


@pytest.fixture
def fake_phoenix(monkeypatch) -> list[tuple[str, list]]:
    calls: list[tuple[str, list]] = []

    def fake_get(self, path, params):
        calls.append((path, params))
        if path.endswith("/spans"):
            return {"data": _SPANS, "next_cursor": None}
        if path.endswith("/span_annotations"):
            return {"data": _ANNOTATIONS, "next_cursor": None}
        raise AssertionError(path)

    monkeypatch.setattr(pt.PhoenixClient, "_get", fake_get)
    return calls


def test_parse_and_select_recent_traces() -> None:
    spans = [pt.parse_span(raw) for raw in _SPANS]
    assert spans[1].duration == 3.0
    assert pt.trace_root([s for s in spans if s.trace_id == "t1"]).name == (
        "work_job.node_grounding"
    )
    recent = pt.select_recent_traces(spans, 1)
    assert {s.trace_id for s in recent} == {"t2"}
    assert len(pt.select_recent_traces(spans, 0)) == len(spans)


def test_llm_peak_concurrency_counts_overlap() -> None:
    llm = [pt.parse_span(raw) for raw in _SPANS if raw["span_kind"] == "LLM"]
    assert pt.llm_peak_concurrency(llm) == 2


def test_full_report(fake_phoenix, capsys) -> None:
    code = pt.run_report(
        base_url="http://phoenix:6006",
        project="knowledge-engine",
        since=_T0,
        until=None,
        trace_ids=[],
        last_traces=0,
        top=5,
        llm_audit=True,
        errors=True,
        evals=True,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2 трейсов" in out
    assert "ChannelWrite" not in out.split("=== Стадии")[1].split("=== LLM")[0]
    assert "Итого токенов: 150" in out
    assert "пик одновременных вызовов: 2" in out
    assert "dedup.pre_map  |  TimeoutError: bge slow" in out
    assert "steering_gate1: n=1 avg_score=0.25" in out
    assert "отклонено всего: 2; по источникам: {'medium': 2}" in out

    spans_path, spans_params = fake_phoenix[0]
    assert spans_path == "/v1/projects/knowledge-engine/spans"
    assert ("start_time", _T0.isoformat()) in spans_params
    _, ann_params = fake_phoenix[1]
    assert ("span_ids", "r2") in ann_params


def test_unreachable_phoenix_returns_error(monkeypatch, capsys) -> None:
    def boom(self, path, params):
        raise RuntimeError("Phoenix недоступен")

    monkeypatch.setattr(pt.PhoenixClient, "_get", boom)
    code = pt.run_report(
        base_url="http://nowhere",
        project="p",
        since=None,
        until=None,
        trace_ids=["t1"],
        last_traces=5,
        top=5,
        llm_audit=False,
        errors=False,
        evals=False,
    )
    assert code == 1
    assert "Phoenix недоступен" in capsys.readouterr().out


def test_log_profiler_phoenix_flag_dispatches(fake_phoenix, monkeypatch, capsys):
    from knowledge_engine.scripts.legacy_research import log_profiler

    monkeypatch.setattr(
        "sys.argv", ["log_profiler.py", "--phoenix", "--trace", "t1", "--evals"]
    )
    assert log_profiler.main() == 0
    _, spans_params = fake_phoenix[0]
    assert ("trace_id", "t1") in spans_params
    assert "=== Оценки" in capsys.readouterr().out
