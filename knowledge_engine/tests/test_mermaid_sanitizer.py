"""Tests for VLM Mermaid post-processing."""

from __future__ import annotations

from knowledge_engine.utils.mermaid_linter import lint_mermaid_code
from knowledge_engine.utils.mermaid_sanitizer import (
    has_mixed_flowchart_xychart,
    is_mermaid_syntax_valid,
    sanitize_mermaid_code,
    sanitize_mermaid_raw_text,
)

RAW_VLM_BENCHMARK_SCHEMA = """%%{init: {"flowchart": {"htmlLabels": true, "useMaxWidth": false, "padding": 28, "nodeSpacing": 56, "rankSpacing": 64, "curve": "basis"}, "themeVariables": {"fontSize": "14px", "fontFamily": "system-ui, sans-serif"}}}%%
flowchart LR
subgraph Benchmark["Streaming<br/>Performance<br/>Benchmark<br/>(QPS)"] direction TB
subgraph ZillizCloud["ZillizCloud<br/>(8cu-perf)"]
ZC_Static["Static: 3957 QPS"]
ZC_500["Ingestion<br/>500<br/>rows/s:<br/>2119<br/>QPS"]
ZC_1000["Ingestion<br/>1000<br/>rows/s:<br/>1860<br/>QPS"]
end
subgraph Pinecone["Pinecone<br/>(p2.x8-1node)"]
PC_Static["Static: 1131 QPS"]
PC_500["Ingestion<br/>500<br/>rows/s:<br/>367.4<br/>QPS"]
PC_1000["Ingestion<br/>1000<br/>rows/s:<br/>369.7<br/>QPS"]
end
subgraph OpenSearch["OpenSearch (16c128g)"]
OS_Static["Static: 505.7 QPS"]
OS_500["Ingestion<br/>500<br/>rows/s:<br/>161.7<br/>QPS"]
OS_1000["Ingestion<br/>1000<br/>rows/s:<br/>149.7<br/>QPS"]
end
end"""


def test_vlm_benchmark_schema_sanitized_and_linted():
    sanitized = sanitize_mermaid_code(RAW_VLM_BENCHMARK_SCHEMA)
    is_valid, errors = lint_mermaid_code(sanitized)
    assert "direction TB" in sanitized
    assert "subgraph Benchmark" in sanitized
    for line in sanitized.split("\n"):
        if "subgraph" in line.lower() and "direction" in line.lower():
            raise AssertionError(f"subgraph+direction still on one line: {line}")
    assert is_valid, f"Linter failed on sanitized code: {errors}"


def test_escape_lt_after_double_br_in_label():
    raw = "flowchart TD\n    A[<br/><<br/>500 QPS]"
    out = sanitize_mermaid_code(raw)
    assert "<<br/>" not in out
    assert "&lt;" in out
    assert "500 QPS" in out


def test_fix_subgraph_direction_node_on_one_line():
    raw = (
        "flowchart TD\n    subgraph Cluster[Workers] direction TB W1[Task] --> W2[Done]"
    )
    out = sanitize_mermaid_code(raw)
    assert "subgraph Cluster[Workers]" in out
    assert "direction TB" in out
    assert "W1[Task]" in out
    lines = out.split("\n")
    dir_line = next(line for line in lines if "direction TB" in line)
    sub_line = next(line for line in lines if line.strip().startswith("subgraph"))
    assert lines.index(sub_line) < lines.index(dir_line)


def test_dedupe_consecutive_br_tags():
    raw = "flowchart LR\n    N[Title<br/><br/><br/>Body]"
    out = sanitize_mermaid_code(raw)
    assert "<br/><br/>" not in out
    assert "Title<br/>Body" in out


def test_arrows_not_broken():
    raw = "flowchart LR\n    A-->B\n    C->>D: msg"
    out = sanitize_mermaid_code(raw)
    assert "A-->B" in out
    assert "C->>D" in out


def test_sanitize_mermaid_raw_text_duplicate_closers():
    raw = 'flowchart LR\n    Server3[ "MCP Server"]"]"'
    out = sanitize_mermaid_raw_text(raw)
    assert '[ "MCP' not in out
    assert 'Server3["MCP Server"]' in out
    assert '"]"]' not in out


def test_sanitize_mermaid_raw_text_hanging_squote():
    raw = "flowchart TD\n    N[label'\"]"
    out = sanitize_mermaid_raw_text(raw)
    assert "label'\"]" not in out
    assert 'label"]' in out


def test_is_mermaid_syntax_valid_rejects_flowchart_xychart_mix():
    mixed = (
        "flowchart LR\n"
        "    Chart[\"xychart-beta title 'QPS' bar [10, 20]\"]\n"
        "    A --> Chart"
    )
    assert has_mixed_flowchart_xychart(mixed) is True
    assert is_mermaid_syntax_valid(mixed) is False


def test_is_mermaid_syntax_valid_accepts_clean_flowchart():
    clean = 'flowchart TD\n    A["Start"] --> B["End"]'
    assert is_mermaid_syntax_valid(clean) is True
    assert has_mixed_flowchart_xychart(clean) is False


def test_is_mermaid_syntax_valid_rejects_unmatched_quotes():
    broken = 'flowchart TD\n    A["Start --> B["End"]'
    assert is_mermaid_syntax_valid(broken) is False


def test_vlm_generation_rules_present():
    from knowledge_engine.services.vlm_batcher import (
        VLM_BATCH_SYSTEM,
        VLM_MERMAID_GENERATION_RULES,
        VLM_SINGLE_SYSTEM,
    )

    assert "MERMAID GENERATION RULES" in VLM_MERMAID_GENERATION_RULES
    assert "xychart-beta" in VLM_MERMAID_GENERATION_RULES
    assert VLM_MERMAID_GENERATION_RULES in VLM_BATCH_SYSTEM
    assert VLM_MERMAID_GENERATION_RULES in VLM_SINGLE_SYSTEM


def test_gemma_sanitizer_prompt_covers_mix_and_quotes():
    from knowledge_engine.services.mermaid_gemma_repair import (
        MERMAID_GEMMA_SANITIZER_PROMPT,
    )

    assert "Mermaid Syntax Fixer" in MERMAID_GEMMA_SANITIZER_PROMPT
    assert "xychart-beta" in MERMAID_GEMMA_SANITIZER_PROMPT
    assert "unmatched quotes" in MERMAID_GEMMA_SANITIZER_PROMPT.lower() or (
        "Fix unmatched quotes" in MERMAID_GEMMA_SANITIZER_PROMPT
    )
