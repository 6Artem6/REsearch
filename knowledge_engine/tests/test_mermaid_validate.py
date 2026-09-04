"""AST Mermaid linter + validate/Gemma error report integration."""

from __future__ import annotations

import re

from knowledge_engine.services.mermaid_validate import (
    format_mermaid_lint_report,
    lint_mermaid_ast,
    sanitize_mermaid_syntax,
    split_spliced_mermaid_lines,
    validate_mermaid_syntax,
)


SPLICED_END_DIAGRAM = """flowchart TD
subgraph C["Task"]
Prompt["Writing"]
end A --> B B --> C
"""

CLEAN_DIAGRAM = """flowchart TD
subgraph C["Task"]
Prompt["Writing"]
end
A --> B
B --> C
"""

PERSPECTIVEGAP_LIKE = """flowchart TD
subgraph A["Scenario"]
f1["f1: problem definition"]
Dispatcher["Dispatcher"]
end
subgraph B["Role-Fragment Assignment Task"]
Task["JSON mapping roles to fragments"]
end
subgraph C["Free-Form Prompt Writing Task"]
Prompt["Role system prompts generation"]
end A --> B B --> C
"""


def test_lint_mermaid_ast_spliced_end_and_unclosed_subgraph():
    ok, errors = lint_mermaid_ast(SPLICED_END_DIAGRAM)
    assert ok is False
    joined = "\n".join(errors)
    assert any(
        "Line 4:" in e and "merged with connections/nodes" in e for e in errors
    ), errors
    assert "end A --> B" in joined or "end A --> B B --> C" in joined
    assert any(
        "Unclosed 'subgraph'" in e and "Line 2:" in e for e in errors
    ), errors


def test_lint_mermaid_ast_perspectivegap_diagram1_style():
    ok, errors = lint_mermaid_ast(PERSPECTIVEGAP_LIKE)
    assert ok is False
    assert any("merged with connections/nodes" in e for e in errors), errors
    assert any("Unclosed 'subgraph'" in e and "('C')" in e for e in errors), errors


def test_lint_mermaid_ast_accepts_clean_subgraph():
    ok, errors = lint_mermaid_ast(CLEAN_DIAGRAM)
    assert ok is True, errors
    assert errors == []


def test_lint_mermaid_ast_subgraph_direction_same_line():
    code = 'flowchart TD\nsubgraph Cluster["Workers"] direction TB\nW1["A"]\nend'
    ok, errors = lint_mermaid_ast(code)
    assert ok is False
    assert any(
        "subgraph" in e and "direction" in e and "separate lines" in e for e in errors
    ), errors


def test_lint_mermaid_ast_unquoted_html_label():
    code = "flowchart TD\nA[Title<br/>Body] --> B[\"Ok\"]"
    ok, errors = lint_mermaid_ast(code)
    assert ok is False
    assert any("double quotes" in e for e in errors), errors


def test_validate_mermaid_syntax_rejects_spliced_end():
    assert validate_mermaid_syntax(SPLICED_END_DIAGRAM) is False
    assert validate_mermaid_syntax(CLEAN_DIAGRAM) is True


def test_lint_mermaid_ast_spliced_statements_like_diagram4():
    """multi_agent_orchestration diagram-4 style: edges glued after node defs."""
    code = """flowchart TD
subgraph cr["cr: Coder-Reviewer"]
coder_cr["coder"]
reviewer_cr["reviewer"]
coder_cr -->|feedback| reviewer_cr reviewer_cr -->|feedback| coder_cr
end
subgraph dcr["dcr"]
dispatcher_dcr["dispatcher"]
coder_dcr["coder"]
reviewer_dcr["reviewer"] dispatcher_dcr -.-> coder_dcr
end
"""
    ok, errors = lint_mermaid_ast(code)
    assert ok is False
    joined = "\n".join(errors)
    assert "multiple Mermaid statements on one line" in joined
    assert any("Line 5:" in e for e in errors), errors
    assert any("Line 10:" in e for e in errors), errors


def test_lint_mermaid_ast_spliced_statements_like_diagram6():
    code = """flowchart TD
subgraph CR["cr"]
coder_cr["coder"]
reviewer_cr["reviewer"] coder_cr <--> reviewer_cr
end
"""
    ok, errors = lint_mermaid_ast(code)
    assert ok is False
    assert any("multiple Mermaid statements on one line" in e for e in errors), errors
    assert validate_mermaid_syntax(code) is False


def test_lint_mermaid_ast_allows_edge_chain():
    code = 'flowchart TD\nA["a"] --> B["b"] --> C["c"]\n'
    ok, errors = lint_mermaid_ast(code)
    assert ok is True, errors


def test_split_spliced_mermaid_lines_diagram4_style():
    raw = """flowchart TD
subgraph cr["cr: Coder-Reviewer"]
coder_cr["coder"]
reviewer_cr["reviewer"]
coder_cr -->|feedback| reviewer_cr reviewer_cr -->|feedback| coder_cr
end
subgraph dcr["dcr"]
dispatcher_dcr["dispatcher"]
coder_dcr["coder"]
reviewer_dcr["reviewer"] dispatcher_dcr -.->|dispatch| coder_dcr dispatcher_dcr -.-> coder_dcr
end
"""
    fixed = split_spliced_mermaid_lines(raw)
    ok, errors = lint_mermaid_ast(fixed)
    assert ok is True, errors
    assert "reviewer_cr reviewer_cr" not in fixed
    assert validate_mermaid_syntax(fixed) is True


def test_split_keeps_subgraph_header_on_one_line():
    raw = 'flowchart TD\nsubgraph CR["cr"]\ncoder_cr["coder"]\nend\n'
    fixed = split_spliced_mermaid_lines(raw)
    assert 'subgraph CR["cr"]' in fixed
    assert re.search(r"^subgraph\s*$", fixed, re.M) is None
    ok, errors = lint_mermaid_ast(fixed)
    assert ok is True, errors


def test_split_subgraph_then_node_splits_after_header():
    raw = 'flowchart TD\nsubgraph CR["cr"] coder_cr["coder"]\nend\n'
    fixed = split_spliced_mermaid_lines(raw)
    assert 'subgraph CR["cr"]' in fixed.split("\n")
    assert 'coder_cr["coder"]' in fixed
    assert 'subgraph CR["cr"] coder_cr' not in fixed
    ok, errors = lint_mermaid_ast(fixed)
    assert ok is True, errors


def test_split_rejoins_broken_bare_subgraph_header():
    broken = 'flowchart TD\nsubgraph\nCR["cr"]\ncoder_cr["coder"]\nend\n'
    fixed = split_spliced_mermaid_lines(broken)
    assert 'subgraph CR["cr"]' in fixed
    assert validate_mermaid_syntax(fixed) is True


def test_lint_bare_subgraph_missing_id():
    code = 'flowchart TD\nsubgraph\nCR["cr"]\nend\n'
    ok, errors = lint_mermaid_ast(code)
    assert ok is False
    assert any("missing its ID/Title" in e for e in errors), errors


def test_sanitize_mermaid_syntax_fixes_diagram4_and_6_without_gemma():
    d4 = """```mermaid
flowchart TD
subgraph cr["cr: Coder-Reviewer"]
coder_cr["coder"]
reviewer_cr["reviewer"]
coder_cr -->|feedback| reviewer_cr reviewer_cr -->|feedback| coder_cr
end
subgraph dcr["dcr: Dispatcher-Coder-Reviewer"]
dispatcher_dcr["dispatcher"]
coder_dcr["coder"]
reviewer_dcr["reviewer"] dispatcher_dcr -.->|dispatch| coder_dcr dispatcher_dcr -.->|dispatch| reviewer_dcr coder_dcr --> reviewer_dcr reviewer_dcr --> coder_dcr
end
subgraph dtc["dtc: Dispatcher-Theory-Coder"]
dispatcher_dtc["dispatcher"]
theorist_dtc["theorist"]
theory_rev_dtc["theory reviewer"]
coder_dtc["coder"]
reviewer_dtc["reviewer"] dispatcher_dtc -.-> theorist_dtc dispatcher_dtc -.-> theory_rev_dtc dispatcher_dtc -.-> coder_dtc dispatcher_dtc -.-> reviewer_dtc theorist_dtc --> theory_rev_dtc theory_rev_dtc --> theorist_dtc coder_dtc --> reviewer_dtc reviewer_dtc --> coder_dtc
end
subgraph spl["spl: Supervisor-PhD-Librarian"]
end
```"""
    d6 = """```mermaid
flowchart TD
subgraph CR["cr"]
coder_cr["coder"]
reviewer_cr["reviewer"] coder_cr <--> reviewer_cr
end
subgraph DCR["dcr"]
dispatcher_dcr["dispatcher"]
coder_dcr["coder"]
reviewer_dcr["reviewer"] dispatcher_dcr -.-> coder_dcr dispatcher_dcr -.-> reviewer_dcr coder_dcr <--> reviewer_dcr
end
```"""
    for raw in (d4, d6):
        out = sanitize_mermaid_syntax(raw)
        assert validate_mermaid_syntax(out) is True, lint_mermaid_ast(out)


def test_format_mermaid_lint_report_for_gemma():
    _ok, errors = lint_mermaid_ast(SPLICED_END_DIAGRAM)
    report = format_mermaid_lint_report(errors)
    assert report.startswith("Validation/Lint Errors Found:")
    assert "- Line " in report
