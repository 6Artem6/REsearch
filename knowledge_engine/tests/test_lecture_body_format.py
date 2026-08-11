"""Sanitize glued Python fences in lecture_body."""

from __future__ import annotations

from knowledge_engine.services.lecture_body_format import sanitize_lecture_body_markdown


def test_unescapes_literal_backslash_n_in_python_fence():
    body = (
        "Intro\n```python\nimport hmacimport hashlib\ndef foo():\n    return 1\n```\n"
    )
    out = sanitize_lecture_body_markdown(body)
    assert "import hmac\nimport hashlib" in out
    assert "import hmacimport" not in out


def test_preserves_prose_outside_fence():
    body = "Paragraph [R1] citation.\n\n```python\nx = 1\n```"
    out = sanitize_lecture_body_markdown(body)
    assert out.startswith("Paragraph [R1]")
