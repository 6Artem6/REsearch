"""Sanitize glued Python fences in lecture_body."""

from __future__ import annotations

import re

from knowledge_engine.services.lecture_body_format import sanitize_lecture_body_markdown
from knowledge_engine.web.linkify import markdown_document_html
from knowledge_engine.web.llm_text_repair import repair_lecture_markdown_layout


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


def test_detach_fence_glued_after_citation():
    body = (
        "пул делится на одноразмерные блоки [R4].```python\n"
        "class PyArena:\n"
        "    def __init__(self):\n"
        "        self.pools = []\n"
        "```\n"
        "Такая организация позволяет избежать фрагментации."
    )
    out = repair_lecture_markdown_layout(body)
    assert re.search(r"^```python\s*$", out, re.M)
    assert "[R4].```python" not in out
    assert "class PyArena:" in out
    html = markdown_document_html(out)
    assert "<pre>" in html
    assert "PyArena" in html
    assert "[R4].```" not in html
