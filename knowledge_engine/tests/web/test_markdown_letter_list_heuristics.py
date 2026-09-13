"""Letter «a)» / «b)» is a list marker only after a real line break."""

from __future__ import annotations

from knowledge_engine.web.linkify import markdown_document_html
from knowledge_engine.web.llm_text_repair import repair_lecture_markdown_layout

_INLINE_MATH = (
    "В динамической типизации CPython любая операция "
    "(например, сложение a + b) требует"
)


def test_inline_parenthetical_plus_b_is_not_a_list() -> None:
    html = markdown_document_html(_INLINE_MATH)
    assert "<ul>" not in html
    assert "<ol>" not in html
    assert "<li>" not in html
    assert "сложение a + b)" in html
    assert "требует" in html


def test_wrapped_parenthetical_plus_b_rejoins_prose() -> None:
    wrapped = (
        "В динамической типизации CPython любая операция "
        "(например, сложение a +\nb) требует"
    )
    repaired = repair_lecture_markdown_layout(wrapped)
    assert "- b)" not in repaired
    assert "сложение a + b) требует" in repaired
    html = markdown_document_html(repaired)
    assert "<li>" not in html
    assert "сложение a + b)" in html


def test_real_letter_subitems_with_newlines_stay_a_list() -> None:
    body = "а) первый пункт\nб) второй пункт"
    repaired = repair_lecture_markdown_layout(body)
    assert "- а) первый пункт" in repaired
    assert "- б) второй пункт" in repaired
    html = markdown_document_html(repaired)
    assert "<li>" in html
    assert "первый пункт" in html
    assert "второй пункт" in html
