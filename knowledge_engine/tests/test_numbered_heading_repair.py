"""Numbered ATX headings must stay on one line (deep_analysis sections)."""

from __future__ import annotations

from knowledge_engine.web.linkify import markdown_document_html
from knowledge_engine.web.llm_text_repair import (
    _rejoin_split_numbered_headings,
    repair_llm_display_text,
)


def test_deep_analysis_section_headings_not_split() -> None:
    sections = [
        "## 1. Анатомия / кишки механизма",
        "## 2. Скрытые архитектурные зависимости",
        "## 3. Точки отказа и узкие места",
        "## 4. Матрица trade-offs",
        "## 5. Кодовая трассировка (Python Code Walkthrough)",
    ]
    body = "\n\n".join(f"{h}\n\nПараграф под секцией." for h in sections)
    repaired = repair_llm_display_text(body)
    for h in sections:
        assert h in repaired, f"heading split or lost: {h!r} in {repaired!r}"
    html = markdown_document_html(repaired)
    assert "<h2>3.</h2>" not in html
    assert "Точки отказа и узкие места" in html
    assert "<h2>3. Точки отказа и узкие места</h2>" in html
    assert "<h2>2. Скрытые архитектурные зависимости</h2>" in html
    assert "<h2>5. Кодовая трассировка (Python Code Walkthrough)</h2>" in html


def test_rejoin_already_broken_numbered_heading() -> None:
    broken = "## 3.\n\nТочки отказа и узкие места\n\nДальше текст."
    fixed = _rejoin_split_numbered_headings(broken)
    assert fixed.startswith("## 3. Точки отказа и узкие места")
    repaired = repair_llm_display_text(broken)
    assert "## 3. Точки отказа и узкие места" in repaired
    html = markdown_document_html(repaired)
    assert "<h2>3.</h2>" not in html
    assert "<h2>3. Точки отказа и узкие места</h2>" in html
