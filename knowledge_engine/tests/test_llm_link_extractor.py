"""strip_html_for_link_llm must tolerate bs4 AttributeValueList on class attrs."""

from knowledge_engine.services.parsers.llm_link_extractor import strip_html_for_link_llm


def test_strip_html_for_link_llm_with_class_attribute():
    html = (
        "<html><body>"
        '<a class="btn pdf-download" href="/files/paper.pdf">PDF</a>'
        "</body></html>"
    )
    out = strip_html_for_link_llm(html)
    assert "paper.pdf" in out
    assert "pdf-download" in out
