"""Coerce BeautifulSoup attribute values to plain strings."""

from __future__ import annotations


def coerce_html_attr(value: object) -> str:
    """Plain str from bs4 Tag attrs (str, list, AttributeValueList, …)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    # list, bs4.element.AttributeValueList, tuple
    if isinstance(value, (list, tuple)):
        parts = [coerce_html_attr(x) for x in value]
        return " ".join(p for p in parts if p).strip()
    mod = type(value).__module__
    if mod == "bs4.element" and type(value).__name__ == "AttributeValueList":
        parts = [coerce_html_attr(x) for x in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()
