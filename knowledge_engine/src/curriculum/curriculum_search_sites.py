"""Приоритетные engineering-домены для curriculum search (единый список)."""

from __future__ import annotations

# site: в SearXNG, Gemini web prompt, API Search grounding
CURRICULUM_PRIORITY_ENGINEERING_SITES: tuple[str, ...] = (
    "martinfowler.com",
    "habr.com",
    "bytebytego.com",
    "netflixtechblog.com",
    "blog.cloudflare.com",
    "eng.uber.com",
    "engineering.fb.com",
)


def format_sites_for_prompt(sep: str = ", ") -> str:
    return sep.join(CURRICULUM_PRIORITY_ENGINEERING_SITES)
