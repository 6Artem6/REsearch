"""Извлечение и описание диаграмм на страницах."""

from __future__ import annotations

import base64
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import urljoin

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_engine.config import MAIN_MODEL
from knowledge_engine.llm import stream_chat
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.search.browser_search import fetch_page_html, human_delay
from knowledge_engine.ui.errors import format_error_location
from knowledge_engine.ui.logger import set_status

VISION_PROMPT = (
    "Describe this architecture diagram/chart: key nodes, topology, data "
    "flows, and bottlenecks. Structure the answer.\n"
    f"{RUSSIAN_OUTPUT_RULE}"
)
"""
RU (пояснение): Vision-промпт для описания схем/графиков со страниц
(legacy analyze CLI) — структурированный технический разбор диаграммы.
"""


class _ImgCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr = {k: v for k, v in attrs if k and v}
        src = attr.get("src", "")
        alt = attr.get("alt", "")
        if src:
            self.images.append((src, alt))


def _resolve_url(base: str, src: str) -> str:
    return urljoin(base, src)


def _is_diagram_candidate(src: str, alt: str) -> bool:
    blob = f"{src} {alt}".lower()
    keywords = ("diagram", "architecture", "schema", "graph", "flow", "схема", "архит")
    return any(k in blob for k in keywords) or bool(alt.strip())


def extract_image_urls(
    html: str, page_url: str, max_images: int = 5
) -> List[tuple[str, str]]:
    parser = _ImgCollector()
    parser.feed(html)
    picked: list[tuple[str, str]] = []
    for src, alt in parser.images:
        if not _is_diagram_candidate(src, alt):
            continue
        full = _resolve_url(page_url, src)
        if full.startswith("http"):
            picked.append((full, alt))
        if len(picked) >= max_images:
            break
    return picked


def _fetch_image_bytes(url: str) -> bytes:
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def describe_diagram_from_url(image_url: str, alt: str = "") -> str:
    set_status(f"[Vision] анализ схемы: {image_url[:80]}…")
    human_delay(0.5, 1.0)
    try:
        raw = _fetch_image_bytes(image_url)
        b64_len = len(base64.b64encode(raw))
    except httpx.HTTPError:
        b64_len = 0

    system = SystemMessage(content=VISION_PROMPT)
    human = HumanMessage(
        content=(
            f"URL изображения: {image_url}\n"
            f"Alt-текст: {alt or '(нет)'}\n"
            f"Размер данных: {b64_len} base64-символов.\n"
            "Дай техническое описание схемы."
        )
    )
    return stream_chat(MAIN_MODEL, [system, human], temperature=0.2, label="vision")


def analyze_page_diagrams(page_url: str, html: Optional[str] = None) -> List[str]:
    if html is None:
        html = fetch_page_html(page_url)
    descriptions: list[str] = []
    for img_url, alt in extract_image_urls(html, page_url):
        try:
            descriptions.append(describe_diagram_from_url(img_url, alt))
        except Exception as exc:
            descriptions.append(
                f"(не удалось описать {img_url}: {format_error_location(exc)})"
            )
    return descriptions
