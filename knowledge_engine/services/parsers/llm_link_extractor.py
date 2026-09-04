"""LLM-fallback: ссылки на PDF из нестандартной вёрстки."""

from __future__ import annotations

from bs4 import BeautifulSoup

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.parsers.article_manifest import PDFLinkValidationResponse
from knowledge_engine.services.parsers.html_attr import coerce_html_attr
from knowledge_engine.ui.run_log import trace

_LLM_DOM_MAX = 14000
_KEEP_ATTRS = frozenset(
    {"href", "src", "id", "class", "title", "name", "content", "type", "rel"}
)


def strip_html_for_link_llm(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines: list[str] = []
    for meta in soup.find_all("meta"):
        attrs = {
            k: coerce_html_attr(meta.get(k))
            for k in meta.attrs
            if k in _KEEP_ATTRS and coerce_html_attr(meta.get(k))
        }
        if attrs:
            lines.append(
                f"<meta {' '.join(f'{k}=\"{v}\"' for k, v in attrs.items())}/>"
            )
    for tag_name in ("a", "button", "iframe"):
        for el in soup.find_all(tag_name):
            attrs = {
                k: coerce_html_attr(el.get(k))
                for k in el.attrs
                if k in _KEEP_ATTRS and coerce_html_attr(el.get(k))
            }
            text = el.get_text(" ", strip=True)[:120]
            if text:
                attrs["_text"] = text
            if attrs:
                lines.append(
                    f"<{tag_name} {' '.join(
                        f'{k}=\"{v}\"' for k, v in attrs.items()
                    )}/>"
                )
    out = "\n".join(lines)
    return out[:_LLM_DOM_MAX]


class LLMShortlinkResolver:
    """Gemini Lite: best PDF / reader URL из урезанного DOM."""

    _SYSTEM = (
        "You are analyzing an HTML fragment of an academic paper page (meta, links, "
        "iframe, buttons).\n"
        "Find a URL to download the full PDF or open a web article reader.\n"
        "direct_pdf — a direct link to .pdf or a binary PDF endpoint.\n"
        "html_reader — a viewer page (epdf, viewer, reader, ReadCube, etc.).\n"
        "If nothing reliable is found — best_pdf_url=null, confidence=0.\n"
        "Respond strictly per the JSON schema.\n"
        f"{RUSSIAN_OUTPUT_RULE}"
    )
    """
    RU (пояснение): fallback-поиск прямой ссылки на PDF/reader в урезанном
    DOM, когда обычный парсинг ссылок не сработал.
    """

    def resolve(
        self,
        page_url: str,
        html: str,
        *,
        anchor: str = "article_pdf_link",
    ) -> PDFLinkValidationResponse | None:
        fragment = strip_html_for_link_llm(html)
        if len(fragment.strip()) < 40:
            return None
        payload = f"Page URL: {page_url[:500]}\n\n" f"DOM fragment:\n{fragment}"
        try:
            from knowledge_engine.src.analytics.gemini_v07 import (
                run_gemini_lite_structured,
            )

            out = run_gemini_lite_structured(
                self._SYSTEM,
                payload,
                anchor,
                PDFLinkValidationResponse,
                f"resource_discovery / {anchor}",
            )
            if out.best_pdf_url and out.confidence >= 0.35:
                trace(
                    f"RESOURCE_DISCOVERY llm ✓ | conf={out.confidence:.2f} "
                    f"kind={out.kind} | {out.best_pdf_url[:70]}"
                )
                return out
            trace(
                f"RESOURCE_DISCOVERY llm ⊘ | low confidence "
                f"{out.confidence:.2f} | {out.reason[:80]}"
            )
            return None
        except Exception as exc:
            trace(f"RESOURCE_DISCOVERY llm ✗ | {exc}")
            return None
