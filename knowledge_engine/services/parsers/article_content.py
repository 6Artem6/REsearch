"""Извлечение основного HTML-тела статьи (trafilatura, без доменных селекторов)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from knowledge_engine.ui.run_log import trace


def extract_article_body_html(raw_html: str, page_url: str) -> str | None:
    """
    DOM основной статьи: без сайдбаров/футеров по эвристике trafilatura.
    Возвращает HTML-фрагмент или None.
    """
    html = (raw_html or "").strip()
    if not html:
        return None
    url = (page_url or "").strip()
    try:
        import trafilatura

        fragment = trafilatura.extract(
            html,
            url=url or None,
            output_format="html",
            include_images=True,
            include_links=False,
            include_tables=True,
        )
        if fragment and len(fragment.strip()) > 80:
            return fragment.strip()
    except Exception as exc:
        trace(f"ARTICLE_BODY trafilatura ✗ | {exc}")
    return None


def article_content_soup(raw_html: str, page_url: str) -> BeautifulSoup:
    """
    Soup для извлечения img: trafilatura, если в фрагменте есть картинки;
    иначе семантические контейнеры article/main (без CSS-классов сайта).
    """
    html = (raw_html or "").strip()
    full = BeautifulSoup(html, "html.parser")
    fragment = extract_article_body_html(html, page_url)
    frag_soup: BeautifulSoup | None = None
    if fragment:
        frag_soup = BeautifulSoup(fragment, "html.parser")
        if frag_soup.find("img"):
            return frag_soup
    for tag in ("article", "main"):
        node = full.find(tag)
        if node is not None and node.find("img") is not None:
            trace(f"ARTICLE_BODY fallback <{tag}> | images in semantic container")
            return BeautifulSoup(str(node), "html.parser")
    if frag_soup is not None:
        return frag_soup
    return full
