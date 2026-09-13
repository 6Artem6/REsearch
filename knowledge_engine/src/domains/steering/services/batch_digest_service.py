"""Batch Digest Generator — Этап 2 Штурвала, шаг 3: сухие поверхностные

дайджесты Gate 1-approved статей ОДНИМ батч-вызовом Flash Lite — без Heavy
LLM и без Map-Reduce. Add-only, план — docs/STEERING_AND_TOPIC_QNA_ROADMAP.md,
контракты Gate 2 — ``src.curriculum.steering_contracts``.
"""

from __future__ import annotations

import asyncio
import json

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.config.settings import (
    LECTURE_PASSAGE_FETCH_CONCURRENCY,
    LECTURE_PASSAGE_MIN_CHARS,
    STEERING_DIGEST_FETCH_TIMEOUT_SEC,
)
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    _lite_structured,
)
from knowledge_engine.src.domains.curriculum.pre_flight_triage import (
    _extract_paragraphs,
)
from knowledge_engine.src.domains.grounding.lecture_passage_fetch import fetch_html
from knowledge_engine.src.domains.steering.services.light_discovery_service import (
    _title_from_html,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    SurfaceDigestResponse,
)

# Бюджет символов на статью в батч-промпте — полный текст в контекст не тянем.
_MAX_CHARS_PER_ARTICLE = 6000

_DIGEST_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a technical digest editor. You receive full texts of several "
    "engineering articles. For EACH article, produce a dry, purely technical "
    "digest for a busy engineer deciding whether to read the full article.\n\n"
    "STRIP all narrative/marketing framing: intro anecdotes, personal "
    'backstory ("how we drank coffee and decided to rewrite the backend"), '
    "jokes, calls to action, and any copywriting filler. Keep ONLY technical "
    "substance: what was built, what problem it solves, what tools/"
    "technologies are used, what the outcome was.\n\n"
    "Return JSON matching SurfaceDigestResponse: exactly one SurfaceDigestItem "
    "per article provided below (skip none, do not invent extra ones):\n"
    "- url: copy verbatim from the provided url.\n"
    "- title: copy verbatim from the provided title, do not invent a new one.\n"
    "- company_or_author: who wrote it, inferred from the text/domain.\n"
    "- problem_solved: the engineering problem this article addresses, one "
    "to two sentences (Russian).\n"
    "- main_tech_stack: technologies/tools explicitly mentioned in the text "
    '(keep original names, e.g. "PostgreSQL", not translated).\n'
    "- two_sentence_summary: dry two-sentence summary of substance/outcome, "
    "no narrative filler (Russian)."
)


def _anchor_digest(urls: list[str]) -> str:
    return "steering_batch_digest:" + ",".join(sorted(urls))[:400]


async def _fetch_article(url: str) -> tuple[str, str] | None:
    """url → (title, полный текст) | None, если fetch/extract не удались.

    Один retry после паузы: некоторые внешние CDN (замечено на
    netflixtechblog.com/Medium) отдают нестабильный 403 именно под
    конкурентной нагрузкой — тот же URL секундами позже (или в одиночном
    запросе) открывается нормально. Раз статья уже approved пользователем
    на Gate 1 (их тут максимум NODE_GATE1_MAX_APPROVED/десяток), одна лишняя
    попытка того стоит — в отличие от лёгкого лекционного fetch_html, где
    такой retry намеренно не добавлен (см. lecture_passage_fetch.py)."""
    html = await fetch_html(url, timeout_sec=STEERING_DIGEST_FETCH_TIMEOUT_SEC)
    if not html:
        await asyncio.sleep(1.5)
        html = await fetch_html(url, timeout_sec=STEERING_DIGEST_FETCH_TIMEOUT_SEC)
    if not html:
        return None
    paragraphs = await asyncio.to_thread(
        _extract_paragraphs, html, url, min_chars=LECTURE_PASSAGE_MIN_CHARS
    )
    if not paragraphs:
        return None
    title = _title_from_html(html, url)
    text = "\n\n".join(paragraphs)[:_MAX_CHARS_PER_ARTICLE]
    return title, text


async def generate_surface_digests(
    approved_urls: list[str],
    *,
    concurrency: int | None = None,
    anchor: str | None = None,
) -> SurfaceDigestResponse:
    """Gate 1 ``approved_urls`` → ``SurfaceDigestResponse`` для Gate 2.

    Полный текст каждой статьи скачивается отдельно (fail-open: URL, которые
    не удалось скачать/распарсить, просто выпадают из батча), но Flash Lite
    вызывается РОВНО ОДИН РАЗ на весь батч утверждённых статей — не по
    одному вызову на URL."""
    urls = [u.strip() for u in approved_urls if (u or "").strip()]
    if not urls:
        return SurfaceDigestResponse()

    conc = max(
        1, concurrency if concurrency is not None else LECTURE_PASSAGE_FETCH_CONCURRENCY
    )
    sem = asyncio.Semaphore(conc)

    async def _bounded(url: str) -> tuple[str, tuple[str, str] | None]:
        async with sem:
            return url, await _fetch_article(url)

    fetched = await asyncio.gather(*[_bounded(u) for u in urls])
    articles: list[dict[str, str]] = []
    for url, result in fetched:
        if result is None:
            continue
        title, text = result
        articles.append({"url": url, "title": title, "text": text})

    trace(
        f"STEERING batch_digest fetch ✓ | approved={len(urls)} "
        f"with_text={len(articles)}"
    )
    if not articles:
        return SurfaceDigestResponse()

    try:
        out = await _lite_structured(
            _DIGEST_SYSTEM,
            json.dumps({"articles": articles}, ensure_ascii=False),
            anchor or _anchor_digest(urls),
            SurfaceDigestResponse,
            "curriculum / steering_batch_digest",
        )
    except Exception as exc:
        trace(f"STEERING batch_digest fallback | skip digest generation | {exc}")
        return SurfaceDigestResponse()

    if isinstance(out, SurfaceDigestResponse):
        return out
    return SurfaceDigestResponse.model_validate(out)


__all__ = ["generate_surface_digests"]
