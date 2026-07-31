"""Практические блоги через веб-чат Gemini (Playwright, persistent profile)."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_MAX_URLS,
    CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC,
    CURRICULUM_GEMINI_WEB_RESPONSE_FIRST_TIMEOUT_SEC,
    CURRICULUM_GEMINI_WEB_RESPONSE_MAX_SEC,
    CURRICULUM_GEMINI_WEB_URL_RETRY_MAX,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
)
from knowledge_engine.services.ai_dialogue.gemini_session import BrowserGeminiDialogueSession
from knowledge_engine.services.curriculum_whitelist_prompt import curriculum_whitelist_prompt_block
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
)
from knowledge_engine.src.curriculum.curriculum_search_sites import format_sites_for_prompt
from knowledge_engine.src.curriculum.url_validate import validate_and_filter_urls
from knowledge_engine.ui.run_log import trace

_MD_LINK_RE = re.compile(r"\[([^\]]{2,200})\]\((https?://[^\)]+)\)")
_BULLET_LINE_RE = re.compile(r"^[\s]*(?:[-*•]|\d+[.)])\s+(.+)$", re.M)


def _normalize_url(url: str) -> str:
    u = (url or "").strip().rstrip(".,);]")
    if not u.startswith("http"):
        return ""
    return u.split("#")[0].rstrip("/")


def _title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path:
        slug = path.split("/")[-1].replace("-", " ").replace("_", " ")
        if len(slug) > 3:
            return slug[:400]
    return url[:400]


def _snippet_for_url(text: str, url: str) -> str:
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if url in line:
            chunk = line.strip()
            if len(chunk) > 60:
                return chunk[:1200]
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if len(nxt) > 40:
                    return nxt[:1200]
    return ""


def _extract_url_title_pairs(text: str) -> list[tuple[str, str, str]]:
    """(url, title, snippet_hint)"""
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []

    for m in _MD_LINK_RE.finditer(text or ""):
        title = (m.group(1) or "").strip()
        url = _normalize_url(m.group(2))
        if not url or url in seen:
            continue
        seen.add(url)
        out.append((url, title or _title_from_url(url), _snippet_for_url(text, url)))

    url_re = re.compile(r"https?://[^\s\]<\"')]+")
    for raw in url_re.findall(text or ""):
        url = _normalize_url(raw)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append((url, _title_from_url(url), _snippet_for_url(text, url)))

    return out


def _build_harvest_prompt(target_goal: str, context_vector: str) -> str:
    sites = format_sites_for_prompt()
    whitelist_block = curriculum_whitelist_prompt_block()
    goal = (target_goal or "").strip()
    ctx = (context_vector or "").strip()
    body = (
        f"{whitelist_block}\n\n"
        "Find 4–8 authoritative engineering articles and practical deep-dives for this goal.\n"
        f"Prioritize known engineering blogs and official docs; also {sites}.\n"
        "Prefer deep technical articles; static whitelist is a hint, not a hard limit.\n"
        "Avoid generic intros, SEO listicles, and beginner-only tutorials.\n"
        "Use web search in Gemini if available.\n\n"
        f"Goal / query:\n{goal}\n"
    )
    if ctx and ctx != goal:
        body += f"\nExpansion / context vector:\n{ctx[:2000]}\n"
    body += (
        "\nOutput format:\n"
        "- Markdown list with [Article title](https://full-url) for each source.\n"
        "- Under each link, one short bullet: why it matters (2–3 sentences).\n"
        "- Only real article URLs, not homepages.\n"
        "- English or Russian titles OK."
    )
    return body


def _build_url_retry_prompt(broken_url: str, article_title: str, goal: str) -> str:
    title = (article_title or "").strip() or _title_from_url(broken_url)
    ctx = (goal or "").strip()[:500]
    return (
        f"Ссылка '{broken_url}' для статьи '{title}' не существует (404). "
        "Выполни повторный Google Search и найди ТОЧНУЮ, реально существующую рабочую ссылку "
        "на эту статью. Если точной ссылки нет — найди аналогичную рабочую статью по этой теме "
        "на habr.com, martinfowler.com или bytebytego.com.\n"
        f"{f'Контекст цели: {ctx}\n' if ctx else ''}\n"
        "Формат: markdown [Article title](https://full-url)."
    )


def _hits_from_answer(answer: str, cap: int) -> list[CurriculumSearchHit]:
    pairs = _extract_url_title_pairs(answer)
    by_url: dict[str, tuple[str, str, str]] = {}
    for url, title, snippet in pairs:
        by_url[url.lower()] = (url, title, snippet)

    hits: list[CurriculumSearchHit] = []
    for _key, (url, title, snippet) in by_url.items():
        if len(hits) >= cap:
            break
        if not is_collectible_article_url(url):
            continue
        bullets = _BULLET_LINE_RE.findall(answer)
        if not snippet and bullets:
            snippet = bullets[min(len(hits), len(bullets) - 1)][:1200]
        hits.append(
            CurriculumSearchHit(
                url=url,
                title=title[:400],
                snippet=(snippet or title)[:1200],
                source_tier="gemini_web",
            )
        )
    return hits


def _harvest_in_session(prompt: str, cap: int, goal: str) -> list[CurriculumSearchHit]:
    session = BrowserGeminiDialogueSession(
        response_max_sec=CURRICULUM_GEMINI_WEB_RESPONSE_MAX_SEC,
        response_first_timeout_sec=CURRICULUM_GEMINI_WEB_RESPONSE_FIRST_TIMEOUT_SEC,
        min_response_chars=80,
    )
    validate_kw = {"timeout": CURRICULUM_URL_VALIDATE_TIMEOUT_SEC}
    try:
        answer = session.send(prompt)
        hits = _hits_from_answer(answer, cap)
        trace(
            f"CURRICULUM gemini_web ✓ | extracted={len(hits)} "
            f"answer_chars={len(answer)}"
        )

        valid, broken = validate_and_filter_urls(hits, **validate_kw)
        by_url: dict[str, CurriculumSearchHit] = {h.url.lower(): h for h in valid}
        retries = 0
        max_retry = max(0, CURRICULUM_GEMINI_WEB_URL_RETRY_MAX)

        for broken_hit in broken:
            if retries >= max_retry or len(by_url) >= cap:
                break
            retries += 1
            trace(
                f"CURRICULUM url_retry ▶ Hallucinated link detected ({broken_hit.url}) "
                "-> Retried in Gemini Chat"
            )
            retry_answer = session.send(
                _build_url_retry_prompt(broken_hit.url, broken_hit.title, goal)
            )
            candidates = _hits_from_answer(retry_answer, 4)
            new_valid, _ = validate_and_filter_urls(candidates, **validate_kw)
            for v in new_valid:
                key = v.url.lower()
                if key in by_url:
                    continue
                by_url[key] = v
                trace(f"CURRICULUM url_retry ✓ Found working link: {v.url}")
                if len(by_url) >= cap:
                    break

        out = list(by_url.values())[:cap]
        trace(
            f"CURRICULUM gemini_web validate ✓ | valid={len(out)} "
            f"broken_unrepaired={max(0, len(broken) - retries)} retries={retries}"
        )
        return out
    finally:
        session.close()


def collect_gemini_web_practical_hits(
    target_goal: str,
    *,
    context_vector: str = "",
    max_urls: int | None = None,
) -> list[CurriculumSearchHit]:
    """
    Playwright → gemini.google.com (профиль .browser_state).
    Не использует вручную открытый Chrome — только автоматический контекст.
    """
    goal = (target_goal or "").strip()
    if len(goal) < 8:
        return []

    cap = max_urls if max_urls is not None else CURRICULUM_GEMINI_GROUNDING_MAX_URLS
    prompt = _build_harvest_prompt(goal, context_vector)
    trace(
        "CURRICULUM gemini_web ▶ | Playwright profile (.browser_state) — "
        f"таймаут {CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC:.0f}s"
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_harvest_in_session, prompt, cap, goal)
            return fut.result(timeout=CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC)
    except FuturesTimeout:
        trace(
            f"CURRICULUM gemini_web ✗ | timeout {CURRICULUM_GEMINI_WEB_HARVEST_TIMEOUT_SEC:.0f}s "
            "(закройте зависший Chromium или уменьшите промпт; SearXNG продолжит сбор)"
        )
        return []
    except Exception as exc:
        trace(
            f"CURRICULUM gemini_web ✗ | {exc} "
            "(login: python -m knowledge_engine.main browser-login)"
        )
        return []
