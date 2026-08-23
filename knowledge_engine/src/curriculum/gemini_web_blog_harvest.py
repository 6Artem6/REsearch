"""Practical blogs via Gemini structured JSON (HarvestedLinksResponse)."""

from __future__ import annotations

from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_MAX_URLS,
    CURRICULUM_GEMINI_WEB_URL_RETRY_MAX,
    CURRICULUM_URL_VALIDATE_TIMEOUT_SEC,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.research_schemas import (
    HarvestedLinksResponse,
)
from knowledge_engine.services.curriculum_whitelist_prompt import (
    curriculum_whitelist_prompt_block,
)
from knowledge_engine.src.analytics.gemini_v07 import run_gemini_flash_structured
from knowledge_engine.src.curriculum.curriculum_search_sites import (
    format_sites_for_prompt,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.url_validate import validate_and_filter_urls
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    is_collectible_article_url,
)
from knowledge_engine.ui.run_log import trace

_HARVEST_SYSTEM = (
    "You are a research librarian collecting practical engineering articles.\n"
    "Return strictly valid JSON matching HarvestedLinksResponse.\n"
    "Required key: items (1–8). Each item: title, url, relevance_reason.\n"
    "URLs MUST be real article pages (not site homepages).\n"
    "Prefer deep technical posts and official docs; skip SEO listicles.\n"
    "No markdown, no prose outside JSON.\n"
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "title and relevance_reason MAY be Russian; urls stay as-is.\n"
)


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


def hits_from_harvest_response(
    payload: HarvestedLinksResponse,
    cap: int,
) -> list[CurriculumSearchHit]:
    """Map a validated Pydantic harvest object to CurriculumSearchHit rows.

    Does not parse Markdown or scrape URLs from free text.
    """
    hits: list[CurriculumSearchHit] = []
    seen: set[str] = set()
    for item in payload.items:
        url = _normalize_url(item.url)
        if not url or url.lower() in seen:
            continue
        if not is_collectible_article_url(url):
            continue
        seen.add(url.lower())
        title = (item.title or "").strip() or _title_from_url(url)
        snippet = (item.relevance_reason or "").strip() or title
        hits.append(
            CurriculumSearchHit(
                url=url,
                title=title[:400],
                snippet=snippet[:1200],
                source_tier="gemini_web",
            )
        )
        if len(hits) >= cap:
            break
    return hits


def _build_harvest_user_payload(target_goal: str, context_vector: str) -> str:
    sites = format_sites_for_prompt()
    whitelist_block = curriculum_whitelist_prompt_block()
    goal = (target_goal or "").strip()
    ctx = (context_vector or "").strip()
    body = (
        f"{whitelist_block}\n\n"
        "Find 4–8 authoritative engineering articles and practical deep-dives.\n"
        f"Prioritize known engineering blogs and official docs; also {sites}.\n"
        "Static whitelist is a hint, not a hard limit.\n\n"
        f"Goal / query:\n{goal}\n"
    )
    if ctx and ctx != goal:
        body += f"\nExpansion / context vector:\n{ctx[:2000]}\n"
    return body


def _build_url_retry_user_payload(
    broken_url: str, article_title: str, goal: str
) -> str:
    title = (article_title or "").strip() or _title_from_url(broken_url)
    ctx = (goal or "").strip()[:500]
    return (
        f"The URL '{broken_url}' for article '{title}' does not exist (HTTP 404).\n"
        "Return HarvestedLinksResponse with 1–4 replacement article URLs that "
        "actually exist (same topic). Prefer habr.com, martinfowler.com, "
        "or bytebytego.com if an exact replacement is missing.\n"
        f"{f'Goal context: {ctx}' if ctx else ''}"
    )


def _structured_harvest(
    user_payload: str,
    anchor: str,
    label: str,
) -> HarvestedLinksResponse | None:
    from knowledge_engine.services.gemini_stateless import (
        GeminiUnavailableError,
        is_gemini_available,
    )

    if not is_gemini_available():
        return None
    try:
        return run_gemini_flash_structured(
            _HARVEST_SYSTEM,
            user_payload,
            anchor,
            HarvestedLinksResponse,
            label,
        )
    except GeminiUnavailableError:
        return None
    except Exception as exc:
        trace(f"CURRICULUM gemini_web structured ✗ | {exc}")
        return None


def collect_gemini_web_practical_hits(
    target_goal: str,
    *,
    context_vector: str = "",
    max_urls: int | None = None,
) -> list[CurriculumSearchHit]:
    """Gemini structured JSON harvest + HTTP URL validation."""
    goal = (target_goal or "").strip()
    if len(goal) < 8:
        return []

    cap = max_urls if max_urls is not None else CURRICULUM_GEMINI_GROUNDING_MAX_URLS
    user_payload = _build_harvest_user_payload(goal, context_vector)
    trace("CURRICULUM gemini_web ▶ | HarvestedLinksResponse structured JSON")

    parsed = _structured_harvest(
        user_payload, goal, "curriculum harvest / HarvestedLinksResponse"
    )
    if parsed is None:
        return []

    hits = hits_from_harvest_response(parsed, cap)
    trace(f"CURRICULUM gemini_web ✓ | structured_items={len(hits)}")

    validate_kw = {"timeout": CURRICULUM_URL_VALIDATE_TIMEOUT_SEC}
    valid, broken = validate_and_filter_urls(hits, **validate_kw)
    by_url: dict[str, CurriculumSearchHit] = {h.url.lower(): h for h in valid}
    retries = 0
    max_retry = max(0, CURRICULUM_GEMINI_WEB_URL_RETRY_MAX)

    for broken_hit in broken:
        if retries >= max_retry or len(by_url) >= cap:
            break
        retries += 1
        trace(
            f"CURRICULUM url_retry ▶ broken link ({broken_hit.url}) "
            "-> HarvestedLinksResponse retry"
        )
        retry_parsed = _structured_harvest(
            _build_url_retry_user_payload(broken_hit.url, broken_hit.title, goal),
            goal,
            "curriculum harvest url retry",
        )
        if retry_parsed is None:
            continue
        candidates = hits_from_harvest_response(retry_parsed, 4)
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
