"""Gemini API + Google Search grounding (практические блоги, без Playwright UI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_MAX_URLS,
    GEMINI_RETRY_BACKOFF_SEC,
    GEMINI_RPM_PAUSE_SEC,
)
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    curriculum_grounding_model_chain,
    gemini_api_key_pool,
    is_gemini_available,
    _client_for_api_key,
    _extract_status_code,
    _google_retry_delay_sec,
    _gemini_error_blob,
    _is_daily_per_model_quota,
    _is_hard_quota_exhausted,
    _is_retryable,
    _rpm_pause_for_model,
    _sleep_with_jitter,
)
from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
    cap_collectible_items,
)
from knowledge_engine.ui.run_log import trace

from knowledge_engine.src.curriculum.curriculum_search_sites import (
    CURRICULUM_PRIORITY_ENGINEERING_SITES,
)
GroundingNextAction = Literal["ok", "next_key", "next_model"]


@dataclass
class GroundingWebHit:
    url: str
    title: str
    snippet: str


@dataclass
class GroundingSearchResult:
    hits: list[GroundingWebHit]
    gemini_exhausted: bool


def _collect_grounding_domains() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in CURRICULUM_PRIORITY_ENGINEERING_SITES:
        d = d.strip().lower()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    for entries in APPROVED_SOURCES_WHITELIST.values():
        for e in entries:
            host = (e or "").split("/")[0].strip().lower()
            if host and host not in seen and len(out) < 24:
                seen.add(host)
                out.append(host)
    return out


def _site_or_clause(domains: list[str], max_sites: int = 10) -> str:
    picked = domains[:max_sites]
    return " OR ".join(f"site:{d}" for d in picked)


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith("http"):
        return ""
    parsed = urlparse(u)
    if not parsed.netloc:
        return ""
    return u.split("#")[0].rstrip("/")


def _extract_grounding_hits(response: Any) -> list[GroundingWebHit]:
    candidates = getattr(response, "candidates", None) or []
    meta = None
    for cand in candidates:
        meta = getattr(cand, "grounding_metadata", None)
        if meta:
            break
    if not meta:
        return []

    snippet_by_index: dict[int, str] = {}
    for sup in getattr(meta, "grounding_supports", None) or []:
        seg = getattr(sup, "segment", None)
        text = (getattr(seg, "text", None) or "").strip()
        if not text:
            continue
        for idx in getattr(sup, "grounding_chunk_indices", None) or []:
            if isinstance(idx, int):
                snippet_by_index.setdefault(idx, text)

    out: list[GroundingWebHit] = []
    seen: set[str] = set()
    chunks = getattr(meta, "grounding_chunks", None) or []
    for i, ch in enumerate(chunks):
        web = getattr(ch, "web", None)
        if not web:
            continue
        uri = _normalize_url(getattr(web, "uri", None) or "")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        title = (getattr(web, "title", None) or uri).strip()[:400]
        snippet = (snippet_by_index.get(i) or "").strip()[:1200]
        out.append(GroundingWebHit(url=uri, title=title, snippet=snippet))
    return out


def _is_rate_limit_429(exc: BaseException) -> bool:
    code = _extract_status_code(exc)
    if code == 429:
        return True
    blob = _gemini_error_blob(exc).lower()
    return "resource_exhausted" in blob or "429" in blob


def _generate_grounding_search(
    client: Any,
    model: str,
    user_prompt: str,
) -> list[GroundingWebHit]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        ),
    )
    return _extract_grounding_hits(response)


def _grounding_with_model_retry(
    client: Any,
    model: str,
    user_prompt: str,
    cap: int,
    *,
    allow_key_switch: bool = False,
) -> tuple[list[GroundingWebHit], GroundingNextAction]:
    """Один model + один API key: retry/backoff; 429 → next_key если allow_key_switch."""
    delays = list(GEMINI_RETRY_BACKOFF_SEC)
    attempt = 0
    label = "curriculum gemini_grounding"

    while True:
        try:
            raw_hits = _generate_grounding_search(client, model, user_prompt)
            accepted = cap_collectible_items(raw_hits, cap)
            trace(
                f"CURRICULUM gemini_grounding ✓ | model={model} | "
                f"raw={len(raw_hits)} whitelist={len(accepted)}"
            )
            return accepted, "ok"
        except Exception as exc:
            code = _extract_status_code(exc)
            if code == 404:
                detail = _gemini_error_blob(exc).replace("\n", " ")[:220]
                trace(
                    f"CURRICULUM gemini_grounding skip | model={model} | HTTP 404 — "
                    f"имя модели не существует в Gemini API (это не «нужен другой ключ»). "
                    f"Проверьте GEMINI_GROUNDING_MODEL в .env (например gemini-2.5-flash). "
                    f"| {detail}"
                )
                return [], "next_model"

            if _is_rate_limit_429(exc) and allow_key_switch:
                trace(
                    "CURRICULUM grounding 429 ▶ Switching to fallback API key..."
                )
                return [], "next_key"

            if not _is_retryable(exc):
                trace(f"CURRICULUM gemini_grounding skip | model={model} | {exc}")
                return [], "next_model"

            google_wait = _google_retry_delay_sec(exc)
            daily = _is_daily_per_model_quota(exc)
            blob_low = _gemini_error_blob(exc).lower()
            if "limit: 0" in blob_low:
                trace(
                    f"CURRICULUM gemini_grounding quota ✗ | model={model} | "
                    f"limit:0 — следующая модель"
                )
                return [], "next_model"

            if daily and google_wait is None and attempt >= len(delays):
                trace(
                    f"CURRICULUM gemini_grounding daily quota ✗ | model={model} | "
                    f"fallback"
                )
                return [], "next_model"

            if _is_hard_quota_exhausted(exc) and attempt >= len(delays):
                trace(
                    f"CURRICULUM gemini_grounding quota ✗ | model={model} | "
                    f"жёсткая квота после backoff"
                )
                return [], "next_model"

            wait: float | None = None
            wait_src = ""
            if google_wait is not None:
                wait = google_wait
                wait_src = "API"
            elif attempt < len(delays):
                wait = delays[attempt]
                wait_src = "backoff"

            if wait is not None:
                trace(
                    f"GEMINI wait {wait:.0f}s ({wait_src}) | {label} | "
                    f"model={model} | {type(exc).__name__}"
                )
                _sleep_with_jitter(wait)
                if google_wait is not None:
                    continue
                attempt += 1
                continue

            if _is_rate_limit_429(exc):
                return [], "next_model"

            trace(f"CURRICULUM gemini_grounding skip | model={model} | {exc}")
            return [], "next_model"


def search_grounded_whitelist_blogs_detailed(
    query: str,
    *,
    context_vector: str = "",
    max_urls: int | None = None,
) -> GroundingSearchResult:
    """
    Gemini + Google Search tool; URL из grounding_metadata + whitelist.
  """
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для Search Grounding")

    goal = (query or "").strip()
    if len(goal) < 8:
        return GroundingSearchResult(hits=[], gemini_exhausted=False)

    cap = max_urls if max_urls is not None else CURRICULUM_GEMINI_GROUNDING_MAX_URLS
    domains = _collect_grounding_domains()
    site_filter = _site_or_clause(domains)

    user_prompt = (
        f"Найди 4–8 авторитетных инженерных статей и практических разборов по теме.\n"
        f"Ограничение поиска: ({site_filter})\n\n"
        f"Тема / запрос: {goal}\n"
    )
    if (context_vector or "").strip():
        user_prompt += f"\nКонтекст вектора расширения:\n{context_vector.strip()[:2000]}\n"
    user_prompt += (
        "\nВерни краткий обзор найденного; ссылки должны быть реальными страницами статей."
    )

    models = curriculum_grounding_model_chain()
    keys = gemini_api_key_pool()
    trace(
        f"CURRICULUM gemini_grounding ▶ | chain={' → '.join(models[:5])} | "
        f"domains={len(domains)} | api_keys={len(keys)}"
    )

    saw_rate_limit = False
    for model in models:
        if GEMINI_RPM_PAUSE_SEC > 0:
            _rpm_pause_for_model(model)
        for key_idx, api_key in enumerate(keys):
            client = _client_for_api_key(api_key)
            allow_switch = key_idx + 1 < len(keys)
            hits, action = _grounding_with_model_retry(
                client,
                model,
                user_prompt,
                cap,
                allow_key_switch=allow_switch,
            )
            if hits:
                return GroundingSearchResult(hits=hits, gemini_exhausted=False)
            if action == "next_key":
                saw_rate_limit = True
                continue
            if action == "next_model":
                break
        trace(f"CURRICULUM gemini_grounding ⊘ | model={model} | нет whitelist URL")

    trace("CURRICULUM gemini_grounding ✗ | chain исчерпан без whitelist hits")
    return GroundingSearchResult(hits=[], gemini_exhausted=saw_rate_limit)


def search_grounded_whitelist_blogs(
    query: str,
    *,
    context_vector: str = "",
    max_urls: int | None = None,
) -> list[GroundingWebHit]:
    return search_grounded_whitelist_blogs_detailed(
        query,
        context_vector=context_vector,
        max_urls=max_urls,
    ).hits
