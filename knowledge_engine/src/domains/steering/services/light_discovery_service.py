"""Light Discovery Service — Этап 2 Штурвала, шаг 2: экспресс-поиск по

хабам компаний из TaxonomySeed, извлечение ТОЛЬКО заголовка и первых 1-2
абзацев (``lead_paragraph``) через Trafilatura — статья целиком не
скачивается. Add-only, план — docs/STEERING_AND_TOPIC_QNA_ROADMAP.md,
контракты Gate 1 — ``src.curriculum.steering_contracts``.

RSS-путь из заголовка задачи в этом проходе не реализован: детальная
спецификация функционала описывает только Exa+Trafilatura, RSS упомянут
лишь в названии раздела — решение задокументировано в roadmap-таблице.

ДОБАВЛЕНО (по прямому указанию пользователя после проверки Gate 1
Штурвала вживую — там были непереведённые английские lead_paragraph):
``_translate_latin_leads`` — та же мгновенная батч-локализация, что уже
была подключена для Node Grounding Gate, через ОБЩИЙ
``translation_service.translate_batch_to_russian``/``needs_ru_translation``
(не дублируется — единая эвристика и единый HTTP-клиент на оба гейта).
В отличие от Node Gate, здесь на Gate 1 нет прохода через Flash Lite
(тот идёт позже, при Batch Digest после approve-gate1) — перевод просто
подставляется в ``lead_paragraph`` сразу после сбора, до возврата
``TaxonomyDiscoveryResponse``."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.adapters.search_providers.exa_client import (
    ExaSearchClient,
    ExaSearchHit,
)
from knowledge_engine.src.config.settings import (
    LECTURE_PASSAGE_FETCH_CONCURRENCY,
    LECTURE_PASSAGE_FETCH_TIMEOUT_SEC,
    LECTURE_PASSAGE_MIN_CHARS,
)
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.pre_flight_triage import (
    _extract_paragraphs,
)
from knowledge_engine.src.domains.grounding.lecture_passage_fetch import fetch_html
from knowledge_engine.src.domains.steering.services.taxonomy_service import (
    TaxonomySeed,
    is_academic_domain,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    CandidateArticleMeta,
    TaxonomyDiscoveryResponse,
)

_LEAD_PARAGRAPHS_N = 2
_MAX_RESULTS_PER_HUB = 5


def _resolve_hub(hub: str) -> tuple[str | None, str]:
    """hub → (hostname для include_domains | None, текстовая подсказка для query).

    ``TaxonomySeed.company_hubs`` может быть как bare hostname/путём
    (``"habr.com/ru/companies/yandex"``), так и голым именем компании
    (``"netflix"``) без резолвимого домена — второй случай ищем без
    ограничения по домену, полагаясь на текст запроса."""
    h = (hub or "").strip().strip("/")
    if not h:
        return None, ""
    hostname = h.split("/")[0]
    if "." in hostname:
        return hostname, h
    return None, h


def _title_from_html(html: str, fallback: str) -> str:
    """Best-effort <title> без парсинга всей страницы — только начало HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html[:20000], "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title
    except Exception:
        pass
    return fallback


async def _search_hub(
    client: ExaSearchClient, hub: str, keywords: list[str]
) -> list[ExaSearchHit]:
    hostname, hint = _resolve_hub(hub)
    if not hint:
        return []
    query = " ".join([hint, *keywords[:4]]).strip()
    try:
        response = await asyncio.to_thread(
            client.search,
            query,
            num_results=_MAX_RESULTS_PER_HUB,
            include_domains=[hostname] if hostname else None,
            allow_unrestricted_fallback=bool(hostname),
        )
        return response.hits
    except Exception as exc:
        trace(f"STEERING light_discovery search ✗ | hub={hub[:60]!r} | {exc}")
        return []


async def _hit_to_candidate(hit: ExaSearchHit, hub: str) -> CandidateArticleMeta | None:
    html = await fetch_html(hit.url, timeout_sec=LECTURE_PASSAGE_FETCH_TIMEOUT_SEC)
    if not html:
        return None
    paragraphs = await asyncio.to_thread(
        _extract_paragraphs, html, hit.url, min_chars=LECTURE_PASSAGE_MIN_CHARS
    )
    if not paragraphs:
        return None
    lead = " ".join(paragraphs[:_LEAD_PARAGRAPHS_N])
    title = _title_from_html(html, hit.title or hit.url)
    return CandidateArticleMeta(
        url=hit.url, title=title, source_hub=hub, lead_paragraph=lead
    )


async def _translate_latin_leads(
    candidates: list[CandidateArticleMeta],
) -> list[CandidateArticleMeta]:
    """Gate 1 показывает ``lead_paragraph`` пользователю "как есть" — без

    прохода через Flash Lite (в отличие от Node Grounding Gate, где перевод
    вставлен перед генерацией Паспортов). Английские/латиница intro
    (обычно так и есть — Exa ищет по company hubs без ограничения языка)
    переводим ОДНИМ батч-вызовом ``translate_batch_to_russian`` до
    возврата ``TaxonomyDiscoveryResponse``. Fail-open: сбой сети/API/
    несовпадение длины — кандидаты возвращаются без изменений."""
    from knowledge_engine.src.utils.translation_service import (
        needs_ru_translation,
        translate_batch_to_russian,
    )

    idx_to_translate = [
        i for i, c in enumerate(candidates) if needs_ru_translation(c.lead_paragraph)
    ]
    if not idx_to_translate:
        return candidates

    texts = [candidates[i].lead_paragraph for i in idx_to_translate]
    try:
        translated = await translate_batch_to_russian(texts)
    except Exception as exc:
        trace(
            f"STEERING light_discovery translate ✗ | fail-open, keep original | {exc}"
        )
        return candidates
    if len(translated) != len(idx_to_translate):
        trace("STEERING light_discovery translate ✗ | length mismatch, fail-open")
        return candidates

    out = list(candidates)
    for pos, i in enumerate(idx_to_translate):
        out[i] = out[i].model_copy(update={"lead_paragraph": translated[pos]})
    trace(f"STEERING light_discovery translate ✓ | n={len(idx_to_translate)}")
    return out


def _dedupe_candidates_by_url(
    candidates: list[CandidateArticleMeta],
) -> list[CandidateArticleMeta]:
    """Один и тот же URL нередко находится через РАЗНЫЕ ``company_hubs``

    (или несколько релевантных страниц одного хаба) — без дедупа он
    попадает в Gate 1 несколько раз отдельными карточками. Пользователь
    отмечает обе (не видя, что это одна и та же статья), и на Gate 2
    ``approved_urls`` приходит с дублями (найдено при живой отладке: 9
    approved_urls, из них только 5 уникальных — 4 дубля одной и той же
    статьи Postgres Pro). Тот же приём, что
    ``node_candidate_collection_service._dedupe_by_url`` (Node Grounding
    Gate) — нормализация регистра/конечного слэша, без BGE-M3 (для целого
    курса разные статьи по смежным темам — это желаемое разнообразие, а не
    дубли, в отличие от одной ноды)."""
    seen: set[str] = set()
    out: list[CandidateArticleMeta] = []
    for c in candidates:
        key = c.url.strip().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


async def discover_candidates(
    seed: TaxonomySeed,
    *,
    source_policy: str = "hybrid",
    concurrency: int | None = None,
) -> TaxonomyDiscoveryResponse:
    """TaxonomySeed → TaxonomyDiscoveryResponse для Gate 1.

    Поиск по каждому ``company_hubs`` через Exa, затем для каждого найденного
    URL — fetch + Trafilatura, но в карточку идут только первые 1-2 абзаца
    (``lead_paragraph``); сама статья не скачивается целиком и не хранится.
    Хабы/URL, которые не удалось обработать, просто отсутствуют в результате
    (fail-open — сбой одного источника не должен рвать всю воронку).

    ``source_policy`` фильтруется повторно уже на итоговых URL (не только на
    входных ``company_hubs``, отфильтрованных в taxonomy_service.py) — Exa
    ``allow_unrestricted_fallback=True`` в ``_search_hub`` может вернуть хит
    ВНЕ заданного домена, если поиск по хабу пуст, и теоретически подсунуть
    академический URL даже при чистых hubs (см. отладку "утечка Consensus/
    arXiv/S2 в Practical")."""
    client = ExaSearchClient()
    if not seed.company_hubs or not client.is_configured():
        trace("STEERING light_discovery skip | no company_hubs or Exa not configured")
        return TaxonomyDiscoveryResponse(
            tags=seed.tags, company_hubs=seed.company_hubs, keywords=seed.keywords
        )

    conc = max(
        1, concurrency if concurrency is not None else LECTURE_PASSAGE_FETCH_CONCURRENCY
    )
    sem = asyncio.Semaphore(conc)

    async def _one_hub(hub: str) -> list[CandidateArticleMeta]:
        hits = await _search_hub(client, hub, seed.keywords)

        async def _bounded(hit: ExaSearchHit) -> CandidateArticleMeta | None:
            async with sem:
                return await _hit_to_candidate(hit, hub)

        results = await asyncio.gather(*[_bounded(h) for h in hits])
        return [c for c in results if c is not None]

    per_hub = await asyncio.gather(*[_one_hub(h) for h in seed.company_hubs])
    candidates = [c for group in per_hub for c in group]
    candidates = _dedupe_candidates_by_url(candidates)

    policy = (source_policy or "hybrid").strip().lower()
    if policy == "practical_only":
        before = len(candidates)
        candidates = [c for c in candidates if not is_academic_domain(c.url)]
        if len(candidates) != before:
            trace(
                "STEERING light_discovery policy_filter | practical_only "
                f"dropped={before - len(candidates)} (unrestricted-fallback leak)"
            )
    elif policy == "academic_only":
        candidates = [c for c in candidates if is_academic_domain(c.url)]

    candidates = await _translate_latin_leads(candidates)

    trace(
        f"STEERING light_discovery ✓ | hubs={len(seed.company_hubs)} "
        f"candidates={len(candidates)}"
    )
    return TaxonomyDiscoveryResponse(
        tags=seed.tags,
        company_hubs=seed.company_hubs,
        keywords=seed.keywords,
        candidate_articles=candidates,
    )


__all__ = ["discover_candidates"]
