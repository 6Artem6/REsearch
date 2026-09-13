"""NodeCandidateCollectionService — Node Grounding Gate, Этап 1: сбор,

дедупликация и сухие паспорта кандидатов для Gate 1 (одна нода). Add-only,
параллельно ``light_discovery_service.py``/``batch_digest_service.py``
(те — для целого курса при Штурвале; этот — для одной ноды).

ИСПРАВЛЕНО после ревью по docs/SOURCE_POOL.md ("Единая логика источников —
не дублировать в pipeline") и docs/ACADEMIC_AND_CONSENSUS.md:

- Academic-сбор (SS/arXiv) теперь идёт через УЖЕ СУЩЕСТВУЮЩИЙ
  ``academic_source_fetch._primary_academic_hits`` (hydrate + hybrid rerank
  + relaxation cascade) — раньше здесь были свои голые
  ``search_semantic_scholar``/``search_arxiv_fallback`` без ``arxiv_params``
  и без rerank.
- Consensus — через УЖЕ СУЩЕСТВУЮЩИЙ
  ``academic_consensus.harvest_consensus_for_node(..., defer_ingest=True)``,
  а не собственный сырой ``ConsensusDirectClient.search()``.
  ``defer_ingest=True`` — штатный, специально для этого предназначенный
  флаг (``curriculum_v08_harvest.py``): останавливается на
  ``_build_discovery_hits`` (заголовок/аннотация из уже пойманных paper'ов
  через SS-обогащение), НЕ доходя до ``fetch_paper_document``/Gemma-инжеста
  — то есть ровно то "метаданные без харвеста", что и раньше, но теперь
  через официальный путь, а не мимо него. Это же и снимает риск повторить
  многоминутную задержку, найденную при отладке.
- Habr остаётся собственной логикой этого файла — в docs нет отдельного
  "Habr hub architect", это и есть то самое "если недостаточно — дополнить".
  ИСПРАВЛЕНО ЕЩЁ РАЗ (по прямому указанию): раньше здесь был Exa-поиск с
  ``include_domains=["habr.com"]`` — рабочий, но не то, что просил Habr per
  se. Правильный путь для Habr — его собственный публичный RSS
  (``https://habr.com/ru/rss/companies/{slug}/articles/?fl=ru`` — формат
  подтверждён вручную через реальный HTTP-запрос и разбор
  ``<link type="application/rss+xml">`` на живой странице компании, не
  выдуман) для ОБНАРУЖЕНИЯ статей + уже существующие ``fetch_html``/
  ``_extract_paragraphs`` (Trafilatura) для прямых ссылок из RSS — та же
  пара примитивов, что уже используется для Exa/академии в этом файле,
  просто без самого Exa-поиска для этого канала. RSS отдаёт только
  хронологию (не релевантность) — простой keyword-фильтр по
  title+description ДО скачивания полной страницы (``_rss_item_matches_keywords``)
  не даёт каналу превратиться в мусор из последних постов компании обо
  всём подряд.

``targeted_node_search.py`` по-прежнему НЕ импортируется и не меняется —
переиспользуются его СОСЕДИ (``academic_source_fetch.py``,
``academic_consensus.py``), а не сам DEEP-пайплайн с фьюженным
search+ingest (см. предыдущую версию этого докстринга и roadmap).

Дедуп — тот же приём, что ``pre_map_deduplicator.py``/
``lecture_passage_fetch.find_near_duplicate_urls``: BGE-M3 pooled-vector +
Union-Find по cosine-порогу, посчитанный in-memory над небольшим (<30)
набором кандидатов — не отдельная персистентная pgvector-таблица (лишняя
инфраструктура ради разового сравнения десятка векторов одной ноды).

ДОБАВЛЕНО: мгновенная внешняя пакетная локализация английских/латиница
введений (``_translate_latin_snippets``, эвристика
``translation_service.needs_ru_translation`` — общая с ``light_discovery_
service.py`` Штурвала, не дублируется) — после отбора сбалансированного
пула, перед ``_generate_passports``, чтобы Flash Lite уже получал русский
текст. Перевод — через
``knowledge_engine.src.utils.translation_service.translate_batch_to_russian``
(бесплатный неофициальный Google REST-эндпоинт, ОДИН HTTP-запрос на весь
батч, БЕЗ локальных LLM/SLM — нулевая нагрузка VRAM/RAM). Русскоязычные
сниппеты (обычно Habr) не отправляются — эвристика по соотношению
кириллица/латиница, никакой новой тяжёлой зависимости (langdetect и т.п.)
не добавлено. Fail-open: сбой сети/API/неожиданный формат ответа — пул
кандидатов возвращается БЕЗ изменений (см. docstring
``translate_batch_to_russian``), ``generate-passports`` никогда не падает
из-за перевода.
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.adapters.search_providers.exa_client import ExaSearchClient
from knowledge_engine.src.config.settings import (
    LECTURE_PASSAGE_FETCH_TIMEOUT_SEC,
    LECTURE_PASSAGE_MIN_CHARS,
)
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.lite_search_pipeline import (
    AcademicSearchPlan,
    _lite_structured,
)
from knowledge_engine.src.domains.curriculum.pre_flight_triage import (
    _extract_paragraphs,
)
from knowledge_engine.src.domains.grounding.lecture_passage_fetch import fetch_html
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NODE_GATE1_POOL_MAX,
    NodeCandidatePassport,
    NodeSearchProfile,
)

if TYPE_CHECKING:
    from knowledge_engine.src.domains.curriculum.schemas import CurriculumNode

_DEDUP_COSINE_THRESHOLD = 0.86
_MAX_PER_CHANNEL = 6
_HABR_RSS_TIMEOUT_SEC = 3.0
_HABR_RSS_MAX_ITEMS = 30
_HABR_RSS_MAX_MATCHED_PER_HUB = 3


@dataclass
class _RawHit:
    url: str
    title: str
    snippet: str
    source_tier: str


def _clean_url(u: str) -> str:
    return (u or "").strip()


def _strip_html_tags(raw: str) -> str:
    """Грубая очистка HTML в RSS description — только для keyword-фильтра

    ниже, не для показа пользователю (тот текст всё равно приходит из
    Trafilatura, не из RSS description)."""
    return re.sub(r"<[^>]+>", " ", raw or "")


def _rss_item_matches_keywords(
    title: str, description: str, keywords: list[str]
) -> bool:
    """RSS отдаёт хронологию компании/хаба, а не релевантность теме ноды —

    без этого фильтра канал почти всегда состоит из случайных последних
    постов. Пустой keywords — намеренно пропускает всё (лучше нерелевантный
    кандидат, который Lite-паспорт потом честно опишет как не по теме, чем
    вообще без Habr-кандидатов)."""
    kws = [k.strip().lower() for k in (keywords or []) if (k or "").strip()]
    if not kws:
        return True
    text = f"{title} {description}".lower()
    return any(kw in text for kw in kws)


async def _fetch_habr_company_rss_items(
    company_slug: str,
) -> list[tuple[str, str, str]]:
    """company_slug → [(url, title, description_text), ...] из RSS хаба

    компании — https://habr.com/ru/rss/companies/{slug}/articles/?fl=ru.
    Формат подтверждён вручную (не выдуман): реальный HTTP-запрос к живой
    странице компании на Habr, разбор <link type="application/rss+xml">,
    затем разбор самого RSS. ``<guid isPermaLink="true">`` — канонический
    URL без ``utm_*``; ``<link>`` — тот же URL, но с ``utm_*`` (fallback,
    если guid вдруг не permalink)."""
    url = f"https://habr.com/ru/rss/companies/{company_slug}/articles/?fl=ru"
    xml_text = await fetch_html(url, timeout_sec=_HABR_RSS_TIMEOUT_SEC)
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        trace(f"NODE_GATE habr rss parse ✗ | {company_slug} | {exc}")
        return []

    out: list[tuple[str, str, str]] = []
    for item in root.iter("item"):
        guid_el = item.find("guid")
        link_el = item.find("link")
        title_el = item.find("title")
        desc_el = item.find("description")
        raw_url = (guid_el.text if guid_el is not None else "") or (
            link_el.text if link_el is not None else ""
        )
        clean_url = _clean_url((raw_url or "").split("?")[0])
        title = (title_el.text or "").strip() if title_el is not None else ""
        description = (
            _strip_html_tags(desc_el.text or "") if desc_el is not None else ""
        )
        if clean_url:
            out.append((clean_url, title, description))
        if len(out) >= _HABR_RSS_MAX_ITEMS:
            break
    return out


async def _collect_habr(profile: NodeSearchProfile) -> list[_RawHit]:
    """Habr — через его собственный RSS (обнаружение статей компании) +

    уже существующие ``fetch_html``/``_extract_paragraphs`` (Trafilatura,
    прямые ссылки из RSS) — НЕ через Exa с доменным ограничением, как было
    раньше (см. модуль docstring: Habr нужно опрашивать через его API/RSS,
    остальные шаги — сбор лида через Trafilatura — уже были и не меняются)."""
    if not profile.habr_hubs:
        return []
    keywords = [*profile.habr_tags, *profile.habr_keywords]
    out: list[_RawHit] = []
    for hub in profile.habr_hubs[:4]:
        slug = (hub or "").strip().strip("/").split("/")[0]
        if not slug:
            continue
        try:
            items = await _fetch_habr_company_rss_items(slug)
        except Exception as exc:
            trace(f"NODE_GATE collect habr rss ✗ | hub={slug} | {exc}")
            continue
        matched = [
            (url, title)
            for url, title, description in items
            if _rss_item_matches_keywords(title, description, keywords)
        ][:_HABR_RSS_MAX_MATCHED_PER_HUB]
        if not matched:
            trace(f"NODE_GATE collect habr rss ⊘ | hub={slug} | no keyword match")
            continue
        for url, title in matched:
            html = await fetch_html(url, timeout_sec=LECTURE_PASSAGE_FETCH_TIMEOUT_SEC)
            if not html:
                continue
            paragraphs = await asyncio.to_thread(
                _extract_paragraphs, html, url, min_chars=LECTURE_PASSAGE_MIN_CHARS
            )
            if not paragraphs:
                continue
            lead = " ".join(paragraphs[:2])
            out.append(_RawHit(url, title or url, lead, "habr"))
    return out[:_MAX_PER_CHANNEL]


async def _collect_exa(profile: NodeSearchProfile) -> list[_RawHit]:
    client = ExaSearchClient()
    if not profile.exa_keywords or not client.is_configured():
        return []
    query = " ".join(profile.exa_keywords[:5])
    try:
        resp = await asyncio.to_thread(
            client.search, query, num_results=_MAX_PER_CHANNEL
        )
    except Exception as exc:
        trace(f"NODE_GATE collect exa ✗ | {exc}")
        return []
    return [
        _RawHit(
            _clean_url(h.url),
            h.title or h.url,
            " ".join(h.highlights or [])[:400],
            "exa",
        )
        for h in resp.hits
        if _clean_url(h.url)
    ]


def _hit_from_curriculum_search_hit(h) -> _RawHit:
    return _RawHit(
        _clean_url(h.url),
        h.title or h.url,
        (h.snippet or " ".join(h.key_extracts or []))[:400],
        (h.source_tier or "academic").strip().lower(),
    )


async def _collect_academic(
    academic_plan: AcademicSearchPlan | None,
    node: "CurriculumNode | None",
    *,
    anchor: str,
) -> list[_RawHit]:
    """Semantic Scholar/arXiv через ``_primary_academic_hits`` (hydrate +

    hybrid rerank + relaxation cascade) и Consensus через
    ``harvest_consensus_for_node(..., defer_ingest=True)`` — оба уже
    существующие, см. модуль docstring. ``defer_ingest=True`` — штатный
    флаг именно для "метаданные без полного харвеста", ничего нового не
    изобретаем."""
    if academic_plan is None or not (academic_plan.academic_query_en or "").strip():
        return []
    query = academic_plan.academic_query_en
    out: list[_RawHit] = []

    async def _primary() -> None:
        from knowledge_engine.src.domains.curriculum.academic_source_fetch import (
            _primary_academic_hits,
        )

        try:
            hits = await _primary_academic_hits(
                query, arxiv_params=academic_plan.arxiv_params, min_hits=3
            )
        except Exception as exc:
            trace(f"NODE_GATE collect academic (SS/arXiv) ✗ | {exc}")
            return
        out.extend(_hit_from_curriculum_search_hit(h) for h in hits)

    async def _consensus() -> None:
        if node is None:
            return
        from knowledge_engine.src.domains.curriculum.academic_consensus import (
            harvest_consensus_for_node,
        )

        try:
            hits = await harvest_consensus_for_node(
                node,
                query,
                anchor,
                "node_gate_discovery",
                on_demand=True,
                defer_ingest=True,
            )
        except Exception as exc:
            trace(f"NODE_GATE collect consensus ✗ | {exc}")
            return
        out.extend(_hit_from_curriculum_search_hit(h) for h in hits)

    await asyncio.gather(_primary(), _consensus())
    return out[:_MAX_PER_CHANNEL]


def _dedupe_by_url(hits: list[_RawHit]) -> list[_RawHit]:
    seen: set[str] = set()
    out: list[_RawHit] = []
    for h in hits:
        key = h.url.strip().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _dedupe_by_embedding(hits: list[_RawHit]) -> list[_RawHit]:
    """BGE-M3 pooled vector + cosine Union-Find — тот же приём, что

    pre_map_deduplicator.py, посчитанный in-memory над этим небольшим
    набором (не персистентная pgvector-таблица)."""
    if len(hits) < 2:
        return hits
    from knowledge_engine.src.domains.ingestion.deduplication.pre_map_deduplicator import (
        _cluster_text_candidates,
        _pool_vector,
    )

    vectors: dict[str, list[float]] = {}
    for h in hits:
        vec = _pool_vector([h.title, h.snippet])
        if vec is not None:
            vectors[h.url] = vec
    if len(vectors) < 2:
        return hits

    groups = _cluster_text_candidates(vectors, threshold=_DEDUP_COSINE_THRESHOLD)
    keep_urls = {g[0] for g in groups}  # первый в группе — представитель
    no_vector = [h.url for h in hits if h.url not in vectors]
    keep_urls.update(no_vector)
    return [h for h in hits if h.url in keep_urls]


def _select_balanced_pool(hits: list[_RawHit]) -> list[_RawHit]:
    """Round-robin по source_tier — «сбалансированный пул разнородных

    кандидатов», не просто первые N по порядку сбора."""
    by_tier: dict[str, list[_RawHit]] = {}
    for h in hits:
        by_tier.setdefault(h.source_tier, []).append(h)
    tiers = list(by_tier.keys())
    out: list[_RawHit] = []
    i = 0
    while len(out) < NODE_GATE1_POOL_MAX and any(by_tier.values()):
        tier = tiers[i % len(tiers)]
        if by_tier[tier]:
            out.append(by_tier[tier].pop(0))
        i += 1
        if i > NODE_GATE1_POOL_MAX * len(tiers) + 8:
            break  # safety valve against pathological empty-tier loops
    return out


async def _translate_latin_snippets(hits: list[_RawHit]) -> list[_RawHit]:
    """Этап 1, перед Паспортами: сниппеты (введения/abstract'ы) на латинице/

    английском (обычно Exa/academic каналы) — ОДНИМ батч-вызовом
    ``translate_batch_to_russian`` до вызова Flash Lite — та же логика,
    что просит задание, только вход/выход остаются ``_RawHit`` (снипет
    подставляется на место оригинала, дальше ``_generate_passports`` не
    знает и не должен знать, что текст был переведён). ``needs_ru_translation``
    — общая эвристика с ``light_discovery_service.py`` (Штурвал), живёт в
    ``translation_service.py``, чтобы не дублировать. Fail-open: сбой
    сети/API/несовпадение длины — хиты возвращаются без изменений (см.
    translation_service.translate_batch_to_russian)."""
    from knowledge_engine.src.utils.translation_service import (
        needs_ru_translation,
        translate_batch_to_russian,
    )

    idx_to_translate = [
        i for i, h in enumerate(hits) if needs_ru_translation(h.snippet)
    ]
    if not idx_to_translate:
        return hits

    texts = [hits[i].snippet for i in idx_to_translate]
    try:
        translated = await translate_batch_to_russian(texts)
    except Exception as exc:
        trace(f"NODE_GATE passports translate ✗ | fail-open, keep original | {exc}")
        return hits
    if len(translated) != len(idx_to_translate):
        trace("NODE_GATE passports translate ✗ | length mismatch, fail-open")
        return hits

    out = list(hits)
    for pos, i in enumerate(idx_to_translate):
        out[i] = replace(out[i], snippet=translated[pos])
    trace(f"NODE_GATE passports translate ✓ | n={len(idx_to_translate)}")
    return out


_PASSPORT_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a technical passport writer. You receive short snippets "
    "(title + highlight/abstract) for several candidate sources for one "
    "curriculum node — NOT full article text. For EACH candidate, write a "
    "dry passport a busy engineer uses to decide whether to pick it for "
    "deeper reading (Gate 1 of a review funnel).\n\n"
    "Return JSON matching a list of passports, one per url actually "
    "provided below (skip none, do not invent extra ones):\n"
    "- url: copy verbatim.\n"
    "- title: copy verbatim from the provided title.\n"
    "- topic: one short topic label (Russian).\n"
    "- tech_stack: technologies/tools mentioned in the snippet, if any "
    '(keep original names, e.g. "PostgreSQL", not translated).\n'
    "- gist: 1-2 dry sentences of substance from the given "
    "snippet/abstract (Russian) — no narrative filler, no guessing beyond "
    "what the snippet says."
)


class _PassportBatch:
    """Плоская обёртка для batch-ответа Flash Lite (список без обёртки

    плохо валидируется некоторыми провайдерами structured output)."""

    pass


async def _generate_passports(
    hits: list[_RawHit], *, anchor: str
) -> list[NodeCandidatePassport]:
    if not hits:
        return []
    from pydantic import BaseModel, Field

    class _Batch(BaseModel):
        passports: list[NodeCandidatePassport] = Field(default_factory=list)

    payload = [
        {
            "url": h.url,
            "title": h.title,
            "snippet": h.snippet,
            "source_tier": h.source_tier,
        }
        for h in hits
    ]
    try:
        out = await _lite_structured(
            _PASSPORT_SYSTEM,
            json.dumps({"candidates": payload}, ensure_ascii=False),
            anchor,
            _Batch,
            "curriculum / node_gate_passports",
        )
    except Exception as exc:
        trace(f"NODE_GATE passports fallback | skip passport generation | {exc}")
        return []

    batch = out if isinstance(out, _Batch) else _Batch.model_validate(out)
    by_url = {h.url: h for h in hits}
    passports: list[NodeCandidatePassport] = []
    for p in batch.passports:
        src = by_url.get(p.url)
        if src is None:
            continue
        passports.append(p.model_copy(update={"source_tier": src.source_tier}))
    return passports


async def collect_node_candidates(
    profile: NodeSearchProfile,
    *,
    curriculum_id: str,
    node_id: str,
    node: "CurriculumNode | None" = None,
    academic_plan: AcademicSearchPlan | None = None,
) -> list[NodeCandidatePassport]:
    """NodeSearchProfile → список паспортов кандидатов для Gate 1 (5-10 шт).

    Собирает Habr/Exa/Academic параллельно, дедуплицирует по URL и по
    BGE-M3 эмбеддингам, отбирает сбалансированный пул, затем batch-Lite
    делает сухие паспорта. Fail-open на каждом шаге — сбой одного канала
    не рвёт остальные, пустой результат — валидный (Gate 1 просто покажет
    "кандидатов не найдено"). ``node``/``academic_plan`` — только для
    academic-канала (``harvest_consensus_for_node`` требует настоящий
    ``CurriculumNode``; ``academic_plan`` — из
    ``node_search_profile_service.build_node_academic_plan``, см. эндпоинт).
    Оба ``None``, если академия выключена для этой ноды — тогда academic-
    канал просто пуст, без единого лишнего вызова."""
    anchor = f"node_gate_academic:{curriculum_id}:{node_id}"[:500]
    results = await asyncio.gather(
        _collect_habr(profile),
        _collect_exa(profile),
        _collect_academic(academic_plan, node, anchor=anchor),
        return_exceptions=True,
    )
    # return_exceptions=True: канал, упавший НЕ через свой внутренний
    # try/except (уже покрывающий реальные сетевые сбои — см. _collect_*),
    # а каким-то неожиданным способом, не должен рвать всю Gate 1 выборку
    # из-за двух других живых каналов.
    channel_names = ("habr", "exa", "academic")
    all_hits: list[_RawHit] = []
    for name, res in zip(channel_names, results):
        if isinstance(res, BaseException):
            trace(f"NODE_GATE collect {name} ✗ | unexpected | {res}")
            continue
        all_hits.extend(res)
    all_hits = _dedupe_by_url(all_hits)
    all_hits = _dedupe_by_embedding(all_hits)
    pool = _select_balanced_pool(all_hits)
    trace(
        f"NODE_GATE collect ✓ | node={node_id} raw={sum(len(r) for r in results if not isinstance(r, BaseException))} "
        f"deduped={len(all_hits)} pool={len(pool)}"
    )
    # Английские/латиница введения (обычно Exa/academic) — в русский ОДНИМ
    # батч-запросом перед тем, как отдать снипеты Flash Lite для паспортов
    # (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md, "мгновенная внешняя
    # пакетная локализация"). Уже русскоязычные (обычно Habr) не трогаем.
    pool = await _translate_latin_snippets(pool)
    passport_anchor = f"node_gate_passports:{curriculum_id}:{node_id}"[:500]
    return await _generate_passports(pool, anchor=passport_anchor)


__all__ = ["collect_node_candidates"]
