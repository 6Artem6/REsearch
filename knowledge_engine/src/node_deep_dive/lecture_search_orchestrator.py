"""Stage 1/2: Exa → (optional) academic search → VERIFIED_EXTERNAL_SOURCES."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from knowledge_engine.config import (
    EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN,
    EXA_FETCH_NUM_RESULTS,
    EXA_MAX_CONCURRENT_SEARCH,
    EXA_RECALL_MAX_PER_DOMAIN,
    EXA_RERANK_LITE_THRESHOLD,
    LECTURE_EXTERNAL_SEARCH_ENABLED,
    LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
    LECTURE_EXTERNAL_SEARCH_TOP_K,
    LECTURE_PASSAGE_BACKFILL_MARGIN,
    MAX_EXTERNAL_SOURCES,
)
from knowledge_engine.services.blocking_pools import pool_net_sync, run_blocking_timed
from knowledge_engine.services.curriculum_whitelist_prompt import (
    enrich_node_learning_materials_from_graph,
)
from knowledge_engine.services.node_source_registry import is_disallowed_source_url
from knowledge_engine.services.search.exa_client import (
    ExaNotConfiguredError,
    ExaSearchClient,
    ExaSearchHit,
)
from knowledge_engine.services.search.exa_transform import (
    ExaQuerySpec,
    _lite_rerank_exa_hits,
    build_exa_query_plan,
    fair_domain_round_robin,
    fill_round_robin_tail,
    merge_multi_vector_exa_hits,
    postprocess_exa_hits_for_external_recall,
)
from knowledge_engine.services.search.providers import (
    ConsensusSearchProvider,
    SemanticScholarProvider,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.ui.run_log import trace
from knowledge_engine.utils.link_sanitizer import (
    extract_urls_from_text,
    normalize_lecture_url,
)

logger = logging.getLogger(__name__)

_SEARCH_TOOL_RE = re.compile(
    r'\{\s*"action"\s*:\s*"search_external_materials"\s*,\s*"query"\s*:\s*"(?P<q>(?:\\.|[^"\\])*)"\s*\}',
    re.I | re.S,
)
_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")


@dataclass(frozen=True)
class VerifiedExternalSource:
    url: str
    title: str
    snippet: str
    provider: str = ""
    score: float = 0.0


def external_source_limit(top_k: int | None = None) -> int:
    if top_k is not None:
        return max(1, int(top_k))
    return max(1, int(MAX_EXTERNAL_SOURCES or LECTURE_EXTERNAL_SEARCH_TOP_K or 3))


def build_external_search_query(
    node: NodeDataInput,
    subtopic: str,
    *,
    query_override: str = "",
) -> str:
    override = (query_override or "").strip()
    if override:
        return override[:500]
    title = (node.title or "").strip()
    sub = (subtopic or "").strip()
    parts = [title, sub, "technical documentation paper architecture"]
    return " ".join(p for p in parts if p)[:500]


def query_needs_en_translation(text: str) -> bool:
    """True when the focus/query contains Cyrillic (RU) for academic indexes."""
    raw = (text or "").strip()
    if not raw:
        return False
    # Any meaningful Cyrillic → translate; pure English stays as-is.
    return len(_CYRILLIC_RE.findall(raw)) >= 3


def translate_to_en_query(focus: str) -> str:
    """
    Translate lecture focus / node topic to English for Consensus / Semantic Scholar.

    Already-English queries are returned unchanged. Russian → Gemini Lite academic sanitize
    (same path as Consensus prep); on failure keep the original focus.
    """
    raw = (focus or "").strip()
    if not raw:
        return ""
    if not query_needs_en_translation(raw):
        return raw[:500]
    try:
        from knowledge_engine.src.processors.consensus_query_prep import (
            extract_preserved_terms_for_consensus,
        )
        from knowledge_engine.src.processors.validator import (
            sanitize_query_for_consensus,
        )

        terms = extract_preserved_terms_for_consensus(raw)
        out = sanitize_query_for_consensus(
            raw,
            f"lecture_external:{raw[:80]}",
            "",
            terms,
        )
        en = (getattr(out, "academic_query_en", None) or "").strip()
        if en:
            trace(
                f"LECTURE_SEARCH translate_en ✓ | in_len={len(raw)} out_len={len(en)}"
            )
            return en[:500]
    except Exception as exc:
        trace(f"LECTURE_SEARCH translate_en skip | {exc}")
    return raw[:500]


def _processed_hits_to_sources(
    processed: list[CurriculumSearchHit],
) -> list[VerifiedExternalSource]:
    out: list[VerifiedExternalSource] = []
    for i, hit in enumerate(processed):
        exa_score = getattr(hit, "exa_relevance_score", None)
        if exa_score is None:
            score = max(0.0, 1.0 - 0.05 * i)
        else:
            try:
                score = float(exa_score)
            except (TypeError, ValueError):
                score = max(0.0, 1.0 - 0.05 * i)
        out.append(
            VerifiedExternalSource(
                url=hit.url,
                title=(hit.title or hit.url)[:400],
                snippet=(hit.snippet or "").strip()[:1200],
                provider="exa",
                score=score,
            )
        )
    return out


def _exa_search_call_sync(
    client: ExaSearchClient,
    query: str,
    num_results: int,
    highlight_query: str | None,
    *,
    include_domains: list[str],
    category: str | None,
) -> list[ExaSearchHit]:
    """Один Exa-проход. include_domains=[] (не None!) отключает ограничение
    по доменам — build_exa_search_kwargs при include_domains=None молча
    подставляет APPROVED_SOURCES_WHITELIST, только пустой список реально
    снимает его. Discovery (Pass 1 validated_domains / Pass 2 broader
    category) собирается вызывающим кодом — см. _exa_sources_multi_vector."""
    kwargs: dict = {
        "num_results": num_results,
        "include_domains": list(include_domains or []),
        "category": category,
    }
    if highlight_query:
        kwargs["highlight_query"] = highlight_query
    resp = client.search(query, **kwargs)
    return list(resp.hits)


async def _exa_search_vector(
    client: ExaSearchClient,
    query: str,
    num_results: int,
    highlight_query: str | None,
    *,
    include_domains: list[str] | None = None,
    category: str | None = None,
) -> list[ExaSearchHit]:
    try:
        return await run_blocking_timed(
            pool_net_sync(),
            LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC,
            _exa_search_call_sync,
            client,
            query,
            num_results,
            highlight_query,
            include_domains=include_domains or [],
            category=category,
        )
    except asyncio.TimeoutError:
        trace(
            f"LECTURE_EXA vector skip | timeout {LECTURE_EXTERNAL_SEARCH_HTTP_TIMEOUT_SEC}s"
        )
        return []
    except Exception as exc:
        trace(f"LECTURE_EXA vector skip | {exc}")
        return []


async def _exa_sources_multi_vector(
    context: str,
    limit: int,
    *,
    anchor: str,
) -> list[VerifiedExternalSource]:
    """Fast & High-Quality lecture waterfall — те же дешёвые/быстрые этапы,
    что и в DEEP (services.search.exa_transform,
    services.search.exa_source_expand, src.curriculum.pre_flight_triage),
    БЕЗ Map-Reduce (никаких генеративных LLM-проходов по каждому документу —
    Gemma summarization и т.п.):

    Domain Discovery (Flash Lite + DOMAIN_REGISTRY, Pass 1 validated / Pass 2
    broader) → multi-vector Exa Search → URL/Slide Guard (practical filters,
    уже внутри postprocess_exa_hits_for_external_recall) → Async Fetch
    (httpx + Trafilatura, короткий таймаут на URL) → Passage Extraction
    (BGE-M3 + greedy MMR, lecture_passage_fetch.py) → Flash Lite Content
    Quality Gate по уже извлечённым абзацам (а не по 900-симв. Exa-хайлайту)
    → финальный round-robin + cap.

    Раньше единственный запрос к search_expanded с num_results=cap не
    оставлял round-robin'у запаса кандидатов — если Exa отдавала все top-N
    с одного домена (реальный баг: 3 версии postgresql.org docs вместо 1x PG
    + 1x SQLite + 1x GitHub), диверсифицировать было не из чего. Recall
    теперь берётся с тем же запасом (fetch_cap), что и в
    fetch_exa_curriculum_hits_for_node."""
    client = ExaSearchClient()
    if not client.is_configured():
        return []

    cap = max(3, limit)
    fetch_cap = max(cap + 8, cap * 2, EXA_FETCH_NUM_RESULTS)

    # --- Domain Discovery (Pass 1 validated domains) — по аналогии с DEEP.
    from knowledge_engine.services.search.exa_domain_validate import (
        prepare_exa_pass1_domains,
    )
    from knowledge_engine.services.search.exa_source_expand import (
        absorb_new_exa_hosts,
        exa_pass2_categories,
        expand_search_context_with_flash_lite,
        filter_pass1_official_hosts,
    )

    expansion = None
    validated_domains: list[str] = []
    try:
        expansion = await asyncio.to_thread(
            expand_search_context_with_flash_lite, context
        )
        live_domains = await prepare_exa_pass1_domains(expansion.primary_domains)
        validated_domains = filter_pass1_official_hosts(live_domains)
        trace(
            f"LECTURE_EXA discovery ✓ | primary={len(expansion.primary_domains)} "
            f"live={len(live_domains)} validated={len(validated_domains)}"
        )
    except Exception as exc:
        trace(f"LECTURE_EXA discovery skip | {exc}")

    try:
        qplan = await build_exa_query_plan(context, anchor=anchor)
    except Exception as exc:
        trace(f"LECTURE_EXA query_plan skip | {exc}")
        qplan = None
    specs: list[ExaQuerySpec] = list(qplan.specs) if qplan else []
    if not specs:
        # Фолбэк: Lite-план не построился (пустой контекст / ошибка) — один
        # синтетический вектор тем же query-текстом, чтобы не дублировать
        # Pass 1/2 логику отдельной веткой.
        specs = [
            ExaQuerySpec(
                role="en_declarative",
                query=(
                    context[:400] if len(context) >= 8 else f"{context} overview"[:400]
                ),
                highlight_query="architecture, internals, trade-offs",
            )
        ]

    per_vector = max(3, EXA_FETCH_NUM_RESULTS // max(1, len(specs)))
    sem = asyncio.Semaphore(max(1, EXA_MAX_CONCURRENT_SEARCH))

    async def _one(
        spec: ExaQuerySpec, *, include_domains: list[str], category: str | None
    ) -> list[ExaSearchHit]:
        async with sem:
            return await _exa_search_vector(
                client,
                spec.query,
                per_vector,
                spec.highlight_query or None,
                include_domains=include_domains,
                category=category,
            )

    batches = list(
        await asyncio.gather(
            *[_one(s, include_domains=validated_domains, category=None) for s in specs]
        )
    )
    raw_total = sum(len(b) for b in batches)
    trace(
        f"LECTURE_EXA pass1 ✓ | vectors={len(specs)} domains={len(validated_domains)} "
        f"hits={raw_total}"
    )
    merged = merge_multi_vector_exa_hits(batches, cap=fetch_cap)

    # --- Pass 2 (broader, no domain restriction) — только если Pass 1 не
    # набрал даже целевой cap.
    if len(merged) < cap and expansion is not None:
        for cat in exa_pass2_categories(expansion) or [None]:
            extra_batches = list(
                await asyncio.gather(
                    *[_one(s, include_domains=[], category=cat) for s in specs]
                )
            )
            extra_total = sum(len(b) for b in extra_batches)
            trace(f"LECTURE_EXA pass2 ✓ | category={cat} hits={extra_total}")
            if extra_total:
                batches = batches + extra_batches
                raw_total += extra_total
                merged = merge_multi_vector_exa_hits(batches, cap=fetch_cap)
                if len(merged) >= cap:
                    break

    capped = fair_domain_round_robin(
        merged,
        fetch_cap,
        max_per_domain=EXA_RECALL_MAX_PER_DOMAIN,
        get_url=lambda h: h.url,
    )
    if len(capped) < cap:
        capped = fill_round_robin_tail(capped, merged, cap, get_url=lambda h: h.url)
    trace(
        f"LECTURE_EXA multi_vector ✓ | vectors={len(specs)} raw={raw_total} "
        f"merged={len(merged)} diversified={len(capped)} cap={cap}"
    )
    if capped:
        absorb_new_exa_hosts([h.url for h in capped])

    # RU: postprocess ранжирует по composite score (URL-эвристика + Exa
    # score), НЕ по домену — топ-N по score мог полностью схлопнуться на
    # 1-2 доменах (реальный баг: 2/3 источника с Habr), даже если capped
    # выше был честно диверсифицирован по доменам В ШИРОКОМ recall-пуле
    # (fetch_cap, max_per_domain=EXA_RECALL_MAX_PER_DOMAIN=2 — рассчитан на
    # DEEP-пайплайн с cap~4). round-robin здесь применяется ВТОРОЙ раз — уже
    # на финальном срезе до cap, как в fetch_exa_curriculum_hits_for_node
    # (_lite_rerank_exa_hits делает то же самое после Flash Lite approve) —
    # но с более строгим EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN (=1): при
    # лекционном cap~3 те же 2 источника с одного домена — это 2/3, а не
    # 2/4, разбавлять нужно жёстче, чем в широком recall-пуле.
    wide_cap = max(cap, EXA_RERANK_LITE_THRESHOLD + 1)
    # RU: reserve_cap берёт wide_cap с запасом (LECTURE_PASSAGE_BACKFILL_
    # MARGIN) специально для near-dup добора ниже — если BGE-кластеризация
    # найдёт почти дубликат внутри processed, есть из чего заменить БЕЗ
    # нового сетевого Exa-запроса (кандидаты уже получены и диверсифицированы
    # round-robin'ом выше, просто ещё не фетчены/не прошли passage extraction).
    reserve_cap = wide_cap + LECTURE_PASSAGE_BACKFILL_MARGIN
    processed_all = postprocess_exa_hits_for_external_recall(capped, cap=reserve_cap)
    processed = processed_all[:wide_cap]
    reserve = processed_all[wide_cap:]

    # --- Async Fetch + Passage Extraction (Trafilatura + BGE-M3 + MMR) —
    # заменяет 900-симв. Exa-хайлайт на реально извлечённые, релевантные
    # ядру темы и разнообразные между собой абзацы страницы. URL, для
    # которых фетч не удался/текст оказался слишком тонким, остаются со
    # своим Exa-snippet (graceful degradation, источник не теряется).
    from knowledge_engine.src.node_deep_dive.lecture_passage_fetch import (
        fetch_and_extract_passages,
        find_near_duplicate_urls,
    )

    passages_by_url = await fetch_and_extract_passages(
        [h.url for h in processed],
        core_theme=context,
    )

    # --- Near-Duplicate Detection + добор из резерва. В отличие от DEEP
    # (deduplicate_before_map_reduce помечает дубликат ALIAS и просто теряет
    # слот — там нет MAP+REDUCE-бюджета для замены, но и резерва на этом
    # этапе тоже нет, см. разбор), здесь дубликат ЗАМЕНЯЕТСЯ следующим по
    # рангу источником из reserve, чтобы не тратить один из скудных
    # финальных слотов на «ту же статью другой версии».
    alias_of_url = await find_near_duplicate_urls(
        passages_by_url, anchor=f"{anchor}:lecture_dedup"
    )
    if alias_of_url:
        dropped = [h for h in processed if h.url in alias_of_url]
        processed = [h for h in processed if h.url not in alias_of_url]
        kept_urls = {h.url for h in processed}
        backfill = [h for h in reserve if h.url not in kept_urls][: len(dropped)]
        if backfill:
            backfill_passages = await fetch_and_extract_passages(
                [h.url for h in backfill], core_theme=context
            )
            passages_by_url.update(backfill_passages)
            processed = processed + backfill
        trace(
            f"LECTURE_EXA dedup ✓ | dropped={len(dropped)} "
            f"backfilled={len(backfill)} kept={len(processed)}"
        )

    if passages_by_url:
        enriched: list[CurriculumSearchHit] = []
        for h in processed:
            passages = passages_by_url.get(h.url)
            if passages:
                enriched.append(
                    h.model_copy(update={"snippet": " ".join(passages)[:1200]})
                )
            else:
                enriched.append(h)
        processed = enriched
        trace(
            f"LECTURE_EXA passages ✓ | sources_with_passages={len(passages_by_url)}"
            f"/{len(processed)}"
        )

    threshold = max(1, EXA_RERANK_LITE_THRESHOLD)
    final_max_per_domain = max(1, EXA_FAIR_ROUND_ROBIN_MAX_PER_DOMAIN)
    if len(processed) > threshold:
        # RU: тот же Flash Lite Content Quality Gate, что и в DEEP-пайплайне
        # ноды (_BATCH_SYSTEM, lite_search_pipeline.py) — с явным критерием
        # на слайд-деки/фрагментированный текст, теперь оценивает уже
        # извлечённые абзацы (после passage extraction выше), а не сырой
        # Exa-хайлайт — раньше в лекционном доборе этого гейта не было
        # вообще.
        final = await _lite_rerank_exa_hits(
            processed,
            context,
            [],
            anchor=anchor,
            cap=cap,
            max_per_domain=final_max_per_domain,
        )
    else:
        final = fair_domain_round_robin(
            processed,
            cap,
            max_per_domain=final_max_per_domain,
            get_url=lambda h: h.url,
        )
        if len(final) < min(cap, len(processed)):
            final = fill_round_robin_tail(
                final, processed, min(cap, len(processed)), get_url=lambda h: h.url
            )
        final = final[:cap]
    return _processed_hits_to_sources(final)


async def _provider_sources(
    provider_name: str,
    query: str,
    limit: int,
) -> list[VerifiedExternalSource]:
    if provider_name == "semantic_scholar":
        provider = SemanticScholarProvider()
    elif provider_name == "consensus":
        provider = ConsensusSearchProvider()
    else:
        return []
    try:
        rows = await provider.search(query, limit=limit)
    except Exception as exc:
        trace(f"LECTURE_SEARCH {provider_name} skip | {exc}")
        return []
    out: list[VerifiedExternalSource] = []
    for i, row in enumerate(rows):
        url = str(row.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        raw_score = row.get("score")
        try:
            score = (
                float(raw_score) if raw_score is not None else max(0.0, 0.55 - 0.04 * i)
            )
        except (TypeError, ValueError):
            score = max(0.0, 0.55 - 0.04 * i)
        out.append(
            VerifiedExternalSource(
                url=url,
                title=str(row.get("title") or url)[:400],
                snippet=str(row.get("snippet") or "")[:1200],
                provider=str(row.get("source") or provider_name),
                score=score,
            )
        )
    return out


def _merge_sources(
    batches: list[list[VerifiedExternalSource]],
    top_k: int,
) -> list[VerifiedExternalSource]:
    """Dedupe by URL; keep highest score; rank by relevance score (desc)."""
    best: dict[str, VerifiedExternalSource] = {}
    for batch in batches:
        for src in batch:
            key = normalize_lecture_url(src.url)
            if not key:
                continue
            prev = best.get(key)
            if prev is None or float(src.score) > float(prev.score):
                best[key] = src
    ranked = sorted(
        best.values(),
        key=lambda s: float(s.score),
        reverse=True,
    )
    return ranked[: max(1, top_k)]


async def fetch_verified_external_sources(
    node: NodeDataInput,
    subtopic: str,
    curriculum_id: str = "",
    *,
    query_override: str = "",
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    """Waterfall: Exa first (early exit) → EN academic Consensus/SS fallback."""
    try:
        return await _fetch_verified_external_sources_impl(
            node,
            subtopic,
            curriculum_id,
            query_override=query_override,
            top_k=top_k,
        )
    except Exception as exc:
        from knowledge_engine.ui.errors import trace_exception

        trace_exception(exc, "LECTURE_SEARCH")
        return []


async def _exa_batch(
    query: str, per_provider: int, *, anchor: str = "lecture_search"
) -> list[VerifiedExternalSource]:
    try:
        return await _exa_sources_multi_vector(query, per_provider, anchor=anchor)
    except (ExaNotConfiguredError, ValueError) as exc:
        trace(f"LECTURE_EXA skip | {exc}")
        return []
    except Exception as exc:
        trace(f"LECTURE_EXA skip | {exc}")
        return []


async def _fetch_verified_external_sources_impl(
    node: NodeDataInput,
    subtopic: str,
    curriculum_id: str = "",
    *,
    query_override: str = "",
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    trace("LECTURE_SEARCH ▶ fetch_verified_external_sources (waterfall)")
    if not LECTURE_EXTERNAL_SEARCH_ENABLED:
        return []
    node = enrich_node_learning_materials_from_graph(node, curriculum_id)
    query = build_external_search_query(node, subtopic, query_override=query_override)
    if not query:
        return []
    cap = external_source_limit(top_k)
    per_provider = max(2, cap)

    # --- Step 1: Exa (early exit) ---
    exa_batch = await _exa_batch(query, per_provider, anchor=f"lecture:{node.node_id}")
    if len(exa_batch) >= cap:
        merged = _merge_sources([exa_batch], cap)
        logger.info(
            "Exa satisfied external source limit (%d/%d). Skipping Consensus/SS.",
            len(exa_batch),
            cap,
        )
        trace(
            f"LECTURE_SEARCH ✓ early_exit exa | query_len={len(query)} "
            f"sources={len(merged)} exa={len(exa_batch)}"
        )
        return merged

    # --- Step 2: Academic fallback (English query) ---
    focus_for_academic = (query_override or subtopic or node.title or query).strip()
    academic_query = await asyncio.to_thread(translate_to_en_query, focus_for_academic)
    if not academic_query:
        academic_query = query
    need = max(1, cap - len(exa_batch))
    trace(
        f"LECTURE_SEARCH ▶ academic fallback | need={need} "
        f"exa={len(exa_batch)} q_en_len={len(academic_query)}"
    )

    async def _provider_batch(name: str, limit: int) -> list[VerifiedExternalSource]:
        try:
            return await _provider_sources(name, academic_query, limit)
        except Exception as exc:
            trace(f"LECTURE_SEARCH {name} skip | {exc}")
            return []

    ss_batch, consensus_batch = await asyncio.gather(
        _provider_batch("semantic_scholar", need),
        _provider_batch("consensus", need),
    )
    merged = _merge_sources([exa_batch, ss_batch, consensus_batch], cap)
    trace(
        f"LECTURE_SEARCH ✓ waterfall | query_len={len(query)} sources={len(merged)} "
        f"exa={len(exa_batch)} ss={len(ss_batch)} cons={len(consensus_batch)}"
    )
    return merged


async def persist_verified_external_sources_to_node(
    curriculum_id: str,
    node: NodeDataInput,
    sources: list[VerifiedExternalSource],
) -> int:
    """Сохраняет найденные на лекции VERIFIED_EXTERNAL_SOURCES как материалы
    ноды — иначе fetch_verified_external_sources ничего не оставляет после
    себя, и КАЖДЫЙ повторный запрос лекции по этой же ноде заново гоняет весь
    Exa/Gemini waterfall с нуля. Источники с пустым/слишком коротким
    сниппетом (< 24 симв.) пропускаются — иначе воспроизводим тот же баг
    stub-материалов, который уже чинили (see: изолированные R1/R2/R3,
    cos=0.000).

    Два слоя привязки, оба обязательны:
    1. document_summaries (VectorStore) + node.resource_urls — чтобы
       Retrieval нашёл материал повторно (см. _lecture_node_needs_retrieval).
    2. curriculum_sources_registry + node.mapped_source_ids — БЕЗ этого
       tutor_source_citations.coerce_references_to_registry видит пустой
       registry и отбрасывает ВСЕ references/used_sources безусловно (см.
       ``if not registry: return []``): лекция цитирует голыми [n] вместо
       [Sn], в правой панели и под ответом — пусто, даже если текст лекции
       реально содержит контент из этих источников (см. разбор реального
       бага: curriculum=indexes_and_data_structures node=b_tree_indexes —
       resource_urls сохранились, но mapped_source_ids остался пустым)."""
    cid = (curriculum_id or "").strip()
    valid = [
        s
        for s in sources
        if (s.url or "").strip().startswith("http")
        and len((s.snippet or "").strip()) >= 24
    ]
    if not cid or not valid:
        return 0

    from knowledge_engine.schemas import DocumentSummary
    from knowledge_engine.services.vector_store import VectorStore

    store = VectorStore()
    saved: list[VerifiedExternalSource] = []
    for s in valid:
        url = s.url.strip()
        ds = DocumentSummary(
            title=(s.title or url).strip()[:400],
            url=url,
            key_takeaways=[s.snippet.strip()[:1200]],
            failure_modes=[],
            cs_concepts=[],
            diagram_descriptions=[],
        )
        try:
            ok = await store.save_summary(ds, skip_rag_ingest=True)
        except Exception as exc:
            trace(f"LECTURE_SEARCH persist ✗ | {url[:60]} | {exc}")
            continue
        if ok:
            saved.append(s)

    if saved:
        await asyncio.to_thread(_attach_sources_to_node_graph, cid, node.node_id, saved)
        trace(
            f"LECTURE_SEARCH persist ✓ | node={node.node_id} "
            f"saved={len(saved)}/{len(valid)} sources"
        )
    return len(saved)


def _registry_entry_dict(source: VerifiedExternalSource) -> dict[str, str]:
    url = source.url.strip()
    return {
        "title": (source.title or url)[:400],
        "whitelist_domain": "",
        "source_type": "verified_external",
        "url": url[:2000],
        "why_read": (source.snippet or "").strip()[:800],
        "snippet": (source.snippet or "").strip()[:1200],
        "key_extracts": [],
        "source_tier": (source.provider or "exa")[:24],
    }


def _attach_sources_to_node_graph(
    curriculum_id: str, node_id: str, sources: list[VerifiedExternalSource]
) -> None:
    from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS
    from knowledge_engine.services.skill_tree_store import (
        get_curriculum_graph,
        patch_curriculum_graph_node,
        patch_curriculum_sources_registry,
    )

    # 1) Регистрируем в curriculum_sources_registry (глобальная библиотека
    # курса) — переиспользует существующий source_id, если URL уже там есть.
    new_ids = patch_curriculum_sources_registry(
        curriculum_id, [_registry_entry_dict(s) for s in sources]
    )
    if not new_ids:
        return

    graph = get_curriculum_graph(curriculum_id) or {}
    existing_mapped: list[str] = []
    existing_urls: list[str] = []
    for raw in graph.get("nodes") or []:
        if str(raw.get("node_id") or "") == node_id:
            existing_mapped = list(raw.get("mapped_source_ids") or [])
            existing_urls = list(raw.get("resource_urls") or [])
            break

    # 2) mapped_source_ids — то, что реально читает tutor_source_citations.
    merged_mapped = list(existing_mapped)
    seen_ids = set(merged_mapped)
    for sid in new_ids:
        if sid not in seen_ids:
            seen_ids.add(sid)
            merged_mapped.append(sid)

    # 3) resource_urls — параллельно, для Retrieval (см. docstring выше).
    merged_urls = list(existing_urls)
    seen_urls = {(u or "").strip().rstrip("/").lower() for u in merged_urls}
    for s in sources:
        key = s.url.strip().rstrip("/").lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        merged_urls.append(s.url.strip())

    # RU: CurriculumNode.mapped_source_ids: max_length=CURRICULUM_DEEP_NODE_
    # MAX_HITS, resource_urls: max_length=12 — те же границы, что и в
    # остальной схеме графа.
    patch_curriculum_graph_node(
        curriculum_id,
        node_id,
        {
            "mapped_source_ids": merged_mapped[:CURRICULUM_DEEP_NODE_MAX_HITS],
            "resource_urls": merged_urls[:12],
        },
    )


def format_verified_external_sources_block(
    sources: list[VerifiedExternalSource],
) -> str:
    if not sources:
        return ""
    lines = [
        "=== ИСТИННЫЕ ПРОВЕРЕННЫЕ ИСТОЧНИКИ (Используй ТОЛЬКО эти ссылки) ===",
        "VERIFIED_EXTERNAL_SOURCES:",
    ]
    for i, src in enumerate(sources, 1):
        snippet = (src.snippet or "").strip().replace("\n", " ")
        if snippet:
            snippet = f'"{snippet[:900]}"'
        else:
            snippet = (
                "(нет сниппета — не выдумывай содержание; упомяни title без ссылки)"
            )
        lines.append(f"- Source [{i}]:")
        lines.append(f"  URL: {src.url}")
        lines.append(f"  Title: {src.title}")
        lines.append(f"  Snippet: {snippet}")
        lines.append("")
    lines.append(
        "В тексте лекции — теги [S1]… по реестру; URL только в JSON used_sources "
        "(копировать из списка выше). Не вставляй http в lecture_body."
    )
    return "\n".join(lines).strip()


def merge_verified_sources(
    existing: list[VerifiedExternalSource],
    new: list[VerifiedExternalSource],
    top_k: int | None = None,
) -> list[VerifiedExternalSource]:
    cap = external_source_limit(top_k)
    return _merge_sources([existing, new], cap)


def collect_lecture_allowed_urls(
    verified: list[VerifiedExternalSource],
    rag_context: str,
    node: NodeDataInput,
    curriculum_id: str = "",
    *,
    skip_graph_enrich: bool = False,
) -> set[str]:
    allowed: set[str] = set()
    for src in verified:
        key = normalize_lecture_url(src.url)
        if key and not is_disallowed_source_url(src.url):
            allowed.add(key)
    allowed |= {
        u
        for u in extract_urls_from_text(rag_context)
        if u and not is_disallowed_source_url(u)
    }
    if not skip_graph_enrich:
        node = enrich_node_learning_materials_from_graph(node, curriculum_id)
    for u in getattr(node, "resource_urls", None) or []:
        raw = str(u)
        key = normalize_lecture_url(raw)
        if key and not is_disallowed_source_url(raw):
            allowed.add(key)
    for lr in getattr(node, "learning_resources", None) or []:
        if isinstance(lr, dict):
            raw = str(lr.get("url") or "")
            key = normalize_lecture_url(raw)
            if key and not is_disallowed_source_url(raw):
                allowed.add(key)
    ref = node.source_ref
    if ref is not None:
        raw = ref.url or ""
        key = normalize_lecture_url(raw)
        if key and not is_disallowed_source_url(raw):
            allowed.add(key)
    return {u for u in allowed if u and not is_disallowed_source_url(u)}


def parse_search_external_materials_request(text: str) -> str | None:
    t = (text or "").strip()
    if not t or len(t) > 900:
        return None
    if "search_external_materials" not in t:
        return None
    m = _SEARCH_TOOL_RE.search(t)
    if m:
        raw = m.group("q").replace('\\"', '"').strip()
        return raw or None
    if t.startswith("{") and t.endswith("}"):
        try:
            data = json.loads(t)
        except json.JSONDecodeError:
            return None
        if str(data.get("action") or "").strip().lower() == "search_external_materials":
            q = str(data.get("query") or "").strip()
            return q or None
    return None


def is_search_tool_only_response(text: str) -> bool:
    t = (text or "").strip()
    if not parse_search_external_materials_request(t):
        return False
    stripped = _SEARCH_TOOL_RE.sub("", t).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            pass
    return len(stripped) < 40
