"""NodeDigestService — Node Grounding Gate, Этап 2: BGE-M3 + Cross-Encoder

top-K смысловых блоков из полного текста ≤NODE_GATE1_MAX_APPROVED
Gate-1-одобренных статей → один батч-вызов Gemma cloud, точечный дайджест
(Архитектура / Практический кейс / Ограничения) на статью — для Gate 2.
Add-only, параллельно ``batch_digest_service.py`` (тот — Штурвал, целый
курс, Flash Lite, без chunk-релевантности; этот — одна нода, Gemma, с
BGE-M3/Cross-Encoder отбором чанков, как явно указано в задании).

Переиспользуется как есть (ничего не дублируется):
- ``fetch_html``/``_extract_paragraphs`` — тот же fetch-путь, что Gate 1
  (``node_candidate_collection_service.py``) и Штурвал
  (``batch_digest_service.py``).
- ``embed_texts_bge_m3`` (``services/search/bge_m3_embed.py``) — грубый
  косинус-отбор ДО дорогого реранкера (векторы уже L2-нормированы, косинус
  = скалярное произведение).
- ``score_relevance_pairs`` (``src/rag_gateway/cross_encoder.py``) — тот же
  локальный Cross-Encoder (``BAAI/bge-reranker-v2-m3``), что Directional RAG
  Gateway (docs/RAG_GATEWAY_MODULE_3.md), детерминированный, без LLM.
- ``GemmaCloudClient.complete_structured`` (``services/llm/gemma_client.py``)
  — тот же примитив одного structured-вызова Gemma cloud, что использует
  ``blog_spatial_summarizer.py`` для MAP/REDUCE окон, но здесь — РОВНО ОДИН
  вызов на весь батч ≤4 статей, без Map-Reduce и без записи в LanceDB/registry
  (это Этап 3, после Gate 2 — сюда не входит).
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from pydantic import BaseModel, Field

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.adapters.llm_providers.gemma_client import GemmaCloudClient
from knowledge_engine.src.config.settings import (
    LECTURE_PASSAGE_FETCH_CONCURRENCY,
    LECTURE_PASSAGE_MIN_CHARS,
    STEERING_DIGEST_FETCH_TIMEOUT_SEC,
)
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.pre_flight_triage import (
    _extract_paragraphs,
)
from knowledge_engine.src.domains.grounding.lecture_passage_fetch import fetch_html
from knowledge_engine.src.domains.steering.services.light_discovery_service import (
    _title_from_html,
)
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.src.shared.ml_runtime.bge_m3_embed import embed_texts_bge_m3
from knowledge_engine.src.shared.node_grounding.node_gate_contracts import (
    NODE_GATE2_COARSE_K,
    NODE_GATE2_TOP_K_CHUNKS,
    NodeDigestItem,
    NodeGate2DigestResponse,
)

_MAX_CHARS_PER_CHUNK = 900
_MAX_CHARS_PER_ARTICLE_PROMPT = 5400  # NODE_GATE2_TOP_K_CHUNKS * _MAX_CHARS_PER_CHUNK

_DIGEST_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are a technical digest editor. You receive, for each of several "
    "engineering/academic articles, ONLY the most relevant excerpts already "
    "pre-selected for a specific curriculum node topic (not the full article "
    "— irrelevant parts were already filtered out). For EACH article, "
    "produce a dry, purely technical point digest with exactly three fields:\n"
    "- architecture: the architecture/approach the excerpts describe "
    "(Russian).\n"
    "- practical_case: the practical case/result of applying it, per the "
    "excerpts (Russian).\n"
    "- limitations: limitations/caveats/applicability conditions mentioned "
    "in the excerpts (if the excerpts do not mention any, say so briefly — "
    "do not invent limitations) (Russian).\n\n"
    "STRIP all narrative/marketing framing. Base every field ONLY on the "
    "given excerpts, do not use outside knowledge.\n\n"
    "Return JSON matching _NodeDigestBatch: exactly one item per article "
    "provided below (skip none, do not invent extra ones):\n"
    "- url: copy verbatim from the provided url.\n"
    "- title: copy verbatim from the provided title."
)


class _NodeDigestBatch(BaseModel):
    items: list[NodeDigestItem] = Field(default_factory=list)


def _anchor_digest(urls: list[str]) -> str:
    return "node_gate_digest:" + ",".join(sorted(urls))[:400]


async def _fetch_article(url: str) -> tuple[str, list[str]] | None:
    """url → (title, абзацы) | None при неудаче fetch/extract.

    Один retry после паузы — см. batch_digest_service.py::_fetch_article
    (тот же root cause: некоторые внешние CDN отдают нестабильный 403 под
    конкурентной нагрузкой, тот же URL секундами позже открывается
    нормально; статья уже approved пользователем на Gate 1 — retry того
    стоит)."""
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
    return title, paragraphs


def _cosine_rank(criterion_vec: list[float], vecs: list[list[float]]) -> list[float]:
    qv = np.asarray(criterion_vec, dtype=np.float64)
    return [float(np.dot(qv, np.asarray(v, dtype=np.float64))) for v in vecs]


async def _top_k_chunks(
    paragraphs: list[str],
    criterion: str,
    *,
    coarse_k: int = NODE_GATE2_COARSE_K,
    top_k: int = NODE_GATE2_TOP_K_CHUNKS,
) -> list[str]:
    """BGE-M3 грубый косинус-отбор → Cross-Encoder точная top-K реранжировка.

    Fail-open на каждом шаге: сбой эмбеддинга — пропускаем грубый отбор
    (реранкер работает на всём пуле, максимум ``coarse_k`` абзацев); сбой
    реранкера — берём первые ``top_k`` абзацев без скоринга (в целом
    относительно безопасно — большинство статей начинают с сути)."""
    cleaned = [p.strip() for p in paragraphs if (p or "").strip()]
    if not cleaned:
        return []
    if len(cleaned) <= top_k:
        return cleaned

    pool = cleaned
    if len(cleaned) > coarse_k:
        try:
            vecs = await asyncio.to_thread(embed_texts_bge_m3, [criterion, *cleaned])
            if len(vecs) == len(cleaned) + 1:
                sims = _cosine_rank(vecs[0], vecs[1:])
                ranked = sorted(zip(cleaned, sims), key=lambda t: t[1], reverse=True)
                pool = [c for c, _ in ranked[:coarse_k]]
            else:
                pool = cleaned[:coarse_k]
        except Exception as exc:
            trace(f"NODE_GATE digest bge_m3 fallback | skip coarse filter | {exc}")
            pool = cleaned[:coarse_k]

    try:
        scores = await asyncio.to_thread(score_relevance_pairs, criterion, pool)
        ranked = sorted(zip(pool, scores), key=lambda t: t[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]
    except Exception as exc:
        trace(f"NODE_GATE digest cross_encoder fallback | first-{top_k} | {exc}")
        return pool[:top_k]


async def generate_node_digests(
    approved_urls: list[str],
    *,
    node_title: str,
    node_summary: str = "",
    curriculum_id: str = "",
    node_id: str = "",
    concurrency: int | None = None,
    anchor: str | None = None,
) -> NodeGate2DigestResponse:
    """Gate 1 ``approved_urls`` (≤NODE_GATE1_MAX_APPROVED) → ``NodeGate2DigestResponse``.

    Fail-open на каждом уровне: URL, которые не удалось скачать/распарсить,
    просто выпадают из батча; сбой самого Gemma-вызова — пустой ответ (без
    исключения наружу), Gate 2 в UI покажет "дайджест недоступен"."""
    urls = [u.strip() for u in approved_urls if (u or "").strip()]
    if not urls:
        return NodeGate2DigestResponse(curriculum_id=curriculum_id, node_id=node_id)

    criterion = f"{(node_title or '').strip()}. {(node_summary or '').strip()}".strip(
        ". "
    )

    conc = max(
        1, concurrency if concurrency is not None else LECTURE_PASSAGE_FETCH_CONCURRENCY
    )
    sem = asyncio.Semaphore(conc)

    async def _bounded(url: str) -> tuple[str, tuple[str, list[str]] | None]:
        async with sem:
            return url, await _fetch_article(url)

    fetched = await asyncio.gather(*[_bounded(u) for u in urls])

    articles: list[dict[str, str]] = []
    for url, result in fetched:
        if result is None:
            continue
        title, paragraphs = result
        top_chunks = await _top_k_chunks(paragraphs, criterion or title)
        if not top_chunks:
            continue
        text = "\n\n".join(c[:_MAX_CHARS_PER_CHUNK] for c in top_chunks)
        articles.append(
            {
                "url": url,
                "title": title,
                "excerpts": text[:_MAX_CHARS_PER_ARTICLE_PROMPT],
            }
        )

    trace(
        f"NODE_GATE digest fetch+rank ✓ | approved={len(urls)} with_chunks={len(articles)}"
    )
    if not articles:
        return NodeGate2DigestResponse(curriculum_id=curriculum_id, node_id=node_id)

    client = GemmaCloudClient()
    prompt = json.dumps(
        {"node_topic": criterion, "articles": articles}, ensure_ascii=False
    )
    try:
        out = await client.complete_structured(
            _DIGEST_SYSTEM,
            prompt,
            _NodeDigestBatch,
            label="curriculum/node_gate_digest",
        )
    except Exception as exc:
        trace(f"NODE_GATE digest ✗ | Gemma call raised | {exc}")
        out = None

    if out is None:
        trace("NODE_GATE digest ✗ | Gemma returned None (fail-open, empty digests)")
        return NodeGate2DigestResponse(curriculum_id=curriculum_id, node_id=node_id)

    known_urls = {a["url"] for a in articles}
    digests = [item for item in out.items if item.url in known_urls]
    trace(
        f"NODE_GATE digest ✓ | node={node_id} articles={len(articles)} "
        f"digests={len(digests)}"
    )
    return NodeGate2DigestResponse(
        curriculum_id=curriculum_id, node_id=node_id, digests=digests
    )


__all__ = ["generate_node_digests"]
