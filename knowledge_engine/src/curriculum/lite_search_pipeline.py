"""Gemini Lite: генерация site: запросов и пакетная фильтрация search hits."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from knowledge_engine.config import (
    CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N,
    CURRICULUM_LITE_BATCH_STRICT,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    ArxivQueryParamsContract,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteAcademicQueryContract as LiteAcademicQueryOut,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteBatchEvalContract as LiteBatchEvalResult,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteHitEvaluationContract as LiteHitEvaluation,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteQueryPlanContract as LiteQueryPlan,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteSourceBatchContract as LiteSourceBatchResult,
)
from knowledge_engine.schemas.llm_contracts.lite_curriculum import (
    LiteSourceEvalItemContract as LiteSourceEvalItem,
)
from knowledge_engine.services.search.arxiv_query_builder import (
    ArxivQueryParams,
    heuristic_arxiv_params_from_keywords,
)
from knowledge_engine.src.curriculum.search_query_builder import (
    build_fallback_quote_queries,
)
from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST
from knowledge_engine.ui.run_log import trace

_QUERY_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Search Query Architect. Тебе дана цель обучения и список разрешённых доменов.\n"
    "На основе своих знаний о содержании этих сайтов выбери 3–5 наиболее подходящих под тему.\n"
    "Сгенерируй короткие (3–5 слов) поисковые запросы для SearXNG, используя оператор "
    "`site:<domain>` для выбранных сайтов, а также 1–2 общих запроса с точными терминами "
    "в кавычках.\n"
    "Отвечай СТРОГО в формате JSON.\n"
    "Схема: selected_domains (string[]), queries (string[], 4–8 элементов)."
)

_BATCH_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Content Quality Gate. Оцени список найденного контента на соответствие учебной цели.\n"
    "Оцени по `title` и `snippet`, содержит ли источник достаточно контекста, релевантен ли "
    "он цели и отсутствует ли в нём SEO-мусор/оффтоп.\n"
    "Отвечай СТРОГО в формате JSON.\n"
    "Схема: evaluations — массив { id (int), is_sufficient (bool), confidence (0–1), "
    "reason (string, русский) } для каждого id из входа."
)


_ACADEMIC_QUERY_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "You are an Academic Query Architect for Semantic Scholar / arXiv.\n"
    "In ONE pass, produce:\n"
    "1) academic_query_en — precise English literature query (1–2 sentences) with "
    "concrete technical terms (asyncio, RabbitMQ, transformers, …). "
    "Do NOT use a single generic word (python, programming). Do NOT change the topic.\n"
    "2) arxiv_params — structured arXiv Atom fields for the SAME topic:\n"
    "   - title_keywords: 1–4 short English phrases for ti:\n"
    "   - abstract_keywords: 2–6 terms/phrases for abs:\n"
    "   - categories: arXiv cats for the node topic (e.g. cs.AI, cs.CL, cs.LG, "
    "stat.ML, cs.DC, cs.SE). Use [] if unsure.\n"
    "   - exclude_terms: noise to ANDNOT (survey, homework, tutorial) when helpful\n"
    "   - start_year / end_year: optional ints for submittedDate window, else null\n"
    "3) notes — brief Russian note about what you translated/kept.\n"
    "JSON keys: academic_query_en, notes, arxiv_params."
)


def _anchor_academic(goal: str) -> str:
    return f"curriculum_lite_academic_query:{(goal or '').strip()[:500]}"


@dataclass(frozen=True)
class AcademicSearchPlan:
    academic_query_en: str
    arxiv_params: ArxivQueryParams
    notes: str = ""


def _plan_from_contract(
    out: LiteAcademicQueryOut,
    *,
    fallback_goal: str,
) -> AcademicSearchPlan:
    q = (out.academic_query_en or "").strip()[:300]
    params = ArxivQueryParams.from_mapping(
        out.arxiv_params or ArxivQueryParamsContract()
    )
    if not params.has_precision():
        from knowledge_engine.src.curriculum.search_query_builder import (
            build_search_queries,
        )

        built = build_search_queries(fallback_goal or q)
        params = heuristic_arxiv_params_from_keywords(
            built.keywords,
            free_text=q or fallback_goal,
        )
    return AcademicSearchPlan(
        academic_query_en=q,
        arxiv_params=params,
        notes=(out.notes or "").strip(),
    )


def _heuristic_academic_plan(goal: str) -> AcademicSearchPlan:
    from knowledge_engine.src.curriculum.search_query_builder import (
        build_search_queries,
    )

    built = build_search_queries(goal)
    q = (built.academic_query or "").strip()
    if len(q) < 8:
        q = goal[:120]
    params = heuristic_arxiv_params_from_keywords(built.keywords, free_text=q or goal)
    return AcademicSearchPlan(academic_query_en=q[:300], arxiv_params=params)


async def build_academic_search_plan(
    learning_goal: str,
    *,
    anchor: str | None = None,
) -> AcademicSearchPlan:
    """One Lite pass → English query + structured arXiv params."""
    goal = (learning_goal or "").strip()
    if len(goal) < 4:
        return AcademicSearchPlan(
            academic_query_en="",
            arxiv_params=ArxivQueryParams(),
        )
    trace("CURRICULUM academic query ▶ | Lite Academic Query Architect")
    try:
        out = await _lite_structured(
            _ACADEMIC_QUERY_SYSTEM,
            json.dumps({"learning_goal": goal[:1200]}, ensure_ascii=False),
            anchor or _anchor_academic(goal),
            LiteAcademicQueryOut,
            "curriculum / lite_academic_query",
        )
        plan = _plan_from_contract(out, fallback_goal=goal)
        q = plan.academic_query_en
        if len(q) < 12 or q.lower().split() == ["python"]:
            raise ValueError(f"weak academic query: {q[:80]}")
        trace(
            f"CURRICULUM academic query ✓ | Lite | {q[:120]} | "
            f"arxiv_cats={plan.arxiv_params.categories[:4]}"
        )
        return plan
    except Exception as exc:
        trace(f"CURRICULUM academic query fallback | {exc}")
        plan = _heuristic_academic_plan(goal)
        trace(
            f"CURRICULUM academic query fallback ✓ | heuristic | "
            f"{plan.academic_query_en[:120]}"
        )
        return plan


async def build_academic_search_query(
    learning_goal: str,
    *,
    anchor: str | None = None,
) -> str:
    plan = await build_academic_search_plan(learning_goal, anchor=anchor)
    return plan.academic_query_en


def build_academic_search_query_sync(
    learning_goal: str, *, anchor: str | None = None
) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build_academic_search_query(learning_goal, anchor=anchor))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            build_academic_search_query(learning_goal, anchor=anchor),
        ).result()


_SOURCE_BATCH_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n\n"
    "Ты — Source Evaluator (пакетный режим). Оцени каждый источник.\n"
    "Поля title/snippet; snippet может содержать «Тезис:» для проверки цитаты в Reasoner.\n"
    "Instant APPROVED для доменов из whitelist (engineering depth).\n"
    "REJECTED: SEO-мусор, оффтоп, источник не подтверждает тезис/цель.\n"
    "JSON: evaluations[] — id, status (APPROVED|REJECTED), confidence, reason (русский), "
    "suggested_action (RETRY_WITH_NEW_SOURCE|REMOVE_LINK|KEEP)."
)


def flatten_whitelist_domains() -> list[str]:
    """Все домены/пути из статического whitelist (без ручного БД)."""
    seen: set[str] = set()
    out: list[str] = []
    for entries in APPROVED_SOURCES_WHITELIST.values():
        for raw in entries:
            d = (raw or "").strip().lower()
            if not d or d in seen:
                continue
            seen.add(d)
            out.append(d)
    return out


def _anchor_queries(goal: str) -> str:
    return f"curriculum_lite_queries:{(goal or '').strip()[:500]}"


def _anchor_batch(goal: str) -> str:
    return f"curriculum_lite_batch:{(goal or '').strip()[:500]}"


async def _lite_structured(
    system: str,
    user_payload: str,
    anchor: str,
    schema: type[BaseModel],
    label: str,
) -> BaseModel:
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    return await asyncio.to_thread(
        run_gemini_lite_structured,
        system,
        user_payload,
        anchor,
        schema,
        label,
    )


async def build_search_queries(
    learning_goal: str,
    whitelist_domains: list[str],
    *,
    anchor: str | None = None,
) -> LiteQueryPlan:
    """
    Lite выбирает 3–5 доменов из whitelist и формирует запросы для SearXNG.
    При ошибке / невалидном JSON — fallback на кавычечные термы из цели.
    """
    goal = (learning_goal or "").strip()
    domains = [d.strip().lower() for d in whitelist_domains if (d or "").strip()]
    if len(goal) < 4:
        return LiteQueryPlan(
            queries=build_fallback_quote_queries(goal),
            selected_domains=domains[:3],
        )

    user_obj = {
        "learning_goal": goal[:1200],
        "whitelist_domains": domains[:80],
    }
    user_payload = json.dumps(user_obj, ensure_ascii=False)
    trace("CURRICULUM lite queries ▶ | Lite Search Query Architect")

    try:
        out = await _lite_structured(
            _QUERY_SYSTEM,
            user_payload,
            anchor or _anchor_queries(goal),
            LiteQueryPlan,
            "curriculum / lite_search_queries",
        )
        queries = [q.strip() for q in (out.queries or []) if (q or "").strip()]
        selected = [d.strip().lower() for d in (out.selected_domains or []) if d]
        if len(queries) < 2:
            raise ValueError("lite returned too few queries")
        trace(
            f"CURRICULUM lite queries ✓ | domains={selected[:5]} "
            f"queries={len(queries)}"
        )
        return LiteQueryPlan(selected_domains=selected[:5], queries=queries[:10])
    except Exception as exc:
        trace(f"CURRICULUM lite queries fallback | {exc}")
        fb = build_fallback_quote_queries(goal)
        return LiteQueryPlan(
            selected_domains=domains[:5],
            queries=fb,
        )


def _hits_to_batch_input(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, h in enumerate(raw_hits, start=1):
        hid = h.get("id")
        if hid is None:
            hid = i
        out.append(
            {
                "id": int(hid),
                "url": str(h.get("url") or "")[:500],
                "title": str(h.get("title") or "")[:400],
                "snippet": str(h.get("snippet") or "")[:1200],
            }
        )
    return out


async def batch_lite_eval_hits(
    learning_goal: str,
    raw_hits: list[dict[str, Any]],
    *,
    anchor: str | None = None,
    fallback_approve_n: int | None = None,
    strict: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Один вызов Lite на все hits. Возвращает подмножество raw_hits (approved).
    strict=True (default из CURRICULUM_LITE_BATCH_STRICT): без аварийного approve.
    """
    goal = (learning_goal or "").strip()
    if not raw_hits:
        return []

    use_strict = strict if strict is not None else CURRICULUM_LITE_BATCH_STRICT
    batch_input = _hits_to_batch_input(raw_hits)
    n_fb = (
        fallback_approve_n
        if fallback_approve_n is not None
        else (CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N)
    )

    if len(goal) < 4:
        if use_strict:
            return []
        return list(raw_hits[: max(1, n_fb)])

    user_obj = {"learning_goal": goal[:1200], "raw_hits": batch_input}
    user_payload = json.dumps(user_obj, ensure_ascii=False)
    trace(f"CURRICULUM lite batch eval ▶ | hits={len(batch_input)}")

    by_id: dict[int, LiteHitEvaluation] = {}
    try:
        out = await _lite_structured(
            _BATCH_SYSTEM,
            user_payload,
            anchor or _anchor_batch(goal),
            LiteBatchEvalResult,
            "curriculum / lite_batch_eval_hits",
        )
        for ev in out.evaluations or []:
            by_id[int(ev.id)] = ev
    except Exception as exc:
        if use_strict:
            trace(f"CURRICULUM lite batch eval strict ⊘ | {exc}")
            return []
        trace(f"CURRICULUM lite batch eval fallback | approve first {n_fb} | {exc}")
        return list(raw_hits[: max(1, n_fb)])

    if not by_id:
        if use_strict:
            trace("CURRICULUM lite batch eval strict ⊘ | empty evaluations")
            return []
        trace(f"CURRICULUM lite batch eval empty ⊘ | approve first {n_fb}")
        return list(raw_hits[: max(1, n_fb)])

    approved: list[dict[str, Any]] = []
    rejected = 0
    for idx, h in enumerate(raw_hits):
        ev_id = int(h.get("id") or idx + 1)
        ev = by_id.get(ev_id)
        if ev and ev.is_sufficient:
            approved.append(h)
        else:
            rejected += 1
            reason = (ev.reason if ev else "no evaluation")[:80]
            url = str(h.get("url") or "")[:60]
            trace(f"CURRICULUM lite batch ⊘ | {url} | {reason}")

    trace(
        f"CURRICULUM lite batch eval ✓ | in={len(raw_hits)} "
        f"approved={len(approved)} rejected={rejected}"
    )
    if not approved and raw_hits and by_id and rejected >= len(raw_hits):
        trace(
            "CURRICULUM lite batch ⊘ | все hits отклонены Lite — "
            "без fallback approve"
        )
        return []
    if not approved and raw_hits:
        if use_strict:
            trace("CURRICULUM lite batch strict ⊘ | approved=0")
            return []
        trace(
            f"CURRICULUM lite batch fallback | approve first {n_fb} "
            "(Lite не дал evaluations)"
        )
        return list(raw_hits[: max(1, n_fb)])
    return approved


def _instant_whitelist_approved(url: str, learning_goal: str) -> bool:
    from knowledge_engine.src.source_evaluator.evaluator import match_whitelist

    matched, category = match_whitelist(url)
    if not matched:
        return False
    from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
        register_curriculum_source,
    )

    register_curriculum_source(
        url,
        (learning_goal or "").strip()[:800],
        category=category,
        trust_score=0.92,
        status="accepted",
        reason="static_whitelist",
    )
    return True


async def batch_evaluate_sources(
    learning_goal: str,
    sources: list[dict[str, Any]],
    *,
    anchor: str | None = None,
    fallback_approve_n: int | None = None,
) -> list[LiteSourceEvalItem]:
    """Пакетная оценка источников (Re-Act, strict archive). Whitelist — без Lite."""
    goal = (learning_goal or "").strip()
    if not sources:
        return []

    n_fb = (
        fallback_approve_n
        if fallback_approve_n is not None
        else (CURRICULUM_LITE_BATCH_EVAL_FALLBACK_N)
    )

    results: list[LiteSourceEvalItem] = []
    need_lite: list[dict[str, Any]] = []

    for idx, src in enumerate(sources, start=1):
        sid = int(src.get("id") or idx)
        url = str(src.get("url") or "").strip()
        if url and _instant_whitelist_approved(url, goal):
            results.append(
                LiteSourceEvalItem(
                    id=sid,
                    status="APPROVED",
                    confidence=1.0,
                    reason="whitelist instant pass",
                    suggested_action="KEEP",
                )
            )
            continue
        need_lite.append(
            {
                "id": sid,
                "url": url[:500],
                "title": str(src.get("title") or "")[:400],
                "snippet": str(src.get("snippet") or "")[:1200],
            }
        )

    if not need_lite:
        return results

    user_obj = {
        "learning_goal": goal[:1200] if goal else "(оценка тезис+источник из snippet)",
        "raw_hits": need_lite,
    }
    user_payload = json.dumps(user_obj, ensure_ascii=False)
    trace(f"SOURCE_EVAL batch ▶ Lite | count={len(need_lite)}")

    by_id: dict[int, LiteSourceEvalItem] = {}
    try:
        out = await _lite_structured(
            _SOURCE_BATCH_SYSTEM,
            user_payload,
            anchor or _anchor_batch(goal or "source_eval_batch"),
            LiteSourceBatchResult,
            "source_evaluator / batch_evaluate_sources",
        )
        for ev in out.evaluations or []:
            by_id[int(ev.id)] = ev
    except Exception as exc:
        trace(f"SOURCE_EVAL batch fallback | approve first {n_fb} | {exc}")
        for src in need_lite[: max(1, n_fb)]:
            by_id[int(src["id"])] = LiteSourceEvalItem(
                id=int(src["id"]),
                status="APPROVED",
                confidence=0.5,
                reason="batch fallback approve",
                suggested_action="KEEP",
            )

    if not by_id:
        for src in need_lite[: max(1, n_fb)]:
            by_id[int(src["id"])] = LiteSourceEvalItem(
                id=int(src["id"]),
                status="APPROVED",
                confidence=0.5,
                reason="empty batch fallback",
                suggested_action="KEEP",
            )

    for src in need_lite:
        sid = int(src["id"])
        ev = by_id.get(sid)
        if ev is None:
            results.append(
                LiteSourceEvalItem(
                    id=sid,
                    status="REJECTED",
                    reason="нет оценки в batch",
                    suggested_action="RETRY_WITH_NEW_SOURCE",
                )
            )
        else:
            results.append(ev)
            if ev.status == "APPROVED" and src.get("url"):
                from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
                    register_curriculum_source,
                )

                register_curriculum_source(
                    str(src["url"]),
                    goal[:800],
                    category="lite_approved",
                    trust_score=min(0.95, float(ev.confidence or 0.86)),
                    reason=(ev.reason or "")[:400],
                )

    trace(
        f"SOURCE_EVAL batch ✓ | approved="
        f"{sum(1 for e in results if e.status == 'APPROVED')} "
        f"rejected={sum(1 for e in results if e.status == 'REJECTED')}"
    )
    return results


def batch_evaluate_sources_sync(
    learning_goal: str,
    sources: list[dict[str, Any]],
    *,
    anchor: str | None = None,
) -> list[LiteSourceEvalItem]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            batch_evaluate_sources(learning_goal, sources, anchor=anchor)
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            batch_evaluate_sources(learning_goal, sources, anchor=anchor),
        ).result()


async def batch_lite_eval_curriculum_hits(
    hits: list[Any],
    learning_goal: str,
    *,
    anchor: str | None = None,
    strict: bool | None = None,
) -> list[Any]:
    """Обертка для CurriculumSearchHit → batch_lite_eval_hits."""
    from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit

    raw: list[dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        if isinstance(h, CurriculumSearchHit):
            raw.append(
                {
                    "id": i,
                    "url": h.url,
                    "title": h.title,
                    "snippet": h.snippet or "",
                    "_hit": h,
                }
            )
        else:
            raw.append(dict(h))

    slim = [
        {
            "id": r["id"],
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
        }
        for r in raw
    ]
    approved_slim = await batch_lite_eval_hits(
        learning_goal,
        slim,
        anchor=anchor,
        strict=strict,
    )
    approved_urls = {str(x.get("url") or "").strip().lower() for x in approved_slim}
    out: list[Any] = []
    for r in raw:
        u = str(r.get("url") or "").strip().lower()
        if u in approved_urls:
            hit = r.get("_hit")
            out.append(hit if hit is not None else r)
    return out
