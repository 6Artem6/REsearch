"""External batch translation for Gate 1 pre-processing — shared by BOTH

HITL funnels that show raw source-language intros to the user before a
gate: Node Grounding Gate (``node_candidate_collection_service.py``, one
node) and Штурвал (``light_discovery_service.py``, whole-course Gate 1).
See docs/STEERING_AND_TOPIC_QNA_ROADMAP.md. Zero VRAM/RAM cost by design:
no local LLM/SLM, no new heavy dependency — one HTTP request to the free,
unofficial Google Translate REST endpoint
(``translate.googleapis.com/translate_a/t``, ``client=gtx``), batched via
repeated ``q=`` params in a single GET. This endpoint is undocumented and
can be rate-limited/blocked without notice (observed from at least one
sandbox network during development) — every failure mode below is
fail-open by design: callers always get back a same-length list, worst
case unchanged from the input.

``needs_ru_translation`` — moved here from
``node_candidate_collection_service.py`` (was private, duplicated once
Штурвал needed the exact same check) so both call sites share one
heuristic instead of two copies drifting apart."""

from __future__ import annotations

import re

import httpx

from knowledge_engine.src.core.run_log import trace

_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/t"
_TRANSLATE_TIMEOUT_SEC = 5.0
_TRANSLATE_MAX_CHARS_PER_TEXT = 5000
_USER_AGENT = "Mozilla/5.0 (compatible; KnowledgeEngineBot/1.0)"

_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def needs_ru_translation(text: str) -> bool:
    """Простая эвристика без новой зависимости (langdetect и т.п.) — снипеты

    короткие (заголовок + 1-2 абзаца/highlight/lead), соотношения латиница
    vs кириллица достаточно для решения "нужен перевод или уже по-русски".
    Русскоязычные источники (Habr в Node Gate, любой русский company hub
    в Штурвале) естественно возвращают False — не нужно хардкодить решение
    по source_tier/hub отдельно, эвристика уже это даёт."""
    t = (text or "").strip()
    if not t:
        return False
    cyr = len(_CYRILLIC_RE.findall(t))
    lat = len(_LATIN_RE.findall(t))
    if cyr + lat == 0:
        return False
    return cyr < lat


def _extract_translations(data: object) -> list[str] | None:
    """``client=gtx``'s ``/translate_a/t`` returns a flat JSON array of

    translated strings when multiple ``q=`` are sent — tolerate the
    single-element-list-per-item variant too ([["t1"], ["t2"]]), observed
    from some proxies/regions. Anything else (including the HTML
    "automated queries" block Google serves when it rate-limits a caller)
    fails the ``isinstance(data, list)`` check and returns ``None``."""
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, list) and item and isinstance(item[0], str):
            out.append(item[0])
        else:
            return None
    return out


async def translate_batch_to_russian(texts: list[str]) -> list[str]:
    """[текст1, текст2, ...] -> [перевод1, перевод2, ...] — тот же порядок,

    ОДИН HTTP-запрос на весь батч (несколько ``q=`` в одном GET; Google это
    официально не документирует, но приём давно и широко используется).

    Fail-open на каждом шаге (сеть недоступна/таймаут/неожиданный формат
    ответа/несовпадение длины) — возвращает ИСХОДНЫЕ тексты без изменений,
    никогда не бросает исключение наружу. Пустые строки во входе не
    отправляются в запрос (нечего переводить) и возвращаются как есть."""
    cleaned = [(t or "") for t in texts]
    non_empty_idx = [i for i, t in enumerate(cleaned) if t.strip()]
    if not non_empty_idx:
        return cleaned

    params = [("client", "gtx"), ("sl", "auto"), ("tl", "ru")]
    params.extend(
        ("q", cleaned[i][:_TRANSLATE_MAX_CHARS_PER_TEXT]) for i in non_empty_idx
    )

    try:
        async with httpx.AsyncClient(timeout=_TRANSLATE_TIMEOUT_SEC) as client:
            resp = await client.get(
                _TRANSLATE_URL,
                params=params,
                headers={"User-Agent": _USER_AGENT},
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        trace(f"TRANSLATE batch ✗ | fail-open, original text kept | {exc}")
        return cleaned

    translated = _extract_translations(data)
    if translated is None or len(translated) != len(non_empty_idx):
        trace(
            "TRANSLATE batch ✗ | unexpected response shape, fail-open | "
            f"got={type(data).__name__} n={len(translated) if translated is not None else '?'} "
            f"expected={len(non_empty_idx)}"
        )
        return cleaned

    out = list(cleaned)
    for pos, idx in enumerate(non_empty_idx):
        item = translated[pos]
        if isinstance(item, str) and item.strip():
            out[idx] = item
    trace(f"TRANSLATE batch ✓ | n={len(non_empty_idx)}")
    return out


__all__ = ["translate_batch_to_russian", "needs_ru_translation"]
