"""Re-Act аудит источников в ответе Reasoner (Gemini Lite)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.ui.run_log import trace

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

SOURCE_EVALUATOR_PROMPT = """Ты — Строгий Аудитор Технической Литературы.
Твоя задача — проверить источник (ссылку, блог, авторов), который предлагает использовать модель-архитектор для подтверждения факта.

ВХОДНЫЕ ДАННЫЕ:
- Утверждение/Тезис: {statement}
- Предложенный источник/ссылка: {source_info}

КРИТЕРИИ ОЦЕНКИ:
1. **Авторитетность (Authority)**: Входит ли источник в белый список (Microsoft Learn `learn.microsoft.com`, Cloudflare Blog `blog.cloudflare.com`, MDN `developer.mozilla.org`, Martin Fowler `martinfowler.com`, AWS Architecture, Uber/Netflix Eng)? Или это поверхностная статья/SEO-мусор?
2. **Глубина (Technical Depth)**: Содержит ли источник конкретные технические детали, схемы, протоколы или бенчмарки?
3. **Релевантность**: Подтверждает ли источник именно данный тезис?

ФОРМАТ ОТВЕТА (СТРОГО JSON):
Если источник качественный:
{{
  "status": "APPROVED",
  "reason": "Источник авторитетен и содержит глубокий разбор механики."
}}

Если источник слабый или сомнительный:
{{
  "status": "REJECTED",
  "reason": "Источник плоховат: это поверхностный материал без разбора подкапотной механики. Замени на авторитетный блог из белого списка (Cloudflare/Microsoft/MDN/Martin Fowler) или сформулируй ответ на базе фундаментальных принципов CS без этой ссылки."
}}
"""

SOURCE_EVALUATOR_SYSTEM = (
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "Отвечай строго на русском языке в поле reason. "
    "status только APPROVED или REJECTED.\n\n"
    "Ты — Строгий Аудитор Технической Литературы. "
    "Проверяешь источник, который архитектор хочет использовать для подтверждения факта.\n\n"
    "КРИТЕРИИ ОЦЕНКИ:\n"
    "1. Авторитетность: белый список — Microsoft Learn, Cloudflare Blog, MDN, Martin Fowler, "
    "AWS Architecture, Uber/Netflix Eng; не SEO-мусор.\n"
    "2. Глубина: конкретные технические детали, схемы, протоколы, бенчмарки.\n"
    "3. Релевантность: источник подтвержает именно данный тезис.\n"
)

MAX_REACT_SOURCE_ITERATIONS = 2
_MAX_CITATIONS_PER_PASS = 8


class SourceEvaluationResult(BaseModel):
    status: Literal["APPROVED", "REJECTED"]
    reason: str = ""


class SourceCitationCandidate(BaseModel):
    statement: str
    source_info: str
    url: str = ""


def _sentence_with_match(text: str, match_start: int) -> str:
    """Ближайшее предложение вокруг позиции ссылки."""
    if not text:
        return ""
    left = text[:match_start]
    parts_left = _SENTENCE_SPLIT_RE.split(left)
    sentence = parts_left[-1] if parts_left else left
    right = text[match_start:]
    parts_right = _SENTENCE_SPLIT_RE.split(right, maxsplit=1)
    if parts_right:
        sentence += parts_right[0]
    return sentence.strip()[:600]


def extract_source_citation_candidates(
    answer: str,
    registry: list[dict[str, Any]] | None = None,
) -> list[SourceCitationCandidate]:
    """Извлечь пары «тезис + источник» из markdown-ссылок и тегов [Sx]."""
    text = (answer or "").strip()
    if not text:
        return []
    by_id = {
        str(e.get("id") or e.get("source_id") or "").upper(): e
        for e in (registry or [])
        if isinstance(e, dict)
    }
    seen: set[str] = set()
    out: list[SourceCitationCandidate] = []

    for m in _MD_LINK_RE.finditer(text):
        title = m.group(1).strip()
        url = m.group(2).strip()
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        statement = _sentence_with_match(text, m.start())
        if len(statement) < 15:
            statement = f"Утверждение рядом со ссылкой «{title}»."
        out.append(
            SourceCitationCandidate(
                statement=statement,
                source_info=f"{title} | {url}",
                url=url,
            )
        )

    for m in _SOURCE_TAG_RE.finditer(text):
        sid = f"S{m.group(1)}".upper()
        key = f"tag:{sid}"
        if key in seen:
            continue
        entry = by_id.get(sid)
        if not entry:
            continue
        seen.add(key)
        url = str(entry.get("url") or "").strip()
        title = str(entry.get("title") or sid)
        statement = _sentence_with_match(text, m.start())
        if len(statement) < 15:
            statement = f"Тезис с опорой на источник [{sid}]."
        out.append(
            SourceCitationCandidate(
                statement=statement,
                source_info=f"[{sid}] {title} | {url or '—'}",
                url=url,
            )
        )

    return out[:_MAX_CITATIONS_PER_PASS]


def evaluate_source(
    statement: str,
    source_info: str,
    global_anchor: str,
) -> SourceEvaluationResult:
    """Один вызов Gemini Lite — оценка авторитетности источника."""
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    st = (statement or "").strip()[:800]
    info = (source_info or "").strip()[:1200]
    user_payload = (
        f"Утверждение/Тезис:\n{st}\n\n"
        f"Предложенный источник/ссылка:\n{info}\n\n"
        "Верни JSON с полями status и reason."
    )
    trace(f"SOURCE_EVAL ▶ Lite | {info[:80]}…")
    result = run_gemini_lite_structured(
        SOURCE_EVALUATOR_SYSTEM,
        user_payload,
        global_anchor,
        SourceEvaluationResult,
        "source_evaluator",
    )
    status = (result.status or "REJECTED").strip().upper()
    if status not in ("APPROVED", "REJECTED"):
        status = "REJECTED"
    trace(f"SOURCE_EVAL ✓ {status} | {result.reason[:100]}")
    return SourceEvaluationResult(status=status, reason=(result.reason or "").strip())


def build_react_feedback(
    candidates: list[SourceCitationCandidate],
    global_anchor: str,
) -> str:
    """
    Проверить источники; собрать текст отклонений для Re-Act коррекции Reasoner.
    Пустая строка — все источники прошли аудит или нечего проверять.
    """
    if not candidates:
        return ""
    lines: list[str] = []
    for cand in candidates:
        ev = evaluate_source(cand.statement, cand.source_info, global_anchor)
        if ev.status == "REJECTED":
            src = cand.url or cand.source_info[:120]
            lines.append(
                f"[Системный отклик: Источник отклонён ({src}). Причина: {ev.reason}]"
            )
    return "\n".join(lines)


def audit_answer_sources_react(
    answer: str,
    registry: list[dict[str, Any]] | None,
    global_anchor: str,
) -> str:
    """Re-Act шаг: аудит всех цитируемых источников в черновике ответа."""
    candidates = extract_source_citation_candidates(answer, registry)
    if not candidates:
        trace("SOURCE_EVAL ⊘ нет ссылок для аудита в ответе")
        return ""
    return build_react_feedback(candidates, global_anchor)
