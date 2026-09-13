"""Steering → тяжёлый Map-Reduce, СТРОГО минуя Discovery Phase.

Этап 5 мастер-плана (см. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md). Add-only:
НЕ модифицирует ``generator.py`` — реиспользует его же строительные блоки
Search-First пути (``assign_source_ids``/``search_hits_as_prompt_json``/
``generate_curriculum_search_first``), пропуская ТОЛЬКО поиск
(``collect_curriculum_source_hits``) и энричмент (``summarize_whitelist_
blog_hits``/``enrich_search_hits_with_extracts``) — у нас уже есть
Gate-2-одобренные ``SurfaceDigestItem`` с готовым company_or_author/
problem_solved/main_tech_stack/two_sentence_summary, повторный
поиск/суммаризация не нужны и не запускаются.
"""

from __future__ import annotations

from knowledge_engine.src.adapters.llm_providers.gemini_stateless import (
    GeminiUnavailableError,
    is_gemini_available,
)
from knowledge_engine.src.domains.curriculum.schemas import (
    CurriculumGenerateInput,
    CurriculumGraph,
    CurriculumSearchHit,
)
from knowledge_engine.src.domains.curriculum.search_first_flash import (
    generate_curriculum_search_first,
)
from knowledge_engine.src.domains.curriculum.search_prestep import (
    assign_source_ids,
    search_hits_as_prompt_json,
)
from knowledge_engine.src.domains.steering.steering_contracts import (
    SurfaceDigestItem,
    SurfaceDigestResponse,
)


def _hit_from_digest(item: SurfaceDigestItem, index: int) -> CurriculumSearchHit:
    extracts: list[str] = []
    if item.problem_solved:
        extracts.append(item.problem_solved.strip())
    if item.main_tech_stack:
        extracts.append("Tech stack: " + ", ".join(item.main_tech_stack))
    if item.two_sentence_summary:
        extracts.append(item.two_sentence_summary.strip())
    extracts = [e for e in extracts if e]
    return CurriculumSearchHit(
        source_id=f"steering_{index}",
        url=item.url,
        title=item.title,
        snippet=(item.two_sentence_summary or "").strip()[:1200],
        key_extracts=extracts[:8],
        source_tier="steering_approved",
        skip_ollama_summary=True,
    )


def generate_curriculum_from_steering_digests(
    model_in: CurriculumGenerateInput,
    digests: SurfaceDigestResponse,
    final_approved_urls: list[str],
    *,
    anchor: str | None = None,
) -> CurriculumGraph:
    """Gate 2 ``final_approved_urls`` + их ``SurfaceDigestItem`` → CurriculumGraph,

    без единого обращения к Discovery-функциям ``generator.py``. URL без
    готового дайджеста (не должно случаться при нормальном Gate 1→2
    потоке, но входные данные — с границы API) просто пропускаются, а не
    роняют всю генерацию."""
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini недоступен для Curriculum Generator")

    by_url = {(d.url or "").strip(): d for d in digests.digests}
    ordered_urls = [u.strip() for u in final_approved_urls if (u or "").strip()]
    hits = [
        _hit_from_digest(by_url[u], i)
        for i, u in enumerate(ordered_urls, start=1)
        if u in by_url
    ]
    if not hits:
        raise ValueError(
            "Steering: ни для одного final_approved_urls нет готового "
            "SurfaceDigestItem — нечего передать в Map-Reduce"
        )
    hits = assign_source_ids(hits)
    parsed_json = search_hits_as_prompt_json(hits)
    anchor_ = anchor or f"steering_generate:{model_in.target_goal.strip()[:400]}"
    return generate_curriculum_search_first(model_in, hits, parsed_json, anchor_)


__all__ = ["generate_curriculum_from_steering_digests"]
