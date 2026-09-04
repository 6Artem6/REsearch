"""Stage 0/1 — Cloud LLM Pipeline guardrails → ValidatedQuerySpec."""

from __future__ import annotations

from knowledge_engine.config import GUARDRAILS_MODEL
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.local_llm_stateless import run_local_structured
from knowledge_engine.src.guardrails.term_guard import extract_user_acronyms
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.state import ValidatedQuerySpec
from knowledge_engine.ui.run_log import trace

_DEFAULT_MODEL = GUARDRAILS_MODEL


def _build_anchor(user_query: str, user_profile_md: str) -> str:
    parts = [f"Задача пользователя: {user_query.strip()}"]
    profile = (user_profile_md or "").strip()
    if profile:
        parts.append(f"Профиль разработчика:\n{profile[:3500]}")
    return "\n\n".join(parts)


def _build_guardrails_system(grounding_context: str, user_query: str) -> str:
    grounding = (
        grounding_context or ""
    ).strip() or "(no web snippets — rely on user query only)"
    return (
        f"{RUSSIAN_OUTPUT_RULE} "
        "You are a technical Search Query Generator for an AI Knowledge Engine.\n\n"
        f"### WEB CONTEXT (RAW SEARCH SNIPPETS):\n{grounding}\n\n"
        f"### USER QUERY:\n{user_query}\n\n"
        "### CRITICAL RULES FOR ACRONYMS & TERMS:\n"
        "1. NEVER guess or expand 2–5 letter technical acronyms (e.g., MCP, RAG, GUI, API) "
        "using internal memory.\n"
        "2. Rely strictly on the WEB CONTEXT above to resolve domain meanings.\n"
        "3. Keep all original acronyms VERBATIM in all generated search queries.\n"
        "4. You MAY append explicit long-form terms alongside the acronym if supported by "
        'the WEB CONTEXT (e.g., "MCP Model Context Protocol"), but NEVER substitute '
        'acronyms with unrelated concepts (e.g., do NOT turn MCP into "CPU" or '
        '"Master Control Processor").\n\n'
        "Generate precise, highly relevant search queries based on the domain context provided.\n\n"
        "Output STRICT JSON ValidatedQuerySpec:\n"
        "- cs_formal_query: 1–3 sentences (Russian OK).\n"
        "- target_keywords: 3–5 entities.\n"
        "- search_queries: 2–3 English-oriented search strings for SearXNG/arXiv.\n"
        "- preserved_terms: acronyms from USER QUERY verbatim.\n"
        "No markdown outside JSON."
    )


def _generate_validated_query_spec_sync(
    user_query: str,
    user_profile_md: str,
    model_name: str,
    grounding_context: str,
) -> ValidatedQuerySpec:
    query = (user_query or "").strip()
    if len(query) < 3:
        raise ValueError("user_query слишком короткий")

    anchor = _build_anchor(query, user_profile_md)
    system = _build_guardrails_system(grounding_context, query)
    user_payload = (
        "Produce ValidatedQuerySpec JSON for the research pipeline.\n"
        f"Detected acronyms in user query: {', '.join(extract_user_acronyms(query)) or '(none)'}"
    )

    trace(f"GUARDRAILS ▶ Cloud LLM structured | model={model_name}")
    spec = run_local_structured(
        model_name,
        ValidatedQuerySpec,
        system,
        user_payload,
        anchor,
        "guardrails / ValidatedQuerySpec",
        temperature=0.05,
    )
    trace(
        f"GUARDRAILS raw ✓ keywords={len(spec.target_keywords)} "
        f"queries={len(spec.search_queries)}"
    )
    return _normalize_spec(spec)


def _normalize_spec(spec: ValidatedQuerySpec) -> ValidatedQuerySpec:
    kw = [k.strip() for k in spec.target_keywords if k and k.strip()][:5]
    sq = [q.strip() for q in spec.search_queries if q and q.strip()][:3]
    formal = (spec.cs_formal_query or "").strip()
    preserved = [p.strip() for p in spec.preserved_terms if p and p.strip()][:12]
    if len(kw) < 3:
        kw = kw + [w for w in formal.split()[:8] if len(w) > 3][: 5 - len(kw)]
    if len(sq) < 2 and formal:
        sq.append(formal[:120])
    return ValidatedQuerySpec(
        cs_formal_query=formal or "Unspecified CS engineering query",
        target_keywords=kw[:5],
        search_queries=sq[:3],
        preserved_terms=preserved,
    )


async def generate_validated_query_spec(
    user_query: str,
    user_profile_md: str = "",
    model_name: str | None = None,
    grounding_context: str = "",
) -> ValidatedQuerySpec:
    """
    Cloud LLM structured JSON under ``uma_resource_lock``.
    Grounding must be fetched outside the lock (see ``manager.run_stage_0_1``).
    """
    model = (model_name or _DEFAULT_MODEL).strip()
    return await run_under_uma_lock(
        _generate_validated_query_spec_sync,
        user_query,
        user_profile_md,
        model,
        grounding_context,
    )
