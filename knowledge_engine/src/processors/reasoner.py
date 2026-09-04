"""Gemini Flash 3.6 (reasoner) — план и финальный ответ пользователю."""

from __future__ import annotations

from typing import Any

from knowledge_engine.config import GEMINI_REASONER_MODEL, GEMINI_RPM_PAUSE_SEC
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.reasoner import FinalResponseContract
from knowledge_engine.services.context_manager import load_personal_orchestrator_focus
from knowledge_engine.services.gemini_stateless import (
    GeminiUnavailableError,
    gemini_reasoner_model_chain,
    is_gemini_available,
    run_gemini_structured_with_chain,
)
from knowledge_engine.src.processors.question_formation_rules import (
    QUESTION_FORMATION_RULES,
)
from knowledge_engine.src.processors.source_anchors import (
    REASONER_SOURCE_ATTRIBUTION_PROMPT,
    format_registry_for_prompt,
    format_valid_docs_for_reasoner,
)
from knowledge_engine.src.processors.source_evaluator import (
    MAX_REACT_SOURCE_ITERATIONS,
    audit_answer_sources_react,
)
from knowledge_engine.src.prompts.engineering_context import GLOBAL_ENGINEERING_CRITERIA
from knowledge_engine.src.source_evaluator.evaluator import (
    format_whitelist_for_reasoner_prompt,
)
from knowledge_engine.ui.run_log import trace

FOLLOW_UP_RULES = (
    """
STRICTLY FORBIDDEN IN NEXT STEPS:
- Never suggest organizational, team, or managerial actions (e.g.: "Audit the codebase", "Adopt CI/CD", "Facilitate an Event Storming session", "Train the team").
- Never give operational tasks or development "homework".

MANDATORY FORMAT FOR NEXT STEPS (Knowledge Navigation):
Produce exactly 3 research vectors that help the user deepen their understanding of the topic:

1. 🔬 **Step deeper (Technical details)**: Suggest examining a specific low-level mechanism, mathematical trade-off, protocol, or internal workings of a technology mentioned in the answer.
2. 🔄 **Step sideways (Alternatives and contrasts)**: Suggest considering an alternative architectural pattern, a competing paradigm, or an edge case that contrasts with the current solution.
3. 🏛️ **Step back (Foundations and first causes)**: Suggest exploring the fundamental origins, history, or base Computer Science principle that led to this technology.

Each item must be phrased as an engaging research question or topic for the next query, NOT as a task to execute.
"""
    + QUESTION_FORMATION_RULES
)
"""
RU (пояснение): запрет менеджерских next-steps + обязательный формат из 3
исследовательских векторов (вглубь/в сторону/назад), не задач на исполнение.
"""

FAST_MODE_REASONER_PROMPT = """You are the Chief Systems Architect. Your task is to explain an architectural concept or pattern in accessible, illustrative, and fact-based language.

{whitelist_block}

FORMATTING AND NAVIGATION RULES:
1. **Material links**: Weave links to whitelist articles into the text in Markdown format: `[Article title](https://...)`.
2. **Illustrative diagrams**: Describe architectural interactions using clear text-based Mermaid or ASCII block diagrams.
3. **STRICTLY FORBIDDEN IN NEXT STEPS**:
   - Never give managerial or organizational assignments ("Audit the codebase", "Adopt CI/CD", "Train the team").

ANSWER STRUCTURE:
- 📌 **Concept "in plain terms"**: The essence of the pattern + an illustrative text diagram of the interaction.
- ⚙️ **How it works under the hood**: A detailed breakdown of the mechanics with links to authoritative sources.
- ⚖️ **Trade-offs and when to apply**: Pros, cons, bottlenecks, and alternatives.
- 📚 **Reference materials**: A list of 2-4 specific whitelist articles/sections to read.
- 🧭 **Research vectors (Next steps)**: 3 questions (Step deeper, Step sideways, Step back).
""".format(
    whitelist_block=format_whitelist_for_reasoner_prompt()
)
"""
RU (пояснение): fast-режим Reasoner (без Consensus) — whitelist_block
подставляется один раз при импорте модуля (детерминированный список,
без per-call данных).
"""

REASONER_REACT_CORRECTION_RULES = """
SOURCE AUDIT CORRECTION (Re-Act):
Below are the Source Evaluator's (Gemini Lite) system responses for rejected links. Rewrite only the problematic fragments:
- replace weak sources with material from the Whitelist Matrix (practitioners, ai_pioneers_labs, engineering_blogs, foundational_docs);
- or remove the link (REMOVE_LINK) and explain the thesis through fundamental CS principles.
Keep the answer's structure and the Russian output language. Do not add managerial recommendations.
"""
"""
RU (пояснение): Re-Act коррекция ответа Reasoner по фидбеку Source Evaluator
— переписать только отклонённые фрагменты, не весь ответ целиком.
"""

REASONER_SYSTEM = (
    f"""You are the Chief Systems Architect of AI systems.
{RUSSIAN_OUTPUT_RULE}
user_final_answer — Russian only (EN terms from sources may be kept as-is).

Based on valid_docs (Consensus material) and developer_profile_context, draft a plan and an answer.

developer_profile_context may be empty — in that case, do not invent developer constraints.

Profile isolation (mandatory):
1. Do NOT weave personal context (Jarvis, Apple Silicon, LanceDB, M-series) into general theoretical explanations.
   If the question is general (CS / architecture), answer only about the industry and academic trade-offs — no local projects or hardware.
2. If personal context is relevant — give the objective general analysis first; local-environment constraints only in
   an optional final "Local environment applicability" block, not in every paragraph.

Requirements:
- Connect theory to engineering implementation; consider developer_profile_context only if it is set
  and apply_personal_profile=true in the payload.
- Structure of user_final_answer:
  context → comparison of approaches → risks and trade-offs → "Research direction" block (see below)
  → optionally "Local environment applicability".
- Do NOT use a "Recommendations" block in the sense of managerial assignments. Use knowledge navigation instead.
{FOLLOW_UP_RULES}
- Formulas and complexity: LaTeX in `$...$` (inline) and `$$...$$` (block), e.g. `$\\mathcal{{O}}(N \\cdot d)$`.
- Formulas must be valid TeX only: `\\text{{}}`, `\\frac{{}}{{}}`; no tabs, form-feed, or duplicated text.
- Code and pseudocode only where appropriate.
- If data is insufficient (partial_data_note), explicitly list the gaps.
- user_final_answer — final text for the user, no meta-explanations about the pipeline.
- fact_nuggets — atomic facts for memory (short, verifiable); **without** [S1]-style tags or URLs — plain text only, for Light RAG.

{GLOBAL_ENGINEERING_CRITERIA}
"""
    + REASONER_SOURCE_ATTRIBUTION_PROMPT
)
"""
RU (пояснение): основной Reasoner (Consensus-режим) — план + финальный ответ
по valid_docs/developer_profile_context, изоляция личного контекста от общих
теоретических вопросов, LaTeX-формулы, навигация вместо менеджерских советов.
"""

FinalResponsePayload = FinalResponseContract


def _invoke_reasoner_structured(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
) -> FinalResponseContract:
    return run_gemini_structured_with_chain(
        GEMINI_REASONER_MODEL,
        system_instruction,
        user_payload,
        global_anchor,
        FinalResponseContract,
        label,
        rpm_pause=GEMINI_RPM_PAUSE_SEC > 0,
        models=gemini_reasoner_model_chain(),
    )


def _build_user_payload(
    *,
    user_query: str,
    profile_block: str,
    apply_personal_profile: bool,
    mode: str,
    registry_section: str,
    papers_section: str,
    valid_section: str,
    raw_consensus_text: str,
    partial_data_note: str,
    light_rag_block: str,
    react_feedback: str = "",
    previous_answer: str = "",
) -> str:
    parts = [
        f"### user_query\n{user_query}",
        f"### apply_personal_profile\n{apply_personal_profile}",
        f"### developer_profile_context\n{profile_block}",
    ]
    if mode == "fast":
        parts.append("### retrieval_mode\nfast (Consensus не вызывался)")
        parts.append(f"### light_rag_facts\n{light_rag_block}")
    else:
        parts.append(f"### SOURCE REGISTRY\n{registry_section}")
        parts.append(f"### scholarly_papers\n{papers_section}")
        parts.append(f"### valid_docs\n{valid_section}")
        parts.append(f"### consensus_raw (fallback)\n{raw_consensus_text[:10000]}")
    parts.append(f"### partial_data_note\n{partial_data_note or 'none'}")
    if previous_answer:
        parts.append(f"### previous_draft_answer\n{previous_answer[:14000]}")
    if react_feedback:
        parts.append(f"### source_audit_feedback\n{react_feedback}")
        parts.append(REASONER_REACT_CORRECTION_RULES)
    return "\n\n".join(parts)


def run_reasoner(
    valid_docs: list[dict[str, Any]],
    user_query: str,
    user_profile: str,
    global_anchor: str,
    *,
    raw_consensus_text: str = "",
    partial_data_note: str = "",
    papers_block: str = "",
    source_registry: list[dict[str, Any]] | None = None,
    apply_personal_profile: bool = True,
    retrieval_mode: str = "consensus",
    light_rag_context: str = "",
) -> FinalResponsePayload:
    if not is_gemini_available():
        raise GeminiUnavailableError("Gemini reasoner недоступен")
    mode = (retrieval_mode or "consensus").strip().lower()
    registry = source_registry or []
    registry_section = format_registry_for_prompt(registry) if registry else ""
    papers_section = (
        papers_block or registry_section or "(нет структурированного списка статей)"
    )
    valid_section = format_valid_docs_for_reasoner(valid_docs, registry)
    profile_block = (
        load_personal_orchestrator_focus()[:2000]
        if apply_personal_profile
        else "(пусто — общий/академический вопрос; apply_personal_profile=false)"
    )
    if user_profile and apply_personal_profile:
        # Совместимость: явно переданный профиль не расширяем — только фокус оркестратора
        _ = user_profile
    light_rag_block = (
        light_rag_context or ""
    ).strip() or "(нет релевантных фактов Light RAG)"

    if mode == "fast":
        system_instruction = (
            f"{RUSSIAN_OUTPUT_RULE}\n\n{FAST_MODE_REASONER_PROMPT}\n\n{FOLLOW_UP_RULES}"
        )
        label_prefix = "fast_reasoner"
    else:
        system_instruction = REASONER_SYSTEM
        label_prefix = "consensus_reasoner"

    user_payload = _build_user_payload(
        user_query=user_query,
        profile_block=profile_block,
        apply_personal_profile=apply_personal_profile,
        mode=mode,
        registry_section=registry_section,
        papers_section=papers_section,
        valid_section=valid_section,
        raw_consensus_text=raw_consensus_text,
        partial_data_note=partial_data_note,
        light_rag_block=light_rag_block,
    )
    result = _invoke_reasoner_structured(
        system_instruction,
        user_payload,
        global_anchor,
        f"{label_prefix} / draft",
    )

    accumulated_feedback = ""
    for react_round in range(MAX_REACT_SOURCE_ITERATIONS):
        feedback = audit_answer_sources_react(
            result.user_final_answer,
            registry,
            global_anchor,
        )
        if not feedback:
            trace(f"REACT ✓ источники прошли аудит | round={react_round}")
            break
        trace(f"REACT ▶ коррекция Reasoner | round={react_round + 1}")
        accumulated_feedback = (
            f"{accumulated_feedback}\n{feedback}".strip()
            if accumulated_feedback
            else feedback
        )
        revision_payload = _build_user_payload(
            user_query=user_query,
            profile_block=profile_block,
            apply_personal_profile=apply_personal_profile,
            mode=mode,
            registry_section=registry_section,
            papers_section=papers_section,
            valid_section=valid_section,
            raw_consensus_text=raw_consensus_text,
            partial_data_note=partial_data_note,
            light_rag_block=light_rag_block,
            react_feedback=accumulated_feedback,
            previous_answer=result.user_final_answer,
        )
        result = _invoke_reasoner_structured(
            system_instruction,
            revision_payload,
            global_anchor,
            f"{label_prefix} / react_{react_round + 1}",
        )

    return result
