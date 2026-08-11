"""Unit tests for Knowledge Triangulation schemas + tutor packing."""

from __future__ import annotations

from knowledge_engine.schemas.extraction import (
    KNOWLEDGE_TRIANGULATION_TUTOR_RULES,
    AggregatedKnowledgeBase,
    KnowledgeAtom,
    ScopeType,
)
from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
    KNOWLEDGE_TRIANGULATION_LECTURE_RULES,
    LECTURE_SYSTEM_PROMPT,
)
from knowledge_engine.src.node_deep_dive.prompt_types import (
    InteractionPromptMode,
    PromptComposeContext,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    compose_system_prompt,
)


def test_knowledge_atom_format_and_parse():
    atom = KnowledgeAtom(
        scope=ScopeType.INSTANCE,
        statement="AJV: задержка 8.3 мс",
        context_quote="latency table",
    )
    tagged = atom.format_tagged()
    assert tagged.startswith("[SCOPE: INSTANCE]")
    parsed = KnowledgeAtom.from_tagged_line(tagged)
    assert parsed is not None
    assert parsed.scope is ScopeType.INSTANCE
    assert "8.3" in parsed.statement


def test_tutor_prompt_includes_triangulation_hierarchy():
    assert "KNOWLEDGE TRIANGULATION" in KNOWLEDGE_TRIANGULATION_TUTOR_RULES
    assert "KNOWLEDGE TRIANGULATION" in KNOWLEDGE_TRIANGULATION_LECTURE_RULES
    assert "KNOWLEDGE TRIANGULATION" in LECTURE_SYSTEM_PROMPT
    assert "[SCOPE: INSTANCE]" in LECTURE_SYSTEM_PROMPT

    dense = compose_system_prompt(
        InteractionPromptMode.LECTURE_DENSE,
        context=PromptComposeContext(),
    )
    assert "KNOWLEDGE TRIANGULATION" in dense
    assert "70%" in dense or "PRINCIPLE" in dense


def test_aggregated_buckets_preserve_order_classes():
    kb = AggregatedKnowledgeBase.from_atoms(
        [
            KnowledgeAtom(
                scope=ScopeType.INSTANCE, statement="Цифра эксперимента 8.3 мс"
            ),
            KnowledgeAtom(
                scope=ScopeType.PRINCIPLE,
                statement="Изоляция агентов снижает blast radius",
            ),
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Перехват вызовов на периметре до исполнения",
            ),
        ]
    )
    assert kb.principles[0].scope is ScopeType.PRINCIPLE
    assert kb.mechanics[0].scope is ScopeType.MECHANIC
    assert kb.evidence_cases[0].scope is ScopeType.INSTANCE
    blocks = kb.format_tutor_blocks()
    # принцип-блок идёт раньше кейсов
    assert blocks.index("FUNDAMENTAL PRINCIPLES") < blocks.index("PRACTICAL CASES")
