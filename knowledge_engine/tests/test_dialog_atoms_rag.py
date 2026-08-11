"""Unit tests for dialog knowledge_atoms RAG (scope filter + format)."""

from __future__ import annotations

from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType
from knowledge_engine.services.dialog_atoms_rag import (
    detect_code_intent,
    filter_atoms_for_dialog,
    format_dialog_atoms_block,
)


def test_detect_code_intent_ru_en() -> None:
    assert detect_code_intent("Покажи код lifecycle hooks")
    assert detect_code_intent("How to implement the function")
    assert detect_code_intent("пример кода для Worker")
    assert not detect_code_intent("Что такое lifecycle hooks?")
    assert not detect_code_intent("Объясни принцип изоляции агентов")


def test_filter_drops_instance_without_code_intent() -> None:
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.PRINCIPLE,
            statement="Isolation reduces blast radius across agents",
        ),
        KnowledgeAtom(
            scope=ScopeType.MECHANIC,
            statement="Hooks run before tool dispatch in the pipeline",
        ),
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Latency measured at 8.3 ms on Apple Silicon M1",
        ),
    ]
    filtered = filter_atoms_for_dialog(atoms, allow_instance=False, limit=6)
    assert len(filtered) == 2
    assert all(a.scope != ScopeType.INSTANCE for a in filtered)


def test_filter_keeps_instance_with_code_intent() -> None:
    atoms = [
        KnowledgeAtom(
            scope=ScopeType.PRINCIPLE,
            statement="Isolation reduces blast radius across agents",
        ),
        KnowledgeAtom(
            scope=ScopeType.INSTANCE,
            statement="Use onRequest hook before fetch in Workers runtime",
        ),
    ]
    filtered = filter_atoms_for_dialog(atoms, allow_instance=True, limit=6)
    assert len(filtered) == 2
    assert filtered[0].scope is ScopeType.PRINCIPLE
    assert filtered[1].scope is ScopeType.INSTANCE


def test_format_dialog_atoms_block() -> None:
    block = format_dialog_atoms_block(
        [
            KnowledgeAtom(
                scope=ScopeType.MECHANIC,
                statement="Validate schema before tool dispatch always",
            )
        ]
    )
    assert "### dialog_knowledge_atoms" in block
    assert "[ФАКТ (MECHANIC)]: Validate schema before tool dispatch always" in block


def test_format_empty() -> None:
    assert format_dialog_atoms_block([]) == ""
