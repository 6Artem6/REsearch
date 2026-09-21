"""Hybrid Context Buffer: 3-layer dialogue history for lite models.

Layer 1 (RAW_HISTORY_DEPTH raw messages at the tail) + Layer 2 (older
messages as message_bullet_summary, existing Windowed History) live in
ChatSessionManager._build_history_content. Layer 3 (block collapsing into a
persistent LEARNER_PROGRESS_SUMMARY) lives in history_block_collapse.py —
see prompt.log task and DIALOG_BLOCK_COLLAPSE_ENABLED in settings.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_engine.src.domains.grounding.chat_session_manager import (
    ChatSessionManager,
    StoredChatSession,
)
from knowledge_engine.src.domains.grounding.history_block_collapse import (
    build_learner_progress_summary_block,
    maybe_collapse_old_block,
)
from knowledge_engine.src.domains.grounding.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)


def _turn(i: int, role: str = "model") -> dict:
    return {"role": role, "content": f"turn-{i} content", "bullet_summary": [f"тезис {i}"]}


def _stored(n_pairs: int) -> StoredChatSession:
    turns: list[dict] = []
    for i in range(n_pairs):
        turns.append({"role": "user", "content": f"user-{i}"})
        turns.append(_turn(i))
    return StoredChatSession(
        session_id="a" * 12,
        model_name="gemini-3.5-flash-lite",
        label="node_deep_dive/tutor",
    ).model_copy(update={"api_turns": turns})


def _memory_with_verified(ids: list[str]) -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(id=cid, label=cid, status="verified") for cid in ids
    ]
    return mem


# --- Layer split (raw tail vs compressed timeline) ---------------------------


def test_build_history_content_raw_tail_uses_raw_history_depth(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_WINDOWED_HISTORY_ENABLED", True
    )
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_BLOCK_COLLAPSE_ENABLED", True
    )
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_SUBTHREAD_ISOLATION_ENABLED", False
    )
    monkeypatch.setattr("knowledge_engine.src.config.settings.RAW_HISTORY_DEPTH", 4)

    mgr = ChatSessionManager("test-scope")
    stored = _stored(5)  # 10 messages total

    class _FakePart:
        def __init__(self, text: str) -> None:
            self.text = text

        @staticmethod
        def from_text(text: str) -> "_FakePart":
            return _FakePart(text)

    class _FakeContent:
        def __init__(self, role: str, parts: list) -> None:
            self.role = role
            self.parts = parts

    class _FakeTypes:
        Part = _FakePart
        Content = _FakeContent

    history = mgr._build_history_content(_FakeTypes, stored.api_turns)

    # Last RAW_HISTORY_DEPTH=4 messages (2 pairs) stay raw, verbatim.
    raw_tail = history[-4:]
    assert [c.parts[0].text for c in raw_tail] == [
        "user-3",
        "turn-3 content",
        "user-4",
        "turn-4 content",
    ]
    # Everything older is compressed: model turns → bullet summary, not raw text.
    older = history[:-4]
    model_texts = [
        c.parts[0].text for c in older if c.role == "model"
    ]
    assert all("тезис" in t for t in model_texts)
    assert all("content" not in t for t in model_texts)


# --- Block collapsing: fires exactly at the configured threshold ------------


def test_maybe_collapse_old_block_does_not_fire_below_threshold(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_BLOCK_COLLAPSE_ENABLED", True
    )
    monkeypatch.setattr("knowledge_engine.src.config.settings.RAW_HISTORY_DEPTH", 4)
    monkeypatch.setattr("knowledge_engine.src.config.settings.COMPRESSED_BLOCK_SIZE", 6)

    stored = _stored(4)  # 8 messages: older_count = 8 - 4 = 4 < COMPRESSED_BLOCK_SIZE=6
    memory = SessionMemory()

    collapsed = maybe_collapse_old_block(stored, memory)

    assert collapsed is False
    assert len(stored.api_turns) == 8
    assert memory.learner_progress_blocks == []


def test_maybe_collapse_old_block_fires_exactly_at_threshold(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_BLOCK_COLLAPSE_ENABLED", True
    )
    monkeypatch.setattr("knowledge_engine.src.config.settings.RAW_HISTORY_DEPTH", 4)
    monkeypatch.setattr("knowledge_engine.src.config.settings.COMPRESSED_BLOCK_SIZE", 6)
    monkeypatch.setattr("knowledge_engine.src.config.settings.MAX_COMPRESSED_BLOCKS", 5)

    # 5 pairs = 10 messages: older_count = 10 - 4 = 6 == COMPRESSED_BLOCK_SIZE → fires.
    stored = _stored(5)
    memory = SessionMemory()

    with patch(
        "knowledge_engine.src.domains.grounding.history_block_collapse._summarize_block",
        return_value="Прошли основы MergeTree, разбор шёл уверенно.",
    ) as fake_summarize:
        collapsed = maybe_collapse_old_block(stored, memory)

    assert collapsed is True
    fake_summarize.assert_called_once()
    # Oldest COMPRESSED_BLOCK_SIZE=6 messages removed, the raw tail (4) remains.
    assert len(stored.api_turns) == 4
    assert stored.api_turns[0]["content"] == "user-3"
    assert memory.learner_progress_blocks == [
        "Прошли основы MergeTree, разбор шёл уверенно."
    ]

    # A second call immediately after (nothing new accumulated) must not re-fire.
    collapsed_again = maybe_collapse_old_block(stored, memory)
    assert collapsed_again is False
    assert len(stored.api_turns) == 4


def test_maybe_collapse_old_block_disabled_by_default(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_BLOCK_COLLAPSE_ENABLED", False
    )
    stored = _stored(20)  # would exceed any reasonable threshold
    memory = SessionMemory()

    collapsed = maybe_collapse_old_block(stored, memory)

    assert collapsed is False
    assert len(stored.api_turns) == 40


# --- No loss of verified_sub_concept_ids across collapses -------------------


def test_collapse_preserves_verified_sub_concept_ids(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.DIALOG_BLOCK_COLLAPSE_ENABLED", True
    )
    monkeypatch.setattr("knowledge_engine.src.config.settings.RAW_HISTORY_DEPTH", 2)
    monkeypatch.setattr("knowledge_engine.src.config.settings.COMPRESSED_BLOCK_SIZE", 4)
    monkeypatch.setattr("knowledge_engine.src.config.settings.MAX_COMPRESSED_BLOCKS", 1)

    memory = _memory_with_verified(["parts_merge", "mergetree_engine"])
    stored = _stored(3)  # 6 messages: older_count = 6-2=4 == COMPRESSED_BLOCK_SIZE

    with patch(
        "knowledge_engine.src.domains.grounding.history_block_collapse._summarize_block",
        return_value="Блок 1 закрыт.",
    ):
        assert maybe_collapse_old_block(stored, memory) is True
    assert memory.learner_progress_verified_ids == ["parts_merge", "mergetree_engine"]
    assert memory.learner_progress_blocks == ["Блок 1 закрыт."]

    # New sub-concept gets verified later, learner keeps chatting, a SECOND
    # block collapses and MAX_COMPRESSED_BLOCKS=1 caps the text list — but
    # the verified-ids union must keep growing regardless.
    memory.sub_concepts.append(
        SubConceptRecord(id="write_amplification", label="WA", status="verified")
    )
    for i in range(3, 6):
        stored.api_turns.append({"role": "user", "content": f"user-{i}"})
        stored.api_turns.append(_turn(i))

    with patch(
        "knowledge_engine.src.domains.grounding.history_block_collapse._summarize_block",
        return_value="Блок 2 закрыт.",
    ):
        assert maybe_collapse_old_block(stored, memory) is True

    # Only the newest block's text survives (MAX_COMPRESSED_BLOCKS=1)...
    assert memory.learner_progress_blocks == ["Блок 2 закрыт."]
    # ...but ALL verified ids across both collapses are still present.
    assert set(memory.learner_progress_verified_ids) == {
        "parts_merge",
        "mergetree_engine",
        "write_amplification",
    }


def test_build_learner_progress_summary_block_empty_when_nothing_collapsed():
    memory = SessionMemory()
    assert build_learner_progress_summary_block(memory) == ""


def test_build_learner_progress_summary_block_includes_ids_and_digests():
    memory = SessionMemory()
    memory.learner_progress_blocks = ["Блок A.", "Блок B."]
    memory.learner_progress_verified_ids = ["parts_merge"]

    block = build_learner_progress_summary_block(memory)

    assert "[LEARNER_PROGRESS_SUMMARY]" in block
    assert "Блок A." in block
    assert "Блок B." in block
    assert "parts_merge" in block


# --- Summary generation uses Gemma (primary → fallback), not Gemini Lite ----


def _fake_gemma_slot(label: str, model: str, *, result=None, raises: bool = False):
    from knowledge_engine.src.domains.grounding.history_block_collapse import (
        LearnerProgressBlockSummary,
    )

    slot = MagicMock()
    slot.label = label
    slot.model = model
    slot.client.estimate_input_tokens.return_value = 42
    if raises:
        slot.client.complete_structured = AsyncMock(side_effect=RuntimeError("429"))
    else:
        slot.client.complete_structured = AsyncMock(
            return_value=result or LearnerProgressBlockSummary(summary="ok, digest here")
        )
    slot.limiter.acquire = AsyncMock()
    return slot


def test_summarize_block_via_gemma_uses_primary_when_available():
    from knowledge_engine.src.domains.grounding.history_block_collapse import (
        LearnerProgressBlockSummary,
        _summarize_block_via_gemma,
    )

    primary = _fake_gemma_slot(
        "primary",
        "gemma-4-31b-it",
        result=LearnerProgressBlockSummary(summary="Прошли основы MergeTree."),
    )
    fallback = _fake_gemma_slot("fallback", "gemma-4-26b-a4b-it")

    with patch(
        "knowledge_engine.src.adapters.llm_providers.gemma_client.build_gemma_model_slots",
        return_value=[primary, fallback],
    ):
        import asyncio

        result = asyncio.run(_summarize_block_via_gemma("Ученик: ..."))

    assert result is not None
    assert result.summary == "Прошли основы MergeTree."
    primary.client.complete_structured.assert_awaited_once()
    fallback.client.complete_structured.assert_not_awaited()


def test_summarize_block_via_gemma_falls_back_when_primary_unavailable():
    from knowledge_engine.src.domains.grounding.history_block_collapse import (
        LearnerProgressBlockSummary,
        _summarize_block_via_gemma,
    )

    primary = _fake_gemma_slot("primary", "gemma-4-31b-it", raises=True)
    fallback = _fake_gemma_slot(
        "fallback",
        "gemma-4-26b-a4b-it",
        result=LearnerProgressBlockSummary(summary="Резюме от fallback-модели."),
    )

    with patch(
        "knowledge_engine.src.adapters.llm_providers.gemma_client.build_gemma_model_slots",
        return_value=[primary, fallback],
    ):
        import asyncio

        result = asyncio.run(_summarize_block_via_gemma("Ученик: ..."))

    assert result is not None
    assert result.summary == "Резюме от fallback-модели."
    primary.client.complete_structured.assert_awaited_once()
    fallback.client.complete_structured.assert_awaited_once()


def test_summarize_block_falls_back_to_digest_when_both_gemma_slots_fail():
    from knowledge_engine.src.domains.grounding.history_block_collapse import (
        _summarize_block,
    )

    primary = _fake_gemma_slot("primary", "gemma-4-31b-it", raises=True)
    fallback = _fake_gemma_slot("fallback", "gemma-4-26b-a4b-it", raises=True)

    with patch(
        "knowledge_engine.src.adapters.llm_providers.gemma_client.build_gemma_model_slots",
        return_value=[primary, fallback],
    ):
        text = _summarize_block(
            [{"role": "user", "content": "Как работает слияние партов?"}]
        )

    assert "сводка недоступна" in text
