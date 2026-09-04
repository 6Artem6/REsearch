"""Фоновая суммаризация истории (dialog_summarize): CAS-запись и Producer/Consumer."""

from __future__ import annotations

import pytest

from knowledge_engine.services import context_compressor_worker as ccw
from knowledge_engine.services.work_job_store import WorkJobKind, work_job_store
from knowledge_engine.src.node_deep_dive import session_store
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    DialogueFactManifest,
    SessionMemory,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeContentBlock


@pytest.fixture(autouse=True)
def _isolated_session_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_store, "_STORE_PATH", tmp_path / "sessions.json")


@pytest.fixture(autouse=True)
def _isolated_work_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    work_job_store._jobs.clear()


def _seed_session(manifest_version: int = 0, history_tag: str = "h0") -> None:
    mem = SessionMemory(
        fact_manifest=DialogueFactManifest(agreed_concepts=["orig"]),
        manifest_version=manifest_version,
    )
    session_store.save_session(
        "cur1",
        "node1",
        "in_progress",
        NodeContentBlock(),
        [{"role": "user", "content": history_tag}],
        memory=mem,
    )


def test_apply_fact_manifest_patch_applies_when_version_matches() -> None:
    _seed_session(manifest_version=0)
    new_manifest = DialogueFactManifest(agreed_concepts=["orig", "new_fact"])

    applied = session_store.apply_fact_manifest_patch("cur1", "node1", 0, new_manifest)
    assert applied is True

    reloaded = session_store.get_session("cur1", "node1")
    assert reloaded.memory is not None
    assert reloaded.memory.fact_manifest.agreed_concepts == ["orig", "new_fact"]
    assert reloaded.memory.manifest_version == 1


def test_apply_fact_manifest_patch_merges_onto_fresh_base_on_version_mismatch() -> None:
    """Real bug from a live run: one turn can call rotate_window_after_message
    TWICE (user- and tutor-message eviction of the SAME turn), both jobs
    enqueue with the same expected_manifest_version — the second job always
    sees a mismatch once the first has applied, even though there is no real
    race with the user. merge_manifest is additive (dedup'd list union), so
    a stale expected_version must still MERGE onto the current state, not
    drop the patch — nothing gets lost, version just keeps advancing."""
    _seed_session(manifest_version=3)
    stale_based_patch = DialogueFactManifest(agreed_concepts=["orig", "stale_fact"])

    applied = session_store.apply_fact_manifest_patch(
        "cur1", "node1", 0, stale_based_patch
    )
    assert applied is True

    reloaded = session_store.get_session("cur1", "node1")
    assert reloaded.memory is not None
    assert reloaded.memory.fact_manifest.agreed_concepts == ["orig", "stale_fact"]
    assert reloaded.memory.manifest_version == 4


def test_apply_fact_manifest_patch_preserves_history_written_after_enqueue() -> None:
    """Другие поля записи (history/content) могли обновиться уже ПОСЛЕ того,
    как job поставили в очередь — CAS-запись не должна их затирать."""
    _seed_session(manifest_version=0, history_tag="turn_at_enqueue")
    # Simulate a later turn's save_session() landing before the background
    # job runs — same manifest_version (no manifest change in that turn).
    mem2 = SessionMemory(
        fact_manifest=DialogueFactManifest(agreed_concepts=["orig"]),
        manifest_version=0,
    )
    session_store.save_session(
        "cur1",
        "node1",
        "in_progress",
        NodeContentBlock(),
        [{"role": "user", "content": "turn_after_enqueue"}],
        memory=mem2,
    )

    new_manifest = DialogueFactManifest(agreed_concepts=["orig", "new_fact"])
    applied = session_store.apply_fact_manifest_patch("cur1", "node1", 0, new_manifest)
    assert applied is True

    reloaded = session_store.get_session("cur1", "node1")
    assert any(
        h.get("content") == "turn_after_enqueue" for h in reloaded.history
    ), "history written after enqueue must survive the CAS write"
    assert reloaded.memory.fact_manifest.agreed_concepts == ["orig", "new_fact"]


def test_enqueue_dialog_summarize_creates_dialog_summarize_job() -> None:
    ccw.enqueue_dialog_summarize(
        "cur1",
        "node1",
        {
            "anchor": "a",
            "role": "user",
            "content": "some evicted text",
            "prev_manifest": {},
            "expected_manifest_version": 0,
        },
    )
    jobs = list(work_job_store._jobs.values())
    assert len(jobs) == 1
    job = jobs[0]
    assert job.kind == WorkJobKind.DIALOG_SUMMARIZE
    assert job.payload["curriculum_id"] == "cur1"
    assert job.payload["node_id"] == "node1"
    assert job.payload["content"] == "some evicted text"


def test_enqueue_dialog_summarize_skips_without_ids() -> None:
    ccw.enqueue_dialog_summarize("", "node1", {"expected_manifest_version": 0})
    assert not work_job_store._jobs


def test_run_dialog_summarize_job_applies_extraction_and_bumps_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_session(manifest_version=0)
    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.fact_manifest.run_fact_manifest_extraction",
        lambda payload: DialogueFactManifest(agreed_concepts=["orig", "extracted"]),
    )

    result = ccw.run_dialog_summarize_job(
        {
            "curriculum_id": "cur1",
            "node_id": "node1",
            "expected_manifest_version": 0,
            "prev_manifest": {"agreed_concepts": ["orig"]},
            "role": "user",
            "content": "text",
            "anchor": "a",
        }
    )
    assert result["applied"] is True

    reloaded = session_store.get_session("cur1", "node1")
    assert reloaded.memory.fact_manifest.agreed_concepts == ["orig", "extracted"]
    assert reloaded.memory.manifest_version == 1


def test_run_dialog_summarize_job_merges_stale_base_instead_of_dropping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_session(manifest_version=7)
    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.fact_manifest.run_fact_manifest_extraction",
        lambda payload: DialogueFactManifest(agreed_concepts=["orig", "late_fact"]),
    )

    result = ccw.run_dialog_summarize_job(
        {
            "curriculum_id": "cur1",
            "node_id": "node1",
            "expected_manifest_version": 0,
            "prev_manifest": {},
            "role": "user",
            "content": "text",
            "anchor": "a",
        }
    )
    assert result["applied"] is True

    reloaded = session_store.get_session("cur1", "node1")
    assert reloaded.memory.fact_manifest.agreed_concepts == ["orig", "late_fact"]
    assert reloaded.memory.manifest_version == 8


def test_two_same_turn_evictions_both_survive_cas() -> None:
    """End-to-end regression for the exact log scenario: two
    dialog_summarize jobs enqueued in the same turn (user- and
    tutor-message eviction), both captured with expected_manifest_version=0.
    Before the fix, job #2 was silently dropped once job #1 applied first."""
    _seed_session(manifest_version=0)

    job1_manifest = DialogueFactManifest(agreed_concepts=["orig", "from_user_evict"])
    applied1 = session_store.apply_fact_manifest_patch(
        "cur1", "node1", 0, job1_manifest
    )
    assert applied1 is True

    job2_manifest = DialogueFactManifest(agreed_concepts=["orig", "from_tutor_evict"])
    applied2 = session_store.apply_fact_manifest_patch(
        "cur1", "node1", 0, job2_manifest
    )
    assert applied2 is True

    reloaded = session_store.get_session("cur1", "node1")
    assert reloaded.memory.fact_manifest.agreed_concepts == [
        "orig",
        "from_user_evict",
        "from_tutor_evict",
    ]
    assert reloaded.memory.manifest_version == 2
