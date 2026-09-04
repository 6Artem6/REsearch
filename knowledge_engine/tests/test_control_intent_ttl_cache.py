"""TTL-кэш classify_control_chip: multi-tenant safe (session_id, text, slot_active)."""

from __future__ import annotations

import time

import pytest

from knowledge_engine.src.node_deep_dive import control_intent
from knowledge_engine.src.node_deep_dive.control_intent import (
    classify_control_chip,
    control_intent_session_scope,
    current_control_intent_session_id,
    is_short_begin_message,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.src.node_deep_dive.vector_intent_router import (
    VectorIntentRouter,
    set_vector_intent_router_for_tests,
)
from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed


@pytest.fixture(autouse=True)
def _isolated_chip_cache():
    control_intent._clear_chip_cache_for_tests()
    yield
    control_intent._clear_chip_cache_for_tests()


@pytest.fixture
def counting_router(tmp_path):
    calls = {"n": 0}

    def counting_embed(text: str):
        calls["n"] += 1
        return lexical_probe_embed(text)

    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=counting_embed,
        enabled=True,
        persist=True,
        db_path=tmp_path / "intent_lance",
        embed_model="probe-embed",
        auto_sync=True,
    )
    set_vector_intent_router_for_tests(router)
    yield calls
    set_vector_intent_router_for_tests(None)


def test_same_session_same_text_hits_vector_once(counting_router):
    with control_intent_session_scope("session_a"):
        first = classify_control_chip("хочу mech")
        n_after_first = counting_router["n"]
        second = classify_control_chip("хочу mech")
        n_after_second = counting_router["n"]

    assert first == "mech"
    assert second == "mech"
    assert n_after_first > 0
    assert n_after_second == n_after_first, "second call must be served from cache"


def test_different_sessions_same_text_do_not_share_cache(counting_router):
    with control_intent_session_scope("session_a"):
        classify_control_chip("хочу mech")
        n_after_a = counting_router["n"]

    with control_intent_session_scope("session_b"):
        classify_control_chip("хочу mech")
        n_after_b = counting_router["n"]

    assert n_after_b > n_after_a, "a different session must not reuse session_a's cache"


def test_cache_key_also_scopes_by_mode_selection_slot(counting_router):
    """Same session+text must still re-classify when slot_active flips —
    slot mode uses a different allowed_intents/threshold in _classify_vector."""
    mem = SessionMemory()
    with control_intent_session_scope("session_a"):
        classify_control_chip("хочу mech", memory=mem)
        n_before_slot = counting_router["n"]

        from knowledge_engine.src.node_deep_dive.control_intent import (
            mark_awaiting_mode_selection,
        )

        mark_awaiting_mode_selection(mem)
        classify_control_chip("хочу mech", memory=mem)
        n_after_slot = counting_router["n"]

    assert n_after_slot > n_before_slot


def test_cache_expires_after_ttl(counting_router, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(control_intent, "_CHIP_CACHE_TTL_SEC", 0.05)
    with control_intent_session_scope("session_a"):
        classify_control_chip("хочу mech")
        n_after_first = counting_router["n"]
        time.sleep(0.15)
        classify_control_chip("хочу mech")
        n_after_ttl = counting_router["n"]

    assert (
        n_after_ttl > n_after_first
    ), "expired entry must be recomputed, not served stale"


def test_scope_resets_session_id_after_exit(counting_router):
    assert current_control_intent_session_id() == ""
    with control_intent_session_scope("session_a"):
        assert current_control_intent_session_id() == "session_a"
    assert current_control_intent_session_id() == ""


def test_wrapper_predicates_transparently_reuse_cache(counting_router):
    """Item 3 of the ticket: surrogate checks (is_short_begin_message etc.)
    must transparently share the same TTL cache via the ContextVar, without
    needing session_id in their own signature."""
    phrase = "готов начать погружение"
    with control_intent_session_scope("session_a"):
        classify_control_chip(phrase)
        n_after_direct = counting_router["n"]
        is_short_begin_message(phrase)
        n_after_wrapper = counting_router["n"]

    assert (
        n_after_wrapper == n_after_direct
    ), "wrapper call must hit the same cache entry"


def test_empty_session_id_bypasses_cache_instead_of_leaking(counting_router):
    """No scope set (session_id="") must never cache — failing safe beats a
    cross-tenant leak if some call site forgets to enter the scope."""
    classify_control_chip("хочу mech")
    n_after_first = counting_router["n"]
    classify_control_chip("хочу mech")
    n_after_second = counting_router["n"]

    assert n_after_second > n_after_first
