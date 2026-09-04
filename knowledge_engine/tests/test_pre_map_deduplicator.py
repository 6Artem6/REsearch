"""Pre-MAP Dedup (BGE clustering + Flash Lite bulk gate) — unit tests."""

from __future__ import annotations

import asyncio
import json
import math
from unittest.mock import AsyncMock

import pytest

from knowledge_engine.src.deduplication import pre_map_deduplicator as m


def _candidate(id_: str, url: str, text: str, is_code: bool | None = None):
    return m.PreMapCandidate(id=id_, url=url, text=text, is_code=is_code)


# ---------------------------------------------------------------------------
# _mmr_top_by_centroid
# ---------------------------------------------------------------------------


def test_mmr_top_by_centroid_passthrough_when_le_top_k():
    assert m._mmr_top_by_centroid([], top_k=5) == []
    paras = ["a", "b", "c"]
    assert m._mmr_top_by_centroid(paras, top_k=5) == paras


def test_mmr_top_by_centroid_selects_distinct_subset(monkeypatch):
    paras = [f"p{i}" for i in range(6)]
    vectors = {
        "p0": [1.0, 0.0],
        "p1": [0.99, 0.01],
        "p2": [0.0, 1.0],
        "p3": [0.98, 0.02],
        "p4": [-1.0, 0.0],
        "p5": [0.5, 0.5],
    }
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [vectors[t] for t in texts],
    )
    out = m._mmr_top_by_centroid(paras, top_k=3)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert all(p in paras for p in out)


# ---------------------------------------------------------------------------
# _pool_vector
# ---------------------------------------------------------------------------


def test_pool_vector_empty_returns_none():
    assert m._pool_vector([]) is None


def test_pool_vector_averages_embeddings(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0], [0.0, 2.0]],
    )
    vec = m._pool_vector(["a", "b"])
    assert vec == pytest.approx([0.5, 1.0])


# ---------------------------------------------------------------------------
# _cluster_text_candidates — чистый Union-Find по косинусному сходству
# ---------------------------------------------------------------------------


def test_cluster_text_candidates_groups_close_pairs():
    vectors = {
        "a": [1.0, 0.0],
        "b": [2.0, 0.0],  # same direction as a -> cosine 1.0
        "c": [0.0, 1.0],  # orthogonal -> cosine 0.0
    }
    groups = sorted(
        sorted(g) for g in m._cluster_text_candidates(vectors, threshold=0.9)
    )
    assert groups == [["a", "b"], ["c"]]


def test_cluster_text_candidates_all_singletons_below_threshold():
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [-1.0, 0.0]}
    groups = m._cluster_text_candidates(vectors, threshold=0.95)
    assert sorted(len(g) for g in groups) == [1, 1, 1]


def test_cluster_text_candidates_transitive_chain_merges():
    # a-b под 30° (cos~0.866), b-c под 30° (cos~0.866), a-c под 60°
    # (cos=0.5) — a и c соединяются только транзитивно ЧЕРЕЗ b.
    a = [1.0, 0.0]
    b = [math.cos(math.radians(30)), math.sin(math.radians(30))]
    c = [math.cos(math.radians(60)), math.sin(math.radians(60))]
    assert m._cosine(a, b) == pytest.approx(0.866, abs=1e-3)
    assert m._cosine(b, c) == pytest.approx(0.866, abs=1e-3)
    assert m._cosine(a, c) == pytest.approx(0.5, abs=1e-3)

    groups = m._cluster_text_candidates({"a": a, "b": b, "c": c}, threshold=0.8)
    assert len(groups) == 1
    assert sorted(groups[0]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _build_bulk_gate_payload
# ---------------------------------------------------------------------------


def test_build_bulk_gate_payload_structure():
    payload = m._build_bulk_gate_payload(
        [["a", "b"]], ["c"], {"a": ["pa"], "b": ["pb"], "c": ["sig c()"]}
    )
    data = json.loads(payload)
    assert data["suspect_text_groups"][0]["group_id"] == "g0"
    ids = {item["id"] for item in data["suspect_text_groups"][0]["candidates"]}
    assert ids == {"a", "b"}
    assert data["code_files"] == [{"id": "c", "extract": "sig c()"}]


# ---------------------------------------------------------------------------
# _sanitize_canonical_map
# ---------------------------------------------------------------------------


def test_sanitize_canonical_map_drops_unknown_ids():
    raw = {"c1": ["a1", "unknown_id"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"c1", "a1"})
    assert clean == {"c1": ["a1"]}


def test_sanitize_canonical_map_drops_self_alias():
    raw = {"c1": ["c1", "a1"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"c1", "a1"})
    assert clean == {"c1": ["a1"]}


def test_sanitize_canonical_map_drops_canonical_with_no_valid_aliases():
    raw = {"c1": ["unknown"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"c1"})
    assert clean == {}


def test_sanitize_canonical_map_first_claim_wins_on_conflict():
    raw = {"c1": ["a1"], "c2": ["a1"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"c1", "c2", "a1"})
    assert clean == {"c1": ["a1"]}


def test_sanitize_canonical_map_rejects_alias_that_is_itself_a_canonical():
    # c2 обрабатывается первым (порядок dict) и закрепляется как собственный
    # canonical; c1 после этого не должен иметь возможность заявить c2 своим alias.
    raw = {"c2": ["a2"], "c1": ["c2"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"c1", "c2", "a2"})
    assert clean == {"c2": ["a2"]}
    assert "c1" not in clean


# ---------------------------------------------------------------------------
# _run_bulk_gate
# ---------------------------------------------------------------------------


def test_run_bulk_gate_noop_when_nothing_to_check():
    out = asyncio.run(m._run_bulk_gate([], [], {}, anchor=""))
    assert out == {}


def test_run_bulk_gate_skips_call_when_gemini_unavailable(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available", lambda: False
    )
    called = {"n": 0}

    def spy(*a, **kw):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        spy,
    )
    out = asyncio.run(
        m._run_bulk_gate([["a", "b"]], [], {"a": ["x"], "b": ["y"]}, anchor="")
    )
    assert out == {}
    assert called["n"] == 0


def test_run_bulk_gate_fail_open_on_gemini_exception(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available", lambda: True
    )

    def boom(*a, **kw):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        boom,
    )
    out = asyncio.run(
        m._run_bulk_gate([["a", "b"]], [], {"a": ["x"], "b": ["y"]}, anchor="")
    )
    assert out == {}


def test_run_bulk_gate_returns_canonical_map_from_contract(monkeypatch):
    from knowledge_engine.schemas.llm_contracts.pre_map_dedup import (
        CanonicalMapContract,
    )

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available", lambda: True
    )

    def fake_call(*a, **kw):
        return CanonicalMapContract(mappings=[{"canonical_id": "a", "aliases": ["b"]}])

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        fake_call,
    )
    out = asyncio.run(
        m._run_bulk_gate([["a", "b"]], [], {"a": ["x"], "b": ["y"]}, anchor="")
    )
    assert out == {"a": ["b"]}


# ---------------------------------------------------------------------------
# deduplicate_before_map_reduce — сквозная (end-to-end) оркестрация
# ---------------------------------------------------------------------------


def test_deduplicate_before_map_reduce_empty_returns_empty():
    result = asyncio.run(m.deduplicate_before_map_reduce([]))
    assert result.decisions == {}
    assert result.alias_map == {}


def test_deduplicate_before_map_reduce_single_autonomous_skips_lite(monkeypatch):
    calls = {"n": 0}

    async def fake_bulk_gate(*a, **kw):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": False)
    monkeypatch.setattr(
        m, "_extract_paragraphs", lambda html, url, min_chars: ["a lone paragraph"]
    )

    candidates = [_candidate("u1", "https://example.com/1", "<html>...</html>")]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    assert result.canonical_ids() == ["u1"]
    assert calls["n"] == 0, "single autonomous candidate must never reach Lite"


def test_deduplicate_before_map_reduce_code_skips_bge_embed(monkeypatch):
    embed_calls = {"n": 0}

    def fake_embed(texts):
        embed_calls["n"] += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", fake_embed
    )
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": True)
    monkeypatch.setattr(
        m, "_ast_semantic_extracts", lambda text, url, min_chars: ["void foo(void)"]
    )

    async def fake_bulk_gate(suspect_groups, code_ids, context_by_id, *, anchor):
        assert suspect_groups == []
        assert code_ids == ["u1"]
        return {}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    candidates = [
        _candidate(
            "u1",
            "https://github.com/x/y/blob/main/f.c",
            "void foo(void) {}",
            is_code=True,
        )
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    assert result.canonical_ids() == ["u1"]
    assert embed_calls["n"] == 0, "code context extraction must never call BGE embed"


def test_deduplicate_before_map_reduce_ast_fallback_to_raw_chunks(monkeypatch):
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": True)
    monkeypatch.setattr(m, "_ast_semantic_extracts", lambda text, url, min_chars: [])

    async def fake_bulk_gate(*a, **kw):
        return {}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    candidates = [
        _candidate("u1", "https://example.com/f.c", "int main(void) { return 0; }" * 3)
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    decision = result.decisions["u1"]
    assert decision.is_code is True
    assert (
        decision.context_units
    ), "empty AST extracts must fall back to _code_paragraphs_from_raw_text"


def test_deduplicate_before_map_reduce_applies_canonical_map(monkeypatch):
    """Two near-duplicate text docs cluster together and get resolved into
    CANONICAL + ALIAS by a (mocked) bulk gate."""
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": False)
    monkeypatch.setattr(
        m,
        "_extract_paragraphs",
        lambda html, url, min_chars: [f"paragraph about {url}"],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0] for _ in texts],  # identical -> cosine 1.0
    )

    captured = {}

    async def fake_bulk_gate(suspect_groups, code_ids, context_by_id, *, anchor):
        captured["suspect_groups"] = suspect_groups
        captured["code_ids"] = code_ids
        group = suspect_groups[0]
        alias = [cid for cid in group if cid != "u1"]
        return {"u1": alias}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    candidates = [
        _candidate("u1", "https://a.example.com/post", "<html>a</html>"),
        _candidate("u2", "https://b.example.com/mirror", "<html>b</html>"),
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))

    assert result.canonical_ids() == ["u1"]
    assert result.is_alias("u2")
    assert result.canonical_of("u2") == "u1"
    assert result.alias_map == {"u1": ["u2"]}
    assert captured["code_ids"] == []
    assert sorted(captured["suspect_groups"][0]) == ["u1", "u2"]


def test_deduplicate_before_map_reduce_fail_open_on_gemini_error(monkeypatch):
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": False)
    monkeypatch.setattr(
        m, "_extract_paragraphs", lambda html, url, min_chars: ["same paragraph text"]
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available", lambda: True
    )

    def boom(*a, **kw):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        boom,
    )

    candidates = [
        _candidate("u1", "https://a.example.com", "<html>a</html>"),
        _candidate("u2", "https://b.example.com", "<html>b</html>"),
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    assert sorted(result.canonical_ids()) == [
        "u1",
        "u2",
    ], "a Flash Lite failure must never remove a candidate — fail-open"
    assert result.alias_map == {}


def test_deduplicate_before_map_reduce_mixed_text_and_code(monkeypatch):
    """Text and code candidates in the same batch reach two different
    comparison paths: text via the Bulk Gate (clustering-gated), code
    unconditionally via the isolated Code Deduplication module."""

    def fake_detect(url, text, method=""):
        return url.endswith(".c")

    monkeypatch.setattr(m, "detect_code_content", fake_detect)
    monkeypatch.setattr(
        m,
        "_extract_paragraphs",
        lambda html, url, min_chars: ["distinct text paragraph"],
    )
    monkeypatch.setattr(
        m, "_ast_semantic_extracts", lambda text, url, min_chars: ["void f(void)"]
    )
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0], [0.0, 1.0]][: len(texts)] or [[1.0, 0.0]],
    )

    code_dedup = AsyncMock(return_value={})
    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.code_deduplicator.deduplicate_code_candidates",
        code_dedup,
    )

    candidates = [
        _candidate("text1", "https://a.example.com/post", "<html>a</html>"),
        _candidate("code1", "https://github.com/x/y/blob/main/f.c", "void f(void) {}"),
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    code_dedup.assert_awaited_once()
    (code_candidates_arg,), _kwargs = code_dedup.await_args
    assert [c.id for c in code_candidates_arg] == ["code1"]
    assert sorted(result.canonical_ids()) == ["code1", "text1"]


# ---------------------------------------------------------------------------
# Закрытие пробелов покрытия: ветки, не задействованные ни одним сценарием выше.
# ---------------------------------------------------------------------------


def test_canonical_of_unknown_id_passes_through():
    result = m.PreMapDedupResult()
    result.decisions["a"] = m.PreMapDedupDecision(
        id="a", is_canonical=True, canonical_id="a", is_code=False
    )
    assert result.canonical_of("never-seen-id") == "never-seen-id"
    assert result.is_alias("never-seen-id") is False


def test_pool_vector_empty_fingerprint_short_circuits_without_embed(monkeypatch):
    called = {"n": 0}

    def fake_embed(texts):
        called["n"] += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", fake_embed
    )
    assert m._pool_vector([]) is None
    assert called["n"] == 0


def test_sanitize_canonical_map_drops_hallucinated_canonical_key():
    raw = {"hallucinated_canonical": ["a1"]}
    clean = m._sanitize_canonical_map(raw, valid_ids={"a1"})
    assert clean == {}


def test_deduplicate_before_map_reduce_respects_explicit_top_k_and_threshold(
    monkeypatch,
):
    """Explicit top_k/cosine_threshold overrides must reach _cluster_text_candidates
    and the fingerprint size, not just the config defaults."""
    monkeypatch.setattr(m, "detect_code_content", lambda url, text, method="": False)
    monkeypatch.setattr(m, "_extract_paragraphs", lambda html, url, min_chars: [html])
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    captured_threshold = {}

    async def fake_bulk_gate(suspect_groups, code_ids, context_by_id, *, anchor):
        return {}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    real_cluster = m._cluster_text_candidates

    def spy_cluster(doc_vectors, *, threshold):
        captured_threshold["value"] = threshold
        return real_cluster(doc_vectors, threshold=threshold)

    monkeypatch.setattr(m, "_cluster_text_candidates", spy_cluster)

    candidates = [
        _candidate("u1", "https://a.example.com", "<html>a</html>"),
        _candidate("u2", "https://b.example.com", "<html>b</html>"),
    ]
    asyncio.run(
        m.deduplicate_before_map_reduce(candidates, top_k=2, cosine_threshold=0.42)
    )
    assert captured_threshold["value"] == 0.42


def test_deduplicate_before_map_reduce_context_extraction_fail_open(monkeypatch):
    """A raising detector (outside any inner try/except) must not abort the
    batch — that one candidate falls back to an empty fingerprint and stays
    CANONICAL, everyone else is processed normally."""

    def flaky_detect(url, text, method=""):
        if url == "https://boom.example.com":
            raise RuntimeError("detector exploded")
        return False

    monkeypatch.setattr(m, "detect_code_content", flaky_detect)
    monkeypatch.setattr(m, "_extract_paragraphs", lambda html, url, min_chars: [html])
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    async def fake_bulk_gate(*a, **kw):
        return {}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    candidates = [
        _candidate("ok", "https://fine.example.com", "<html>fine</html>"),
        _candidate("boom", "https://boom.example.com", "<html>boom</html>"),
    ]
    result = asyncio.run(m.deduplicate_before_map_reduce(candidates))
    assert sorted(result.canonical_ids()) == ["boom", "ok"]
    assert result.decisions["boom"].context_units == []


# ---------------------------------------------------------------------------
# Triage (Step 1a, реальный Flash Lite через paper_structure_analyzer.py) ->
# MMR (Step 1b, немодифицированный greedy_mmr_select). Никакой локальной
# эвристики здесь нет — _flash_lite_triage_core_units всегда идёт через
# PaperStructureAnalyzer.
# ---------------------------------------------------------------------------


@pytest.mark.real_triage
def test_flash_lite_triage_core_units_keeps_only_core(monkeypatch):
    from knowledge_engine.src.parsers.paper_structure_schema import (
        PaperStructureAnalysis,
        ParagraphPriority,
        ParagraphStructureVerdict,
    )

    def fake_analyze(self, target_topic, input_paper, *, label="", anchor=""):
        rows = []
        for page in input_paper.pages:
            for p in page.paragraphs:
                priority = (
                    ParagraphPriority.CORE
                    if "core" in p.text
                    else ParagraphPriority.DROP
                )
                rows.append(
                    ParagraphStructureVerdict(
                        paragraph_id=p.paragraph_id,
                        page_number=1,
                        section_title=p.section_title,
                        priority=priority,
                        topic_relevance=5,
                        reason="",
                    )
                )
        return PaperStructureAnalysis(paragraphs=rows)

    monkeypatch.setattr(
        "knowledge_engine.src.parsers.paper_structure_analyzer.PaperStructureAnalyzer.analyze",
        fake_analyze,
    )

    units = [
        "This paragraph explains the core algorithm in detail with real substance.",
        "This paragraph is just boilerplate filler with no substance at all here.",
    ]
    out = asyncio.run(m._flash_lite_triage_core_units(units))
    assert out == [units[0]]


@pytest.mark.real_triage
def test_flash_lite_triage_core_units_fails_open_on_error(monkeypatch):
    def boom(self, target_topic, input_paper, **kw):
        raise RuntimeError("flash lite unavailable")

    monkeypatch.setattr(
        "knowledge_engine.src.parsers.paper_structure_analyzer.PaperStructureAnalyzer.analyze",
        boom,
    )
    units = [
        "A perfectly normal paragraph with plenty of real content in it here.",
        "Another normal paragraph with a completely different topic in it too.",
    ]
    out = asyncio.run(m._flash_lite_triage_core_units(units))
    assert out == units, "a Triage failure must fall back to the full input list"


@pytest.mark.real_triage
def test_flash_lite_triage_core_units_empty_input_no_call(monkeypatch):
    called = {"n": 0}

    def spy(self, *a, **kw):
        called["n"] += 1
        raise AssertionError("should not be called for empty input")

    monkeypatch.setattr(
        "knowledge_engine.src.parsers.paper_structure_analyzer.PaperStructureAnalyzer.analyze",
        spy,
    )
    assert asyncio.run(m._flash_lite_triage_core_units([])) == []
    assert called["n"] == 0


def test_mmr_top_by_centroid_is_pure_mmr_no_filtering(monkeypatch):
    """MMR (Step 1b) must not itself filter anything — Triage already ran
    before it. Given N > top_k paragraphs, it selects a diverse top_k subset
    of exactly the paragraphs it was handed, nothing dropped beforehand."""
    paragraphs = [f"p{i}" for i in range(6)]
    vectors = {
        "p0": [1.0, 0.0],
        "p1": [0.99, 0.01],
        "p2": [0.0, 1.0],
        "p3": [0.98, 0.02],
        "p4": [-1.0, 0.0],
        "p5": [0.5, 0.5],
    }
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        lambda texts: [vectors[t] for t in texts],
    )
    out = m._mmr_top_by_centroid(paragraphs, top_k=3)
    assert len(out) == 3
    assert all(p in paragraphs for p in out)


# ---------------------------------------------------------------------------
# Group Batching + TPM guard: _pack_bulk_gate_sub_batches / _run_bulk_gate
# ---------------------------------------------------------------------------


def test_pack_bulk_gate_sub_batches_single_batch_when_small():
    groups = [["a", "b"]]
    code_ids = ["c1"]
    context_by_id = {"a": ["short"], "b": ["short"], "c1": ["def f(): pass"]}
    batches = m._pack_bulk_gate_sub_batches(
        groups, code_ids, context_by_id, max_tokens=250_000
    )
    assert len(batches) == 1
    assert batches[0] == (groups, code_ids)


def test_pack_bulk_gate_sub_batches_splits_when_over_budget():
    # 3 код-юнита, каждый по отдельности маленький, но крошечный max_tokens
    # заставляет открывать новый суб-батч каждый раз, когда юнит превысил бы бегущий итог.
    code_ids = ["c1", "c2", "c3"]
    context_by_id = {cid: ["x" * 200] for cid in code_ids}  # ~одинаковый размер у всех
    per_unit_tokens = m._bulk_gate_unit_tokens({"id": "c1", "extract": "x" * 200})
    assert per_unit_tokens > 0
    batches = m._pack_bulk_gate_sub_batches(
        [], code_ids, context_by_id, max_tokens=per_unit_tokens  # места на ~1 юнит
    )
    assert len(batches) >= 2, "must split across multiple sub-batches"
    all_codes = [cid for _groups, codes in batches for cid in codes]
    assert sorted(all_codes) == code_ids, "no unit dropped by splitting"


def test_pack_bulk_gate_sub_batches_never_splits_a_group_across_batches():
    group = ["a", "b", "c"]
    context_by_id = {cid: ["x" * 500] for cid in group}
    batches = m._pack_bulk_gate_sub_batches(
        [group], [], context_by_id, max_tokens=1  # абсурдно мало — всё равно один юнит
    )
    assert len(batches) == 1
    assert batches[0][0] == [
        group
    ], "a suspect group is one indivisible unit — never split across sub-batches"


def test_run_bulk_gate_splits_into_multiple_real_calls_and_merges(monkeypatch):
    """With a tiny max_tpm, 3 code candidates must trigger 3 separate Flash
    Lite calls (Group Batching), each contributing to the merged canonical_map."""
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available",
        lambda: True,
    )

    call_log = []

    def fake_call(primary_model, system, payload, anchor, schema, label, **kw):
        call_log.append(label)
        import json as _json

        data = _json.loads(payload)
        code_id = data["code_files"][0]["id"]
        from knowledge_engine.schemas.llm_contracts.pre_map_dedup import (
            CanonicalMapContract,
        )

        # каждый код-файл заявляет себя canonical без alias, кроме c1,
        # который "модель" объявляет canonical для c2 в этом тесте.
        if code_id == "c1":
            return CanonicalMapContract(
                mappings=[{"canonical_id": "c1", "aliases": ["c2"]}]
            )
        return CanonicalMapContract(mappings=[])

    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        fake_call,
    )

    context_by_id = {cid: [f"signature for {cid}"] for cid in ("c1", "c2", "c3")}
    per_unit_tokens = m._bulk_gate_unit_tokens(
        {"id": "c1", "extract": "signature for c1"}
    )
    out = asyncio.run(
        m._run_bulk_gate(
            [],
            ["c1", "c2", "c3"],
            context_by_id,
            anchor="",
            max_tpm=per_unit_tokens,
        )
    )
    assert len(call_log) == 3, "one Flash Lite call per sub-batch"
    assert out == {"c1": ["c2"]}, "partial results from each sub-batch must merge"
