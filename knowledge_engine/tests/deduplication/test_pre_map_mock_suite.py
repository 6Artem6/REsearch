"""Pre-MAP Dedup — comprehensive isolated mock test suite (no network I/O).

Covers:
  Group 1 — TEXT pipeline: paraphrase/copy clustering, BGE-suspect-but-Lite-
            keeps-both, below-threshold skips Lite entirely.
  Group 2 — CODE pipeline: git-branch mirrors, cross-language same algorithm,
            genuinely different code (BGE never called for code candidates).
  Group 3 — Pool Replenishment: dedup collapsing a duplicate pair frees no
            "slots" per se, but naturally yields exactly the right number of
            CANONICAL sources once uniques are counted alongside it.
  Group 4 — Fail-Open (BGE error / Flash Lite error/timeout) and
            canonical_map sanitization (hallucinated / self / cyclic alias).

All external boundaries (BGE-M3 embed, Flash Lite bulk gate) are intercepted
via unittest.mock — MagicMock for the synchronous BGE embed call, AsyncMock
for the async Flash Lite bulk-gate boundary (_run_bulk_gate). Zero network
requests; all "sources" below are curated in-memory content snippets, not
fetched from the real URLs referenced in their comments/ids.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

from knowledge_engine.src.deduplication import pre_map_deduplicator as m

# ---------------------------------------------------------------------------
# Общие хелперы
# ---------------------------------------------------------------------------


def _candidate(id_: str, url: str, text: str, *, is_code: bool) -> m.PreMapCandidate:
    return m.PreMapCandidate(id=id_, url=url, text=text, is_code=is_code)


def _cos_pair(cosine: float) -> tuple[list[float], list[float]]:
    """Two 2D unit vectors with EXACTLY the given pairwise cosine similarity."""
    theta = math.acos(cosine)
    return [1.0, 0.0], [math.cos(theta), math.sin(theta)]


def _use_identity_paragraph_extraction(monkeypatch) -> None:
    """TEXT candidates in this suite carry their one representative
    paragraph directly in .text — no real Trafilatura/HTML parsing needed,
    just treat the raw text as the (single) extracted paragraph."""
    monkeypatch.setattr(
        m, "_extract_paragraphs", lambda html, url, min_chars: [html.strip()]
    )


def _embed_lookup_mock(vector_by_text: dict[str, list[float]]) -> MagicMock:
    return MagicMock(side_effect=lambda texts: [vector_by_text[t] for t in texts])


# ---------------------------------------------------------------------------
# Group 1: текстовый pipeline
# ---------------------------------------------------------------------------


def test_group1_1_paraphrase_copy_becomes_alias(monkeypatch):
    """1.1 — Original (Habr RAG overview) vs a near-verbatim repost (Dzen).
    BGE must cluster them (cosine >= 0.80); Flash Lite marks one CANONICAL,
    the other ALIAS."""
    original = _candidate(
        "habr_original",
        "https://habr.com/ru/companies/vtb/articles/788172/",
        "RAG объединяет извлечение релевантного контекста из векторной базы "
        "знаний с генерацией ответа языковой моделью, что снижает галлюцинации.",
        is_code=False,
    )
    repost = _candidate(
        "dzen_repost",
        "https://dzen.ru/a/ZcbP7X3d8xS2F9fU",
        "RAG объединяет извлечение релевантного контекста из векторной базы "
        "знаний с генерацией ответа языковой моделью, что снижает галлюцинации.",
        is_code=False,
    )
    _use_identity_paragraph_extraction(monkeypatch)
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock({original.text: [1.0, 0.0], repost.text: [1.0, 0.0]}),
    )

    bulk_gate = AsyncMock(return_value={"habr_original": ["dzen_repost"]})
    monkeypatch.setattr(m, "_run_bulk_gate", bulk_gate)

    result = asyncio.run(m.deduplicate_before_map_reduce([original, repost]))

    bulk_gate.assert_awaited_once()
    suspect_groups = bulk_gate.await_args.args[0]
    assert sorted(suspect_groups[0]) == ["dzen_repost", "habr_original"]

    assert result.canonical_ids() == ["habr_original"]
    assert result.is_alias("dzen_repost")
    assert result.canonical_of("dzen_repost") == "habr_original"
    assert result.alias_map == {"habr_original": ["dzen_repost"]}


def test_group1_2_suspect_group_but_lite_keeps_both_canonical(monkeypatch):
    """1.2 — Two different-authors' GraphRAG overviews land in a suspect
    group (cosine == 0.82 >= 0.80) but Flash Lite judges them NOT duplicates
    (different angle on the same topic) -> canonical_map stays empty -> both
    remain CANONICAL, neither is lost."""
    author_a = _candidate(
        "medium_graphrag",
        "https://medium.com/@zilliz_learn/what-is-graphrag-a-complete-guide-813f88f28c2e",
        "GraphRAG extracts entities and relations into a knowledge graph, then "
        "retrieves subgraphs to ground the LLM's answer in structured facts.",
        is_code=False,
    )
    author_b = _candidate(
        "tds_graphrag",
        "https://towardsdatascience.com/graphrag-the-unreasonable-effectiveness-of-knowledge-graphs-6825c317f7d1",
        "Knowledge graphs built from entity/relation extraction let GraphRAG "
        "retrieve structured context, grounding generation beyond flat vector search.",
        is_code=False,
    )
    _use_identity_paragraph_extraction(monkeypatch)
    vec_a, vec_b = _cos_pair(0.82)
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock({author_a.text: vec_a, author_b.text: vec_b}),
    )

    bulk_gate = AsyncMock(return_value={})  # Lite: не настоящие дубли
    monkeypatch.setattr(m, "_run_bulk_gate", bulk_gate)

    result = asyncio.run(m.deduplicate_before_map_reduce([author_a, author_b]))

    bulk_gate.assert_awaited_once()  # cosine 0.82 >= 0.80 всё равно должен дойти до Lite
    assert sorted(result.canonical_ids()) == ["medium_graphrag", "tds_graphrag"]
    assert result.alias_map == {}


def test_group1_3_below_threshold_skips_lite_entirely(monkeypatch):
    """1.3 — Same-author, different-topic posts (agents vs similarity
    metrics), cosine < 0.80 -> never reaches Flash Lite at all (0 LLM
    calls), both instantly CANONICAL."""
    agents = _candidate(
        "weng_agents",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "LLM-powered autonomous agents combine planning, memory and tool use "
        "to decompose complex tasks into executable sub-steps.",
        is_code=False,
    )
    similarity_metrics = _candidate(
        "weng_similarity",
        "https://lilianweng.github.io/posts/2020-10-29-similarities/",
        "Cosine similarity, Jaccard index and other distance metrics quantify "
        "how close two embedding vectors or sets are to each other.",
        is_code=False,
    )
    _use_identity_paragraph_extraction(monkeypatch)
    vec_a, vec_b = _cos_pair(0.5)
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock({agents.text: vec_a, similarity_metrics.text: vec_b}),
    )

    bulk_gate = AsyncMock(return_value={"should_not": ["be_called"]})
    monkeypatch.setattr(m, "_run_bulk_gate", bulk_gate)

    result = asyncio.run(m.deduplicate_before_map_reduce([agents, similarity_metrics]))

    bulk_gate.assert_not_awaited()
    assert sorted(result.canonical_ids()) == ["weng_agents", "weng_similarity"]
    assert result.alias_map == {}


# ---------------------------------------------------------------------------
# Group 2: код-pipeline — курированные сниппеты, не реальные (грязные,
# насыщенные препроцессором) исходные файлы; проверено, что чисто парсятся Tree-Sitter'ом.
# ---------------------------------------------------------------------------

_C_MAIN = """
/* GIL acquisition — main branch */
static void take_gil(PyThreadState *tstate) {
    int err;
    while (gil_locked) {
        err = wait_on_condvar(&gil_cond, &gil_mutex);
        if (err) {
            handle_wait_error(err);
        }
    }
    gil_locked = 1;
    signal_active_thread(tstate);
}

static void drop_gil(PyThreadState *tstate) {
    gil_locked = 0;
    notify_waiters(&gil_cond);
}
"""

_C_312 = """
/* GIL acquisition — 3.12 branch */
static void take_gil(PyThreadState *tstate) {
    int err;
    while (gil_locked) {
        err = wait_on_condvar(&gil_cond, &gil_mutex);
        if (err) {
            report_wait_failure(err);
        }
    }
    gil_locked = 1;
    signal_active_thread(tstate);
}

static void drop_gil(PyThreadState *tstate) {
    gil_locked = 0;
    notify_waiters(&gil_cond);
}
"""

_CPP_DSU = """
class DisjointSet {
public:
    std::vector<int> parent;

    int find_set(int v) {
        if (v == parent[v]) {
            return v;
        }
        return parent[v] = find_set(parent[v]);
    }

    void union_sets(int a, int b) {
        a = find_set(a);
        b = find_set(b);
        if (a != b) {
            parent[b] = a;
        }
    }
};
"""

_PY_DSU = """
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find_set(self, v):
        if v == self.parent[v]:
            return v
        self.parent[v] = self.find_set(self.parent[v])
        return self.parent[v]

    def union_sets(self, a, b):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            self.parent[b] = a
"""

_PY_QUICKSORT = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
"""

_PY_BINARY_SEARCH = """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""


def test_group2_1_git_branch_mirrors_dedupe_via_ast(monkeypatch):
    """2.1 — Two branches of the same C file (ceval_gil.c main vs 3.12).
    BGE must never be called for code; the pair is unconditionally sent to
    Flash Lite, which resolves them via AST signatures into 1 CANONICAL +
    1 ALIAS."""
    main_branch = _candidate(
        "ceval_gil_main",
        "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c",
        _C_MAIN,
        is_code=True,
    )
    branch_312 = _candidate(
        "ceval_gil_312",
        "https://raw.githubusercontent.com/python/cpython/3.12/Python/ceval_gil.c",
        _C_312,
        is_code=True,
    )
    embed_spy = MagicMock()
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", embed_spy
    )
    code_dedup = AsyncMock(return_value={"ceval_gil_main": ["ceval_gil_312"]})
    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.code_deduplicator.deduplicate_code_candidates",
        code_dedup,
    )

    result = asyncio.run(m.deduplicate_before_map_reduce([main_branch, branch_312]))

    embed_spy.assert_not_called()
    code_dedup.assert_awaited_once()
    (code_candidates,), _kwargs = code_dedup.await_args
    assert sorted(c.id for c in code_candidates) == [
        "ceval_gil_312",
        "ceval_gil_main",
    ], "code must be routed to the isolated Code Deduplication module"

    assert result.canonical_ids() == ["ceval_gil_main"]
    assert result.is_alias("ceval_gil_312")
    assert result.canonical_of("ceval_gil_312") == "ceval_gil_main"


def test_group2_2_cross_language_same_algorithm_merges(monkeypatch):
    """2.2 — Same Disjoint Set Union algorithm in C++ and Python. AST
    extracts succeed for both languages; Flash Lite recognizes the
    cross-language duplicate and merges them into one group."""
    cpp_dsu = _candidate(
        "cpp_dsu",
        "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/data_structures/disjoint_set.cpp",
        _CPP_DSU,
        is_code=True,
    )
    py_dsu = _candidate(
        "py_dsu",
        "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/disjoint_set/disjoint_set.py",
        _PY_DSU,
        is_code=True,
    )
    embed_spy = MagicMock()
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", embed_spy
    )
    code_dedup = AsyncMock(return_value={"cpp_dsu": ["py_dsu"]})
    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.code_deduplicator.deduplicate_code_candidates",
        code_dedup,
    )

    result = asyncio.run(m.deduplicate_before_map_reduce([cpp_dsu, py_dsu]))

    embed_spy.assert_not_called()
    (code_candidates,), _kwargs = code_dedup.await_args
    assert sorted(c.id for c in code_candidates) == ["cpp_dsu", "py_dsu"]

    assert result.canonical_ids() == ["cpp_dsu"]
    assert result.is_alias("py_dsu")
    assert result.canonical_of("py_dsu") == "cpp_dsu"


def test_group2_3_different_code_both_stay_canonical(monkeypatch):
    """2.3 — Two genuinely different Python functions (quicksort vs binary
    search). Flash Lite finds no duplicate -> both stay CANONICAL."""
    quicksort = _candidate(
        "quicksort", "https://example.com/quicksort.py", _PY_QUICKSORT, is_code=True
    )
    binsearch = _candidate(
        "binary_search",
        "https://example.com/binary_search.py",
        _PY_BINARY_SEARCH,
        is_code=True,
    )
    code_dedup = AsyncMock(return_value={})
    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.code_deduplicator.deduplicate_code_candidates",
        code_dedup,
    )

    result = asyncio.run(m.deduplicate_before_map_reduce([quicksort, binsearch]))

    code_dedup.assert_awaited_once()
    assert sorted(result.canonical_ids()) == ["binary_search", "quicksort"]
    assert result.alias_map == {}


# ---------------------------------------------------------------------------
# Group 3: пополнение пула (Pool Replenishment)
# ---------------------------------------------------------------------------


def test_group3_pool_replenishment_dedup_frees_slots_for_uniques(monkeypatch):
    """2 near-duplicate BGE-M3 paper mirrors (arxiv abs vs html rendering)
    + 4 independently unique sources. Dedup collapses the mirror pair into
    1 CANONICAL + 1 ALIAS; the 4 uniques never cluster with anything, so the
    final CANONICAL batch for MAP-REDUCE naturally lands at exactly 5
    sources (1 + 4). The ALIAS keeps its URL, gets alias_of set (verified at
    the decision level here — end-to-end propagation into
    CurriculumSearchHit.alias_of is covered by
    test_source_material_pipeline_dedup.py) and does not consume a second
    MAP-REDUCE run."""
    arxiv_abs = _candidate(
        "arxiv_abs",
        "https://arxiv.org/abs/2402.03216",
        "BGE-M3 is an embedding model distinguished by versatility in "
        "multi-lingual, multi-granularity and multi-functionality retrieval.",
        is_code=False,
    )
    arxiv_html = _candidate(
        "arxiv_html",
        "https://arxiv.org/html/2402.03216v1",
        "BGE-M3 is an embedding model distinguished by versatility in "
        "multi-lingual, multi-granularity and multi-functionality retrieval.",
        is_code=False,
    )
    flag_embedding = _candidate(
        "flag_embedding_repo",
        "https://github.com/FlagOpen/FlagEmbedding",
        "Official repository for BGE embedding models, training scripts and "
        "evaluation benchmarks maintained by the FlagOpen team.",
        is_code=False,
    )
    llm_patterns = _candidate(
        "llm_patterns_blog",
        "https://eugeneyan.com/writing/llm-patterns/",
        "A survey of practical design patterns for building LLM-powered "
        "systems in production: evals, RAG, guardrails, caching.",
        is_code=False,
    )
    cloudflare_vectorize = _candidate(
        "cloudflare_vectorize",
        "https://blog.cloudflare.com/vectorize-vector-database-open-beta/",
        "Cloudflare Vectorize is a globally distributed vector database now "
        "available in open beta for storing and querying embeddings.",
        is_code=False,
    )
    fastapi_async = _candidate(
        "fastapi_async_docs",
        "https://fastapi.tiangolo.com/async/",
        "FastAPI lets you define path operations with async def, running "
        "them concurrently on the event loop without blocking the server.",
        is_code=False,
    )
    all_candidates = [
        arxiv_abs,
        arxiv_html,
        flag_embedding,
        llm_patterns,
        cloudflare_vectorize,
        fastapi_async,
    ]

    _use_identity_paragraph_extraction(monkeypatch)
    # arxiv_abs/arxiv_html имеют абсолютно одинаковый текст -> идентичный
    # вектор (cosine 1.0); у каждого остального источника своя ортогональная
    # ось, чтобы ничего не сгруппировалось случайно.
    vector_by_text = {
        arxiv_abs.text: [1.0, 0.0, 0.0, 0.0, 0.0],
        arxiv_html.text: [1.0, 0.0, 0.0, 0.0, 0.0],
        flag_embedding.text: [0.0, 1.0, 0.0, 0.0, 0.0],
        llm_patterns.text: [0.0, 0.0, 1.0, 0.0, 0.0],
        cloudflare_vectorize.text: [0.0, 0.0, 0.0, 1.0, 0.0],
        fastapi_async.text: [0.0, 0.0, 0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock(vector_by_text),
    )

    async def fake_bulk_gate(suspect_groups, code_ids, context_by_id, *, anchor):
        assert len(suspect_groups) == 1, "only the arxiv mirror pair must cluster"
        assert sorted(suspect_groups[0]) == ["arxiv_abs", "arxiv_html"]
        assert code_ids == []
        return {"arxiv_abs": ["arxiv_html"]}

    monkeypatch.setattr(m, "_run_bulk_gate", fake_bulk_gate)

    result = asyncio.run(m.deduplicate_before_map_reduce(all_candidates))

    canonical = sorted(result.canonical_ids())
    assert canonical == sorted(
        [
            "arxiv_abs",
            "flag_embedding_repo",
            "llm_patterns_blog",
            "cloudflare_vectorize",
            "fastapi_async_docs",
        ]
    )
    assert len(canonical) == 5, "exactly 5 CANONICAL sources reach MAP-REDUCE"

    assert result.is_alias("arxiv_html")
    assert result.canonical_of("arxiv_html") == "arxiv_abs"
    assert result.alias_map == {"arxiv_abs": ["arxiv_html"]}
    alias_decision = result.decisions["arxiv_html"]
    assert alias_decision.canonical_id == "arxiv_abs"
    assert alias_decision.context_units, "alias must still carry its own fingerprint"


# ---------------------------------------------------------------------------
# Group 4: Fail-Open + санитизация canonical_map
# ---------------------------------------------------------------------------


def test_group4_1a_bge_error_fails_open_all_canonical(monkeypatch):
    """4.1 — BGE embed() raises during both fingerprinting and clustering.
    Every candidate must stay CANONICAL and the pipeline must not raise."""
    docs = [
        _candidate(
            f"doc{i}",
            f"https://example.com/{i}",
            f"unique paragraph content {i}",
            is_code=False,
        )
        for i in range(3)
    ]
    # >top_k абзацев, чтобы _mmr_top_by_centroid реально попытался вызвать
    # embed(), а не срезал путь на ветке passthrough.
    monkeypatch.setattr(
        m,
        "_extract_paragraphs",
        lambda html, url, min_chars: [f"{html} sentence {i}" for i in range(7)],
    )
    embed_boom = MagicMock(side_effect=RuntimeError("BGE service unavailable"))
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", embed_boom
    )
    trace_spy = MagicMock()
    monkeypatch.setattr(m, "trace", trace_spy)

    result = asyncio.run(m.deduplicate_before_map_reduce(docs))

    assert sorted(result.canonical_ids()) == ["doc0", "doc1", "doc2"]
    assert result.alias_map == {}
    assert any(
        "✗" in str(call.args[0]) for call in trace_spy.call_args_list
    ), "a BGE failure must be logged (warning), not silently swallowed"


def test_group4_1b_flash_lite_timeout_fails_open_all_canonical(monkeypatch):
    """4.1 — Flash Lite (Gemini) raises/times out. Both candidates that
    would otherwise have clustered must still stay CANONICAL."""
    a = _candidate(
        "a", "https://example.com/a", "shared paragraph text here", is_code=False
    )
    b = _candidate(
        "b", "https://example.com/b", "shared paragraph text here", is_code=False
    )
    _use_identity_paragraph_extraction(monkeypatch)
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock({a.text: [1.0, 0.0], b.text: [1.0, 0.0]}),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.is_gemini_available",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.gemini_stateless.run_gemini_structured_with_chain",
        MagicMock(side_effect=TimeoutError("flash lite timed out")),
    )
    trace_spy = MagicMock()
    monkeypatch.setattr(m, "trace", trace_spy)

    result = asyncio.run(m.deduplicate_before_map_reduce([a, b]))

    assert sorted(result.canonical_ids()) == ["a", "b"]
    assert result.alias_map == {}
    assert any("✗" in str(call.args[0]) for call in trace_spy.call_args_list)


def test_group4_2_sanitizes_hallucinated_self_and_cyclic_aliases(monkeypatch):
    """4.2 — Flash Lite response contains a hallucinated id, a self-alias,
    and a genuine cycle (a claims b, b claims a back). _sanitize_canonical_map
    must resolve all of this into a clean, acyclic assignment."""
    a = _candidate("a", "https://example.com/a", "paragraph a text", is_code=False)
    b = _candidate("b", "https://example.com/b", "paragraph b text", is_code=False)
    _use_identity_paragraph_extraction(monkeypatch)
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _embed_lookup_mock({a.text: [1.0, 0.0], b.text: [1.0, 0.0]}),
    )
    bulk_gate = AsyncMock(
        return_value={
            "a": ["b", "a", "nonexistent_id"],  # self-alias + галлюцинированный id
            "b": ["a"],  # цикл: b пытается заново заявить a, который уже canonical
        }
    )
    monkeypatch.setattr(m, "_run_bulk_gate", bulk_gate)

    result = asyncio.run(m.deduplicate_before_map_reduce([a, b]))

    assert result.alias_map == {"a": ["b"]}
    assert result.canonical_ids() == ["a"]
    assert result.is_alias("b")
    assert result.canonical_of("b") == "a"
