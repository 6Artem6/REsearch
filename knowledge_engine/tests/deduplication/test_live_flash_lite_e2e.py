"""Honest, no-mock end-to-end run of the EXISTING Pre-MAP Dedup pipeline
against REAL Gemini Flash Lite (3.1/3.5, via GEMINI_LITE_MODEL's own
fallback chain) and the REAL local BGE-M3 model — no AsyncMock/MagicMock
anywhere on the LLM layer. Triage/MMR logic itself is NOT reimplemented
here: this test calls the production functions directly —
`_flash_lite_triage_core_units` (which wraps the existing
paper_structure_analyzer.PaperStructureAnalyzer CORE/CONTEXT/DROP pass),
the unmodified `greedy_mmr_select`/`_mmr_top_by_centroid`, real BGE-M3
clustering, and the real TPM-guarded Bulk Gate. Costs real API quota and
wall-clock time; run explicitly:

    pytest tests/deduplication/test_live_flash_lite_e2e.py -s

Skipped automatically when GEMINI_API_KEY/GOOGLE_API_KEY is not set.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

import knowledge_engine.services.search.bge_m3_embed as bge_mod
from knowledge_engine.src.deduplication import pre_map_deduplicator as m

_HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

pytestmark = pytest.mark.skipif(
    not _HAS_GEMINI,
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required for the live Flash Lite e2e run",
)


def _candidate(id_, url, text, *, is_code):
    return m.PreMapCandidate(id=id_, url=url, text=text, is_code=is_code)


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _html(text: str) -> str:
    """Wrap plain text in minimal HTML so the real Trafilatura extractor
    (pre_flight_triage._extract_paragraphs, used by Triage Step 1a for TEXT
    candidates) can find paragraphs — it errors on bare plain-text input."""
    return f"<html><body><article><p>{text}</p></article></body></html>"


# ---------------------------------------------------------------------------
# Реальный набор данных (курированные, содержательные абзацы — не крошечные
# игрушечные предложения). Текст: репост Habr/Dzen, GraphRAG (2 автора),
# FastAPI (контрольная, отдельная тема). Код: ceval_gil.c main/3.12,
# DSU C++/Python, quicksort/binary_search.
# ---------------------------------------------------------------------------

_HABR_RAG = (
    "RAG (Retrieval-Augmented Generation) — архитектурный паттерн, объединяющий "
    "поиск релевантного контекста в векторной базе знаний с генерацией финального "
    "ответа большой языковой моделью. Вместо того чтобы полагаться исключительно "
    "на параметрическую память модели, система на каждом запросе извлекает "
    "наиболее похожие по смыслу фрагменты документов через косинусное сходство "
    "эмбеддингов, а затем передаёт их в промпт как дополнительный контекст. Это "
    "заметно снижает частоту галлюцинаций и позволяет модели отвечать на вопросы "
    "о свежих или узкоспециализированных данных, которых не было в обучающей "
    "выборке. Качество RAG-системы напрямую зависит от качества чанкинга "
    "документов, выбора эмбеддинг-модели и стратегии реранкинга кандидатов перед "
    "финальной генерацией."
)
_DZEN_RAG_REPOST = (
    "RAG (Retrieval-Augmented Generation) — архитектурный паттерн, объединяющий "
    "поиск релевантного контекста в векторной базе знаний с генерацией финального "
    "ответа большой языковой моделью. Вместо того чтобы полагаться исключительно "
    "на параметрическую память модели, система на каждом запросе извлекает "
    "наиболее похожие по смыслу фрагменты документов через косинусное сходство "
    "эмбеддингов и передаёт их в промпт как дополнительный контекст. Это заметно "
    "снижает частоту галлюцинаций и позволяет модели отвечать на вопросы о "
    "свежих или узкоспециализированных данных, отсутствовавших в обучающей "
    "выборке. Качество RAG-системы напрямую зависит от качества чанкинга "
    "документов, выбора эмбеддинг-модели и стратегии реранкинга кандидатов перед "
    "финальной генерацией ответа."
)
_GRAPHRAG_AUTHOR_A = (
    "GraphRAG расширяет классический RAG, добавляя явный граф знаний: сущности "
    "и связи между ними извлекаются из корпуса документов заранее, а не ищутся "
    "на лету по плоскому векторному индексу. Ответ на сложный многошаговый "
    "вопрос строится через обход подграфа вокруг релевантных сущностей, что "
    "позволяет агрегировать факты из разных документов, которые классический "
    "vector-similarity search никогда не свяжет вместе. Это особенно полезно "
    "для вопросов вида «как сущность А связана с сущностью Б через несколько "
    "промежуточных шагов», где обычный RAG теряется из-за отсутствия явной "
    "структуры в индексе."
)
_GRAPHRAG_AUTHOR_B = (
    "В отличие от плоского векторного поиска, GraphRAG сначала строит граф "
    "знаний из корпуса: узлы — сущности, рёбра — отношения между ними, "
    "извлечённые LLM на этапе индексации. При ответе на запрос система находит "
    "релевантные узлы и обходит их локальные окрестности в графе, собирая "
    "связанные факты, которые были бы недостижимы для наивного top-k поиска по "
    "эмбеддингам. Такой подход особенно эффективен там, где ответ требует "
    "агрегации информации из нескольких документов, логически связанных через "
    "общие сущности графа."
)
_FASTAPI_ASYNC = (
    "FastAPI позволяет объявлять асинхронные обработчики через async def, "
    "которые исполняются на event loop без блокировки остальных запросов, пока "
    "ожидается I/O — обращение к базе данных, внешний HTTP-запрос или файловая "
    "операция. Синхронные обработчики (def) автоматически диспетчеризуются в "
    "отдельный тредпул, чтобы не блокировать event loop даже если внутри есть "
    "блокирующий вызов. Выбор между async def и def определяется тем, вызывает "
    "ли тело функции реально асинхронные библиотеки — иначе смешивание "
    "блокирующего кода внутри async def даёт обратный эффект и замедляет весь "
    "сервер целиком."
)

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
_C_312 = _C_MAIN.replace("handle_wait_error", "report_wait_failure").replace(
    "main branch", "3.12 branch"
)
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


def _build_candidates() -> list[m.PreMapCandidate]:
    return [
        _candidate(
            "habr_rag",
            "https://habr.com/ru/companies/vtb/articles/788172/",
            _html(_HABR_RAG),
            is_code=False,
        ),
        _candidate(
            "dzen_rag_repost",
            "https://dzen.ru/a/ZcbP7X3d8xS2F9fU",
            _html(_DZEN_RAG_REPOST),
            is_code=False,
        ),
        _candidate(
            "medium_graphrag",
            "https://medium.com/@zilliz_learn/what-is-graphrag-a-complete-guide-813f88f28c2e",
            _html(_GRAPHRAG_AUTHOR_A),
            is_code=False,
        ),
        _candidate(
            "tds_graphrag",
            "https://towardsdatascience.com/graphrag-the-unreasonable-effectiveness-of-knowledge-graphs-6825c317f7d1",
            _html(_GRAPHRAG_AUTHOR_B),
            is_code=False,
        ),
        _candidate(
            "fastapi_async_docs",
            "https://fastapi.tiangolo.com/async/",
            _html(_FASTAPI_ASYNC),
            is_code=False,
        ),
        _candidate(
            "ceval_gil_main",
            "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c",
            _C_MAIN,
            is_code=True,
        ),
        _candidate(
            "ceval_gil_312",
            "https://raw.githubusercontent.com/python/cpython/3.12/Python/ceval_gil.c",
            _C_312,
            is_code=True,
        ),
        _candidate(
            "cpp_dsu",
            "https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/data_structures/disjoint_set.cpp",
            _CPP_DSU,
            is_code=True,
        ),
        _candidate(
            "py_dsu",
            "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/disjoint_set/disjoint_set.py",
            _PY_DSU,
            is_code=True,
        ),
        _candidate(
            "quicksort", "https://example.com/quicksort.py", _PY_QUICKSORT, is_code=True
        ),
        _candidate(
            "binary_search",
            "https://example.com/binary_search.py",
            _PY_BINARY_SEARCH,
            is_code=True,
        ),
    ]


@pytest.mark.integration
def test_live_flash_lite_e2e():
    candidates = _build_candidates()

    # -- Считаем РЕАЛЬНЫЕ вызовы BGE embed() (pass-through шпион — данные не подделывает) --
    orig_embed = bge_mod.embed_texts_bge_m3
    embed_calls = {"n": 0}

    def counting_embed(texts):
        embed_calls["n"] += 1
        return orig_embed(texts)

    bge_mod.embed_texts_bge_m3 = counting_embed

    # -- Захватываем СЫРОЙ (до санитизации) ответ Flash Lite на каждый суб-батч --
    orig_call_once = m._call_bulk_gate_once
    raw_responses: list[dict] = []

    async def capturing_call_once(suspect_groups, code_ids, context_by_id, **kw):
        t0 = time.monotonic()
        result = await orig_call_once(suspect_groups, code_ids, context_by_id, **kw)
        elapsed = time.monotonic() - t0
        payload = m._build_bulk_gate_payload(suspect_groups, code_ids, context_by_id)
        tokens = m._bulk_gate_unit_tokens(json.loads(payload))
        raw_responses.append(result)
        _hr(f"REAL Flash Lite Bulk Gate call — sub-batch {kw.get('batch_label', '?')}")
        print(f"  groups={len(suspect_groups)} code_files={len(code_ids)}")
        print(
            f"  actual token consumption ≈ {tokens} (TPM guard = {m.PRE_MAP_DEDUP_BULK_GATE_MAX_TPM})"
        )
        print(f"  latency = {elapsed:.2f}s")
        print(f"  RAW canonical_map (before sanitization): {result!r}")
        print(f"  Payload sent to Flash Lite (verbatim): {payload}")
        return result

    m._call_bulk_gate_once = capturing_call_once

    try:
        # === Step 1a: РЕАЛЬНЫЙ Triage через Flash Lite (CORE/CONTEXT/DROP) =====
        # Вызывает СУЩЕСТВУЮЩУЮ продакшн-функцию Triage напрямую — тот же
        # проход paper_structure_analyzer.PaperStructureAnalyzer, который
        # использует продакшн-пайплайн, никакой локальной эвристики/фильтра.
        _hr("STEP 1a — Triage via real Flash Lite (paper_structure_analyzer CORE/CONTEXT/DROP)")
        context_by_id: dict[str, list[str]] = {}
        is_code_by_id: dict[str, bool] = {}
        for c in candidates:
            is_code_by_id[c.id] = c.is_code
            if c.is_code:
                raw_units = m._ast_semantic_extracts(c.text, c.url, min_chars=20)
                if not raw_units:
                    raw_units = m._code_paragraphs_from_raw_text(c.text, min_chars=20)
            else:
                raw_units = m._extract_paragraphs(c.text, c.url, min_chars=20)

            core_units = asyncio.run(
                m._flash_lite_triage_core_units(
                    raw_units, label=f"live_e2e:{c.id}"
                )
            )
            kind = "CODE" if c.is_code else "TEXT"
            print(f"\n  [{kind}] {c.id} — Triage: {len(core_units)}/{len(raw_units)} kept as CORE")
            for unit in core_units:
                print(f"      CORE: {unit[:100]!r}")

            # === Step 1b: центроидный MMR / AST top-k по CORE-юнитам ===
            if c.is_code:
                context_by_id[c.id] = core_units[:5]
            else:
                context_by_id[c.id] = m._mmr_top_by_centroid(core_units, top_k=5)
            print(f"  [{kind}] {c.id} — final MMR/AST-capped fingerprint ({len(context_by_id[c.id])} unit(s)):")
            for unit in context_by_id[c.id]:
                print(f"      {unit[:100]!r}")

        text_ids = [c.id for c in candidates if not is_code_by_id[c.id]]
        code_ids = [c.id for c in candidates if is_code_by_id[c.id]]

        # === Step 2: РЕАЛЬНАЯ BGE-кластеризация ===
        _hr("STEP 2 — REAL BGE-M3 Cosine Similarity + Union-Find suspect groups")
        doc_vectors: dict[str, list[float]] = {}
        for cid in text_ids:
            vec = m._pool_vector(context_by_id.get(cid) or [])
            if vec is not None:
                doc_vectors[cid] = vec

        for i in range(len(text_ids)):
            for j in range(i + 1, len(text_ids)):
                a, b = text_ids[i], text_ids[j]
                if a in doc_vectors and b in doc_vectors:
                    sim = m._cosine(doc_vectors[a], doc_vectors[b])
                    tag = (
                        "≥ 0.80 -> SUSPECT"
                        if sim >= m.PRE_MAP_DEDUP_COSINE_THRESHOLD
                        else "< 0.80"
                    )
                    print(f"  cosine({a}, {b}) = {sim:.4f}   [{tag}]")

        groups = m._cluster_text_candidates(
            doc_vectors, threshold=m.PRE_MAP_DEDUP_COSINE_THRESHOLD
        )
        print("\n  Union-Find groups:")
        for g in groups:
            tag = "SUSPECT GROUP" if len(g) > 1 else "autonomous"
            print(f"    {sorted(g)}  [{tag}]")
        print(f"\n  Real BGE embed() call count so far = {embed_calls['n']}")

        # === Step 3+4: полный pipeline (реальный Flash Lite Bulk Gate + санитизация + пулинг) ===
        _hr(
            "STEP 3/4 — Full deduplicate_before_map_reduce() run (REAL Triage + REAL Bulk Gate, TPM-guarded)"
        )
        t0 = time.monotonic()
        result = asyncio.run(
            m.deduplicate_before_map_reduce(candidates, anchor="pre_map_dedup_live_e2e")
        )
        total_elapsed = time.monotonic() - t0

        _hr("STEP 5 — Final report")
        print(
            f"  Total wall time for deduplicate_before_map_reduce(): {total_elapsed:.2f}s"
        )
        print(
            f"  Total REAL BGE embed() calls across the whole run: {embed_calls['n']}"
        )
        print(f"  Sanitized alias_map: {result.alias_map!r}")
        print("\n  Final decisions:")
        for c in candidates:
            d = result.decisions[c.id]
            role = "CANONICAL" if d.is_canonical else f"ALIAS -> {d.canonical_id}"
            print(f"    {c.id:20s} [{'CODE' if c.is_code else 'TEXT'}] -> {role}")

        # ------------------------------------------------------------------
        # Ассерты — строгие там, где мы полностью контролируем механику. Проверки
        # DSU/FastAPI ниже проверяют РЕАЛЬНОЕ живое суждение модели о дублях,
        # которое не гарантированно детерминировано между прогонами/версиями
        # модели; падение здесь — находка о живом поведении модели, не
        # обязательно баг пайплайна — см. RAW canonical_map/payload выше.
        # ------------------------------------------------------------------
        assert code_ids, "code candidates must be present in this fixture set"
        for cid in code_ids:
            assert context_by_id[cid], f"AST extraction must be non-empty for {cid}"
        for cid in text_ids:
            assert context_by_id[cid], f"Triage->MMR fingerprint must be non-empty for {cid}"

        # Механическая гарантия: никто не теряется молча.
        assert set(result.decisions.keys()) == {c.id for c in candidates}

        # Явно запрошенные проверки: межъязыковой дубль DSU (C++ vs Python)
        # схлопывается в CANONICAL + ALIAS; кандидаты с разной темой текста
        # (RAG vs GraphRAG vs FastAPI) остаются CANONICAL.
        cpp_decision = result.decisions["cpp_dsu"]
        py_decision = result.decisions["py_dsu"]
        dsu_collapsed = (
            not cpp_decision.is_canonical and cpp_decision.canonical_id == "py_dsu"
        ) or (not py_decision.is_canonical and py_decision.canonical_id == "cpp_dsu")
        print(
            f"\n  Cross-language DSU (C++ vs Python) collapsed to ALIAS: {dsu_collapsed}"
        )

        fastapi_decision = result.decisions["fastapi_async_docs"]
        print(
            f"  fastapi_async_docs (distinct topic) stayed CANONICAL: {fastapi_decision.is_canonical}"
        )

        assert dsu_collapsed, (
            "REAL Flash Lite failed to recognize the C++/Python DSU pair as the "
            "same algorithm — see the RAW canonical_map printed above for what "
            "it actually returned"
        )
        assert fastapi_decision.is_canonical, (
            "FastAPI async docs is a distinct topic and must stay CANONICAL, not "
            "be folded into an unrelated group by the real Bulk Gate"
        )
    finally:
        bge_mod.embed_texts_bge_m3 = orig_embed
        m._call_bulk_gate_once = orig_call_once
