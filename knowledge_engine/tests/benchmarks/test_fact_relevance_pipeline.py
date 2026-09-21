"""A/B benchmark: Two-Stage Fact Relevance & Anchor Filtering vs current MAP->REDUCE.

Validated A/B harness (see ``anchor_relevance.py``, now wired into
``blog_spatial_summarizer.py`` / ``dialog_atoms_rag.py`` behind feature
flags, both off by default). Runs the REAL production MAP/REDUCE pipeline
(Gemma cloud) plus the real Cross-Encoder reranker on two real articles
pulled from Postgres/pgvector (``tests/fixtures/articles/*.json``):

1. Baseline            — current MAP -> REDUCE, no filtering.
2. Variant A            — MAP -> Synthetic Anchor (0 LLM calls) rerank & cut -> REDUCE.
3. Variant B            — MAP -> Fast LLM Anchor rerank & cut -> REDUCE.
4. Variant C            — MAP -> REDUCE -> Anchor rerank & cut (post-hoc).
5. Fact Weighting demo — pure query rerank vs hybrid (query + core_relevance_score)
   rerank of the same candidate atoms, to see whether PRINCIPLE/MECHANIC atoms
   rise above narrow INSTANCE atoms at equal query relevance.

Real Gemma/Gemini cloud calls + real Cross-Encoder inference — not free, not
instant. Skipped by default; run explicitly:

    RUN_FACT_RELEVANCE_BENCHMARK=1 PYTHONPATH=. ./.venv/bin/python -m pytest \\
        knowledge_engine/tests/benchmarks/test_fact_relevance_pipeline.py -v -s
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import httpx
import pytest

from knowledge_engine.src.adapters.llm_providers.gemma_client import (
    RateLimitedLLMClient,
)
from knowledge_engine.src.config.settings import (
    BLOG_SPATIAL_TIMEOUT_SEC,
    GEMMA_MAP_FORCE_PER_MODEL_LIMITS,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
    generate_answer_from_facts,
    get_fast_llm_anchor,
    get_synthetic_anchor,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_schemas import (
    FinalArticleSummaryResponse,
    MapWindowResponse,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    map_reduce_jobs_pooled_async,
    run_reduce,
)
from knowledge_engine.src.domains.ingestion.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "articles"
_THRESHOLD = 0.35
_ALPHA = 0.7

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_FACT_RELEVANCE_BENCHMARK", "").strip() != "1",
    reason=(
        "Real Gemma/Gemini cloud + Cross-Encoder calls, not mocked, not free. "
        "Opt in explicitly: RUN_FACT_RELEVANCE_BENCHMARK=1 pytest "
        "knowledge_engine/tests/benchmarks/test_fact_relevance_pipeline.py -v -s"
    ),
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _load_fixture(name: str) -> dict[str, Any]:
    with open(_FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _job_from_fixture(article: dict[str, Any]) -> MapReduceArticleJob:
    windows = [
        TokenWindowChunk(window_index=w["window_index"], body=w["body"])
        for w in article["windows"]
    ]
    return MapReduceArticleJob(
        job_id=article["url"],
        title=article["title"],
        url=article["url"],
        windows=windows,
    )


def _new_http_client_and_rl() -> Tuple[httpx.AsyncClient, RateLimitedLLMClient]:
    client = httpx.AsyncClient(timeout=httpx.Timeout(BLOG_SPATIAL_TIMEOUT_SEC))
    rl = RateLimitedLLMClient(map_parallel_streams=GEMMA_MAP_FORCE_PER_MODEL_LIMITS)
    return client, rl


def _raw_atoms(map_results: List[MapWindowResponse | None]) -> List[KnowledgeAtom]:
    out: List[KnowledgeAtom] = []
    for m in map_results:
        if m is not None:
            out.extend(m.knowledge_atoms or [])
    return out


async def _filtered_map_results(
    map_results: List[MapWindowResponse | None],
    anchor: str,
    threshold: float,
    service: AtomAnchorFilterService,
) -> Tuple[List[MapWindowResponse | None], List[KnowledgeAtom]]:
    """Per-window Static Anchor Filtering — fresh atom copies so scoring for
    one variant's anchor never leaks into another variant sharing the same
    baseline map_results. Returns (filtered_map_results, all_scored_atoms)
    — the second list carries `core_relevance_score` on every atom (kept
    AND dropped), since the caller needs it to report what was cut."""
    out: List[MapWindowResponse | None] = []
    all_scored: List[KnowledgeAtom] = []
    for mw in map_results:
        if mw is None:
            out.append(None)
            continue
        window_atoms = [a.model_copy() for a in (mw.knowledge_atoms or [])]
        kept = await service.filter_atoms_by_anchor(window_atoms, anchor, threshold)
        all_scored.extend(window_atoms)  # scored in place by filter_atoms_by_anchor
        out.append(mw.model_copy(update={"knowledge_atoms": kept}))
    return out, all_scored


@dataclass
class ScenarioResult:
    name: str
    latency_sec: float
    atoms_before_cut_or_reduce: int
    atoms_after: int
    dropped_atoms: List[KnowledgeAtom] = field(default_factory=list)

    @property
    def compression_ratio_pct(self) -> float:
        if self.atoms_before_cut_or_reduce == 0:
            return 0.0
        return 100.0 * (1 - self.atoms_after / self.atoms_before_cut_or_reduce)


def _print_scenarios_table(article_label: str, results: List[ScenarioResult]) -> None:
    print(f"\n=== {article_label}: scenarios ===")
    header = (
        f"{'scenario':28} {'latency_s':>10} {'before':>8} {'after':>8} "
        f"{'compress%':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:28} {r.latency_sec:10.1f} {r.atoms_before_cut_or_reduce:8d} "
            f"{r.atoms_after:8d} {r.compression_ratio_pct:10.1f}"
        )
    for r in results:
        if not r.dropped_atoms:
            continue
        print(f"\n-- {r.name}: dropped atoms ({len(r.dropped_atoms)}) --")
        for a in r.dropped_atoms:
            score = a.core_relevance_score
            score_s = f"{score:.2f}" if score is not None else "?"
            print(f"  [{a.scope.value}] score={score_s} {a.statement[:100]}")


async def _run_baseline(
    job: MapReduceArticleJob,
) -> Tuple[ScenarioResult, List[MapWindowResponse | None], FinalArticleSummaryResponse]:
    t0 = time.perf_counter()
    pooled = await map_reduce_jobs_pooled_async([job], force_gemma_cloud=True)
    outcome = pooled.get(job.job_id)
    latency = time.perf_counter() - t0
    assert (
        outcome is not None and outcome.final is not None
    ), "Baseline MAP-REDUCE failed"
    raw_atoms = _raw_atoms(outcome.map_results)
    result = ScenarioResult(
        name="Baseline (no filter)",
        latency_sec=latency,
        atoms_before_cut_or_reduce=len(raw_atoms),
        atoms_after=len(outcome.final.knowledge_atoms or []),
    )
    return result, list(outcome.map_results), outcome.final


async def _run_pre_reduce_cut_variant(
    name: str,
    job: MapReduceArticleJob,
    map_results: List[MapWindowResponse | None],
    anchor: str,
    service: AtomAnchorFilterService,
) -> ScenarioResult:
    t0 = time.perf_counter()
    filtered_results, all_scored = await _filtered_map_results(
        map_results, anchor, _THRESHOLD, service
    )
    kept_atoms = _raw_atoms(filtered_results)
    dropped = [
        a
        for a in all_scored
        if a.core_relevance_score is not None and a.core_relevance_score < _THRESHOLD
    ]
    http_client, gemma_rl = _new_http_client_and_rl()
    try:
        final = await run_reduce(
            job, filtered_results, http_client=http_client, gemma_rl=gemma_rl
        )
    finally:
        await http_client.aclose()
    latency = time.perf_counter() - t0
    assert final is not None, f"{name}: REDUCE failed on filtered atoms"
    return ScenarioResult(
        name=name,
        latency_sec=latency,
        atoms_before_cut_or_reduce=len(kept_atoms),
        atoms_after=len(final.knowledge_atoms or []),
        dropped_atoms=dropped,
    )


async def _run_post_reduce_cut_variant(
    baseline_final: FinalArticleSummaryResponse,
    anchor: str,
    service: AtomAnchorFilterService,
) -> ScenarioResult:
    atoms = [a.model_copy() for a in (baseline_final.knowledge_atoms or [])]
    t0 = time.perf_counter()
    kept = await service.filter_atoms_by_anchor(atoms, anchor, _THRESHOLD)
    latency = time.perf_counter() - t0
    kept_ids = {id(a) for a in kept}
    dropped = [a for a in atoms if id(a) not in kept_ids]
    return ScenarioResult(
        name="Variant C (post-REDUCE cut)",
        latency_sec=latency,
        atoms_before_cut_or_reduce=len(atoms),
        atoms_after=len(kept),
        dropped_atoms=dropped,
    )


async def _run_fact_weighting_demo(
    article_label: str,
    raw_atoms: List[KnowledgeAtom],
    anchor: str,
    query: str,
    service: AtomAnchorFilterService,
) -> None:
    """Outcome 1 (pure query rerank) vs Outcome 2 (hybrid: query + static
    core_relevance_score, precomputed at "ingest time" — not part of the
    timed retrieval step, matching how it would run in production) —
    latency of the retrieval step itself, plus a downstream answer-quality
    probe (what would the tutor answer using each outcome's Top-5 as
    context). No fabricated NDCG ground truth — no labeled relevance
    judgments exist for these real articles, so comparison is visual."""
    unscored = [a.model_copy() for a in raw_atoms]  # core_relevance_score=None
    t0 = time.perf_counter()
    outcome_1 = await service.rerank_atoms_for_query(query, unscored, alpha=_ALPHA)
    retrieval_latency_1 = time.perf_counter() - t0

    # "ingest time" static scoring — NOT counted in outcome_2's retrieval latency,
    # since in production core_relevance_score is already stored before any query.
    scored = [a.model_copy() for a in raw_atoms]
    await service.filter_atoms_by_anchor(scored, anchor, threshold=0.0)  # stamp only
    t0 = time.perf_counter()
    outcome_2 = await service.rerank_atoms_for_query(query, scored, alpha=_ALPHA)
    retrieval_latency_2 = time.perf_counter() - t0

    top5_1 = [atom for atom, _ in outcome_1[:5]]
    top5_2 = [atom for atom, _ in outcome_2[:5]]
    t0 = time.perf_counter()
    answer_1 = await generate_answer_from_facts(query, top5_1)
    answer_latency_1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    answer_2 = await generate_answer_from_facts(query, top5_2)
    answer_latency_2 = time.perf_counter() - t0

    print(f"\n### {article_label}: Fact Weighting — query={query!r}\n")
    print(
        f"| outcome | retrieval_s | answer_gen_s | top1_score | top1_scope |\n"
        f"|---|---|---|---|---|\n"
        f"| 1 (pure query) | {retrieval_latency_1:.3f} | {answer_latency_1:.1f} | "
        f"{outcome_1[0][1]:.3f} | {outcome_1[0][0].scope.value} |\n"
        f"| 2 (hybrid query+core) | {retrieval_latency_2:.3f} | "
        f"{answer_latency_2:.1f} | {outcome_2[0][1]:.3f} | {outcome_2[0][0].scope.value} |"
    )
    print("\n**Outcome 1 — Top 5 (pure query rerank):**")
    for atom, score in outcome_1[:5]:
        print(f"- {score:.3f} [{atom.scope.value}] {atom.statement[:90]}")
    print("\n**Outcome 2 — Top 5 (hybrid query+core_relevance_score):**")
    for atom, score in outcome_2[:5]:
        print(f"- {score:.3f} [{atom.scope.value}] {atom.statement[:90]}")
    print(f"\n**Answer from Outcome 1 context:**\n{answer_1}")
    print(f"\n**Answer from Outcome 2 context:**\n{answer_2}")


async def _run_full_benchmark_for_article(
    fixture_name: str, article_label: str
) -> None:
    article = _load_fixture(fixture_name)
    service = AtomAnchorFilterService()

    job_baseline = _job_from_fixture(article)
    baseline_result, map_results, baseline_final = await _run_baseline(job_baseline)

    synthetic_anchor = get_synthetic_anchor(article["title"], article["lead_text"])
    full_text = "\n\n".join(w["body"] for w in article["windows"])
    llm_anchor = await get_fast_llm_anchor(full_text)
    print(f"\n[{article_label}] synthetic_anchor={synthetic_anchor[:120]!r}")
    print(f"[{article_label}] llm_anchor={llm_anchor[:120]!r}")

    variant_a = await _run_pre_reduce_cut_variant(
        "Variant A (synthetic anchor cut)",
        _job_from_fixture(article),
        map_results,
        synthetic_anchor,
        service,
    )
    variant_b = await _run_pre_reduce_cut_variant(
        "Variant B (fast LLM anchor cut)",
        _job_from_fixture(article),
        map_results,
        llm_anchor,
        service,
    )
    variant_c = await _run_post_reduce_cut_variant(
        baseline_final, synthetic_anchor, service
    )

    _print_scenarios_table(
        article_label, [baseline_result, variant_a, variant_b, variant_c]
    )

    raw_atoms = _raw_atoms(map_results)
    demo_query = article.get(
        "demo_query", f"О чём главная техническая идея статьи «{article['title']}»?"
    )
    await _run_fact_weighting_demo(
        article_label, raw_atoms, synthetic_anchor, demo_query, service
    )

    for r in (baseline_result, variant_a, variant_b, variant_c):
        assert r.atoms_after >= 0
        assert r.latency_sec > 0.0


@pytest.mark.anyio
async def test_fact_relevance_pipeline_dense_technical() -> None:
    await _run_full_benchmark_for_article(
        "dense_technical.json", "dense_technical (Habr/PostgresPro, партиционирование)"
    )


@pytest.mark.anyio
async def test_fact_relevance_pipeline_regular_longread() -> None:
    await _run_full_benchmark_for_article(
        "regular_longread.json", "regular_longread (kariernik.ru, рекурсивные CTE)"
    )
