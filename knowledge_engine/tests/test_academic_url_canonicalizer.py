"""academic_url_canonicalizer — RegEx DOI + optional HEAD fallback."""

from __future__ import annotations

import asyncio

from knowledge_engine.db.domain_blocklist import load_blocked_domain_set
from knowledge_engine.src.curriculum.academic_url_canonicalizer import (
    academic_source_dedupe_key,
    canonicalize_academic_url,
    canonicalize_academic_url_pure,
)
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.curriculum.targeted_hit_replenishment import (
    precheck_candidate_hit,
    precheck_candidate_url,
)


def test_arxiv_doi_pure_new_format():
    url = "https://doi.org/10.48550/arxiv.2512.08290"
    out = canonicalize_academic_url_pure(url)
    assert out == "https://arxiv.org/pdf/2512.08290.pdf"


def test_arxiv_doi_pure_with_version():
    url = "https://dx.doi.org/10.48550/arxiv.2203.08975v2"
    out = canonicalize_academic_url_pure(url)
    assert out == "https://arxiv.org/pdf/2203.08975v2.pdf"


def test_arxiv_doi_pure_old_subject_format():
    url = "https://doi.org/10.48550/arxiv.hep-th/9901001"
    out = canonicalize_academic_url_pure(url)
    assert out == "https://arxiv.org/pdf/hep-th/9901001.pdf"


def test_abs_and_pdf_same_dedupe_key():
    abs_u = "https://arxiv.org/abs/2512.08290"
    pdf_u = "https://arxiv.org/pdf/2512.08290.pdf"
    assert academic_source_dedupe_key(abs_u) == academic_source_dedupe_key(pdf_u)


def test_zenodo_doi_pure():
    url = "https://doi.org/10.5281/zenodo.1234567"
    out = canonicalize_academic_url_pure(url)
    assert out == "https://zenodo.org/record/1234567"


def test_precheck_passes_after_pure_canonicalize_with_doi_blocklist():
    blocked = load_blocked_domain_set()
    if "doi.org" not in blocked:
        blocked = set(blocked) | {"doi.org"}
    canon = canonicalize_academic_url_pure("https://doi.org/10.48550/arxiv.2512.08290")
    assert canon is not None
    reason = precheck_candidate_url(
        canon,
        blocked_domains=blocked,
        skip_practical_filter=True,
    )
    assert reason is None


def test_hit_precheck_after_async_canonicalize():
    blocked = load_blocked_domain_set()
    hit = CurriculumSearchHit(
        url="https://doi.org/10.48550/arxiv.2512.08290",
        title="paper",
        snippet="s",
        key_extracts=["x"],
        source_tier="arxiv",
    )
    canon_hit = asyncio.run(canonicalize_academic_url(hit.url))
    assert canon_hit.startswith("https://arxiv.org/pdf/")
    hit2 = hit.model_copy(update={"url": canon_hit})
    assert precheck_candidate_hit(hit2, blocked_domains=blocked) is None


def test_async_canonicalize_pure_path_no_network():
    async def _run():
        out = await canonicalize_academic_url(
            "https://doi.org/10.48550/arxiv.2301.03724v1"
        )
        assert out == "https://arxiv.org/pdf/2301.03724v1.pdf"

    asyncio.run(_run())
