"""Pre-Flight Triage: greedy MMR, Stage 1-4 gates, In-Memory Replenishment,
Safety Fallback, Zero-Waste Handover."""

from __future__ import annotations

import asyncio

import pytest

from knowledge_engine.src.curriculum import pre_flight_triage as mod
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit


def _hit(url: str, *, extracts: list[str] | None = None, tier: str = "exa"):
    return CurriculumSearchHit(
        url=url,
        title=url,
        snippet="snippet",
        key_extracts=extracts or ["some highlight text about the topic"],
        source_tier=tier,
    )


# ---------------------------------------------------------------------------
# greedy_mmr_select — чистая функция
# ---------------------------------------------------------------------------


def test_greedy_mmr_select_picks_top_relevance_first():
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    scores = [0.9, 0.8, 0.1]
    idx = mod.greedy_mmr_select(vectors, scores, top_k=1, lambda_param=0.65)
    assert idx == [0]


def test_greedy_mmr_select_prefers_diversity_on_second_pick():
    # Два почти-дублирующихся вектора с высокой релевантностью + один с
    # меньшей релевантностью, но ортогональный. При lambda=0.5 2-й выбор
    # должен предпочесть разнообразный элемент почти-дублю 1-го выбора.
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    scores = [0.9, 0.85, 0.6]
    idx = mod.greedy_mmr_select(vectors, scores, top_k=2, lambda_param=0.5)
    assert idx[0] == 0
    assert idx[1] == 2, "near-duplicate of #0 should lose to the diverse candidate"


def test_greedy_mmr_select_respects_top_k_and_empty_input():
    assert mod.greedy_mmr_select([], [], top_k=5) == []
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scores = [0.5, 0.5, 0.5]
    idx = mod.greedy_mmr_select(vectors, scores, top_k=2)
    assert len(idx) == 2
    assert len(set(idx)) == 2


def test_greedy_mmr_select_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        mod.greedy_mmr_select([[1.0, 0.0]], [0.1, 0.2], top_k=1)


# ---------------------------------------------------------------------------
# Stage 1: полное ранжирование (без отсева) -> RAM priority queue для Phase 2
# ---------------------------------------------------------------------------


def test_stage1_ranks_all_candidates_without_dropping_any(monkeypatch):
    hits = [_hit(f"https://example.com/{i}") for i in range(4)]
    # Релевантность убывает по индексу: 0 — лучший, 3 — худший.
    fake_scores = [0.9, 0.7, 0.4, 0.1]

    def fake_score_relevance_pairs(criterion, texts):
        return fake_scores

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    out = mod.stage1_zero_http_gate("core theme", hits)
    assert len(out) == 4, "Stage 1 must not drop any candidate"
    assert [h.url for h in out] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]


def test_stage1_pins_whitelisted_candidates_ahead_of_higher_score(monkeypatch):
    good = _hit("https://example.com/good")
    whitelisted_low_score = _hit("https://official-docs.example.com/low")

    def fake_score_relevance_pairs(criterion, texts):
        return [0.9, 0.01]

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )
    monkeypatch.setattr(
        mod,
        "_is_whitelisted",
        lambda url: "official-docs" in url,
    )

    out = mod.stage1_zero_http_gate("core theme", [good, whitelisted_low_score])
    assert len(out) == 2, "Stage 1 must not drop any candidate"
    assert out[0].url == whitelisted_low_score.url, (
        "whitelisted domain must be pinned ahead in the queue despite a low score, "
        "so it reaches Stage 2 in the first replenishment batch"
    )


def test_stage1_empty_core_theme_keeps_all():
    hits = [_hit("https://example.com/a"), _hit("https://example.com/b")]
    out = mod.stage1_zero_http_gate("", hits)
    assert out == hits


# ---------------------------------------------------------------------------
# Stage 2: Parallel Fetch + разбиение на абзацы через Trafilatura
# ---------------------------------------------------------------------------


def test_stage2_drops_anti_bot_and_thin_body(monkeypatch):
    ok_hit = _hit("https://example.com/ok")
    bot_hit = _hit("https://example.com/bot")
    thin_hit = _hit("https://example.com/thin")

    def fake_fetch(url):
        if "bot" in url:
            return "<html>captcha challenge</html>", "httpx"
        if "thin" in url:
            return "short", "httpx"
        return "<html><body>" + ("word " * 200) + "</body></html>", "httpx"

    def fake_extract(html, url=None, **kwargs):
        if "word" in html:
            return "\n".join(
                [
                    "This is a sufficiently long paragraph about the topic at hand, "
                    "with plenty of words in it.",
                    "short",
                    "Another sufficiently long paragraph covering the details, "
                    "also well past the sixty character minimum threshold.",
                ]
            )
        return None

    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.smart_fetch_page_html", fake_fetch
    )
    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.is_anti_bot_fetch_result",
        lambda text, method, html=None, http_status=None: "captcha" in (html or ""),
    )
    import trafilatura

    monkeypatch.setattr(trafilatura, "extract", fake_extract)

    out, dead = asyncio.run(
        mod.stage2_parallel_fetch(
            [ok_hit, bot_hit, thin_hit], top_k=3, concurrency=3, min_paragraph_chars=60
        )
    )
    assert set(out.keys()) == {ok_hit.url}
    assert dead == [], "anti-bot/thin-body are hard drops, not fallback candidates"
    _hit_obj, _html, paragraphs, is_code = out[ok_hit.url]
    assert len(paragraphs) == 2
    assert all(len(p) >= 60 for p in paragraphs)
    assert is_code is False


def test_stage2_respects_top_k():
    hits = [_hit(f"https://example.com/{i}") for i in range(5)]

    async def run():
        return await mod.stage2_parallel_fetch(
            hits, top_k=2, concurrency=2, timeout=0.01
        )

    # Без monkeypatch на fetch => реальный сетевой вызов зависнет/быстро упадёт
    # на плохом хосте; проверять таймингом не нужно — используем monkeypatch-
    # счётчик, чтобы убедиться, что попытка fetch ограничена top_k=2.
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(args[0] if args else None)
        return "", "failed"

    import knowledge_engine.src.curriculum.pre_flight_triage as pft

    orig = asyncio.to_thread
    asyncio.to_thread = fake_to_thread  # type: ignore[assignment]
    try:
        asyncio.run(
            pft.stage2_parallel_fetch(hits, top_k=2, concurrency=2, timeout=1.0)
        )
    finally:
        asyncio.to_thread = orig  # type: ignore[assignment]
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Stage 3: отбор абзацев MMR
# ---------------------------------------------------------------------------


def test_stage3_mmr_paragraphs_selects_top_k(monkeypatch):
    paragraphs = ["para about gil", "para about gil twin", "para about memory model"]

    def fake_embed(texts):
        # сначала вектор темы, затем по вектору на абзац — почти ортогональны.
        vecs = [[1.0, 0.0, 0.0]]  # тема близка к абзацам про "gil"
        for t in texts[1:]:
            if "twin" in t:
                vecs.append([0.99, 0.01, 0.0])
            elif "gil" in t:
                vecs.append([0.95, 0.05, 0.0])
            else:
                vecs.append([0.0, 0.0, 1.0])
        return vecs

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3", fake_embed
    )

    out = mod.stage3_mmr_paragraphs(
        "gil internals", paragraphs, top_k=2, lambda_param=0.5
    )
    assert len(out) == 2
    assert "para about gil" in out[0] or "para about gil" == out[0]


def test_stage3_empty_paragraphs_returns_empty():
    assert mod.stage3_mmr_paragraphs("theme", []) == []


# ---------------------------------------------------------------------------
# Stage 4: quality gate + формула Triage Score
# ---------------------------------------------------------------------------


def test_stage4_triage_score_formula(monkeypatch):
    paragraphs = [
        "p1 mentions asyncio and GIL",
        "p2 talks about something else",
        "p3 x",
    ]

    def fake_score_relevance_pairs(criterion, texts):
        return [0.8, 0.6, 0.4]

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )

    score = mod.stage4_quality_gate("gil internals", ["asyncio", "GIL"], paragraphs)
    peak = 0.8
    mean_top3 = (0.8 + 0.6 + 0.4) / 3
    coverage = 1.0  # оба ключевых слова присутствуют в объединённых абзацах
    expected = 0.5 * peak + 0.3 * mean_top3 + 0.2 * coverage
    assert score == pytest.approx(expected)


def test_stage4_empty_theme_or_paragraphs_returns_zero():
    assert mod.stage4_quality_gate("", ["kw"], ["p"]) == 0.0
    assert mod.stage4_quality_gate("theme", ["kw"], []) == 0.0


def test_stage4_quality_gate_batch_makes_single_cross_encoder_call(monkeypatch):
    """OPTIMIZATION STEP 3: N URL раньше уходили в N отдельных model.predict()
    (см. лог-аудит — 4 URL ≈26s последовательно). Батчевая версия обязана
    сделать ровно один вызов score_relevance_pairs на весь раунд, а не по
    одному на каждый hit."""
    calls: list[list[str]] = []

    def fake_score_relevance_pairs(criterion, texts):
        calls.append(list(texts))
        # По одному «релевантному» скору на каждый текст, по порядку.
        return [0.9 - 0.1 * i for i in range(len(texts))]

    monkeypatch.setattr(
        "knowledge_engine.src.rag_gateway.cross_encoder.score_relevance_pairs",
        fake_score_relevance_pairs,
    )

    paragraphs_per_hit = [
        ["hit0 p0", "hit0 p1"],
        ["hit1 p0"],
        ["hit2 p0", "hit2 p1", "hit2 p2"],
    ]
    scores = mod.stage4_quality_gate_batch("theme", [], paragraphs_per_hit)

    assert len(calls) == 1, "ожидается ровно один батчевый вызов cross-encoder на раунд"
    assert calls[0] == [
        "hit0 p0",
        "hit0 p1",
        "hit1 p0",
        "hit2 p0",
        "hit2 p1",
        "hit2 p2",
    ], "тексты всех hits должны уйти одним плоским списком, в исходном порядке"
    assert len(scores) == 3


def test_keyword_coverage_partial_credit_for_multiword_keyword():
    # "reference counting" не встречается как точная фраза целиком, и
    # присутствует только одно из двух слов — частичный зачёт, а не жёсткий ноль.
    text = "this article explains reference semantics in detail"
    cov_partial = mod._keyword_coverage(["reference counting"], text)
    assert 0.0 < cov_partial < 1.0
    assert cov_partial == pytest.approx(0.5)

    cov_full = mod._keyword_coverage(
        ["reference counting"], "a deep dive into reference counting internals"
    )
    assert cov_full == pytest.approx(1.0)

    cov_zero = mod._keyword_coverage(
        ["reference counting"], "completely unrelated content"
    )
    assert cov_zero == 0.0


# ---------------------------------------------------------------------------
# Phase 2: Hard Gate + Zero-Waste Handover
# ---------------------------------------------------------------------------


def test_run_pre_flight_triage_hard_gate_excludes_when_quota_already_met(monkeypatch):
    survivor = _hit("https://example.com/good")
    dropped = _hit("https://example.com/bad")

    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hits, **kw: hits)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    async def fake_stage2(hits, **kw):
        return {
            survivor.url: (survivor, "<html>good</html>", ["p1", "p2"], False),
            dropped.url: (dropped, "<html>bad</html>", ["p1", "p2"], False),
        }, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(
        mod,
        "stage3_mmr_paragraphs_batch",
        lambda theme, paragraphs, **kw: paragraphs,
    )

    scores_iter = iter([0.9, 0.1])

    def fake_stage4_seq_batch(theme, keywords, paragraphs_per_hit):
        return [next(scores_iter) for _ in paragraphs_per_hit]

    monkeypatch.setattr(mod, "stage4_quality_gate_batch", fake_stage4_seq_batch)

    out = asyncio.run(
        mod.run_pre_flight_triage(
            [survivor, dropped],
            core_theme="gil internals",
            keywords=["GIL"],
            final_articles=1,
            hard_gate_threshold=0.35,
        )
    )
    urls = {h.url for h in out}
    assert len(out) == 1
    assert urls == {survivor.url}
    assert mod.pop_preflight_html(survivor.url) is not None
    assert mod.pop_preflight_html(dropped.url) is None


# ---------------------------------------------------------------------------
# Phase 2: In-Memory Replenishment (следующий батч берётся из ТОЙ ЖЕ уже
# ранжированной очереди — без повторных вызовов Exa / Flash Lite)
# ---------------------------------------------------------------------------


def test_run_pre_flight_triage_in_memory_replenishment_pulls_next_batch(monkeypatch):
    """Round 1 (Top-3) clears only 2 of the 4 required survivors — the loop
    must pull round 2 straight from the queue Stage 1 already built, not
    trigger a new Exa search."""
    hits = [_hit(f"https://example.com/{i}") for i in range(6)]

    stage1_calls = {"n": 0}

    def fake_stage1(theme, hs, **kw):
        stage1_calls["n"] += 1
        return hs

    monkeypatch.setattr(mod, "stage1_zero_http_gate", fake_stage1)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    stage2_batches: list[list[str]] = []

    async def fake_stage2(batch_hits, **kw):
        stage2_batches.append([h.url for h in batch_hits])
        return {
            h.url: (h, f"<html>{h.url}</html>", [h.url], False) for h in batch_hits
        }, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)

    # индексы 0,1 (раунд 1) и 3,4 (раунд 2) проходят гейт; 2 и 5 — нет.
    pass_urls = {hits[0].url, hits[1].url, hits[3].url, hits[4].url}

    def fake_stage4_batch(theme, keywords, paragraphs_per_hit):
        return [
            0.9 if paragraphs and paragraphs[0] in pass_urls else 0.1
            for paragraphs in paragraphs_per_hit
        ]

    monkeypatch.setattr(mod, "stage4_quality_gate_batch", fake_stage4_batch)

    out = asyncio.run(
        mod.run_pre_flight_triage(
            hits,
            core_theme="theme",
            keywords=[],
            final_articles=4,
            hard_gate_threshold=0.35,
            replenish_batch_size=3,
        )
    )

    assert stage1_calls["n"] == 1, "Stage 1 (one Exa fetch) must run exactly once"
    assert len(stage2_batches) == 2, "round 1 fell short (2/4) — must pull round 2"
    assert stage2_batches[0] == [h.url for h in hits[0:3]]
    assert stage2_batches[1] == [h.url for h in hits[3:6]]
    assert {h.url for h in out} == pass_urls
    assert len(out) == 4


# ---------------------------------------------------------------------------
# Phase 3: Safety Fallback
# ---------------------------------------------------------------------------


def test_run_pre_flight_triage_safety_fallback_whitelist_relaxed_threshold(
    monkeypatch,
):
    whitelisted = _hit("https://official-docs.example.com/page")
    plain = _hit("https://example.com/other")

    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hs, **kw: hs)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: "official-docs" in url)

    async def fake_stage2(batch_hits, **kw):
        return {h.url: (h, "<html>x</html>", [h.url], False) for h in batch_hits}, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)

    def fake_stage4_batch(theme, keywords, paragraphs_per_hit):
        return [
            0.25 if "official-docs" in paragraphs[0] else 0.05
            for paragraphs in paragraphs_per_hit
        ]

    monkeypatch.setattr(mod, "stage4_quality_gate_batch", fake_stage4_batch)

    out = asyncio.run(
        mod.run_pre_flight_triage(
            [whitelisted, plain],
            core_theme="theme",
            keywords=[],
            final_articles=1,
            hard_gate_threshold=0.35,
            whitelist_hard_gate_threshold=0.20,
        )
    )
    assert len(out) == 1
    assert out[0].url == whitelisted.url, (
        "0.25 clears the relaxed 0.20 whitelist threshold even though it "
        "fails the strict 0.35 hard gate"
    )


def test_run_pre_flight_triage_safety_fallback_best_of_rest(monkeypatch):
    a = _hit("https://example.com/a")
    b = _hit("https://example.com/b")

    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hs, **kw: hs)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    async def fake_stage2(batch_hits, **kw):
        return {h.url: (h, "<html>x</html>", [h.url], False) for h in batch_hits}, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)

    def fake_stage4_batch(theme, keywords, paragraphs_per_hit):
        return [
            0.15 if paragraphs[0] == a.url else 0.05 for paragraphs in paragraphs_per_hit
        ]

    monkeypatch.setattr(mod, "stage4_quality_gate_batch", fake_stage4_batch)

    out = asyncio.run(
        mod.run_pre_flight_triage(
            [a, b],
            core_theme="theme",
            keywords=[],
            final_articles=2,
            hard_gate_threshold=0.35,
            whitelist_hard_gate_threshold=0.20,
        )
    )
    assert {h.url for h in out} == {a.url, b.url}, (
        "Best-of-Rest must guarantee TARGET_CAP sources when the pool has "
        "enough candidates, even if none cleared any threshold"
    )


def test_run_pre_flight_triage_limits_to_final_articles(monkeypatch):
    hits = [_hit(f"https://example.com/{i}") for i in range(5)]
    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hits, **kw: hits)

    async def fake_stage2(hits, **kw):
        return {h.url: (h, "<html>x</html>", ["p"], False) for h in hits}, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)
    monkeypatch.setattr(
        mod, "stage4_quality_gate_batch", lambda theme, kws, plist: [0.9] * len(plist)
    )

    out = asyncio.run(
        mod.run_pre_flight_triage(
            hits, core_theme="theme", keywords=[], final_articles=2
        )
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Code Preservation Policy: сырой исходник (github_blob passthrough) и
# встроенные блоки <pre>/<code> должны пережить Stage 2, а чисто-код-источники
# должны быть иммунны к занижению оценки NL cross-encoder'ом на Stage 4.
# ---------------------------------------------------------------------------


def test_stage2_preserves_raw_code_via_github_blob_passthrough(monkeypatch):
    code_hit = _hit("https://github.com/python/cpython/blob/main/Python/ceval_gil.c")
    raw_c_source = "\n\n".join(
        [
            "#include <Python.h>",
            "#include <pycore_ceval.h>",
            "void take_gil(PyThreadState *tstate) {\n"
            "    int err;\n"
            "    if (tstate == NULL) {\n"
            '        Py_FatalError("take_gil: NULL tstate");\n'
            "    }\n"
            "    return;\n"
            "}",
            "static int gil_locked = 0;\n" "static unsigned long switch_number = 0;",
        ]
    )

    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.smart_fetch_page_html",
        lambda url: (raw_c_source, "github_blob"),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.is_anti_bot_fetch_result",
        lambda text, method, html=None, http_status=None: False,
    )

    out, dead = asyncio.run(
        mod.stage2_parallel_fetch([code_hit], top_k=1, concurrency=1)
    )
    assert dead == []
    assert code_hit.url in out, "raw source must not be dropped as no_paragraphs"
    _hit_obj, _html, paragraphs, is_code = out[code_hit.url]
    assert is_code is True
    assert paragraphs, "Trafilatura is bypassed — chunked raw text must be non-empty"
    assert any("take_gil" in p for p in paragraphs)


def test_stage2_preserves_embedded_code_blocks_in_html(monkeypatch):
    html_hit = _hit("https://docs.python.org/3/c-api/threads.html")
    code_block = "int x = 1;\nint y = 2;\nfor (i = 0; i < 10; i++) {\n    y += i;\n}"
    html = (
        "<html><body>"
        "<p>" + ("word " * 30) + "</p>"
        f"<pre><code>{code_block}</code></pre>"
        "</body></html>"
    )

    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.smart_fetch_page_html",
        lambda url: (html, "httpx"),
    )
    monkeypatch.setattr(
        "knowledge_engine.services.web_extract.is_anti_bot_fetch_result",
        lambda text, method, html=None, http_status=None: False,
    )
    import trafilatura

    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, url=None, **kw: "word " * 30,
    )

    out, dead = asyncio.run(
        mod.stage2_parallel_fetch([html_hit], top_k=1, concurrency=1)
    )
    assert dead == []
    _hit_obj, _html, paragraphs, is_code = out[html_hit.url]
    assert is_code is False, "a mostly-prose article is not itself a code source"
    assert any(
        "for (i = 0; i < 10; i++)" in p for p in paragraphs
    ), "the <pre><code> block must survive whole, not be shredded line-by-line"


def test_run_pre_flight_triage_code_source_immune_to_hard_gate(monkeypatch):
    code_hit = _hit("https://github.com/python/cpython/blob/main/Python/ceval_gil.c")

    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hs, **kw: hs)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    async def fake_stage2(batch_hits, **kw):
        return {h.url: (h, "<html>x</html>", [h.url], True) for h in batch_hits}, []

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)
    # Враждебный score, далеко ниже любого порога — иммунитет всё равно должен сработать.
    monkeypatch.setattr(
        mod, "stage4_quality_gate_batch", lambda theme, kws, plist: [0.02] * len(plist)
    )

    out = asyncio.run(
        mod.run_pre_flight_triage(
            [code_hit],
            core_theme="theme",
            keywords=[],
            final_articles=1,
            hard_gate_threshold=0.35,
        )
    )
    assert {h.url for h in out} == {
        code_hit.url
    }, "code source must survive Stage 4 despite scoring 0.02 < 0.35 hard gate"


def test_run_pre_flight_triage_recovers_stage2_no_paragraph_rejects_via_fallback(
    monkeypatch,
):
    """The exact In-Memory Replenishment bug: Stage 2 dropping a candidate as
    no_paragraphs must not make it invisible to Phase 3 Best-of-Rest — it
    must still count toward TARGET_CAP if the queue has nothing better."""
    good = _hit("https://example.com/good")
    unparsable = _hit("https://example.com/unparsable")

    monkeypatch.setattr(mod, "stage1_zero_http_gate", lambda theme, hs, **kw: hs)
    monkeypatch.setattr(mod, "_is_whitelisted", lambda url: False)

    async def fake_stage2(batch_hits, **kw):
        out = {good.url: (good, "<html>good</html>", [good.url], False)}
        return out, [unparsable]

    monkeypatch.setattr(mod, "stage2_parallel_fetch", fake_stage2)
    monkeypatch.setattr(mod, "stage3_mmr_paragraphs_batch", lambda theme, p, **kw: p)
    # `good` проходит hard gate сразу; `unparsable` вообще не доходит до
    # stage4 (Stage 2 его отбросил) — единственный путь назад — Best-of-Rest.
    monkeypatch.setattr(
        mod, "stage4_quality_gate_batch", lambda theme, kws, plist: [0.9] * len(plist)
    )

    out = asyncio.run(
        mod.run_pre_flight_triage(
            [good, unparsable],
            core_theme="theme",
            keywords=[],
            final_articles=2,
            hard_gate_threshold=0.35,
            whitelist_hard_gate_threshold=0.20,
        )
    )
    assert {h.url for h in out} == {good.url, unparsable.url}, (
        "Stage 2's no_paragraphs reject must still be recoverable by "
        "Best-of-Rest when the queue can't provide a better replacement"
    )


# ---------------------------------------------------------------------------
# detect_code_content: 3-слойный детектор (URL/домен -> вектор bge-m3 ->
# Tree-Sitter AST) — заменяет старую эвристику на NL-регэкспах, которая
# никогда не срабатывала на чистом тексте raw.githubusercontent.com или на
# React-таблицах blob-UI GitHub (в обоих случаях разметка <pre>/<code> не
# переживает Trafilatura).
# ---------------------------------------------------------------------------


def test_detect_code_content_layer1_raw_githubusercontent_domain():
    url = "https://raw.githubusercontent.com/python/cpython/main/Python/ceval_gil.c"
    assert mod.detect_code_content(url, "irrelevant plain text body") is True


def test_detect_code_content_layer1_github_blob_path_without_extension():
    # Расширения файла нет вообще (например, Makefile) — поймать это может
    # только сигнал пути /blob/; одно лишь сравнение по расширению бы это пропустило.
    url = "https://github.com/python/cpython/blob/main/Makefile"
    assert mod.detect_code_content(url, "irrelevant plain text body") is True


def test_detect_code_content_layer1_known_extension_off_github():
    url = "https://example.com/snippets/gil.rs"
    assert mod.detect_code_content(url, "irrelevant plain text body") is True


def test_detect_code_content_layer1_github_fetch_method_short_circuits():
    # одного method (от web_extract.smart_fetch_page_html) достаточно, даже
    # на URL, который собственная проверка domain/extension у Layer 1 не поймала бы.
    assert (
        mod.detect_code_content(
            "https://example.com/weird-path-no-extension",
            "irrelevant",
            method="github_blob",
        )
        is True
    )


def test_detect_code_content_layer2_and_3_confirm_real_code_on_unrelated_url(
    monkeypatch,
):
    """Layer 1 is inconclusive (plain blog URL) — Layer 2 (vector) flags it,
    Layer 3 (real Tree-Sitter parse, not mocked) confirms a genuine, parseable
    Python function -> True."""

    def fake_embed(texts):
        # порядок: [code_anchor, prose_anchor, sample] — code_vec ближе.
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        fake_embed,
    )

    url = "https://example.com/blog/some-post"
    code = "def add(a, b):\n    return a + b\n"
    assert mod.detect_code_content(url, code) is True


def test_detect_code_content_layer2_flag_without_ast_confirmation_returns_false(
    monkeypatch,
):
    """Layer 2 alone must not grant immunity — a Layer-2 false positive
    (vector says 'code-like') is rejected unless Layer 3 finds a real,
    error-free syntax tree in some language."""

    def fake_embed(texts):
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        fake_embed,
    )

    url = "https://example.com/blog/some-post"
    prose = (
        "This is a normal sentence, that talks about the weather today. "
        "It is not source code at all, just plain writing about clouds "
        "and rain in the afternoon."
    )
    assert mod.detect_code_content(url, prose) is False


def test_detect_code_content_prose_rejected_at_layer2(monkeypatch):
    def fake_embed(texts):
        # prose_vec ближе, чем code_vec -> Layer 2 отклоняет, Layer 3 не запускается.
        return [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        fake_embed,
    )
    ast_calls = {"n": 0}
    monkeypatch.setattr(
        mod,
        "_ast_validate_code",
        lambda text, url: ast_calls.update(n=ast_calls["n"] + 1) or True,
    )

    url = "https://example.com/blog/some-post"
    assert mod.detect_code_content(url, "a perfectly ordinary sentence") is False
    assert ast_calls["n"] == 0, "Layer 3 must not run when Layer 2 rejects"


# ---------------------------------------------------------------------------
# AST Collapsing для BGE: сигнатуры/docstring'и/комментарии/неглубокие вызовы
# вместо сырых тел функций — объём embed() на Stage 3 MMR для больших
# код-источников должен сжаться с сотен сырых чанков до горстки юнитов.
# ---------------------------------------------------------------------------


def test_code_paragraphs_for_embedding_collapses_function_bodies():
    url = "https://github.com/python/cpython/blob/main/Python/ceval_gil.c"
    source = (
        "#include <Python.h>\n\n"
        "/* Acquire the GIL for the given thread state. */\n"
        "void take_gil(PyThreadState *tstate) {\n"
        "    int err;\n"
        "    while (gil_locked) {\n"
        "        err = wait_on_condvar();\n"
        "        if (err) {\n"
        "            handle_error(err);\n"
        "        }\n"
        "    }\n"
        "    gil_locked = 1;\n"
        "    signal_active_thread(tstate);\n"
        "}\n\n"
        "void drop_gil(PyThreadState *tstate) {\n"
        "    gil_locked = 0;\n"
        "    notify_waiters();\n"
        "}\n"
    )
    paragraphs = mod._code_paragraphs_for_embedding(source, url, min_chars=20)
    assert paragraphs, "AST collapsing must produce at least the two signatures"
    joined = "\n".join(paragraphs)
    assert "take_gil" in joined and "drop_gil" in joined
    assert any(
        "calls:" in p and "wait_on_condvar" in p for p in paragraphs
    ), "shallow (depth<=2) calls inside the body must still be listed"
    assert not any(
        "while (gil_locked)" in p for p in paragraphs
    ), "the loop body itself must be collapsed, not copied verbatim"
    assert len(paragraphs) < 10, "collapsed to signature units, not raw line chunks"


def test_code_paragraphs_for_embedding_falls_back_when_unparsable():
    url = "https://example.com/data.xyz"
    garbage = "%%% not valid syntax $$$ {{{ ]]] ??? " * 3
    paragraphs = mod._code_paragraphs_for_embedding(garbage, url, min_chars=10)
    assert paragraphs, (
        "AST layer must yield [] for genuinely unparsable text, and the "
        "raw blank-line-block splitter must still produce a result"
    )


def test_ast_semantic_extracts_empty_for_pure_prose():
    url = "https://example.com/blog/post"
    prose = (
        "This is a normal sentence, that talks about the weather today. "
        "It is not source code at all, just plain writing about clouds "
        "and rain in the afternoon, with more sentences following it."
    )
    assert mod._ast_semantic_extracts(prose, url, min_chars=10) == []


# ---------------------------------------------------------------------------
# Батчинг встроенных блоков <code>: фикс N+1 — один вызов embed() на всю
# страницу вместо одного вызова на каждый отдельный тег <code>.
# ---------------------------------------------------------------------------


def test_extract_embedded_code_blocks_batches_single_embed_call(monkeypatch):
    url = "https://docs.python.org/3/some-page.html"
    code_snippets = [f"int compute_{i}(int x) {{ return x + {i}; }}" for i in range(5)]
    html = (
        "<html><body>"
        + "".join(f"<code>{s}</code>" for s in code_snippets)
        + "</body></html>"
    )

    embed_calls = {"n": 0}

    def fake_embed(texts):
        embed_calls["n"] += 1
        return [[1.0, 0.0], [0.0, 1.0]] + [[1.0, 0.0]] * (len(texts) - 2)

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        fake_embed,
    )

    out = mod._extract_embedded_code_blocks(html, url, min_chars=10)
    assert embed_calls["n"] == 1, "must batch all <code> tags into ONE embed() call"
    assert len(out) == 5


def test_extract_embedded_code_blocks_no_candidates_skips_embed_call(monkeypatch):
    url = "https://docs.python.org/3/some-page.html"
    html = "<html><body>" + ("word " * 30) + "</body></html>"

    embed_calls = {"n": 0}

    def fake_embed(texts):
        embed_calls["n"] += 1
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        fake_embed,
    )

    out = mod._extract_embedded_code_blocks(html, url, min_chars=10)
    assert out == []
    assert embed_calls["n"] == 0, "no <code> tags -> no embed() call at all"


def test_stash_and_pop_preflight_html_roundtrip():
    mod.stash_preflight_html("https://example.com/x", "<html>content</html>")
    assert mod.pop_preflight_html("https://example.com/x") == "<html>content</html>"
    assert mod.pop_preflight_html("https://example.com/x") is None


def test_run_pre_flight_triage_no_hits_returns_empty():
    out = asyncio.run(mod.run_pre_flight_triage([], core_theme="t", keywords=[]))
    assert out == []
