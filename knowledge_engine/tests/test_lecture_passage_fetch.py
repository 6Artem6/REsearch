"""lecture_passage_fetch: async fetch + Trafilatura + BGE-M3/MMR — лёгкая
версия pre_flight_triage без Code Preservation Policy/AST/TOC-триажа.
Реальный кейс: контур добора лекции раньше опирался только на 900-симв.
Exa-highlight; здесь проверяем, что при реальном фетче извлекаются связные
абзацы, MMR отбирает разнообразные/релевантные, а URL с неудачным
фетчем/тонким текстом просто выпадает из результата (вызывающий код должен
фолбэкнуться на Exa-снипет для него, не терять источник целиком)."""

from __future__ import annotations

import asyncio

import knowledge_engine.src.node_deep_dive.lecture_passage_fetch as mod

_ARTICLE_HTML = """
<html><body><article>
<p>B-Tree indexes organize data in a balanced tree structure where every leaf
node sits at the same depth, guaranteeing logarithmic lookup time regardless
of which key is queried against the index.</p>
<p>Leaf pages in a B-Tree are linked together via a doubly linked list, which
lets range queries scan consecutive keys without repeatedly walking back up
through the internal nodes of the tree.</p>
<p>Internal nodes only store separator keys and child pointers; they never
hold the actual row data, which keeps them small enough to stay cached in
memory across most workloads.</p>
</article></body></html>
"""

_SLIDE_HTML = """
<html><body><article>
<p>Definition.</p>
<p>Example.</p>
<p>Theorem.</p>
</article></body></html>
"""


def _fake_embed(texts):
    """Тема и абзацы про b-tree близки друг к другу; посторонний текст —
    ортогонален."""
    vecs = []
    for t in texts:
        low = t.lower()
        if "leaf" in low or "b-tree" in low or "balanced tree" in low:
            vecs.append([1.0, 0.0, 0.0])
        elif "internal nodes" in low:
            vecs.append([0.9, 0.1, 0.0])
        else:
            vecs.append([0.0, 0.0, 1.0])
    return vecs


def test_fetch_and_extract_passages_selects_relevant_paragraphs(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _fake_embed,
    )

    async def fake_fetch_html(url, *, timeout_sec):
        if url == "https://good.example/btree":
            return _ARTICLE_HTML
        return ""  # симулирует таймаут/недоступный хост

    monkeypatch.setattr(mod, "fetch_html", fake_fetch_html)

    out = asyncio.run(
        mod.fetch_and_extract_passages(
            [
                "https://good.example/btree",
                "https://timeout.example/dead",
            ],
            core_theme="B-Tree index internals",
            top_k=2,
            min_chars=20,
        )
    )

    assert "https://good.example/btree" in out
    assert out["https://good.example/btree"]
    assert "https://timeout.example/dead" not in out  # graceful degradation


def test_fragmented_slide_style_text_yields_no_useful_passages(monkeypatch):
    """Слайд-стиль (Definition. Example. Theorem.) — короче min_chars,
    отсеивается ещё на этапе _extract_paragraphs, до MMR/Flash Lite."""
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _fake_embed,
    )

    async def fake_fetch_html(url, *, timeout_sec):
        return _SLIDE_HTML

    monkeypatch.setattr(mod, "fetch_html", fake_fetch_html)

    out = asyncio.run(
        mod.fetch_and_extract_passages(
            ["https://slides.example/ch11.pdf"],
            core_theme="B-Tree index internals",
        )
    )
    assert out == {}


def test_empty_url_list_returns_empty_without_network(monkeypatch):
    async def _boom(url, *, timeout_sec):
        raise AssertionError("must not fetch when url list is empty")

    monkeypatch.setattr(mod, "fetch_html", _boom)
    out = asyncio.run(mod.fetch_and_extract_passages([], core_theme="anything"))
    assert out == {}


_WISC_SLIDE_URL = (
    "https://pages.cs.wisc.edu/~dbbook/openAccess/thirdEdition/slides/"
    "slides3ed-english/Ch11_Hash_Index.pdf"
)


def test_wisc_slide_deck_yields_no_passages_by_content(monkeypatch):
    """Реальный кейс задачи: PDF со слайдами (wisc.edu, .edu — авторитетный
    домен) не должен отбраковываться по URL/расширению (пользователь
    попросил отделять по содержанию, не по формату) — отсекается здесь
    естественно, т.к. фрагментированный слайд-текст короче min_chars на
    абзац; даже если бы прошёл — Flash Lite Content Quality Gate
    (_BATCH_SYSTEM) добраковал бы его по narrative-density."""
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _fake_embed,
    )

    async def fake_fetch_html(url, *, timeout_sec):
        assert url == _WISC_SLIDE_URL
        return _SLIDE_HTML

    monkeypatch.setattr(mod, "fetch_html", fake_fetch_html)

    out = asyncio.run(
        mod.fetch_and_extract_passages(
            [_WISC_SLIDE_URL], core_theme="Hash index internals"
        )
    )
    assert out == {}


def _near_identical_embed(texts):
    """URL A/B почти идентичны (cos~1.0), C ортогонален — для теста
    Union-Find кластеризации в find_near_duplicate_urls."""
    vecs = []
    for t in texts:
        if "unrelated" in t.lower():
            vecs.append([0.0, 0.0, 1.0])
        else:
            vecs.append([1.0, 0.001, 0.0])
    return vecs


def test_find_near_duplicate_urls_clusters_and_confirms_via_bulk_gate(monkeypatch):
    """Реальный кейс: две почти одинаковые статьи (например, разные версии
    документации). BGE-M3 кластеризация должна свести их в suspect group,
    Flash Lite Bulk Gate (переиспользованный из pre_map_deduplicator.py, тот
    же, что DEEP гоняет перед MAP+REDUCE) — подтвердить дубликат."""
    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _near_identical_embed,
    )

    async def fake_bulk_gate(
        suspect_groups, code_ids, context_by_id, *, anchor, max_tpm=None
    ):
        assert code_ids == []
        assert len(suspect_groups) == 1
        assert set(suspect_groups[0]) == {
            "https://a.example/v18",
            "https://a.example/v17",
        }
        return {"https://a.example/v18": ["https://a.example/v17"]}

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator._run_bulk_gate",
        fake_bulk_gate,
    )

    passages_by_url = {
        "https://a.example/v18": ["Same content, version 18 of the docs."],
        "https://a.example/v17": ["Same content, version 17 of the docs."],
        "https://b.example/unrelated": ["Completely unrelated topic here."],
    }
    out = asyncio.run(mod.find_near_duplicate_urls(passages_by_url))
    assert out == {"https://a.example/v17": "https://a.example/v18"}


def test_find_near_duplicate_urls_no_suspects_returns_empty(monkeypatch):
    """Разные по содержанию источники — кластеризация не находит suspect
    group, Flash Lite вообще не вызывается."""

    def _distinct_embed(texts):
        return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts)]

    monkeypatch.setattr(
        "knowledge_engine.services.search.bge_m3_embed.embed_texts_bge_m3",
        _distinct_embed,
    )

    async def _must_not_be_called(*a, **kw):
        raise AssertionError("Flash Lite bulk gate must not run without suspects")

    monkeypatch.setattr(
        "knowledge_engine.src.deduplication.pre_map_deduplicator._run_bulk_gate",
        _must_not_be_called,
    )

    out = asyncio.run(
        mod.find_near_duplicate_urls(
            {
                "https://a.example": ["Topic A content."],
                "https://b.example": ["Topic B content."],
            }
        )
    )
    assert out == {}


def test_find_near_duplicate_urls_skips_below_two_candidates():
    out = asyncio.run(
        mod.find_near_duplicate_urls({"https://a.example": ["some text here"]})
    )
    assert out == {}
