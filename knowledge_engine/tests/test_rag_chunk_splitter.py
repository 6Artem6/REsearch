from knowledge_engine.services.rag_chunk_splitter import split_sliding_window


def test_split_sliding_window_overlap():
    text = "a" * 1500
    chunks = split_sliding_window(text, chunk_size=600, overlap=100)
    assert len(chunks) >= 2
    assert all(48 <= len(c) <= 600 for c in chunks)


def test_split_sliding_window_snaps_to_word_boundary():
    # "switch statement" / "leaner internal" sit on a 600-char cut.
    pad = ("block " * 98)  # 588 chars
    text = (
        pad
        + "switch statement is generated for the leaner internal compiler path. "
        + ("tail " * 80)
    )
    chunks = split_sliding_window(text, chunk_size=600, overlap=100)
    heads = [c[:24] for c in chunks]
    assert "h statement" not in " ".join(heads)
    assert "er internal" not in " ".join(heads)
    blob = "\n".join(chunks)
    assert "statement" in blob
    assert "leaner internal" in blob
    for c in chunks:
        assert not c.startswith(("tement", "ement "))
        assert not c.endswith(("statemen", "switch statem"))

