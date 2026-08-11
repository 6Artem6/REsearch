from knowledge_engine.services.rag_chunk_splitter import split_sliding_window


def test_split_sliding_window_overlap():
    text = "a" * 1500
    chunks = split_sliding_window(text, chunk_size=600, overlap=100)
    assert len(chunks) >= 2
    assert all(48 <= len(c) <= 600 for c in chunks)
