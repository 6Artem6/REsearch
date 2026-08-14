"""Sliding-window text chunking for LanceDB rag_chunks."""

from __future__ import annotations

from knowledge_engine.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


def split_sliding_window(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int = 48,
) -> list[str]:
    """~500–800 char windows with overlap (defaults from config)."""
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []
    size = int(chunk_size or RAG_CHUNK_SIZE)
    ov = int(overlap if overlap is not None else RAG_CHUNK_OVERLAP)
    size = max(200, min(size, 1200))
    ov = max(0, min(ov, size // 2))
    step = max(1, size - ov)
    out: list[str] = []
    start = 0
    while start < len(raw):
        piece = raw[start : start + size].strip()
        if len(piece) >= min_chunk_chars:
            out.append(piece)
        if start + size >= len(raw):
            break
        start += step
    return out
