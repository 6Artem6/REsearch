"""Sliding-window text chunking for LanceDB rag_chunks."""

from __future__ import annotations

from knowledge_engine.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE

_BOUNDARY_LOOKBACK = 80


def _last_whitespace_index(text: str) -> int:
    best = -1
    for ch in (" ", "\n", "\t"):
        best = max(best, text.rfind(ch))
    return best


def split_sliding_window(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int = 48,
) -> list[str]:
    """~500–800 char windows with overlap; snap cuts to whitespace when possible."""
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []
    size = int(chunk_size or RAG_CHUNK_SIZE)
    ov = int(overlap if overlap is not None else RAG_CHUNK_OVERLAP)
    size = max(200, min(size, 1200))
    ov = max(0, min(ov, size // 2))
    out: list[str] = []
    start = 0
    n = len(raw)
    while start < n:
        hard_end = min(start + size, n)
        end = hard_end
        if hard_end < n:
            floor = start + min_chunk_chars
            lookback_from = max(floor, hard_end - _BOUNDARY_LOOKBACK)
            region = raw[lookback_from:hard_end]
            br = _last_whitespace_index(region)
            if br >= 0:
                end = lookback_from + br
        if end <= start:
            end = hard_end
        piece = raw[start:end].strip()
        if len(piece) >= min_chunk_chars:
            out.append(piece)
        if end >= n:
            break
        nxt = end - ov
        if nxt <= start:
            nxt = end
        else:
            while nxt < end and nxt < n and not raw[nxt].isspace():
                nxt += 1
            while nxt < n and raw[nxt].isspace():
                nxt += 1
        if nxt <= start:
            nxt = end
        start = nxt
    return out
