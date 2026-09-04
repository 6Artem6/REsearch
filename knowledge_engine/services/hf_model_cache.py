"""Resolve HuggingFace snapshots from the local hub cache before any download."""

from __future__ import annotations

from knowledge_engine.ui.run_log import trace


def resolve_hf_snapshot(repo_id: str, *, revision: str = "") -> str:
    """Return a local snapshot directory. Network only on a true cache miss.

    SentenceTransformer/CrossEncoder with a Hub id may fetch ``model.safetensors``
    even when ``pytorch_model.bin`` is already cached (new revision / weight
    format). Loading from this path skips that Hub round-trip.
    """
    from huggingface_hub import snapshot_download

    repo = (repo_id or "").strip()
    if not repo:
        raise ValueError("HuggingFace repo_id is empty")
    rev = (revision or "").strip() or None
    kwargs: dict[str, object] = {"repo_id": repo}
    if rev:
        kwargs["revision"] = rev
    label = f"{repo}@{rev or 'main'}"
    try:
        path = snapshot_download(local_files_only=True, **kwargs)
        trace(f"HF cache hit ✓ | {label} | {path}")
        return path
    except Exception as exc:
        trace(f"HF cache miss ▶ download | {label} | {type(exc).__name__}")
        path = snapshot_download(**kwargs)
        trace(f"HF cache save ✓ | {label} | {path}")
        return path
