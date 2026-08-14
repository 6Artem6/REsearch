"""Проиндексировать knowledge_engine/user_profile.md → LanceDB (light_rag_facts)."""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(force: bool) -> int:
    from knowledge_engine.config import USER_PROFILE_PATH
    from knowledge_engine.src.memory.light_rag import (
        _PROFILE_SYNC_HASH_FILE,
        LightRAG,
        sync_profile_from_markdown_if_needed,
    )

    if force and _PROFILE_SYNC_HASH_FILE.is_file():
        _PROFILE_SYNC_HASH_FILE.unlink()
        print("[PERSONAL_RAG] force: cleared stored profile hash")

    synced = await sync_profile_from_markdown_if_needed()
    rag = LightRAG()
    segments = await rag.count_profile_segments()
    total = await rag.count_indexed_rows()
    print(
        f"[PERSONAL_RAG] profile_file={USER_PROFILE_PATH} "
        f"synced={synced} profile_segments={segments} total_rows={total}"
    )
    return 0 if segments > 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync user_profile.md into LanceDB light_rag_facts"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if SHA256 hash matches (full profile refresh)",
    )
    args = parser.parse_args()
    try:
        code = asyncio.run(_run(args.force))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
