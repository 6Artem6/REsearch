"""Smoke runner for Knowledge Engine v0.7 LangGraph."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from knowledge_engine.src.graph import compile_v07_graph, run_knowledge_engine_v07


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke v0.7 LangGraph")
    parser.add_argument("--query", required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--thread-id", default="v07-smoke")
    args = parser.parse_args()

    profile_md = ""
    if args.profile and args.profile.is_file():
        profile_md = args.profile.read_text(encoding="utf-8")

    # Проверка compile без полного прогона
    compile_v07_graph()
    print("compile_v07_graph: ok", file=sys.stderr)

    result = asyncio.run(
        run_knowledge_engine_v07(args.query, profile_md, args.thread_id)
    )
    out = {
        "current_step": result.get("current_step"),
        "search_depth": result.get("search_depth"),
        "density_delta": result.get("density_delta"),
        "documents": len(result.get("documents") or []),
        "structured_chunks": len(result.get("structured_chunks") or []),
        "tradeoff_matrix": result.get("tradeoff_matrix"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
