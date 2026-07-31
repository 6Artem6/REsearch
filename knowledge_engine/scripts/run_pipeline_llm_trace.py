"""Тестовый прогон v0.8 RAG-пайплайна с полным LLM trace (KE_LLM_FULL_TRACE)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# До импорта config — env для trace
os.environ.setdefault("KE_TRACE_STDOUT", "1")
os.environ.setdefault("KE_LOG_PLAIN", "1")
os.environ.setdefault("KE_LLM_FULL_TRACE", "1")

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.src.agent.local_orchestrator import run_knowledge_engine_v08
from knowledge_engine.ui.llm_trace import reset_llm_trace_steps
from knowledge_engine.ui.run_log import get_run_log_path, init_run_log, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.8 pipeline + full LLM trace")
    parser.add_argument(
        "--query",
        default="Локальный легковесный RAG в векторной базе данных",
        help="Тестовая тема",
    )
    parser.add_argument(
        "--mode",
        choices=("consensus", "fast"),
        default="consensus",
        help="consensus = полный граф L2a–L2c; fast = Light RAG + Reasoner",
    )
    parser.add_argument("--thread-id", default="llm-trace-run")
    parser.add_argument(
        "--profile",
        type=Path,
        default=PACKAGE_ROOT / "user_profile.md",
    )
    args = parser.parse_args()

    profile_md = ""
    if args.profile.is_file():
        profile_md = args.profile.read_text(encoding="utf-8")

    reset_llm_trace_steps()
    log_path = init_run_log(f"trace-{args.query[:48]}")
    trace(f"LLM trace run | mode={args.mode} | log={log_path}")

    try:
        result = asyncio.run(
            run_knowledge_engine_v08(
                args.query,
                profile_md,
                args.thread_id,
                retrieval_mode=args.mode,
            )
        )
    except Exception as exc:
        trace(f"PIPELINE ✗ | {exc}")
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise

    trace(
        f"PIPELINE done | step={result.get('current_step')} | "
        f"validation={result.get('validation_status')}"
    )
    out = get_run_log_path() or log_path
    print(f"\nFull trace log: {out}", file=sys.stderr)
    # stdout — только путь к логу (полный trace уже в файле и был в консоли)
    print(str(out))


if __name__ == "__main__":
    main()
