"""Curriculum Search-First: сбор (Consensus/SearXNG/…) → Flash DAG. Лог в .runs/*.log."""

from __future__ import annotations

import argparse
import os
import sys

# Пайплайн «как лонгриды»: материалы из Consensus → агрегация Flash (search_first_flash)
os.environ.setdefault("CURRICULUM_TARGETED_NODE_GROUNDING_ENABLED", "false")
os.environ.setdefault("CURRICULUM_SEARCH_FIRST_ENABLED", "true")
os.environ.setdefault("KE_TRACE_STDOUT", "1")
os.environ.setdefault("KE_LOG_PLAIN", "1")
os.environ.setdefault("KE_LLM_FULL_TRACE", "1")
os.environ.setdefault("CONSENSUS_FORCE_HEADED", "true")
# Полный trace в файл .runs/*.log (не только Redis)
os.environ["KE_REDIS_LOGS"] = "false"

from knowledge_engine.src.curriculum.generator import generate_curriculum_graph
from knowledge_engine.src.curriculum.schemas import CurriculumGenerateInput
from knowledge_engine.ui.llm_trace import reset_llm_trace_steps
from knowledge_engine.ui.run_log import get_run_log_path, init_run_log, trace


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search-First curriculum: sources + Flash graph (Consensus when mode=consensus)"
    )
    parser.add_argument(
        "--goal",
        default="Локальный легковесный RAG в векторной базе данных",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "consensus"),
        default="consensus",
    )
    parser.add_argument(
        "--policy",
        choices=("hybrid", "academic_only", "practical_only"),
        default="hybrid",
    )
    parser.add_argument("--depth", default="Deep Mechanics")
    args = parser.parse_args()

    reset_llm_trace_steps()
    init_run_log(f"curriculum SF trace | {args.goal[:40]}")
    inp = CurriculumGenerateInput(
        target_goal=args.goal.strip(),
        user_level="Intermediate/Advanced",
        depth_level=args.depth,
        generation_mode=args.mode,
        source_policy=args.policy,
    )
    trace(
        f"CLI curriculum Search-First | mode={args.mode} policy={args.policy} "
        f"TARGETED_OFF SEARCH_FIRST_ON"
    )
    try:
        graph = generate_curriculum_graph(inp)
    except Exception as exc:
        trace(f"CLI curriculum ✗ | {exc}")
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    log_path = get_run_log_path()
    print(f"curriculum_id={graph.curriculum_id} nodes={graph.total_nodes}")
    print(f"log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
