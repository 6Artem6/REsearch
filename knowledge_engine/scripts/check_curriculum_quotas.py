"""Показать локальные лимиты curriculum API (CSE / Semantic Scholar)."""

from __future__ import annotations

import argparse
import json

from knowledge_engine.services.curriculum_api_quota_store import get_quota_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum API local quota state")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--clear-ss-block",
        action="store_true",
        help="Сбросить blocked_until_day и счётчик Semantic Scholar (тот же UTC-день)",
    )
    args = parser.parse_args()
    if args.clear_ss_block:
        from knowledge_engine.config import PACKAGE_ROOT

        path = (PACKAGE_ROOT / ".runs" / "curriculum_api_quota_state.json").resolve()
        if path.is_file():
            path.unlink()
            print(f"removed {path}")
        else:
            print("quota state file not found")
        return
    summary = get_quota_summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(f"day_utc={summary.get('day_utc')} updated={summary.get('updated_at')}")
    for name in ("google_cse", "semantic_scholar"):
        row = summary.get(name) or {}
        used = row.get("requests_today", 0)
        lim = row.get("daily_limit", 0)
        status = row.get("last_status", "")
        blocked = row.get("blocked_until_day")
        print(f"{name}: {used}/{lim} last={status} blocked_until={blocked}")


if __name__ == "__main__":
    main()
