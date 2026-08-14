#!/usr/bin/env python3
"""Сброс explicit Gemini context cache (system + layer1) и локального реестра."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Инвалидация explicit Gemini prompt cache (google-genai cached_content).",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Только очистить knowledge_engine/.runs/gemini_explicit_cache_registry.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать записи реестра без удаления",
    )
    args = parser.parse_args()

    from knowledge_engine.config import PACKAGE_ROOT
    from knowledge_engine.services.gemini_cache_manager import (
        _load_registry,
        invalidate_all_explicit_caches,
        registry_clear_all,
    )

    registry_path = PACKAGE_ROOT / ".runs" / "gemini_explicit_cache_registry.json"
    data = _load_registry()
    print(f"Registry: {registry_path} ({len(data)} entries)")

    if args.dry_run:
        for digest, row in data.items():
            if not isinstance(row, dict):
                continue
            print(
                f"  {digest[:12]}… | {row.get('cache_name', '')} | "
                f"label={row.get('label', '')} | session={row.get('node_session_key', '')}"
            )
        return 0

    if args.local_only:
        rows = registry_clear_all()
        print(
            json.dumps(
                {
                    "registry_entries_cleared": len(rows),
                    "remote_deleted": 0,
                    "local_only": True,
                },
                ensure_ascii=False,
            )
        )
        return 0

    client = None
    try:
        from knowledge_engine.services.gemini_stateless import _client

        client = _client()
    except Exception as exc:
        print(
            f"Gemini client unavailable ({exc}); clearing local registry only.",
            file=sys.stderr,
        )
        rows = registry_clear_all()
        print(
            json.dumps(
                {
                    "registry_entries_cleared": len(rows),
                    "remote_deleted": 0,
                    "remote_failed": len(rows),
                    "warning": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1

    stats = invalidate_all_explicit_caches(client, delete_remote=True)
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats.get("remote_failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
