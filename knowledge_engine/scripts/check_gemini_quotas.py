"""Проверка доступности Gemini моделей и разбор 429 / quota из ответа API.

Загружает GEMINI_API_KEY и имена моделей из .env (через knowledge_engine.config).
Для каждой модели — один короткий generate_content (без retry), затем отчёт.

Usage:
  python -m knowledge_engine.scripts.check_gemini_quotas
  python -m knowledge_engine.scripts.check_gemini_quotas --models gemini-3.6-flash,gemini-3.5-flash
  python -m knowledge_engine.scripts.check_gemini_quotas --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from knowledge_engine.config import (
    GEMINI_API_KEY,
    GEMINI_CLIENT,
    GEMINI_FALLBACK_MODELS,
    GEMINI_FLASH_MODEL,
    GEMINI_HIGH_QUOTA_MODEL,
    GEMINI_LITE_MODEL,
    GEMINI_MODEL,
    GEMINI_REASONER_MODEL,
    SKIP_GEMINI,
)
from knowledge_engine.services.gemini_quota_store import (
    apply_probe_result,
    default_daily_limit_rpd,
    extract_quota_fields_from_blob,
)
from knowledge_engine.services.gemini_stateless import (
    _gemini_error_blob,
    _google_retry_delay_sec,
    _is_daily_per_model_quota,
    is_gemini_available,
)


def _unique_models_from_env() -> list[str]:
    chain: list[str] = []
    for name in (
        GEMINI_MODEL,
        GEMINI_REASONER_MODEL,
        GEMINI_FLASH_MODEL,
        GEMINI_LITE_MODEL,
        GEMINI_HIGH_QUOTA_MODEL,
        *GEMINI_FALLBACK_MODELS,
    ):
        m = (name or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def _extract_quota_fields(blob: str) -> dict[str, Any]:
    return extract_quota_fields_from_blob(blob)


def _interpret(result: dict[str, Any]) -> str:
    if result["status"] == "ok":
        return "запрос прошёл — квота на этот вызов не исчерпана (лимит дня мог быть частично использован)"
    if not result.get("is_rate_limit"):
        return result.get("summary") or "ошибка не похожа на квоту"
    if result.get("likely_daily_per_model"):
        return (
            "дневной free-tier лимит на модель (GenerateRequestsPerDayPerModel); "
            "ждать секунды из RetryInfo обычно не восстанавливает дневной счётчик"
        )
    if result.get("retry_after_sec") is not None:
        return (
            f"rate limit / RESOURCE_EXHAUSTED с retry ~{result['retry_after_sec']:.0f}s "
            "(скорее RPM/короткое окно, не обязательно «день»)"
        )
    return "RESOURCE_EXHAUSTED без явного retry — смотрите quota_id / metric в деталях"


def probe_model(client: Any, model: str, timeout_sec: float) -> dict[str, Any]:
    from google.genai import types

    start = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly one word: OK",
            config=types.GenerateContentConfig(
                max_output_tokens=16,
                temperature=0,
            ),
        )
        elapsed = time.monotonic() - start
        text = (response.text or "").strip()
        return {
            "model": model,
            "status": "ok",
            "elapsed_sec": round(elapsed, 2),
            "response_preview": text[:80],
            "is_rate_limit": False,
            "likely_daily_per_model": False,
            "retry_after_sec": None,
            "quota_details": {},
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        blob = _gemini_error_blob(exc)
        retry = _google_retry_delay_sec(exc)
        daily = _is_daily_per_model_quota(exc)
        details = _extract_quota_fields(blob)
        msg_low = blob.lower()
        is_rl = (
            daily
            or retry is not None
            or "resource_exhausted" in msg_low
            or "quota exceeded" in msg_low
            or details.get("http_like_code") == 429
        )
        return {
            "model": model,
            "status": "error",
            "elapsed_sec": round(elapsed, 2),
            "error_type": type(exc).__name__,
            "is_rate_limit": is_rl,
            "likely_daily_per_model": daily,
            "retry_after_sec": retry,
            "quota_details": details,
            "error_preview": blob[:1200],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Gemini models for quota / 429")
    parser.add_argument(
        "--models",
        help="Comma-separated model ids (default: all from .env chain)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Pause between models (sec) to reduce RPM noise",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить результат в knowledge_engine/.runs/gemini_quota_state.json",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Показать локальный store без probe",
    )
    args = parser.parse_args()

    if args.summary_only:
        from knowledge_engine.services.gemini_quota_store import get_quota_summary

        summary = get_quota_summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if SKIP_GEMINI:
        print("SKIP_GEMINI=true — проверка отменена", file=sys.stderr)
        sys.exit(2)
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY не задан", file=sys.stderr)
        sys.exit(2)
    if not is_gemini_available() or GEMINI_CLIENT is None:
        print("Gemini client недоступен (ключ / google-genai)", file=sys.stderr)
        sys.exit(2)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = _unique_models_from_env()

    client = GEMINI_CLIENT
    results: list[dict[str, Any]] = []
    for i, model in enumerate(models):
        if i > 0 and args.pause > 0:
            time.sleep(args.pause)
        row = probe_model(client, model, timeout_sec=60.0)
        row["interpretation"] = _interpret(row)
        results.append(row)
        if args.save:
            apply_probe_result(row, count_probe=True)

    if args.json:
        print(
            json.dumps(
                {"models_checked": models, "results": results},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    key_hint = (
        f"{GEMINI_API_KEY[:6]}…{GEMINI_API_KEY[-4:]}"
        if len(GEMINI_API_KEY) > 12
        else "(set)"
    )
    print(f"GEMINI_API_KEY: {key_hint}")
    print(f"Модели ({len(models)}): {', '.join(models)}")
    print()
    for row in results:
        model = row["model"]
        rpd = default_daily_limit_rpd(model)
        print(f"── {model} (ориентир RPD ~{rpd}) ──")
        if row["status"] == "ok":
            print(
                f"  ✓ OK ({row['elapsed_sec']}s) preview={row.get('response_preview')!r}"
            )
        else:
            print(f"  ✗ {row['error_type']} ({row['elapsed_sec']}s)")
            if row.get("quota_details"):
                for k, v in row["quota_details"].items():
                    print(f"    {k}: {v}")
            if row.get("retry_after_sec") is not None:
                print(f"    retry_after_sec: {row['retry_after_sec']}")
            print(f"    daily_per_model: {row.get('likely_daily_per_model')}")
        print(f"  → {row['interpretation']}")
        if row["status"] == "error" and row.get("error_preview"):
            print("  snippet:")
            for line in row["error_preview"].splitlines()[:6]:
                print(f"    {line[:200]}")
        print()

    ok = sum(1 for r in results if r["status"] == "ok")
    rl = sum(1 for r in results if r.get("is_rate_limit"))
    print(
        f"Итого: {ok} OK, {rl} rate-limit/quota errors, {len(results) - ok - rl} other errors"
    )
    if args.save:
        from knowledge_engine.services.gemini_quota_store import (
            get_quota_summary as _summary,
        )

        print()
        print("Локальный store:", _summary()["state_path"])


if __name__ == "__main__":
    main()
