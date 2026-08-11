"""Проверка Gemini моделей с Google Search grounding (tooling).

Один короткий запрос с types.Tool(google_search=...) — как в curriculum fast/expand.
Помогает отделить: неверное имя модели (404), квота (429), tool не поддержан, пустой grounding.

Usage (из корня REsearch, с .env):
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_gemini_grounding
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_gemini_grounding --json
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_gemini_grounding --all-candidates
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.check_gemini_grounding --compare-plain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_CLIENT,
    GEMINI_GROUNDING_MODEL,
    SKIP_GEMINI,
)
from knowledge_engine.services.gemini_quota_store import extract_quota_fields_from_blob
from knowledge_engine.services.gemini_search_grounding import (
    _extract_grounding_hits,
    _whitelist_grounding_hits,
)
from knowledge_engine.services.gemini_stateless import (
    _client_for_api_key,
    _extract_status_code,
    _gemini_error_blob,
    _google_retry_delay_sec,
    _is_daily_per_model_quota,
    _is_search_grounding_model,
    curriculum_grounding_model_chain,
    gemini_api_key_pool,
    is_gemini_available,
)

_DEFAULT_QUERY = (
    "Find one authoritative article about HTTP caching. "
    "Prefer martinfowler.com or blog.cloudflare.com."
)

# Кандидаты для --all-candidates (не все поддерживают Search tool)
_CANDIDATE_GROUNDING_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
)


def _key_hint(key: str) -> str:
    k = (key or "").strip()
    if len(k) <= 12:
        return "(set)" if k else "(empty)"
    return f"{k[:6]}…{k[-4:]}"


def _interpret_grounding(row: dict[str, Any]) -> str:
    preview = (row.get("error_preview") or "").lower()
    if row["status"] == "ok":
        if row.get("grounding_chunks", 0) == 0:
            return (
                "API OK, но grounding_chunks=0 — Search tool не вернул веб-ссылки "
                "(модель или ключ без Search grounding?)"
            )
        if row.get("whitelist_hits", 0) == 0:
            return (
                "Search grounding работает, но URL не прошли whitelist "
                "(нормально для узкого запроса; смотрите sample_urls)"
            )
        return "Search grounding + whitelist OK — модель подходит для curriculum"
    code = row.get("http_code")
    if code == 404:
        if "no longer available to new users" in preview:
            return (
                "HTTP 404 — модель в списке API, но закрыта для «новых» ключей Google. "
                "Попробуйте gemini-2.0-flash / gemini-3.1-flash-lite / gemini-flash-latest "
                "или обычный AI Studio ключ (AIza…)"
            )
        return (
            "HTTP 404 — имя модели не найдено или generateContent не поддержан "
            "(не «нужен другой ключ»). См. --list-models"
        )
    if code == 429 or row.get("is_rate_limit"):
        if "limit: 0" in preview or row.get("quota_details", {}).get("limit") == "0":
            return (
                "квота Search/generate для этой модели = 0 на вашем ключе "
                "(limit:0 в 429) — другая модель или billing / новый AI Studio ключ"
            )
        if row.get("likely_daily_per_model"):
            return (
                "дневная квота на модель / Search tier — ждать до сброса (Pacific) "
                "или другая модель"
            )
        if row.get("retry_after_sec") is not None:
            return (
                f"rate limit — retry ~{row['retry_after_sec']:.0f}s (RPM/короткое окно)"
            )
        return "RESOURCE_EXHAUSTED — квота Search grounding или generate_content"
    if "google_search" in (row.get("error_preview") or "").lower():
        return "ошибка явно связана с google_search tool — модель может не поддерживать tooling"
    return row.get("error_type") or "ошибка API"


def probe_plain_generate(client: Any, model: str) -> dict[str, Any]:
    from google.genai import types

    start = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly one word: OK",
            config=types.GenerateContentConfig(max_output_tokens=8, temperature=0),
        )
        elapsed = time.monotonic() - start
        return {
            "status": "ok",
            "elapsed_sec": round(elapsed, 2),
            "preview": (response.text or "").strip()[:40],
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        blob = _gemini_error_blob(exc)
        return {
            "status": "error",
            "elapsed_sec": round(elapsed, 2),
            "http_code": _extract_status_code(exc),
            "error_type": type(exc).__name__,
            "error_preview": blob[:800],
        }


def probe_grounding_search(
    client: Any,
    model: str,
    query: str,
) -> dict[str, Any]:
    from google.genai import types

    start = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
        elapsed = time.monotonic() - start
        raw_hits = _extract_grounding_hits(response)
        wh = _whitelist_grounding_hits(raw_hits, cap=8)
        text_preview = (getattr(response, "text", None) or "").strip()[:200]
        candidates = getattr(response, "candidates", None) or []
        meta = None
        for cand in candidates:
            meta = getattr(cand, "grounding_metadata", None)
            if meta:
                break
        chunks = getattr(meta, "grounding_chunks", None) or [] if meta else []
        return {
            "model": model,
            "status": "ok",
            "elapsed_sec": round(elapsed, 2),
            "http_code": None,
            "grounding_chunks": len(chunks),
            "raw_urls": len(raw_hits),
            "whitelist_hits": len(wh),
            "sample_urls": [h.url for h in raw_hits[:5]],
            "whitelist_urls": [h.url for h in wh[:5]],
            "text_preview": text_preview,
            "is_rate_limit": False,
            "likely_daily_per_model": False,
            "retry_after_sec": None,
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        blob = _gemini_error_blob(exc)
        retry = _google_retry_delay_sec(exc)
        daily = _is_daily_per_model_quota(exc)
        code = _extract_status_code(exc)
        msg_low = blob.lower()
        is_rl = (
            daily
            or retry is not None
            or code == 429
            or "resource_exhausted" in msg_low
            or "quota exceeded" in msg_low
        )
        return {
            "model": model,
            "status": "error",
            "elapsed_sec": round(elapsed, 2),
            "http_code": code,
            "error_type": type(exc).__name__,
            "is_rate_limit": is_rl,
            "likely_daily_per_model": daily,
            "retry_after_sec": retry,
            "quota_details": extract_quota_fields_from_blob(blob),
            "error_preview": blob[:1200],
            "grounding_chunks": 0,
            "raw_urls": 0,
            "whitelist_hits": 0,
            "sample_urls": [],
            "whitelist_urls": [],
            "text_preview": "",
        }


def probe_model_metadata(client: Any, model: str) -> dict[str, Any]:
    try:
        info = client.models.get(model=model)
        name = getattr(info, "name", None) or str(info)
        supported = getattr(info, "supported_generation_methods", None)
        return {
            "listed": True,
            "name": str(name)[:120],
            "supported_generation_methods": list(supported or []),
        }
    except Exception as exc:
        return {
            "listed": False,
            "error": _gemini_error_blob(exc)[:400],
            "http_code": _extract_status_code(exc),
        }


def _models_for_run(args: argparse.Namespace) -> list[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    chain = curriculum_grounding_model_chain()
    if args.all_candidates:
        merged: list[str] = []
        for m in (*chain, *_CANDIDATE_GROUNDING_MODELS):
            if m and m not in merged:
                merged.append(m)
        return merged
    return chain


def list_flash_models(client: Any) -> list[str]:
    names: list[str] = []
    for m in client.models.list():
        name = (getattr(m, "name", None) or "").strip()
        low = name.lower()
        if "gemini" in low and "flash" in low:
            names.append(name.replace("models/", ""))
    return sorted(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Gemini models with Google Search grounding tool",
    )
    parser.add_argument("--models", help="Comma-separated model ids")
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Env chain + известные кандидаты (2.x / 3.x lite)",
    )
    parser.add_argument("--json", action="store_true", help="JSON only")
    parser.add_argument(
        "--pause",
        type=float,
        default=3.0,
        help="Пауза между моделями (сек)",
    )
    parser.add_argument(
        "--query", default=_DEFAULT_QUERY, help="Текст grounding запроса"
    )
    parser.add_argument(
        "--compare-plain",
        action="store_true",
        help="Для каждой модели: plain generate_content без tools",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="models.get() перед probe (доп. запрос на модель)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить в knowledge_engine/.runs/gemini_grounding_probe.json",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Показать models.list() (flash) и выйти",
    )
    args = parser.parse_args()

    if SKIP_GEMINI:
        print("SKIP_GEMINI=true — проверка отменена", file=sys.stderr)
        sys.exit(2)

    keys = gemini_api_key_pool()
    if not keys:
        print("GEMINI_API_KEY / GEMINI_API_KEYS не задан", file=sys.stderr)
        sys.exit(2)

    client = GEMINI_CLIENT or _client_for_api_key(keys[0])
    if args.list_models:
        if client is None:
            print("Gemini client недоступен", file=sys.stderr)
            sys.exit(2)
        for name in list_flash_models(client):
            print(name)
        return

    if not is_gemini_available() and GEMINI_CLIENT is None:
        print("Gemini client недоступен (google-genai)", file=sys.stderr)
        sys.exit(2)

    models = _models_for_run(args)
    report: dict[str, Any] = {
        "env_grounding_model": GEMINI_GROUNDING_MODEL,
        "env_curriculum_grounding_model": CURRICULUM_GEMINI_GROUNDING_MODEL,
        "chain_from_env": curriculum_grounding_model_chain(),
        "models_probed": models,
        "query": args.query,
        "keys": [_key_hint(k) for k in keys],
        "results": [],
    }

    for key_idx, api_key in enumerate(keys):
        client = (
            _client_for_api_key(api_key) if api_key != GEMINI_API_KEY else GEMINI_CLIENT
        )
        if client is None:
            client = _client_for_api_key(api_key)

        key_block: dict[str, Any] = {
            "key_hint": _key_hint(api_key),
            "key_index": key_idx,
            "probes": [],
        }

        for i, model in enumerate(models):
            if i > 0 and args.pause > 0:
                time.sleep(args.pause)

            row: dict[str, Any] = {
                "model": model,
                "search_grounding_tier_ok": _is_search_grounding_model(model),
            }
            if args.metadata:
                row["metadata"] = probe_model_metadata(client, model)
            if args.compare_plain:
                row["plain"] = probe_plain_generate(client, model)
            row["grounding"] = probe_grounding_search(client, model, args.query)
            g = row["grounding"]
            g["interpretation"] = _interpret_grounding(g)
            key_block["probes"].append(row)

        report["results"].append(key_block)

    if args.save:
        from knowledge_engine.config import PACKAGE_ROOT

        out_path = (PACKAGE_ROOT / ".runs" / "gemini_grounding_probe.json").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["saved_path"] = str(out_path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print("=== Gemini Search Grounding probe ===")
    print(f"GEMINI_GROUNDING_MODEL (env): {GEMINI_GROUNDING_MODEL}")
    print(f"CURRICULUM chain: {' → '.join(curriculum_grounding_model_chain())}")
    print(f"Keys: {', '.join(report['keys'])}")
    print(f"Query: {args.query[:100]}…")
    print()

    ok_ground = 0
    ok_whitelist = 0
    err_404 = 0
    err_429 = 0

    for block in report["results"]:
        print(f"── API key {block['key_hint']} ──")
        for row in block["probes"]:
            model = row["model"]
            tier = "tier-ok" if row["search_grounding_tier_ok"] else "tier-skip-in-app"
            g = row["grounding"]
            print(f"  {model} [{tier}]")
            if args.metadata and row.get("metadata"):
                meta = row["metadata"]
                if meta.get("listed"):
                    print(f"    models.get: OK {meta.get('name', '')[:80]}")
                else:
                    print(f"    models.get: FAIL code={meta.get('http_code')}")
            if args.compare_plain and row.get("plain"):
                p = row["plain"]
                if p["status"] == "ok":
                    print(f"    plain generate: OK ({p['elapsed_sec']}s)")
                else:
                    print(
                        f"    plain generate: FAIL {p.get('http_code')} "
                        f"{p.get('error_type')} ({p['elapsed_sec']}s)"
                    )
            if g["status"] == "ok":
                ok_ground += 1 if g.get("grounding_chunks", 0) > 0 else 0
                ok_whitelist += g.get("whitelist_hits", 0)
                print(
                    f"    grounding: OK ({g['elapsed_sec']}s) "
                    f"chunks={g['grounding_chunks']} raw_urls={g['raw_urls']} "
                    f"whitelist={g['whitelist_hits']}"
                )
                if g.get("sample_urls"):
                    for u in g["sample_urls"][:3]:
                        print(f"      url: {u}")
            else:
                code = g.get("http_code")
                if code == 404:
                    err_404 += 1
                if code == 429 or g.get("is_rate_limit"):
                    err_429 += 1
                print(
                    f"    grounding: FAIL http={code} {g.get('error_type')} "
                    f"({g['elapsed_sec']}s)"
                )
                if g.get("quota_details"):
                    for k, v in g["quota_details"].items():
                        print(f"      {k}: {v}")
            print(f"    → {g['interpretation']}")
            if g["status"] == "error" and g.get("error_preview"):
                line = g["error_preview"].splitlines()[0][:200]
                print(f"    snippet: {line}")
            print()

    print(
        f"Итого: grounding с chunks>0 на {ok_ground} моделей, "
        f"whitelist hits суммарно {ok_whitelist}, 404={err_404}, 429/quota={err_429}"
    )
    if report.get("saved_path"):
        print(f"Saved: {report['saved_path']}")


if __name__ == "__main__":
    main()
