#!/usr/bin/env python3
"""Анализ HAR от Consensus Playwright: найти JSON API со списком статей.

Usage:
  PYTHONPATH=. python -m knowledge_engine.scripts.analyze_consensus_har \\
    --har consensus_network_trace.har
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_PAPER_KEYS = (
    "title",
    "doi",
    "paper_id",
    "paperId",
    "authors",
    "abstract",
    "results",
    "displayName",
    "sourceUrl",
)
_AUTH_HEADER_RE = re.compile(
    r"^(authorization|cookie|user-agent|referer|origin|x-[\w-]*token|x-[\w-]*auth|"
    r"cf-[\w-]+|x-csrf|x-xsrf)",
    re.I,
)


def _decode_har_text(content: dict[str, Any] | None) -> str:
    if not content:
        return ""
    text = content.get("text")
    if not isinstance(text, str):
        return ""
    encoding = (content.get("encoding") or "").lower()
    if encoding == "base64":
        try:
            return base64.b64decode(text).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return text


def _json_load_loose(text: str) -> Any | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _walk_has_paper_signals(node: Any, hits: dict[str, int], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(node, dict):
        keys_lower = {str(k).lower() for k in node}
        for key in _PAPER_KEYS:
            if key.lower() in keys_lower:
                hits[key] = hits.get(key, 0) + 1
        for v in node.values():
            _walk_has_paper_signals(v, hits, depth + 1)
    elif isinstance(node, list):
        for item in node[:80]:
            _walk_has_paper_signals(item, hits, depth + 1)


def _score_payload(data: Any) -> tuple[int, dict[str, int], int]:
    hits: dict[str, int] = {}
    _walk_has_paper_signals(data, hits)
    list_len = 0
    if isinstance(data, list):
        list_len = len(data)
    elif isinstance(data, dict):
        for key in ("results", "papers", "items", "data", "hits", "documents"):
            val = data.get(key)
            if isinstance(val, list):
                list_len = max(list_len, len(val))
            elif isinstance(val, dict):
                for nested in ("results", "papers", "items", "hits"):
                    nv = val.get(nested)
                    if isinstance(nv, list):
                        list_len = max(list_len, len(nv))
    score = sum(hits.values()) * 3 + min(list_len, 50)
    if (
        hits.get("title")
        or hits.get("abstract")
        or hits.get("doi")
        or hits.get("paperId")
        or hits.get("paper_id")
    ):
        score += 20
    if list_len >= 3:
        score += 15
    return score, hits, list_len


def _headers_map(headers: list[dict[str, Any]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in headers or []:
        name = str(h.get("name") or "").strip()
        if not name:
            continue
        out[name] = str(h.get("value") or "")
    return out


def _interesting_headers(headers: dict[str, str]) -> dict[str, str]:
    picked: dict[str, str] = {}
    for name, value in headers.items():
        if _AUTH_HEADER_RE.match(name):
            picked[name] = value
    return picked


def _query_params(url: str) -> dict[str, Any]:
    qs = parse_qs(urlparse(url).query, keep_blank_values=True)
    return {k: (v[0] if len(v) == 1 else v) for k, v in qs.items()}


def _post_payload(request: dict[str, Any]) -> Any:
    post = request.get("postData") or {}
    text = post.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    parsed = _json_load_loose(text)
    return parsed if parsed is not None else text


def _curl_equivalent(
    method: str,
    url: str,
    headers: dict[str, str],
    post_data: Any,
) -> str:
    parts = [f"curl -X {method} {json.dumps(url)}"]
    for name in (
        "Authorization",
        "Cookie",
        "User-Agent",
        "Referer",
        "Origin",
        "Content-Type",
    ):
        # case-insensitive lookup
        for hk, hv in headers.items():
            if hk.lower() == name.lower() and hv:
                parts.append(f"  -H {json.dumps(f'{hk}: {hv[:500]}')}")
                break
    for hk, hv in headers.items():
        low = hk.lower()
        if low.startswith("x-") and ("token" in low or "auth" in low or "csrf" in low):
            parts.append(f"  -H {json.dumps(f'{hk}: {hv[:500]}')}")
    if post_data is not None:
        body = (
            json.dumps(post_data, ensure_ascii=False)
            if not isinstance(post_data, str)
            else post_data
        )
        parts.append(f"  --data-raw {json.dumps(body[:4000])}")
    return " \\\n".join(parts)


def analyze_har(har_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(har_path.read_text(encoding="utf-8"))
    entries = (raw.get("log") or {}).get("entries") or []
    candidates: list[dict[str, Any]] = []

    for entry in entries:
        request = entry.get("request") or {}
        response = entry.get("response") or {}
        status = int(response.get("status") or 0)
        if status != 200:
            continue
        url = str(request.get("url") or "")
        method = str(request.get("method") or "GET").upper()
        resp_headers = _headers_map(response.get("headers"))
        ct = (
            resp_headers.get("content-type") or resp_headers.get("Content-Type") or ""
        ).lower()
        body_text = _decode_har_text(response.get("content"))
        if "json" not in ct and not url.rstrip("/").endswith(".json"):
            # всё равно пробуем, если тело — JSON
            if not body_text.lstrip().startswith(("{", "[")):
                continue
        data = _json_load_loose(body_text)
        if data is None:
            continue
        score, hits, list_len = _score_payload(data)
        if score < 20 and not hits:
            continue
        req_headers = _headers_map(request.get("headers"))
        item = {
            "score": score,
            "method": method,
            "url": url,
            "status": status,
            "paper_field_hits": hits,
            "result_list_len": list_len,
            "query_params": _query_params(url),
            "request_headers_interesting": _interesting_headers(req_headers),
            "request_headers_all": req_headers,
            "post_payload": _post_payload(request),
            "response_preview": body_text[:400],
            "curl": _curl_equivalent(method, url, req_headers, _post_payload(request)),
        }
        candidates.append(item)

    candidates.sort(key=lambda x: (-x["score"], -x["result_list_len"]))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find Consensus paper-list JSON endpoints in a HAR file"
    )
    parser.add_argument(
        "--har",
        default="consensus_network_trace.har",
        help="Path to Playwright HAR (default: ./consensus_network_trace.har)",
    )
    parser.add_argument(
        "--out",
        default="consensus_api_endpoint.json",
        help="Write best endpoint descriptor JSON here",
    )
    parser.add_argument("--top", type=int, default=5, help="Print top-N candidates")
    parser.add_argument("--json", action="store_true", help="Machine-readable dump")
    args = parser.parse_args()

    har_path = Path(args.har).expanduser()
    if not har_path.is_file():
        print(f"HAR not found: {har_path}", file=sys.stderr)
        return 1

    candidates = analyze_har(har_path)
    if not candidates:
        print("No paper-list JSON endpoints found in HAR.", file=sys.stderr)
        return 2

    best = candidates[0]
    out_path = Path(args.out).expanduser()
    slim = {
        "method": best["method"],
        "url": best["url"],
        "query_params": best["query_params"],
        "post_payload": best["post_payload"],
        "headers": best["request_headers_interesting"],
        "paper_field_hits": best["paper_field_hits"],
        "result_list_len": best["result_list_len"],
        "score": best["score"],
        "curl": best["curl"],
        "source_har": str(har_path.resolve()),
    }
    out_path.write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.json:
        print(json.dumps(candidates[: args.top], ensure_ascii=False, indent=2))
    else:
        print(f"Found {len(candidates)} candidate JSON endpoint(s)\n")
        for i, c in enumerate(candidates[: args.top], 1):
            print("=" * 72)
            print(f"#{i} score={c['score']} list_len={c['result_list_len']}")
            print(f"Method: {c['method']}")
            print(f"URL:    {c['url']}")
            print(f"Hits:   {c['paper_field_hits']}")
            print("Interesting request headers:")
            for k, v in (c["request_headers_interesting"] or {}).items():
                shown = v if len(v) <= 160 else v[:157] + "..."
                print(f"  {k}: {shown}")
            if c["query_params"]:
                print(
                    f"Query:  {json.dumps(c['query_params'], ensure_ascii=False)[:500]}"
                )
            if c["post_payload"] is not None:
                print(
                    "Body:   " + json.dumps(c["post_payload"], ensure_ascii=False)[:500]
                )
            print("\ncURL:")
            print(c["curl"])
            print()
        print(f"Best endpoint written → {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
