#!/usr/bin/env python3
"""POC: прямой HTTP-запрос к Consensus paper-search API без DOM-парсинга.

Hybrid cookie approach:
  1) короткий Playwright (persistent profile) → Cloudflare/session cookies + storage_state
  2) curl_cffi (Chrome TLS fingerprint) → replay найденного JSON endpoint

Usage:
  # 1) снять HAR (нужен свободный browser profile):
  PYTHONPATH=. python -m knowledge_engine.scripts.check_consensus_playwright \\
    --send --record-har --query "retrieval augmented generation"

  # 2) найти ручку:
  PYTHONPATH=. python -m knowledge_engine.scripts.analyze_consensus_har \\
    --har consensus_network_trace.har

  # 3) POC:
  PYTHONPATH=. python -m knowledge_engine.scripts.poc_consensus_api \\
    --endpoint consensus_api_endpoint.json --query "retrieval augmented generation"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover
    curl_requests = None  # type: ignore[assignment]

from knowledge_engine.config import (
    BROWSER_PROFILE_PATH,
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_QUICK_BASE_URL,
    CONSENSUS_QUICK_OPEN_ACCESS,
    CONSENSUS_START_URL,
)
from knowledge_engine.src.retrieval.consensus_capture import papers_from_json_text

_TOKEN_KEYS = (
    "access_token",
    "accessToken",
    "id_token",
    "idToken",
    "auth_token",
    "authToken",
    "token",
    "jwt",
    "Bearer",
)

# Cookie jar из persistent profile содержит Google/YouTube/ads — nginx режет огромный Cookie.
_CONSENSUS_COOKIE_DOMAIN_RE = re.compile(
    r"(^|\.)consensus\.app$|(^|\.)clerk\.consensus\.app$",
    re.I,
)
_CONSENSUS_COOKIE_NAME_RE = re.compile(
    r"^(cf_clearance|__cf_bm|__session|__client|__refresh|__consensus|"
    r"clerk_|searchMode|aws-waf-token)",
    re.I,
)


def _cookie_relevant_for_consensus(cookie: dict[str, Any]) -> bool:
    domain = str(cookie.get("domain") or "").lstrip(".").lower()
    name = str(cookie.get("name") or "")
    if _CONSENSUS_COOKIE_DOMAIN_RE.search(domain):
        return True
    if domain.endswith("consensus.app"):
        return True
    # fallback: известные auth/CF имена без чужих рекламных доменов
    if _CONSENSUS_COOKIE_NAME_RE.match(name) and not domain:
        return True
    return False


def _filter_consensus_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [c for c in cookies or [] if _cookie_relevant_for_consensus(c)]
    # дедуп по name (предпочесть consensus.app host-only)
    by_name: dict[str, dict[str, Any]] = {}
    for c in filtered:
        name = str(c.get("name") or "")
        if not name:
            continue
        prev = by_name.get(name)
        if prev is None:
            by_name[name] = c
            continue
        prev_dom = str(prev.get("domain") or "")
        cur_dom = str(c.get("domain") or "")
        if "consensus.app" in cur_dom and "consensus.app" not in prev_dom:
            by_name[name] = c
    return list(by_name.values())


def _pick_bearer_from_storage(origins: list[dict[str, Any]]) -> Optional[str]:
    for origin in origins or []:
        origin_url = str(origin.get("origin") or "")
        if origin_url and "consensus.app" not in origin_url:
            continue
        for item in origin.get("localStorage") or []:
            name = str(item.get("name") or "")
            value = str(item.get("value") or "")
            if not value:
                continue
            low = name.lower()
            if any(k.lower() in low for k in _TOKEN_KEYS):
                if value.startswith("{") or value.startswith("["):
                    try:
                        parsed = json.loads(value)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        for tk in _TOKEN_KEYS:
                            tv = parsed.get(tk)
                            if isinstance(tv, str) and tv.strip():
                                return tv.strip()
                    continue
                return value.strip()
            # JWT-looking blob
            if re.match(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.", value):
                return value.strip()
    return None


def _cookies_to_header(cookies: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for c in _filter_consensus_cookies(cookies):
        name = c.get("name")
        value = c.get("value")
        if name is None or value is None:
            continue
        parts.append(f"{name}={value}")
    return "; ".join(parts)


def _has_cf_clearance(cookies: list[dict[str, Any]]) -> bool:
    return any(
        str(c.get("name") or "") == "cf_clearance"
        for c in _filter_consensus_cookies(cookies)
    )


def _profile_locked() -> bool:
    lock = BROWSER_PROFILE_PATH / "Default" / "SingletonLock"
    return lock.is_file()


async def extract_playwright_session(
    *,
    headless: bool,
    storage_out: Path,
    warmup_url: Optional[str] = None,
) -> dict[str, Any]:
    """Короткий Playwright: CF challenge + cookies/localStorage → storage_state."""
    from playwright.async_api import async_playwright

    from knowledge_engine.services.search.playwright_launch import (
        launch_persistent_context_async,
    )

    url = (warmup_url or CONSENSUS_START_URL).rstrip("/")
    if not url.endswith("/home") and "quick" not in url:
        # landing достаточно для CF + auth cookies
        url = CONSENSUS_START_URL

    async with async_playwright() as p:
        context = await launch_persistent_context_async(p, headless=headless)
        page = context.pages[0] if context.pages else await context.new_page()
        print(f"[poc] Playwright warmup → {url}", flush=True)
        await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        # дать Cloudflare / SPA hydrate
        for _ in range(20):
            title = (await page.title()) or ""
            body = ""
            try:
                body = await page.evaluate(
                    "() => (document.body && document.body.innerText) || ''"
                )
            except Exception:
                pass
            challenge = bool(
                re.search(
                    r"just a moment|checking your browser|cf-browser",
                    title + body,
                    re.I,
                )
            )
            if not challenge:
                break
            print("[poc] waiting Cloudflare challenge…", flush=True)
            await page.wait_for_timeout(1500)
        try:
            await page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:
            pass

        storage_out.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(storage_out))
        state = json.loads(storage_out.read_text(encoding="utf-8"))
        cookies = state.get("cookies") or []
        origins = state.get("origins") or []
        bearer = _pick_bearer_from_storage(origins)
        ua = await page.evaluate("() => navigator.userAgent")
        await context.close()

    filtered = _filter_consensus_cookies(cookies)
    bearer = bearer or _session_jwt_from_cookies(filtered)
    return {
        "storage_path": str(storage_out.resolve()),
        "cookies": filtered,
        "origins": origins,
        "cookie_header": _cookies_to_header(filtered),
        "bearer": bearer,
        "user_agent": ua,
        "has_cf_clearance": _has_cf_clearance(filtered),
        "cookie_names": sorted({str(c.get("name") or "") for c in filtered}),
    }


def _load_endpoint(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("url") or not data.get("method"):
        raise ValueError(f"endpoint JSON missing url/method: {path}")
    return data


def _rewrite_query_param(url: str, query: str) -> str:
    """Подставить новый q= в URL поиска, сохранив остальные query params."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    if "q" in qs or "query" in qs:
        if "q" in qs:
            qs["q"] = [query]
        if "query" in qs:
            qs["query"] = [query]
        new_query = urlencode(
            {k: v[0] if len(v) == 1 else v for k, v in qs.items()}, doseq=True
        )
        return urlunparse(parsed._replace(query=new_query))
    # fallback: Consensus quick URL shape
    params = {"q": query}
    if CONSENSUS_QUICK_OPEN_ACCESS:
        params["oa"] = "true"
    base = f"{CONSENSUS_QUICK_BASE_URL}/"
    return f"{base}?{urlencode(params)}"


def _rewrite_post_payload(payload: Any, query: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return payload
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    for key in ("q", "query", "search", "searchQuery", "text", "prompt"):
        if key in out and isinstance(out[key], str):
            out[key] = query
    # nested common shapes
    for nest in ("input", "variables", "params", "data"):
        child = out.get(nest)
        if isinstance(child, dict):
            for key in ("q", "query", "search", "searchQuery", "text"):
                if key in child and isinstance(child[key], str):
                    child = dict(child)
                    child[key] = query
                    out[nest] = child
                    break
    return out


def _diagnose_failure(status: int, body: str, headers_sent: dict[str, str]) -> str:
    low = (body or "").lower()
    missing: list[str] = []
    if status == 400 and "cookie too large" in low:
        missing.append("Cookie header too large — filter to consensus.app domains only")
    if status in (307, 302) and "clerk" in low:
        missing.append(
            "fresh Clerk __session (JWT ~60s TTL); "
            "GET-prefetch refresh required before POST "
            "(reason: session-token-expired-refresh-non-eligible-non-get)"
        )
    if status in (401, 403):
        if "cf-" in low or "cloudflare" in low or "just a moment" in low:
            missing.append(
                "cf_clearance / Cloudflare clearance cookie (refresh via Playwright)"
            )
        if "unauthorized" in low or status == 401:
            if "authorization" not in {k.lower() for k in headers_sent}:
                missing.append(
                    "Clerk session cookies (__session / __client_*) "
                    "or Authorization bearer"
                )
            else:
                missing.append("valid/fresh Authorization bearer token")
        if "csrf" in low:
            missing.append("CSRF / X-*-Token header")
        if not headers_sent.get("Cookie"):
            missing.append("Cookie header (session cookies from storage_state)")
    if status == 405:
        missing.append(
            "likely followed Clerk 307 handshake on POST "
            "(disable redirects; refresh __session via GET first)"
        )
    if not missing:
        missing.append("unknown — inspect response body / Set-Cookie")
    return "; ".join(missing)


def _session_jwt_from_cookies(
    cookies: list[dict[str, Any]] | dict[str, str],
) -> Optional[str]:
    """Clerk JWT лежит в cookie __session (короткий TTL ~60s)."""
    if isinstance(cookies, dict):
        for key in ("__session", "__session_gvFLUf2l"):
            val = cookies.get(key)
            if isinstance(val, str) and val.startswith("eyJ"):
                return val
        return None
    jar = {str(c.get("name") or ""): str(c.get("value") or "") for c in cookies or []}
    return _session_jwt_from_cookies(jar)


def _cookies_list_to_jar(cookies: list[dict[str, Any]]) -> dict[str, str]:
    jar: dict[str, str] = {}
    for c in _filter_consensus_cookies(cookies):
        name = str(c.get("name") or "")
        value = c.get("value")
        if name and value is not None:
            jar[name] = str(value)
    return jar


def _merge_set_cookie(jar: dict[str, str], resp) -> None:
    """Подтянуть Set-Cookie из ответа в jar (Clerk refresh __session)."""
    try:
        # curl_cffi may expose cookies on response
        for name, value in (getattr(resp, "cookies", None) or {}).items():
            jar[str(name)] = str(value)
    except Exception:
        pass
    raw = resp.headers.get("set-cookie") or resp.headers.get("Set-Cookie") or ""
    if not raw:
        return
    # может быть несколько через запятую — грубый парсер name=value
    for part in re.split(r", (?=[A-Za-z0-9_\-]+=)", raw):
        nv = part.split(";", 1)[0].strip()
        if "=" not in nv:
            continue
        name, value = nv.split("=", 1)
        if name:
            jar[name] = value


def replay_consensus_api(
    *,
    endpoint: dict[str, Any],
    session: dict[str, Any],
    query: str,
    impersonate: str = "chrome124",
) -> dict[str, Any]:
    if curl_requests is None:
        raise RuntimeError(
            "curl_cffi is not installed. Run: pip install 'curl_cffi>=0.7,<0.14'"
        )

    method = str(endpoint.get("method") or "GET").upper()
    url = str(endpoint["url"])
    post_payload = endpoint.get("post_payload")

    if method == "GET":
        url = _rewrite_query_param(url, query)
    else:
        post_payload = _rewrite_post_payload(post_payload, query)
        if post_payload is None:
            # дефолтный body quick search из HAR
            post_payload = {
                "query": query,
                "product_feature": "quick_search",
                "filters": {"open_access": "true"},
            }

    headers: dict[str, str] = {}
    for k, v in (endpoint.get("headers") or {}).items():
        lk = str(k).lower()
        if lk in {"cookie", "authorization", "content-length"}:
            continue
        if isinstance(v, str) and v.strip():
            headers[k] = v

    ua = (
        session.get("user_agent")
        or headers.get("User-Agent")
        or headers.get("user-agent")
    )
    if ua:
        headers["User-Agent"] = ua
    headers.setdefault("Referer", f"{CONSENSUS_QUICK_BASE_URL}/")
    headers.setdefault("Origin", "https://consensus.app")
    headers.setdefault("Accept", "application/json, text/plain, */*")

    jar = _cookies_list_to_jar(session.get("cookies") or [])
    if not jar and session.get("cookie_header"):
        for part in str(session["cookie_header"]).split(";"):
            part = part.strip()
            if "=" in part:
                n, v = part.split("=", 1)
                jar[n.strip()] = v.strip()

    # Браузер шлёт только Cookie; curl_cffi без Authorization ловит Clerk 307
    # (session-token-expired-refresh-non-eligible-non-get). Bearer = cookie __session.
    bearer = session.get("bearer") or _session_jwt_from_cookies(jar)
    cookie_header = _cookies_to_header(
        [{"name": k, "value": v, "domain": ".consensus.app"} for k, v in jar.items()]
    )
    if cookie_header:
        headers["Cookie"] = cookie_header
    if bearer:
        token = (
            bearer if str(bearer).lower().startswith("bearer ") else f"Bearer {bearer}"
        )
        headers["Authorization"] = token

    print(f"[poc] {method} {url[:180]}", flush=True)
    print(
        f"[poc] cookies={len(jar)} "
        f"cf_clearance={'cf_clearance' in jar} "
        f"session={'__session' in jar or '__session_gvFLUf2l' in jar} "
        f"bearer={'yes' if bearer else 'no'} "
        f"impersonate={impersonate}",
        flush=True,
    )

    req_headers = dict(headers)
    if method in {"POST", "PUT", "PATCH"}:
        req_headers.setdefault("Content-Type", "application/json")

    kwargs: dict[str, Any] = {
        "headers": req_headers,
        "impersonate": impersonate,
        "timeout": 60,
        # Не следовать на Clerk handshake: POST→GET даёт 405
        "allow_redirects": False,
    }
    if method in {"POST", "PUT", "PATCH"}:
        if isinstance(post_payload, (dict, list)):
            kwargs["json"] = post_payload
        elif isinstance(post_payload, str):
            kwargs["data"] = post_payload
        resp = curl_requests.request(method, url, **kwargs)
    else:
        resp = curl_requests.request(method, url, **kwargs)

    if resp.status_code in (307, 302):
        loc = resp.headers.get("location") or resp.headers.get("Location") or ""
        print(
            f"[poc] Clerk redirect {resp.status_code} → {loc[:120]} "
            "(need fresh __session via Playwright warmup)",
            flush=True,
        )

    body = resp.text or ""
    ct = (resp.headers.get("content-type") or "").lower()
    result: dict[str, Any] = {
        "ok": False,
        "status": resp.status_code,
        "content_type": ct,
        "url": str(getattr(resp, "url", url)),
        "body_preview": body[:500],
        "papers": 0,
        "missing": None,
        "location": resp.headers.get("location") or resp.headers.get("Location"),
        "auth_mode": "cookie+bearer_session_jwt" if bearer else "cookie_only",
    }

    if (
        resp.status_code in (401, 403, 307, 302)
        or "just a moment" in body.lower()
        or ("cloudflare" in body.lower() and resp.status_code != 200)
        or (resp.status_code == 400 and "cookie too large" in body.lower())
        or (resp.status_code == 405)
    ):
        result["missing"] = _diagnose_failure(
            resp.status_code, body + (result["location"] or ""), req_headers
        )
        print(
            f"[poc] FAIL status={resp.status_code} | missing: {result['missing']}",
            flush=True,
        )
        return result

    if resp.status_code != 200:
        result["missing"] = f"unexpected HTTP {resp.status_code}"
        print(f"[poc] FAIL status={resp.status_code}", flush=True)
        print(body[:400], flush=True)
        return result

    papers = papers_from_json_text(body)
    result["papers"] = len(papers)
    result["ok"] = bool(papers) or body.lstrip().startswith(("{", "["))
    result["paper_titles"] = [p.title for p in papers[:8]]
    if result["ok"]:
        print(
            f"[poc] OK JSON | papers={len(papers)} | preview={body[:200]!r}",
            flush=True,
        )
        for t in result["paper_titles"]:
            print(f"  - {t[:120]}", flush=True)
    else:
        result["missing"] = "200 but no parseable paper list"
        print("[poc] WARN: 200 but paper list not detected", flush=True)
    return result


async def replay_via_playwright_request(
    *,
    endpoint: dict[str, Any],
    query: str,
    headless: bool,
) -> dict[str, Any]:
    """Fallback: fetch() внутри живой страницы (CF + Clerk cookies, без DOM-парсинга)."""
    from playwright.async_api import async_playwright

    from knowledge_engine.services.search.playwright_launch import (
        launch_persistent_context_async,
    )

    method = str(endpoint.get("method") or "POST").upper()
    url = str(endpoint.get("url") or "https://consensus.app/api/paper_search/")
    payload = _rewrite_post_payload(endpoint.get("post_payload"), query)
    if payload is None:
        payload = {
            "query": query,
            "product_feature": "quick_search",
            "filters": {"open_access": "true"},
        }

    async with async_playwright() as p:
        context = await launch_persistent_context_async(p, headless=headless)
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(
            CONSENSUS_START_URL, wait_until="domcontentloaded", timeout=90_000
        )
        for _ in range(15):
            title = (await page.title()) or ""
            if not re.search(r"just a moment|checking your browser", title, re.I):
                break
            await page.wait_for_timeout(1000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        print(f"[poc] in-page fetch {method} {url}", flush=True)
        result = await page.evaluate(
            """async ({url, method, payload}) => {
                const resp = await fetch(url, {
                    method,
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                    },
                    body: method === 'GET' ? undefined : JSON.stringify(payload),
                    credentials: 'include',
                });
                const text = await resp.text();
                return {status: resp.status, body: text, ct: resp.headers.get('content-type') || ''};
            }""",
            {"url": url, "method": method, "payload": payload},
        )
        await context.close()

    status = int(result.get("status") or 0)
    body = str(result.get("body") or "")
    papers = papers_from_json_text(body) if status == 200 else []
    ok = status == 200 and (bool(papers) or body.lstrip().startswith(("{", "[")))
    print(
        f"[poc] in-page fetch status={status} papers={len(papers)} ok={ok}",
        flush=True,
    )
    for t in [p.title for p in papers[:8]]:
        print(f"  - {t[:120]}", flush=True)
    return {
        "ok": ok,
        "status": status,
        "papers": len(papers),
        "paper_titles": [p.title for p in papers[:8]],
        "body_preview": body[:500],
        "content_type": result.get("ct"),
        "via": "playwright_inpage_fetch",
        "missing": None if ok else f"in-page fetch HTTP {status}",
    }


async def _async_main(args: argparse.Namespace) -> dict[str, Any]:
    storage_out = Path(args.storage_state).expanduser()
    endpoint_path = Path(args.endpoint).expanduser() if args.endpoint else None

    if _profile_locked() and not args.skip_playwright:
        print(
            "WARN: SingletonLock — другой Chromium держит profile. "
            "Остановите make dev / Consensus browser или передайте --skip-playwright "
            "со свежим storage_state.",
            file=sys.stderr,
        )

    if not endpoint_path or not endpoint_path.is_file():
        print(
            "[poc] endpoint JSON missing — defaulting to POST /api/paper_search/",
            file=sys.stderr,
        )
        endpoint = {
            "method": "POST",
            "url": "https://consensus.app/api/paper_search/",
            "headers": {},
            "post_payload": {
                "query": args.query,
                "product_feature": "quick_search",
                "filters": {"open_access": "true"},
            },
        }
    else:
        endpoint = _load_endpoint(endpoint_path)

    via = (args.via or "auto").lower()
    if via == "playwright":
        result = await replay_via_playwright_request(
            endpoint=endpoint,
            query=args.query,
            headless=args.headless,
        )
        result["endpoint_url"] = endpoint.get("url")
        result["endpoint_method"] = endpoint.get("method")
        return result

    session: dict[str, Any]
    if args.skip_playwright:
        if not storage_out.is_file():
            raise FileNotFoundError(f"storage_state not found: {storage_out}")
        state = json.loads(storage_out.read_text(encoding="utf-8"))
        cookies = _filter_consensus_cookies(state.get("cookies") or [])
        origins = state.get("origins") or []
        session = {
            "storage_path": str(storage_out.resolve()),
            "cookies": cookies,
            "origins": origins,
            "cookie_header": _cookies_to_header(cookies),
            "bearer": _pick_bearer_from_storage(origins),
            "user_agent": args.user_agent
            or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "has_cf_clearance": _has_cf_clearance(cookies),
            "cookie_names": sorted({str(c.get("name") or "") for c in cookies}),
        }
    else:
        session = await extract_playwright_session(
            headless=args.headless,
            storage_out=storage_out,
            warmup_url=args.warmup_url,
        )

    result = replay_consensus_api(
        endpoint=endpoint,
        session=session,
        query=args.query,
        impersonate=args.impersonate,
    )
    result["via"] = "curl_cffi"
    result["session"] = {
        "storage_path": session.get("storage_path"),
        "has_cf_clearance": session.get("has_cf_clearance"),
        "cookie_names": session.get("cookie_names"),
        "bearer_present": bool(session.get("bearer")),
    }
    result["endpoint_url"] = endpoint.get("url")
    result["endpoint_method"] = endpoint.get("method")

    if not result.get("ok") and via == "auto" and not args.skip_playwright:
        print(
            "[poc] curl_cffi failed — falling back to Playwright APIRequestContext",
            flush=True,
        )
        pw = await replay_via_playwright_request(
            endpoint=endpoint,
            query=args.query,
            headless=args.headless,
        )
        result["curl_cffi"] = {
            "ok": result.get("ok"),
            "status": result.get("status"),
            "missing": result.get("missing"),
        }
        result.update(pw)
        result["endpoint_url"] = endpoint.get("url")
        result["endpoint_method"] = endpoint.get("method")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="POC: Consensus search API via curl_cffi + Playwright cookies"
    )
    parser.add_argument(
        "--endpoint",
        default="consensus_api_endpoint.json",
        help="Descriptor from analyze_consensus_har.py",
    )
    parser.add_argument(
        "--query",
        default="retrieval augmented generation vector database",
        help="Search query to inject into the API request",
    )
    parser.add_argument(
        "--storage-state",
        default="consensus_storage_state.json",
        help="Where to save/load Playwright storage_state",
    )
    parser.add_argument(
        "--skip-playwright",
        action="store_true",
        help="Reuse existing storage_state (no browser)",
    )
    parser.add_argument(
        "--via",
        choices=("auto", "curl", "playwright"),
        default="auto",
        help="Transport: curl_cffi, Playwright request, or auto-fallback",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--warmup-url", default=None)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--impersonate", default="chrome124")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        args.headless = CONSENSUS_BROWSER_HEADLESS

    try:
        result = asyncio.run(_async_main(args))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "ok": result.get("ok"),
                    "via": result.get("via"),
                    "status": result.get("status"),
                    "papers": result.get("papers"),
                    "missing": result.get("missing"),
                    "endpoint_method": result.get("endpoint_method"),
                    "endpoint_url": result.get("endpoint_url"),
                    "session": result.get("session"),
                    "paper_titles": result.get("paper_titles"),
                    "body_preview": result.get("body_preview"),
                    "curl_cffi": result.get("curl_cffi"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
