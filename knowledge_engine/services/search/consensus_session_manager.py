"""Singleton-кэш Clerk/CF сессии Consensus + быстрый Playwright warmup.

Отдельно от UI-менеджера в ``src/retrieval/consensus_session.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge_engine.config import (
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_DIRECT_SESSION_MAX_AGE_SEC,
    CONSENSUS_DIRECT_WARMUP_URL,
)
from knowledge_engine.ui.run_log import trace

SESSION_TTL_SEC = 45.0
WARMUP_GOTO_TIMEOUT_MS = 7_000
WARMUP_POLL_JWT_MS = 2_500
WARMUP_URL_DEFAULT = "https://consensus.app/"
_LAUNCH_FAST_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
]

EXCLUDED_EXTENSIONS = {
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".gif",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".ico",
    ".map",
    ".mp4",
    ".webm",
    ".avi",
}
# Трекеры/виджеты из consensus_network_trace.har (+ типовой мусор).
# Матч: substring в URL (`host in url`), поэтому достаточно корня домена.
EXCLUDED_HOSTS = {
    # --- HAR (Consensus quick search capture) ---
    "browser-intake-us3-datadoghq.com",
    "datadoghq.com",
    "analytics.google.com",
    "www.googletagmanager.com",
    "googletagmanager.com",
    "api-iam.intercom.io",
    "widget.intercom.io",
    "nexus-websocket-a.intercom.io",
    "js.intercomcdn.com",
    "intercom.io",
    "intercomcdn.com",
    "prodregistryv2.org",  # Statsig
    "featureassets.org",  # Statsig assets
    "px.ads.linkedin.com",
    "snap.licdn.com",
    "mpc-prod-23-s6uit34pua-ue.a.run.app",  # Intercom/MPC beacon
    "www.facebook.com",
    "connect.facebook.net",
    "facebook.net",
    "facebook.com",
    "track.consensus.app",
    "static.cloudflareinsights.com",
    "www.google.kz",  # GA pixel
    "google.kz",
    # --- общий мусор ---
    "google-analytics.com",
    "googleadservices.com",
    "doubleclick.net",
    "segment.io",
    "segment.com",
    "cdn.segment.com",
    "sentry.io",
    "mixpanel.com",
    "hotjar.com",
    "linkedin.com",
    "ads-twitter.com",
    "adservice.google.com",
    "clarity.ms",
    "fullstory.com",
    "amplitude.com",
    "hackage.haskell.org",
    "js.hs-scripts.com",
    "js.hs-analytics.net",
    "js.hscollectedforms.net",
    "bat.bing.com",
    "statsigapi.net",
    "featuregates.org",
}
EXCLUDED_RESOURCE_TYPES = {"image", "stylesheet", "font", "media"}

_CONSENSUS_COOKIE_DOMAIN_RE = re.compile(
    r"(^|\.)consensus\.app$",
    re.I,
)


def _b64url_json(segment: str) -> dict[str, Any] | None:
    raw = segment + "=" * (-len(segment) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(raw.encode("ascii")))
    except Exception:
        return None


def jwt_iat_age_sec(token: str) -> Optional[float]:
    parts = (token or "").split(".")
    if len(parts) < 2:
        return None
    payload = _b64url_json(parts[1])
    if not payload or "iat" not in payload:
        return None
    try:
        return max(0.0, time.time() - float(payload["iat"]))
    except (TypeError, ValueError):
        return None


def jwt_seconds_to_exp(token: str) -> Optional[float]:
    parts = (token or "").split(".")
    if len(parts) < 2:
        return None
    payload = _b64url_json(parts[1])
    if not payload or "exp" not in payload:
        return None
    try:
        return float(payload["exp"]) - time.time()
    except (TypeError, ValueError):
        return None


def _cookie_domain_ok(domain: str) -> bool:
    d = (domain or "").lstrip(".").lower()
    if not d:
        return False
    # только *.consensus.app (включая clerk.consensus.app)
    return bool(_CONSENSUS_COOKIE_DOMAIN_RE.search(d)) or d.endswith("consensus.app")


def filter_consensus_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Только куки доменов *.consensus.app — защита от nginx 400 Cookie Too Large."""
    filtered = [
        c for c in cookies or [] if _cookie_domain_ok(str(c.get("domain") or ""))
    ]
    by_name: dict[str, dict[str, Any]] = {}
    for c in filtered:
        name = str(c.get("name") or "")
        if not name:
            continue
        prev = by_name.get(name)
        if prev is None:
            by_name[name] = c
            continue
        # предпочесть host-only / более специфичный consensus.app
        prev_dom = str(prev.get("domain") or "")
        cur_dom = str(c.get("domain") or "")
        if len(cur_dom) >= len(prev_dom):
            by_name[name] = c
    return list(by_name.values())


def cookies_list_to_dict(cookies: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(c.get("name") or ""): str(c.get("value") or "")
        for c in filter_consensus_cookies(cookies)
        if c.get("name") is not None
    }


def cookies_dict_to_header(cookies_dict: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookies_dict.items() if k and v is not None)


def session_jwt_from_cookies_dict(cookies_dict: dict[str, str]) -> Optional[str]:
    for key in ("__session", "__session_gvFLUf2l"):
        val = (cookies_dict.get(key) or "").strip()
        if val.startswith("eyJ"):
            return val
    for name, val in cookies_dict.items():
        if name.startswith("__session") and str(val).startswith("eyJ"):
            return str(val)
    return None


def _should_block_url(url: str, resource_type: str = "") -> bool:
    u = (url or "").lower()
    rt = (resource_type or "").lower()

    # Критичные хосты/ручки — никогда не режем (Clerk refresh + CF + API).
    if any(
        tok in u
        for tok in (
            "clerk.consensus.app",
            "clerk.com",
            "challenges.cloudflare.com",
            "cdn.cloudflare.com",
            "consensus.app/api/",
            "consensus.app/cdn-cgi/",
        )
    ):
        return False
    if rt in {"document", "xhr", "fetch", "websocket", "eventsource"}:
        if any(host in u for host in EXCLUDED_HOSTS):
            return True
        return False
    if rt == "script":
        if "consensus.app" in u or "clerk" in u or "cloudflare" in u:
            return False
        # сторонние аналитические скрипты
        return True

    if rt in EXCLUDED_RESOURCE_TYPES:
        return True
    for ext in EXCLUDED_EXTENSIONS:
        if u.endswith(ext) or f"{ext}?" in u:
            return True
    for host in EXCLUDED_HOSTS:
        if host in u:
            return True
    if any(
        tok in u
        for tok in (
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "cdnjs.cloudflare.com",
            "/_next/static/media/",
            "/_next/image",
            "/_next/static/css/",
        )
    ):
        return True
    return False


async def block_unnecessary_requests(route) -> None:
    req = route.request
    url = req.url or ""
    try:
        rtype = req.resource_type or ""
    except Exception:
        rtype = ""
    if _should_block_url(url, rtype):
        await route.abort()
        return
    await route.continue_()


@dataclass
class ConsensusSession:
    """Кэш auth-артефактов для curl_cffi Direct API."""

    cf_clearance: str = ""
    session_jwt: str = ""
    cookies_dict: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    @property
    def age_sec(self) -> float:
        if not self.created_at:
            return 1e9
        return max(0.0, time.time() - self.created_at)

    def is_fresh(self, ttl_sec: float = SESSION_TTL_SEC) -> bool:
        if not self.session_jwt or not self.cookies_dict:
            return False
        if self.age_sec >= ttl_sec:
            return False
        age_iat = jwt_iat_age_sec(self.session_jwt)
        if age_iat is not None and age_iat >= ttl_sec:
            return False
        ttl_left = jwt_seconds_to_exp(self.session_jwt)
        if ttl_left is not None and ttl_left < 5.0:
            return False
        return True


class ConsensusSessionManager:
    """Singleton in-memory cache + быстрый Playwright warmup (commit + block static)."""

    def __init__(
        self,
        *,
        ttl_sec: float | None = None,
        headless: bool | None = None,
        warmup_url: str | None = None,
    ) -> None:
        self.ttl_sec = float(
            ttl_sec
            if ttl_sec is not None
            else (CONSENSUS_DIRECT_SESSION_MAX_AGE_SEC or SESSION_TTL_SEC)
        )
        # Clerk не ротирует __session в headless на этом profile — берём UI-режим по умолчанию.
        self.headless = (
            CONSENSUS_BROWSER_HEADLESS if headless is None else bool(headless)
        )
        self.warmup_url = (
            warmup_url or CONSENSUS_DIRECT_WARMUP_URL or WARMUP_URL_DEFAULT
        ).rstrip("/") + "/"
        # предпочтительно корень — меньше SPA JS, чем /search
        if "consensus.app" not in self.warmup_url:
            self.warmup_url = WARMUP_URL_DEFAULT
        self._cache: Optional[ConsensusSession] = None
        self._lock = asyncio.Lock()
        self._warmup_task: Optional[asyncio.Task[ConsensusSession]] = None

    def peek_session(self) -> Optional[ConsensusSession]:
        sess = self._cache
        if sess and sess.is_fresh(self.ttl_sec):
            return sess
        return None

    async def get_active_session(self, *, force: bool = False) -> ConsensusSession:
        """Свежий кэш → мгновенно; иначе оптимизированный Playwright warmup."""
        async with self._lock:
            if not force:
                cached = self.peek_session()
                if cached is not None:
                    trace(
                        f"Consensus Session ✓ cache hit | age={cached.age_sec:.1f}s "
                        f"cookies={len(cached.cookies_dict)}"
                    )
                    return cached
            if self._warmup_task is not None and not self._warmup_task.done():
                task = self._warmup_task
            else:
                task = asyncio.create_task(self._run_fast_warmup())
                self._warmup_task = task
        # вне lock — не блокируем параллельные get на время warmup
        session = await task
        async with self._lock:
            self._cache = session
            if self._warmup_task is task:
                self._warmup_task = None
        return session

    async def ensure_warmup_async(self) -> None:
        """Фоновый prefetch к старту пайплайна (не блокирует caller дольше create_task)."""
        async with self._lock:
            if self.peek_session() is not None:
                return
            if self._warmup_task is not None and not self._warmup_task.done():
                return
            self._warmup_task = asyncio.create_task(self._warmup_and_store())

    async def _warmup_and_store(self) -> ConsensusSession:
        session = await self._run_fast_warmup()
        async with self._lock:
            self._cache = session
            self._warmup_task = None
        return session

    def invalidate(self) -> None:
        self._cache = None

    async def close(self) -> None:
        async with self._lock:
            self._cache = None
            task = self._warmup_task
            self._warmup_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_fast_warmup(self) -> ConsensusSession:
        from playwright.async_api import async_playwright

        from knowledge_engine.config import BROWSER_PROFILE_PATH, PLAYWRIGHT_BROWSER
        from knowledge_engine.services.search.playwright_browsers import (
            ensure_playwright_browsers_path,
        )

        t0 = time.perf_counter()
        url = self.warmup_url
        if not url.endswith("/"):
            url += "/"
        if url.rstrip("/").endswith("/search"):
            url = "https://consensus.app/"
        trace(
            f"Consensus Session ▶ fast warmup | {url} "
            f"commit+block headless={self.headless}"
        )

        ensure_playwright_browsers_path()
        BROWSER_PROFILE_PATH.mkdir(parents=True, exist_ok=True)

        playwright = await async_playwright().start()
        context = None
        captured_auth: list[str] = []
        try:
            launch_kwargs: dict[str, Any] = {
                "user_data_dir": str(BROWSER_PROFILE_PATH),
                "headless": self.headless,
                "viewport": {"width": 800, "height": 600},
            }
            if PLAYWRIGHT_BROWSER == "firefox":
                context = await playwright.firefox.launch_persistent_context(
                    **launch_kwargs
                )
            else:
                context = await playwright.chromium.launch_persistent_context(
                    **launch_kwargs,
                    args=_LAUNCH_FAST_ARGS,
                )
            page = context.pages[0] if context.pages else await context.new_page()

            async def _on_request(request) -> None:
                try:
                    headers = await request.all_headers()
                except Exception:
                    return
                auth = (
                    headers.get("authorization") or headers.get("Authorization") or ""
                )
                if auth.lower().startswith("bearer eyj"):
                    token = auth.split(None, 1)[-1].strip()
                    if token.startswith("eyJ"):
                        captured_auth.append(token)

            page.on("request", lambda req: asyncio.create_task(_on_request(req)))
            await context.route("**/*", block_unnecessary_requests)

            pre = filter_consensus_cookies(await context.cookies())
            pre_dict = cookies_list_to_dict(pre)
            pre_jwt = session_jwt_from_cookies_dict(pre_dict)
            if pre_jwt and ConsensusSession(
                session_jwt=pre_jwt,
                cookies_dict=pre_dict,
                created_at=time.time() - (jwt_iat_age_sec(pre_jwt) or 0.0),
                cf_clearance=pre_dict.get("cf_clearance", ""),
            ).is_fresh(self.ttl_sec):
                elapsed = (time.perf_counter() - t0) * 1000.0
                sess = ConsensusSession(
                    cf_clearance=pre_dict.get("cf_clearance", ""),
                    session_jwt=pre_jwt,
                    cookies_dict=pre_dict,
                    created_at=time.time(),
                )
                trace(
                    f"Consensus Session ✓ profile cookies fresh | "
                    f"{elapsed:.0f}ms cookies={len(pre_dict)}"
                )
                return sess

            try:
                await page.goto(
                    url,
                    wait_until="commit",
                    timeout=WARMUP_GOTO_TIMEOUT_MS,
                )
            except Exception as exc:
                trace(f"Consensus Session ⊘ goto commit | {exc}")
                if pre_jwt:
                    return ConsensusSession(
                        cf_clearance=pre_dict.get("cf_clearance", ""),
                        session_jwt=pre_jwt,
                        cookies_dict=pre_dict,
                        created_at=time.time(),
                    )
                raise

            cookies = filter_consensus_cookies(await context.cookies())
            cookies_dict = cookies_list_to_dict(cookies)
            jwt = session_jwt_from_cookies_dict(cookies_dict)

            deadline = time.perf_counter() + (WARMUP_POLL_JWT_MS / 1000.0)
            while time.perf_counter() < deadline:
                if jwt:
                    ttl_left = jwt_seconds_to_exp(jwt)
                    age = jwt_iat_age_sec(jwt)
                    if (ttl_left is None or ttl_left >= 5) and (
                        age is None or age < self.ttl_sec
                    ):
                        break
                await asyncio.sleep(0.08)
                cookies = filter_consensus_cookies(await context.cookies())
                cookies_dict = cookies_list_to_dict(cookies)
                jwt = session_jwt_from_cookies_dict(cookies_dict) or jwt
                if captured_auth:
                    jwt = captured_auth[-1]

            if not jwt or (jwt_seconds_to_exp(jwt) or 0) < 5:
                try:
                    stored = await asyncio.wait_for(
                        page.evaluate(
                            """() => {
                                const out = {};
                                try {
                                    for (const store of [localStorage, sessionStorage]) {
                                        for (let i = 0; i < store.length; i++) {
                                            const k = store.key(i);
                                            const v = store.getItem(k) || '';
                                            if (/session|token|jwt|clerk/i.test(k) || /^eyJ/.test(v)) {
                                                out[k] = v.slice(0, 4000);
                                            }
                                        }
                                    }
                                } catch (e) {}
                                return out;
                            }"""
                        ),
                        timeout=0.35,
                    )
                    if isinstance(stored, dict):
                        for v in stored.values():
                            if isinstance(v, str) and v.startswith("eyJ"):
                                jwt = v
                                break
                except Exception:
                    pass

            if captured_auth:
                cand = captured_auth[-1]
                if not jwt or (jwt_seconds_to_exp(cand) or 0) >= (
                    jwt_seconds_to_exp(jwt) or 0
                ):
                    jwt = cand

            if not jwt:
                raise RuntimeError(
                    "Consensus Session: нет __session после fast warmup — "
                    "выполните consensus-login"
                )

            async def _refresh_cookies_from_context() -> None:
                nonlocal jwt, cookies_dict
                cookies = filter_consensus_cookies(await context.cookies())
                cookies_dict = cookies_list_to_dict(cookies)
                jwt = session_jwt_from_cookies_dict(cookies_dict) or jwt
                if captured_auth:
                    jwt = captured_auth[-1]

            def _jwt_ok(token: str | None) -> bool:
                if not token:
                    return False
                age_v = jwt_iat_age_sec(token)
                ttl_v = jwt_seconds_to_exp(token)
                if age_v is not None and age_v >= self.ttl_sec:
                    return False
                if ttl_v is not None and ttl_v < 5:
                    return False
                return True

            # Не кэшируем просроченный Clerk JWT (иначе curl ловит 307).
            if not _jwt_ok(jwt):
                trace("Consensus Session ▶ Clerk JWT stale — force refresh")
                try:
                    try:
                        await page.wait_for_load_state(
                            "domcontentloaded", timeout=3_000
                        )
                    except Exception:
                        pass
                    # Явный handshake/client touch — ротация __session без полной SPA.
                    await page.evaluate(
                        """async () => {
                            const urls = [
                              'https://clerk.consensus.app/v1/client?__clerk_api_version=2025-11-10',
                              'https://consensus.app/',
                            ];
                            for (const u of urls) {
                              try {
                                await fetch(u, { credentials: 'include', mode: 'cors' });
                              } catch (e) {}
                            }
                        }"""
                    )
                    refresh_deadline = time.perf_counter() + 2.5
                    while time.perf_counter() < refresh_deadline:
                        await _refresh_cookies_from_context()
                        if _jwt_ok(jwt):
                            break
                        await asyncio.sleep(0.12)
                except Exception as exc:
                    trace(f"Consensus Session ⊘ jwt refresh | {exc}")

            if not _jwt_ok(jwt):
                age = jwt_iat_age_sec(jwt) if jwt else None
                ttl_left = jwt_seconds_to_exp(jwt) if jwt else None
                raise RuntimeError(
                    f"Consensus Session: __session stale after warmup "
                    f"(age={age} ttl_left={ttl_left}) — login/refresh required"
                )

            elapsed = (time.perf_counter() - t0) * 1000.0
            sess = ConsensusSession(
                cf_clearance=cookies_dict.get("cf_clearance", ""),
                session_jwt=jwt,
                cookies_dict=cookies_dict,
                created_at=time.time(),
            )
            trace(
                f"Consensus Session ✓ fast warmup | {elapsed:.0f}ms "
                f"cookies={len(cookies_dict)} "
                f"cf={bool(sess.cf_clearance)} "
                f"jwt_age={jwt_iat_age_sec(jwt)}"
            )
            return sess
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
            try:
                await playwright.stop()
            except Exception:
                pass


_manager_lock = asyncio.Lock()
_shared_manager: Optional[ConsensusSessionManager] = None


async def get_consensus_session_manager() -> ConsensusSessionManager:
    global _shared_manager
    async with _manager_lock:
        if _shared_manager is None:
            _shared_manager = ConsensusSessionManager()
        return _shared_manager


async def shutdown_consensus_session_manager() -> None:
    global _shared_manager
    async with _manager_lock:
        if _shared_manager is None:
            return
        await _shared_manager.close()
        _shared_manager = None
        trace("Consensus Session ✓ manager closed")
