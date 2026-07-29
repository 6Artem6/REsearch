"""HTTP-проверка URL статей (404 / soft-404) для curriculum pipeline."""

from __future__ import annotations

import asyncio
import re
from typing import Sequence

import httpx

from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace

_DEFAULT_TIMEOUT = 10.0
_USER_AGENT = "REsearch-CurriculumUrlValidate/1.0"

_SOFT_404_TITLE_RE = re.compile(
    r"(404|not\s+found|page\s+not\s+found|article\s+does\s+not\s+exist|"
    r"doesn'?t\s+exist|no\s+longer\s+available|страница\s+не\s+найдена|"
    r"не\s+найдена|материал\s+не\s+найден)",
    re.I,
)
_SOFT_404_BODY_SNIPPETS = (
    "page not found",
    "404 not found",
    "article does not exist",
    "this page doesn't exist",
    "страница не найдена",
    "материал не найден",
)


def _normalize_check_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith("http"):
        return ""
    return u.split("#")[0].rstrip("/")


async def _fetch_head_or_get(client: httpx.AsyncClient, url: str) -> tuple[int, str, str]:
    """(status_code, title_hint, body_snippet)"""
    status = 0
    title_hint = ""
    body_snippet = ""
    try:
        head = await client.head(url)
        status = head.status_code
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
    except Exception:
        status = 0

    if status == 404:
        return status, "", ""

    need_get = status in (0, 405, 501) or (200 <= status < 400)
    if not need_get and status >= 400:
        return status, "", ""

    try:
        resp = await client.get(url, headers={"Range": "bytes=0-8192"})
        status = resp.status_code
        if status < 400:
            raw = resp.text[:8192] if resp.text else ""
            body_snippet = raw[:2000]
            m = re.search(r"<title[^>]*>([^<]{1,200})</title>", raw, re.I | re.S)
            if m:
                title_hint = m.group(1).strip()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
    except Exception:
        if status == 0:
            status = 0

    return status, title_hint, body_snippet


def _is_soft_404(status: int, title_hint: str, body_snippet: str) -> bool:
    if status == 404:
        return True
    if status < 200 or status >= 400:
        return status != 0
    blob = f"{title_hint}\n{body_snippet}".lower()
    if _SOFT_404_TITLE_RE.search(title_hint or ""):
        return True
    for needle in _SOFT_404_BODY_SNIPPETS:
        if needle in blob:
            return True
    return False


async def check_url_live(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[bool, str]:
    """True если URL отвечает 2xx и не soft-404."""
    u = _normalize_check_url(url)
    if not u:
        return False, "invalid_url"
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        status, title_hint, body = await _fetch_head_or_get(client, u)
    if _is_soft_404(status, title_hint, body):
        reason = "404" if status == 404 else "soft_404"
        return False, reason
    if status < 200 or status >= 400:
        return False, f"http_{status}"
    return True, "ok"


async def validate_and_filter_urls_async(
    hits: Sequence[CurriculumSearchHit],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_concurrency: int = 8,
) -> tuple[list[CurriculumSearchHit], list[CurriculumSearchHit]]:
    """Разделить hits на живые (200 OK) и битые."""
    if not hits:
        return [], []

    sem = asyncio.Semaphore(max(1, max_concurrency))
    valid: list[CurriculumSearchHit] = []
    broken: list[CurriculumSearchHit] = []

    async def one(hit: CurriculumSearchHit) -> tuple[CurriculumSearchHit, bool, str]:
        async with sem:
            ok, reason = await check_url_live(hit.url, timeout=timeout)
            return hit, ok, reason

    results = await asyncio.gather(*(one(h) for h in hits))
    for hit, ok, reason in results:
        if ok:
            valid.append(hit)
        else:
            broken.append(hit)
            trace(f"CURRICULUM url_validate ⊘ | {hit.url[:70]} | {reason}")

    trace(
        f"CURRICULUM url_validate ✓ | in={len(hits)} valid={len(valid)} broken={len(broken)}"
    )
    return valid, broken


def validate_and_filter_urls(
    hits: Sequence[CurriculumSearchHit],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[list[CurriculumSearchHit], list[CurriculumSearchHit]]:
    """Sync wrapper для worker / ThreadPool."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(validate_and_filter_urls_async(hits, timeout=timeout))

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(
            asyncio.run,
            validate_and_filter_urls_async(hits, timeout=timeout),
        )
        return fut.result()

