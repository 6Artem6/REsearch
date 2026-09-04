#!/usr/bin/env python3
"""Isolated Trafilatura quality probe: fetch HTML, extract markdown, write output/.

Run from repo root:

  ./.venv/bin/python knowledge_engine/scripts/trafilatura_test.py
  ./.venv/bin/python knowledge_engine/scripts/trafilatura_test.py --async
  ./.venv/bin/python knowledge_engine/scripts/trafilatura_test.py --url https://peps.python.org/pep-0703/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trafilatura_test")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT_SEC = 25.0
MAX_CONCURRENCY = 4

# Mix of article / docs / blog / GitHub blob / listing-with-nav / expected 404.
TEST_URLS: list[str] = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://peps.python.org/pep-0703/",
    "https://docs.python.org/3/howto/free-threading-python.html",
    "https://github.com/python/cpython/blob/main/InternalDocs/interpreter.md",
    "https://realpython.com/python-gil/",
    "https://python.langchain.com/docs/introduction/",
    "https://news.ycombinator.com/",
    "https://peps.python.org/pep-99999/",
]


class ExtractionResult(BaseModel):
    url: str
    title: str | None = None
    author: str | None = None
    date: str | None = None
    content_markdown: str | None = None
    raw_length: int = 0
    extraction_time_ms: float = 0.0
    extract_time_ms: float = 0.0
    http_status: int | None = None
    status: str = Field(default="success")
    error: str | None = None


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (name or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return (cleaned or "page")[:80]


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "page").replace("www.", "")
    path = (parsed.path or "").strip("/") or "index"
    return sanitize_filename(f"{host}_{path.replace('/', '_')}")


def fetch_html(url: str, client: httpx.Client) -> tuple[str, int | None, str | None]:
    """Return (html, http_status, error_status_label)."""
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        return "", None, f"failed_to_fetch:{type(exc).__name__}"
    status = int(resp.status_code)
    if status in (401, 403, 404, 429, 503):
        return resp.text or "", status, f"http_{status}"
    if status >= 400:
        return resp.text or "", status, f"http_{status}"
    return resp.text or "", status, None


async def fetch_html_async(
    url: str, client: httpx.AsyncClient
) -> tuple[str, int | None, str | None]:
    try:
        resp = await client.get(url)
    except httpx.HTTPError as exc:
        return "", None, f"failed_to_fetch:{type(exc).__name__}"
    status = int(resp.status_code)
    if status >= 400:
        return resp.text or "", status, f"http_{status}"
    return resp.text or "", status, None


def extract_markdown(html: str, url: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (markdown, title, author, date)."""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (getattr(metadata, "title", None) or "").strip() or None
    author = (getattr(metadata, "author", None) or "").strip() or None
    date = (getattr(metadata, "date", None) or "").strip() or None
    content = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_images=False,
        include_links=True,
        favor_precision=True,
    )
    markdown = (content or "").strip() or None
    return markdown, title, author, date


def process_downloaded(
    url: str,
    html: str,
    http_status: int | None,
    fetch_error: str | None,
    started: float,
) -> ExtractionResult:
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if fetch_error and not html.strip():
        return ExtractionResult(
            url=url,
            extraction_time_ms=elapsed_ms,
            http_status=http_status,
            status=fetch_error,
            error=fetch_error,
        )
    if fetch_error and http_status in (401, 403, 404):
        return ExtractionResult(
            url=url,
            extraction_time_ms=elapsed_ms,
            http_status=http_status,
            status=fetch_error,
            error=fetch_error,
        )
    if not html.strip():
        return ExtractionResult(
            url=url,
            extraction_time_ms=elapsed_ms,
            http_status=http_status,
            status="failed_to_fetch",
            error="empty_html",
        )
    extract_started = time.perf_counter()
    markdown, title, author, date = extract_markdown(html, url)
    extract_ms = round((time.perf_counter() - extract_started) * 1000, 2)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if not markdown:
        return ExtractionResult(
            url=url,
            title=title,
            author=author,
            date=date,
            raw_length=0,
            extraction_time_ms=elapsed_ms,
            extract_time_ms=extract_ms,
            http_status=http_status,
            status="empty_content",
            error="trafilatura_empty",
        )
    return ExtractionResult(
        url=url,
        title=title,
        author=author,
        date=date,
        content_markdown=markdown,
        raw_length=len(markdown),
        extraction_time_ms=elapsed_ms,
        extract_time_ms=extract_ms,
        http_status=http_status,
        status="success",
    )


def process_url(url: str, client: httpx.Client) -> ExtractionResult:
    started = time.perf_counter()
    html, http_status, fetch_error = fetch_html(url, client)
    if fetch_error:
        log.warning("fetch %s | %s | http=%s", url, fetch_error, http_status)
    else:
        log.info("fetch %s | http=%s | html=%s chars", url, http_status, len(html))
    return process_downloaded(url, html, http_status, fetch_error, started)


async def process_url_async(
    url: str,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> ExtractionResult:
    started = time.perf_counter()
    async with sem:
        html, http_status, fetch_error = await fetch_html_async(url, client)
    if fetch_error:
        log.warning("fetch %s | %s | http=%s", url, fetch_error, http_status)
    else:
        log.info("fetch %s | http=%s | html=%s chars", url, http_status, len(html))
    return process_downloaded(url, html, http_status, fetch_error, started)


def write_markdown_file(output_dir: Path, index: int, result: ExtractionResult) -> Path | None:
    if result.status != "success" or not result.content_markdown:
        return None
    stem = sanitize_filename(result.title) if result.title else _filename_from_url(result.url)
    path = output_dir / f"{index:02d}_{stem}.md"
    header = [f"# {result.title or stem}", "", f"**URL**: {result.url}"]
    if result.author:
        header.append(f"**Author**: {result.author}")
    if result.date:
        header.append(f"**Date**: {result.date}")
    header.append(f"**Length**: {result.raw_length} chars")
    header.append(f"**Time**: {result.extraction_time_ms} ms")
    header.extend(["", "---", "", result.content_markdown.rstrip(), ""])
    path.write_text("\n".join(header), encoding="utf-8")
    return path


def build_summary(results: list[ExtractionResult]) -> dict:
    success = [r for r in results if r.status == "success"]
    empty = [r for r in results if r.status == "empty_content"]
    http_err = [r for r in results if r.status.startswith("http_")]
    fetch_err = [r for r in results if r.status.startswith("failed_to_fetch")]
    extract_times = [r.extraction_time_ms for r in results]
    dump_rows = []
    for row in results:
        payload = row.model_dump()
        md = payload.get("content_markdown") or ""
        payload["content_markdown"] = None
        payload["content_preview"] = md[:400]
        dump_rows.append(payload)
    return {
        "stats": {
            "total": len(results),
            "success": len(success),
            "empty_content": len(empty),
            "http_errors": len(http_err),
            "fetch_errors": len(fetch_err),
            "avg_raw_length": round(
                sum(r.raw_length for r in success) / len(success), 1
            )
            if success
            else 0,
            "avg_extraction_time_ms": round(sum(extract_times) / len(extract_times), 2)
            if extract_times
            else 0,
            "avg_extract_only_ms": round(
                sum(r.extract_time_ms for r in results) / len(results), 2
            )
            if results
            else 0,
            "max_extraction_time_ms": max(extract_times) if extract_times else 0,
        },
        "results": dump_rows,
    }


def run_sync(urls: list[str]) -> list[ExtractionResult]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en,ru;q=0.8"}
    with httpx.Client(
        timeout=HTTP_TIMEOUT_SEC,
        follow_redirects=True,
        headers=headers,
    ) as client:
        results: list[ExtractionResult] = []
        for idx, url in enumerate(urls, 1):
            log.info("[%s/%s] %s", idx, len(urls), url)
            results.append(process_url(url, client))
        return results


async def run_async(urls: list[str]) -> list[ExtractionResult]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en,ru;q=0.8"}
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SEC,
        follow_redirects=True,
        headers=headers,
    ) as client:
        tasks = [process_url_async(url, client, sem) for url in urls]
        return list(await asyncio.gather(*tasks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trafilatura markdown extraction probe")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=[],
        help="Extra URL (repeatable). Replaces the default list if passed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for .md files and summary.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Fetch HTML concurrently with httpx.AsyncClient",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = list(args.urls) if args.urls else list(TEST_URLS)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_async:
        results = asyncio.run(run_async(urls))
    else:
        results = run_sync(urls)

    for idx, result in enumerate(results, 1):
        saved = write_markdown_file(output_dir, idx, result)
        if saved:
            log.info(
                "saved %s | %s chars | total=%.1f ms extract=%.1f ms | title=%s",
                saved.name,
                result.raw_length,
                result.extraction_time_ms,
                result.extract_time_ms,
                (result.title or "")[:70],
            )
        else:
            log.warning(
                "no markdown | %s | status=%s | http=%s",
                result.url,
                result.status,
                result.http_status,
            )

    summary = build_summary(results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats = summary["stats"]
    log.info(
        "summary %s | success=%s/%s empty=%s http_err=%s avg_ms=%.1f",
        summary_path,
        stats["success"],
        stats["total"],
        stats["empty_content"],
        stats["http_errors"],
        stats["avg_extraction_time_ms"],
    )
    return 0 if stats["success"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
