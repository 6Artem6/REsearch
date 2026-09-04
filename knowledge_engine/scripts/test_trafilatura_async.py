import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
import trafilatura

DEFAULT_TARGET_URLS = [
    "https://docs.python.org/3/c-api/threads.html",
    "https://github.com/python/cpython/blob/f23a1837/InternalDocs/interpreter.md",
    "https://raw.githubusercontent.com/python/cpython/f23a1837/InternalDocs/interpreter.md",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def normalize_github_url(url: str) -> str | None:
    """Генерирует RAW ссылку, если передан GitHub blob URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return None


async def inspect_url(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    print(f"[FETCHING] {url}")
    result: Dict[str, Any] = {
        "url": url,
        "suggested_raw_url": normalize_github_url(url),
        "http_status": None,
        "content_type": None,
        "raw_response_char_len": 0,
        "error": None,
        "trafilatura_default": {
            "word_count": 0,
            "char_count": 0,
            "preview": "",
        },
        "trafilatura_advanced": {
            "word_count": 0,
            "char_count": 0,
            "preview": "",
        },
    }

    try:
        response = await client.get(url, headers=HEADERS, timeout=15.0, follow_redirects=True)
        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("Content-Type")
        html_content = response.text
        result["raw_response_char_len"] = len(html_content)

        if response.status_code != 200:
            result["error"] = f"HTTP Error {response.status_code}"
            return result

        # 1. Trafilatura DEFAULT
        ext_default = trafilatura.extract(html_content) or ""
        result["trafilatura_default"]["char_count"] = len(ext_default)
        result["trafilatura_default"]["word_count"] = len(ext_default.split())
        result["trafilatura_default"]["preview"] = ext_default[:300].replace("\n", " ")

        # 2. Trafilatura ADVANCED
        ext_advanced = (
            trafilatura.extract(
                html_content,
                include_links=False,
                include_formatting=True,
                include_tables=True,
                include_images=False,
                output_format="txt",
                target_language="en",
            )
            or ""
        )
        result["trafilatura_advanced"]["char_count"] = len(ext_advanced)
        result["trafilatura_advanced"]["word_count"] = len(ext_advanced.split())
        result["trafilatura_advanced"]["preview"] = ext_advanced[:300].replace("\n", " ")

        print(
            f"[OK] {url} -> Default Words: {result['trafilatura_default']['word_count']}, "
            f"Advanced Words: {result['trafilatura_advanced']['word_count']}"
        )

    except Exception as e:
        print(f"[ERROR] {url} -> {str(e)}")
        result["error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Async Trafilatura extractor debug tool.")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=DEFAULT_TARGET_URLS,
        help="Список URL для проверки",
    )
    parser.add_argument(
        "--output",
        default="trafilatura_report.json",
        help="Путь к файлу итогового отчета (JSON)",
    )
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        tasks = [inspect_url(client, url) for url in args.urls]
        results = await asyncio.gather(*tasks)

    # Формируем итоговый отчёт с ключом "urls_report" на массив результатов
    report = {
        "total_urls": len(results),
        "urls_report": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Отчет успешно сохранен в файл: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
