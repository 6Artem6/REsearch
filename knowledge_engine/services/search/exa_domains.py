"""Очистка доменов whitelist для Exa `include_domains`."""

from __future__ import annotations

from urllib.parse import urlparse


def clean_domain_for_exa(entry: str) -> str:
    """
    Одна запись whitelist → домен для Exa API (без пути, query, www.).

    Примеры:
      habr.com/ru/companies/yandex → habr.com
      https://openai.com/research → openai.com
      lilianweng.github.io → lilianweng.github.io
    """
    s = (entry or "").strip()
    if not s:
        return ""
    if "://" in s:
        parsed = urlparse(s if "://" in s else f"https://{s}")
        host = (parsed.netloc or "").strip()
        if not host and parsed.path:
            host = parsed.path.split("/")[0]
    else:
        host = s.split("/")[0].split("?")[0].split("#")[0].strip()
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def get_clean_exa_domains(whitelist_dict: dict[str, list[str]]) -> list[str]:
    """Плоский уникальный список доменов из всех категорий APPROVED_SOURCES_WHITELIST."""
    seen: set[str] = set()
    out: list[str] = []
    for entries in whitelist_dict.values():
        if not isinstance(entries, list):
            continue
        for raw in entries:
            if not isinstance(raw, str):
                continue
            domain = clean_domain_for_exa(raw)
            if not domain or domain in seen:
                continue
            seen.add(domain)
            out.append(domain)
    return out
