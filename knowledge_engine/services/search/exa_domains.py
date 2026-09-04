"""Очистка доменов whitelist для Exa `include_domains` + dynamic extras."""

from __future__ import annotations

import threading
from urllib.parse import urlparse

from knowledge_engine.schemas.llm_contracts.exa_search import AUTHORITY_KEEP_CLASSES
from knowledge_engine.src.source_evaluator.whitelist import APPROVED_SOURCES_WHITELIST

_DYNAMIC_EXA_DOMAINS: dict[str, str] = {}
_DYNAMIC_LOCK = threading.Lock()


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


def foundational_docs_domains(
    whitelist_dict: dict[str, list[str]] | None = None,
) -> list[str]:
    """Hosts from the `foundational_docs` whitelist bucket only."""
    wl = whitelist_dict if whitelist_dict is not None else APPROVED_SOURCES_WHITELIST
    entries = wl.get("foundational_docs") if isinstance(wl, dict) else None
    if not isinstance(entries, list):
        return []
    return get_clean_exa_domains({"foundational_docs": entries})


def add_dynamic_exa_domain(host: str, classification: str = "OFFICIAL_DOCS") -> str:
    """Remember a Lite-approved hostname for later include_domains unions."""
    domain = clean_domain_for_exa(host)
    if not domain or "." not in domain:
        return ""
    cls = (classification or "OFFICIAL_DOCS").strip().upper() or "OFFICIAL_DOCS"
    with _DYNAMIC_LOCK:
        _DYNAMIC_EXA_DOMAINS[domain] = cls
    return domain


def list_dynamic_exa_domains(*, keep_only: bool = True) -> list[str]:
    with _DYNAMIC_LOCK:
        items = list(_DYNAMIC_EXA_DOMAINS.items())
    if not keep_only:
        return [h for h, _ in items]
    return [h for h, cls in items if cls in AUTHORITY_KEEP_CLASSES]


def get_clean_exa_domains(whitelist_dict: dict[str, list[str]]) -> list[str]:
    """Плоский уникальный список доменов из whitelist + динамический архив."""
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
    for domain in list_dynamic_exa_domains():
        if domain not in seen:
            seen.add(domain)
            out.append(domain)
    return out


def is_official_docs_host(host_or_url: str) -> bool:
    """True for foundational_docs whitelist or classifier-assigned OFFICIAL_DOCS."""
    host = clean_domain_for_exa(host_or_url)
    if not host:
        return False
    with _DYNAMIC_LOCK:
        dyn = _DYNAMIC_EXA_DOMAINS.get(host, "")
    if dyn == "OFFICIAL_DOCS":
        return True
    return host in set(foundational_docs_domains())
