"""Проверка, что байты — читаемый PDF (PyMuPDF)."""

from __future__ import annotations

import re

_ACM_EPDF_RE = re.compile(r"/doi/epdf/", re.I)


def is_parseable_pdf(data: bytes | None, *, min_pages: int = 1) -> bool:
    if not data or len(data) < 2000 or data[:5] != b"%PDF-":
        return False
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return doc.page_count >= min_pages
        finally:
            doc.close()
    except Exception:
        return False


def is_acm_doi_pdf_url(url: str) -> bool:
    low = (url or "").lower()
    return "/doi/pdf/" in low and "/epdf/" not in low


def acm_epdf_to_pdf_url(url: str) -> str:
    u = (url or "").strip()
    if _ACM_EPDF_RE.search(u):
        return _ACM_EPDF_RE.sub("/doi/pdf/", u, count=1)
    return u


def acm_doi_pdf_download_variants(url: str) -> list[str]:
    """ACM: /doi/pdf/{doi} с query ?download=true и без."""
    u = (url or "").strip()
    if not u.startswith("http"):
        return []
    pdf = acm_epdf_to_pdf_url(u) if _ACM_EPDF_RE.search(u) else u
    if "/doi/pdf/" not in pdf.lower() or "/epdf/" in pdf.lower():
        return []
    base = pdf.split("?", 1)[0].rstrip("/")
    out: list[str] = []
    seen: set[str] = set()
    for v in (base, f"{base}?download=true"):
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def prefer_acm_pdf_endpoint(url: str) -> str:
    """ACM: явный /doi/pdf/ не заменяется на epdf."""
    u = (url or "").strip()
    if is_acm_doi_pdf_url(u):
        return u
    if "/doi/epdf/" in u.lower():
        return acm_epdf_to_pdf_url(u)
    return u


def doi_pdf_url_variants(url: str) -> list[str]:
    """Варианты fetch: epdf→pdf; для pdf-URL epdf не добавляется."""
    u = prefer_acm_pdf_endpoint(url)
    if not u:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = (v or "").strip()
        if v.startswith("http") and v not in seen:
            seen.add(v)
            out.append(v)

    add(u)
    low = u.lower()
    if "/doi/epdf/" in low:
        add(acm_epdf_to_pdf_url(u))
    if "/doi/pdf/" in low and "/epdf/" not in low:
        for v in acm_doi_pdf_download_variants(u):
            add(v)
    # Не добавлять epdf, если пользователь/источник уже указал /doi/pdf/
    return out
