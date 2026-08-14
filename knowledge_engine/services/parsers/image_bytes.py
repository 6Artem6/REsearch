"""Загрузка байтов изображения из src / data URI."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from knowledge_engine.services.parsers.html_attr import coerce_html_attr

_DATA_URI_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$",
    re.DOTALL | re.IGNORECASE,
)

_TRACKING_IMG_RE = re.compile(r"1x1|pixel|spacer|blank\.gif", re.I)


def sniff_mime(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def pick_img_src(img: object, base_url: str = "") -> str:
    """Best URL from img tag (lazy load, srcset, picture)."""
    from bs4 import Tag

    if not isinstance(img, Tag):
        return ""
    base = (base_url or "").strip()
    for attr in (
        "src",
        "data-src",
        "data-lazy-src",
        "data-original",
        "data-image",
        "data-lazy",
    ):
        raw = coerce_html_attr(img.get(attr))
        if not raw or raw.startswith("data:image/gif"):
            continue
        if _TRACKING_IMG_RE.search(raw):
            continue
        resolved = resolve_image_url(raw, base)
        if resolved:
            return resolved
    srcset = coerce_html_attr(img.get("srcset") or img.get("data-srcset"))
    if srcset:
        first_part = srcset.split(",")[0].strip()
        url_part = first_part.split()[0] if first_part else ""
        if url_part and not _TRACKING_IMG_RE.search(url_part):
            resolved = resolve_image_url(url_part, base)
            if resolved:
                return resolved
    picture = img.find_parent("picture")
    if picture is not None:
        for src in picture.find_all("source"):
            ss = coerce_html_attr(src.get("srcset") or src.get("src"))
            if not ss:
                continue
            url_part = ss.split(",")[0].strip().split()[0]
            resolved = resolve_image_url(url_part, base)
            if resolved:
                return resolved
    return ""


def resolve_image_url(src: str, base_url: str = "") -> str:
    raw = (src or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https", "data"):
        return raw
    base = (base_url or "").strip()
    if base:
        return urljoin(base, raw)
    if raw.startswith("/"):
        return raw
    return raw


def load_image_bytes(
    src: str,
    *,
    base_path: Path | None = None,
    base_url: str = "",
    timeout_sec: float = 12.0,
) -> tuple[bytes, str] | None:
    raw = resolve_image_url(src, base_url)
    if not raw:
        return None
    m = _DATA_URI_RE.match(raw)
    if m:
        mime = m.group(1).lower()
        try:
            data = base64.b64decode(m.group(2), validate=False)
        except Exception:
            return None
        return data, mime
    if raw.startswith("file://"):
        path = Path(raw[7:])
    elif raw.startswith("/") or (len(raw) > 2 and raw[1] == ":"):
        path = Path(raw)
    else:
        parsed = urlparse(raw)
        if parsed.scheme in ("http", "https"):
            try:
                with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
                    resp = client.get(raw)
                    resp.raise_for_status()
                    data = resp.content
                    ctype = (
                        (resp.headers.get("content-type") or "").split(";")[0].strip()
                    )
                    mime = ctype if ctype.startswith("image/") else sniff_mime(data)
                    return data, mime
            except Exception:
                return None
        path = (base_path or Path.cwd()) / raw
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return data, sniff_mime(data)
