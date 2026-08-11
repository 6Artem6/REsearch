"""ArticleIngestionPipeline: parse → sanitize → batch VLM → store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from knowledge_engine.services.article_diagram_store import (
    phash_exists,
    save_diagram,
)
from knowledge_engine.services.article_ingestion.smart_filter import (
    filter_structural_diagram_candidates,
)
from knowledge_engine.services.image_filter import ImageSanitizer
from knowledge_engine.services.mermaid_validate import (
    is_misclassified_benchmark_flowchart,
    process_mermaid_for_ingest,
    strip_mermaid_fences,
    validate_mermaid_syntax,
)
from knowledge_engine.services.parsers.base import ExtractedDiagram, ExtractedImage
from knowledge_engine.services.parsers.html_parser import HtmlArticleParser
from knowledge_engine.services.parsers.md_parser import MarkdownArticleParser
from knowledge_engine.services.parsers.pdf_parser import PdfArticleParser
from knowledge_engine.services.vlm_batcher import chunk_batches, run_vlm_images_parallel
from knowledge_engine.ui.run_log import trace


class ArticleFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    MD = "markdown"


@dataclass
class IngestedDiagram:
    image_phash: str
    caption: str
    mermaid_code: str
    summary: str
    title: str


def _finalize_vlm_mermaid(vlm_mermaid: str, *, phash_hint: str = "") -> str:
    """sanitize → validate → Gemma repair; fenced mermaid or reject."""
    raw = (vlm_mermaid or "").strip()
    if not raw:
        return ""
    fenced = process_mermaid_for_ingest(raw)
    if not fenced:
        tag = phash_hint[:12] + "…" if phash_hint else "?"
        trace(f"ARTICLE_INGEST spatial ⊘ | mermaid rejected phash={tag}")
    return fenced


class ArticleIngestionPipeline:
    def __init__(
        self,
        *,
        batch_min: int = 3,
        batch_max: int = 5,
        base_path: Path | None = None,
    ) -> None:
        self._batch_min = batch_min
        self._batch_max = batch_max
        self._base_path = base_path

    def ingest(
        self,
        article_id: str,
        data: bytes,
        content_type: ArticleFormat | str,
        *,
        page_url: str = "",
    ) -> list[IngestedDiagram]:
        fmt = self._normalize_format(content_type)
        aid = (article_id or "").strip()
        images, inline_diagrams = self._parse_all(data, fmt, page_url=page_url)
        saved: list[IngestedDiagram] = []
        saved.extend(self._save_inline_diagrams(aid, inline_diagrams))

        sanitizer = ImageSanitizer()
        clean: list[ExtractedImage] = []
        for img in images:
            ok, ph = sanitizer.accept(
                img.image_bytes, img.layout, phash=img.phash or None
            )
            if not ok:
                continue
            img.phash = ph
            if phash_exists(ph):
                trace(f"ARTICLE_INGEST skip cache phash={ph[:12]}…")
                continue
            clean.append(img)

        candidates = len(clean)
        clean = filter_structural_diagram_candidates(clean)
        trace(
            f"ARTICLE_INGEST filter | sanitizer_ok={candidates} "
            f"qwen_ok={len(clean)}"
        )

        if not clean:
            trace(
                f"ARTICLE_INGEST ✓ {aid} | inline={len(saved)} "
                f"images_after_filter=0"
            )
            return saved

        batches = chunk_batches(clean, self._batch_min, self._batch_max)
        trace(
            f"ARTICLE_INGEST ▶ {aid} | images={len(clean)} "
            f"batches={len(batches)} inline_saved={len(saved)} (VLM parallel per image)"
        )
        pairs = run_vlm_images_parallel(clean, label=f"article_ingest/{aid[:16]}")
        for img, item in pairs:
            if item is None or not item.is_diagram:
                continue
            mermaid = _finalize_vlm_mermaid(
                (item.mermaid or "").strip(),
                phash_hint=img.phash,
            )
            if not mermaid:
                continue
            caption_probe = f"{item.title} {img.caption} {img.context_text}"
            if is_misclassified_benchmark_flowchart(
                strip_mermaid_fences(mermaid),
                caption_probe,
            ):
                trace(
                    f"ARTICLE_INGEST skip misclassified benchmark flowchart "
                    f"phash={img.phash[:12]}…"
                )
                continue
            caption = (item.title or img.caption or "")[:2000]
            summary = (item.summary or "").strip()
            save_diagram(
                aid,
                img.phash,
                caption,
                mermaid,
                summary,
            )
            saved.append(
                IngestedDiagram(
                    image_phash=img.phash,
                    caption=caption,
                    mermaid_code=mermaid,
                    summary=summary,
                    title=(item.title or "")[:300],
                )
            )
        trace(f"ARTICLE_INGEST ✓ {aid} | saved={len(saved)}")
        return saved

    def ingest_vlm_targets(
        self,
        article_id: str,
        images: list[ExtractedImage],
    ) -> list[IngestedDiagram]:
        """VLM для уже отфильтрованного списка (spatial FIG dispatch)."""
        aid = (article_id or "").strip()
        if not aid or not images:
            return []
        sanitizer = ImageSanitizer()
        clean: list[ExtractedImage] = []
        for img in images:
            ok, ph = sanitizer.accept(
                img.image_bytes, img.layout, phash=img.phash or None
            )
            if not ok:
                trace("ARTICLE_INGEST spatial ⊘ | sanitizer rejected image")
                continue
            img.phash = ph
            if phash_exists(ph):
                trace(f"ARTICLE_INGEST spatial skip cache phash={ph[:12]}…")
                continue
            clean.append(img)
        if not clean:
            return []
        saved: list[IngestedDiagram] = []
        trace(f"ARTICLE_INGEST spatial ▶ | images={len(clean)} VLM parallel per image")
        pairs = run_vlm_images_parallel(clean, label="article_ingestion/spatial_vlm")
        for img, item in pairs:
            if item is None:
                trace(
                    f"ARTICLE_INGEST spatial ⊘ | missing VLM item phash={img.phash[:12]}…"
                )
                continue
            if not item.is_diagram:
                trace(
                    f"ARTICLE_INGEST spatial ⊘ | is_diagram=false "
                    f"phash={img.phash[:12]}…"
                )
                continue
            mermaid = _finalize_vlm_mermaid(
                (item.mermaid or "").strip(),
                phash_hint=img.phash,
            )
            if not mermaid:
                continue
            caption_probe = f"{item.title} {img.caption} {img.context_text}"
            if is_misclassified_benchmark_flowchart(
                strip_mermaid_fences(mermaid),
                caption_probe,
            ):
                trace(
                    f"ARTICLE_INGEST spatial ⊘ | misclassified benchmark "
                    f"phash={img.phash[:12]}…"
                )
                continue
            caption = (item.title or img.caption or "")[:2000]
            summary = (item.summary or img.context_text or caption).strip()[:2000]
            save_diagram(aid, img.phash, caption, mermaid, summary)
            saved.append(
                IngestedDiagram(
                    image_phash=img.phash,
                    caption=caption,
                    mermaid_code=mermaid,
                    summary=summary,
                    title=(item.title or "")[:300],
                )
            )
        return saved

    def ingest_vlm_targets_multi_article(
        self,
        works: list[tuple[str, ExtractedImage]],
    ) -> dict[str, int]:
        """Один VLM-пул на все изображения (разные article_id), лимиты Flash Lite."""
        if not works:
            return {}
        sanitizer = ImageSanitizer()
        clean: list[tuple[str, ExtractedImage]] = []
        for aid, img in works:
            ok, ph = sanitizer.accept(
                img.image_bytes, img.layout, phash=img.phash or None
            )
            if not ok:
                continue
            img.phash = ph
            if phash_exists(ph):
                trace(f"ARTICLE_INGEST spatial skip cache phash={ph[:12]}…")
                continue
            clean.append(((aid or "").strip(), img))
        if not clean:
            return {}
        images = [img for _, img in clean]
        trace(
            f"ARTICLE_INGEST spatial pool ▶ | images={len(images)} "
            f"articles={len({a for a, _ in clean})}"
        )
        pairs = run_vlm_images_parallel(
            images, label="article_ingestion/spatial_vlm_pool"
        )
        saved_count: dict[str, int] = {}
        for (aid, _img), (img_out, item) in zip(clean, pairs):
            if item is None or not item.is_diagram:
                continue
            mermaid = _finalize_vlm_mermaid(
                (item.mermaid or "").strip(),
                phash_hint=img_out.phash,
            )
            if not mermaid:
                continue
            caption_probe = f"{item.title} {img_out.caption} {img_out.context_text}"
            if is_misclassified_benchmark_flowchart(
                strip_mermaid_fences(mermaid),
                caption_probe,
            ):
                continue
            caption = (item.title or img_out.caption or "")[:2000]
            summary = (item.summary or img_out.context_text or caption).strip()[:2000]
            save_diagram(aid, img_out.phash, caption, mermaid, summary)
            saved_count[aid] = saved_count.get(aid, 0) + 1
        return saved_count

    def _save_inline_diagrams(
        self,
        article_id: str,
        diagrams: list[ExtractedDiagram],
    ) -> list[IngestedDiagram]:
        out: list[IngestedDiagram] = []
        for d in diagrams:
            if not d.is_inline_mermaid:
                continue
            mermaid = _finalize_vlm_mermaid((d.mermaid_code or "").strip())
            if not mermaid:
                continue
            inner = strip_mermaid_fences(mermaid)
            if not validate_mermaid_syntax(inner):
                continue
            ph = hashlib.sha256(inner.encode("utf-8")).hexdigest()[:48]
            if phash_exists(ph):
                trace(f"ARTICLE_INGEST inline skip phash={ph[:12]}…")
                continue
            caption = (d.caption or "")[:2000]
            summary = caption[:400]
            save_diagram(article_id, ph, caption, mermaid, summary)
            out.append(
                IngestedDiagram(
                    image_phash=ph,
                    caption=caption,
                    mermaid_code=mermaid,
                    summary=summary,
                    title=caption[:300],
                )
            )
        if out:
            trace(f"ARTICLE_INGEST inline ✓ | n={len(out)}")
        return out

    def _normalize_format(self, content_type: ArticleFormat | str) -> ArticleFormat:
        if isinstance(content_type, ArticleFormat):
            return content_type
        raw = str(content_type or "").strip().lower()
        if raw in ("pdf", "application/pdf"):
            return ArticleFormat.PDF
        if raw in ("html", "htm", "text/html"):
            return ArticleFormat.HTML
        if raw in ("md", "markdown", "text/markdown", "text/x-markdown"):
            return ArticleFormat.MD
        return ArticleFormat.MD

    def _parse_all(
        self,
        data: bytes,
        fmt: ArticleFormat,
        *,
        page_url: str = "",
    ) -> tuple[list[ExtractedImage], list[ExtractedDiagram]]:
        if fmt == ArticleFormat.PDF:
            return PdfArticleParser().parse(data), []
        if fmt == ArticleFormat.HTML:
            return (
                HtmlArticleParser(
                    self._base_path,
                    page_url=page_url,
                ).parse(data),
                [],
            )
        parser = MarkdownArticleParser(self._base_path)
        return parser.parse_all(data)

    def _parse(self, data: bytes, fmt: ArticleFormat) -> list[ExtractedImage]:
        images, _ = self._parse_all(data, fmt)
        return images
