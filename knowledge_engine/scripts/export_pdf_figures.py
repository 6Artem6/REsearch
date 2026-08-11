"""QA: тот же путь, что build_annotated_pdf → PNG из AnnotatedArticle.fig_bytes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_RENDER_DPI = 300


def _safe_name(key: str) -> str:
    return re.sub(r"[^\w.-]+", "_", key.strip())[:80]


def _fig_sort_key(key: str) -> tuple[int, str]:
    m = re.match(r"^FIG_(\d+)$", key, re.I)
    if m:
        return (0, f"{int(m.group(1)):06d}")
    m = re.match(r"^FIG_SEQ_(\d+)$", key, re.I)
    if m:
        return (1, f"{int(m.group(1)):06d}")
    return (2, key)


def _dpi_label(data: bytes, mime: str) -> str:
    from knowledge_engine.services.parsers.vector_pdf_cropper import png_pixel_size

    if not data or "png" not in (mime or "").lower():
        return "—"
    dims = png_pixel_size(data)
    if dims is None:
        return "—"
    return str(_RENDER_DPI)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export FIG_* via build_annotated_pdf (production path)",
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <pdf_stem>_figures next to PDF)",
    )
    args = parser.parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        print(f"not found: {pdf_path}", file=sys.stderr)
        return 2

    out_dir = args.out
    if out_dir is None:
        out_dir = pdf_path.parent / f"{pdf_path.stem}_figures"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from knowledge_engine.services.parsers.pdf_annotator import build_annotated_pdf
    from knowledge_engine.services.parsers.pdf_bytes import is_parseable_pdf
    from knowledge_engine.services.parsers.vector_pdf_cropper import export_status_label

    raw = pdf_path.read_bytes()
    if not is_parseable_pdf(raw):
        print(
            f"PDF not parseable (need valid PDF, >=2000 B, page_count>=1): "
            f"{len(raw)} bytes",
            file=sys.stderr,
        )
        return 3

    import fitz

    doc = fitz.open(stream=raw, filetype="pdf")
    page_count = doc.page_count
    text_len = sum(len((doc[i].get_text() or "").strip()) for i in range(page_count))
    doc.close()
    print(
        f"PDF: {page_count} pages, {len(raw)} bytes, "
        f"extractable_text_chars={text_len}"
    )
    if text_len < 200:
        print(
            "warning: almost no text layer — likely raster/scanned pages",
            file=sys.stderr,
        )

    annotated = build_annotated_pdf(raw)
    sources = annotated.fig_extract_source or {}
    topology_map = annotated.fig_extract_topology or {}
    fig_keys = sorted(
        set(annotated.fig_bytes.keys()) | set(sources.keys()),
        key=_fig_sort_key,
    )
    if not fig_keys:
        print("no figures in AnnotatedArticle", file=sys.stderr)
        return 4

    rows: list[tuple[str, int, str, str, str]] = []
    written = 0
    for key in fig_keys:
        source = sources.get(key, "unknown")
        topo = topology_map.get(key)
        status = export_status_label(source, topo)
        payload = annotated.fig_bytes.get(key)
        if payload and not source.startswith("invalid:"):
            data, mime = payload
            ext = "png" if "png" in (mime or "") else "bin"
            path = out_dir / f"{_safe_name(key)}.{ext}"
            path.write_bytes(data)
            written += 1
            nbytes = len(data)
            dpi = _dpi_label(data, mime)
        else:
            nbytes = 0
            dpi = "—"

        rows.append((key, nbytes, dpi, source, status))

    n_ok = sum(1 for *_, st in rows if st == "OK" or st.startswith("OK:"))
    n_warn = sum(1 for *_, st in rows if st.startswith("WARN:"))
    n_rej = sum(1 for *_, st in rows if st.startswith("REJECTED:"))
    n_residue = sum(
        1 for *_, st in rows if "zero_visual_residue" in st or "duplicate_visual" in st
    )

    print()
    hdr = f"{'FIG_ID':<14} {'BYTES':>8} {'DPI':>5}  {'SOURCE':<24} STATUS"
    print(hdr)
    print("-" * min(120, len(hdr) + 40))
    for fig_id, nbytes, dpi, source, status in rows:
        print(f"{fig_id:<14} {nbytes:>8} {dpi:>5}  {source:<24} {status}")

    print(
        f"\nwritten={written} rows={len(rows)} | "
        f"OK={n_ok} WARN={n_warn} REJECTED={n_rej} (residue/dup={n_residue}) → {out_dir}"
    )
    print(
        "note: STATUS uses fig_extract_topology from VectorPDFCropper; "
        "xref skips density; WARN does not block export.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
