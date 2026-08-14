"""Benchmark Gemma MAP: Pydantic schema success vs input window size.

Runs A–D (≈2k / 4k / 8k / 10k token windows) sequentially against the real
MAP system prompt + ``MapWindowResponse`` schema. No Gemini Flash fallback —
failures are true Gemma parse/validation misses.

Usage (repo root, ``.env`` with Gemma key):

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.benchmark_gemma_chunk_sizes \\
    --file /path/to/article.txt

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.benchmark_gemma_chunk_sizes \\
    --url https://arxiv.org/pdf/2512.20660.pdf --max-chunks 3

  # Chunk plan only (no API):
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.benchmark_gemma_chunk_sizes \\
    --file article.txt --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from pydantic import ValidationError

from knowledge_engine.config import (
    GEMMA_API_BASE,
    GEMMA_API_KEY,
    GEMMA_MAP_MAX_OUTPUT_TOKENS,
    GEMMA_PRIMARY_MODEL,
)
from knowledge_engine.services.article_ingestion.blog_spatial_schemas import (
    MapWindowResponse,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    _MAP_SYSTEM,
)
from knowledge_engine.services.llm.gemma_client import (
    GemmaRateLimitError,
    _gemma_user_content,
    _strip_gemma_thought_wrapper,
)
from knowledge_engine.ui.run_log import trace

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
_DEFAULT_WINDOW_SIZES = (2000, 4000, 8000, 10000)
_TIKTOKEN_ENC: Any = None


def _tiktoken_enc() -> Any:
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is not None:
        return _TIKTOKEN_ENC
    try:
        import tiktoken

        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TIKTOKEN_ENC = False
    return _TIKTOKEN_ENC


def _estimate_tokens(text: str) -> int:
    """Fast local estimate — tiktoken cl100k ×1.15, else chars/4.

    Avoids ``estimate_text_tokens`` (Qwen HF Autotokenizer can hang offline).
    """
    t = text or ""
    if not t:
        return 0
    enc = _tiktoken_enc()
    if enc:
        try:
            return max(1, int(len(enc.encode(t)) * 1.15))
        except Exception:
            pass
    return max(1, len(t) // 4)


_SAMPLE_PARAGRAPH = (
    "[P_1] Governed agent pipelines attach hooks before and after tool calls so "
    "policy engines can veto, rewrite, or audit actions. The mechanic is a "
    "middleware chain: ingress validator → planner → tool gateway → egress "
    "redactor. In one INSTANCE, a bank MCP server rejected 12% of write tools "
    "when the session lacked dual-control approval. "
    "Principles include least privilege, explicit capability tokens, and "
    "deterministic replay of the decision log. "
)


@dataclass
class ChunkCallResult:
    run_label: str
    window_target: int
    chunk_index: int
    chunks_total: int
    in_tokens: int
    latency_sec: float
    success: bool
    atoms: int
    error: str | None = None
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    raw_preview: str = ""


@dataclass
class RunSummary:
    window_target: int
    label: str
    chunks: int
    total_time_sec: float
    avg_latency_sec: float
    success_rate_pct: float
    total_atoms: int
    avg_atoms_per_1k: float
    calls: list[ChunkCallResult] = field(default_factory=list)


def _load_text(
    *, file: str | None, url: str | None, pdf: str | None
) -> tuple[str, str]:
    """Return (text, source_label)."""
    if file:
        path = Path(file).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--file not found: {path}")
        raw = path.read_bytes()
        if raw[:5] == b"%PDF-":
            return _pdf_bytes_to_text(raw), str(path)
        return raw.decode("utf-8", errors="replace"), str(path)

    if pdf:
        path = Path(pdf).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--pdf not found: {path}")
        return _pdf_bytes_to_text(path.read_bytes()), str(path)

    if url:
        import httpx

        with httpx.Client(timeout=90.0, follow_redirects=True) as client:
            resp = client.get(url.strip())
            resp.raise_for_status()
            data = resp.content
        if data[:5] == b"%PDF-":
            return _pdf_bytes_to_text(data), url.strip()
        ctype = (resp.headers.get("content-type") or "").lower()
        if "html" in ctype:
            # crude strip — enough for a size/latency benchmark
            text = re.sub(
                r"(?is)<script.*?>.*?</script>", " ", data.decode("utf-8", "replace")
            )
            text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text, url.strip()
        return data.decode("utf-8", errors="replace"), url.strip()

    # Synthetic corpus large enough for 10k windows (repeat sample).
    body = "\n\n".join(
        _SAMPLE_PARAGRAPH.replace("[P_1]", f"[P_{i}]") for i in range(1, 220)
    )
    return body, "synthetic:sample_paragraphs"


def _pdf_bytes_to_text(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise SystemExit("PyMuPDF (fitz) required for PDF input") from exc
    doc = fitz.open(stream=data, filetype="pdf")
    parts: list[str] = []
    try:
        for i, page in enumerate(doc):
            t = (page.get_text("text") or "").strip()
            if t:
                parts.append(f"[P_{i + 1}] {t}")
    finally:
        doc.close()
    return "\n\n".join(parts)


def split_into_token_windows(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Greedy character windows sized by ``estimate_text_tokens`` ≈ target."""
    body = (text or "").strip()
    if not body:
        return []
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")

    # Start with char budget ≈ 4 * tokens; refine by measured tokens.
    approx_chars = max(200, target_tokens * 4)
    chunks: list[str] = []
    pos = 0
    n = len(body)
    while pos < n:
        end = min(n, pos + approx_chars)
        # Prefer paragraph / blank-line boundaries near the end.
        if end < n:
            window = body[pos:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > len(window) * 0.55:
                end = pos + cut + (2 if window[cut : cut + 2] == "\n\n" else 1)
        piece = body[pos:end].strip()
        if not piece:
            pos = end
            continue
        # Grow / shrink to approach target_tokens (±15%).
        toks = _estimate_tokens(piece)
        grow_guard = 0
        while toks < int(target_tokens * 0.85) and end < n and grow_guard < 40:
            end = min(n, end + approx_chars // 4)
            piece = body[pos:end].strip()
            toks = _estimate_tokens(piece)
            grow_guard += 1
        shrink_guard = 0
        while (
            toks > int(target_tokens * 1.15) and len(piece) > 200 and shrink_guard < 40
        ):
            end = pos + int(len(piece) * 0.9)
            piece = body[pos:end].strip()
            toks = _estimate_tokens(piece)
            shrink_guard += 1
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        if overlap_tokens > 0:
            back = min(len(piece), overlap_tokens * 4)
            pos = max(pos + 1, end - back)
        else:
            pos = end
    return chunks


def build_map_user_prompt(
    window_text: str,
    *,
    title: str,
    source: str,
    window_index: int,
) -> str:
    return "\n".join(
        [
            "<article_context>",
            f"ARTICLE_TITLE: {(title or 'benchmark')[:300]}",
            f"ARTICLE_URL: {(source or '')[:500]}",
            "SECTION: —",
            f"WINDOW_INDEX: {window_index}",
            "</article_context>",
            "",
            "<window_text>",
            window_text.strip(),
            "</window_text>",
        ]
    )


def parse_map_response(
    raw: str,
) -> tuple[MapWindowResponse | None, str | None, list[dict[str, Any]]]:
    """Return (parsed, error_summary, validation_error_dicts)."""
    text = _strip_gemma_thought_wrapper((raw or "").strip())
    if not text:
        return None, "empty model content", []
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}", []
    try:
        return MapWindowResponse.model_validate(data), None, []
    except ValidationError as exc:
        errs = exc.errors()
        return None, f"ValidationError ({len(errs)} issue(s))", list(errs)


async def gemma_map_once(
    *,
    system: str,
    prompt: str,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    label: str,
    timeout_sec: float,
) -> tuple[str, float]:
    """Single Gemma chat/completions JSON call. Returns (raw_content, latency_sec)."""
    import httpx

    user_prompt = _gemma_user_content(prompt)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    timeout = httpx.Timeout(timeout_sec)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra else None
            raise GemmaRateLimitError(retry_after)
        resp.raise_for_status()
        data = resp.json()
    latency = time.perf_counter() - t0
    content = (data.get("choices") or [{}])[0].get("message", {}).get(
        "content", ""
    ) or ""
    usage = data.get("usage") or {}
    trace(
        f"BENCH {label} ✓ http | latency={latency:.1f}s "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )
    return str(content), latency


async def run_window_config(
    *,
    text: str,
    source: str,
    title: str,
    target_tokens: int,
    max_chunks: int | None,
    overlap_tokens: int,
    model: str,
    api_base: str,
    api_key: str,
    max_out: int,
    timeout_sec: float,
    sleep_sec: float,
    dry_run: bool,
) -> RunSummary:
    label = f"{target_tokens // 1000}k tokens"
    chunks = split_into_token_windows(
        text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
    )
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]

    calls: list[ChunkCallResult] = []
    trace(
        f"BENCH ▶ {label} | chunks={len(chunks)} "
        f"targets≈{target_tokens} model={model} dry_run={dry_run}"
    )

    for i, chunk in enumerate(chunks):
        in_tok = _estimate_tokens(chunk)
        prompt = build_map_user_prompt(
            chunk, title=title, source=source, window_index=i
        )
        # Include system in reported in_tokens (window body + prompt overhead).
        in_tok_full = _estimate_tokens(f"{_MAP_SYSTEM}\n{prompt}")
        if dry_run:
            calls.append(
                ChunkCallResult(
                    run_label=label,
                    window_target=target_tokens,
                    chunk_index=i,
                    chunks_total=len(chunks),
                    in_tokens=in_tok_full,
                    latency_sec=0.0,
                    success=True,
                    atoms=0,
                    error="dry-run",
                )
            )
            print(
                f"  [{label}] chunk {i + 1}/{len(chunks)} "
                f"body≈{in_tok} full≈{in_tok_full} (dry-run)"
            )
            continue

        err: str | None = None
        val_errs: list[dict[str, Any]] = []
        atoms = 0
        success = False
        latency = 0.0
        raw_preview = ""
        try:
            raw, latency = await gemma_map_once(
                system=_MAP_SYSTEM,
                prompt=prompt,
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=max_out,
                label=f"map/{label}/c{i}",
                timeout_sec=timeout_sec,
            )
            raw_preview = (raw or "")[:240].replace("\n", " ")
            parsed, err, val_errs = parse_map_response(raw)
            if parsed is not None:
                success = True
                atoms = len(parsed.knowledge_atoms or [])
                err = None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            latency = latency or 0.0

        call = ChunkCallResult(
            run_label=label,
            window_target=target_tokens,
            chunk_index=i,
            chunks_total=len(chunks),
            in_tokens=in_tok_full,
            latency_sec=latency,
            success=success,
            atoms=atoms,
            error=err,
            validation_errors=val_errs,
            raw_preview=raw_preview,
        )
        calls.append(call)
        status = "✓" if success else "✗"
        print(
            f"  [{label}] chunk {i + 1}/{len(chunks)} {status} "
            f"in_tokens={in_tok_full} latency_sec={latency:.2f} "
            f"atoms={atoms}" + (f" | {err}" if err else "")
        )
        if val_errs:
            # Compact ValidationError dump — where Gemma breaks the schema.
            compact = [
                {
                    "loc": e.get("loc"),
                    "type": e.get("type"),
                    "msg": e.get("msg"),
                }
                for e in val_errs[:12]
            ]
            print(f"      validation: {json.dumps(compact, ensure_ascii=False)}")
        if sleep_sec > 0 and i + 1 < len(chunks):
            await asyncio.sleep(sleep_sec)

    ok = [c for c in calls if c.success and c.error != "dry-run"]
    # For dry-run, treat planned chunks as "success" for rate display = 100%.
    if dry_run:
        ok = list(calls)
    n = len(calls) or 1
    total_time = sum(c.latency_sec for c in calls)
    total_atoms = sum(c.atoms for c in calls)
    total_in = sum(c.in_tokens for c in calls) or 1
    return RunSummary(
        window_target=target_tokens,
        label=label,
        chunks=len(calls),
        total_time_sec=total_time,
        avg_latency_sec=(total_time / n) if calls else 0.0,
        success_rate_pct=(100.0 * len(ok) / n) if calls else 0.0,
        total_atoms=total_atoms,
        avg_atoms_per_1k=(1000.0 * total_atoms / total_in) if calls else 0.0,
        calls=calls,
    )


def print_summary_table(runs: list[RunSummary]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Gemma MAP chunk-size benchmark")
        table.add_column("Window Size", justify="left")
        table.add_column("Chunks Count", justify="right")
        table.add_column("Total Time (s)", justify="right")
        table.add_column("Avg Latency / Chunk", justify="right")
        table.add_column("Success Rate (%)", justify="right")
        table.add_column("Total Atoms Extracted", justify="right")
        table.add_column("Avg Atoms / 1k Tokens", justify="right")
        for r in runs:
            table.add_row(
                r.label,
                str(r.chunks),
                f"{r.total_time_sec:.1f}",
                f"{r.avg_latency_sec:.2f}",
                f"{r.success_rate_pct:.1f}%",
                str(r.total_atoms),
                f"{r.avg_atoms_per_1k:.2f}",
            )
        Console().print(table)
        return
    except Exception:
        pass

    # Markdown fallback
    print()
    print(
        "| Window Size | Chunks Count | Total Time (s) | Avg Latency / Chunk | "
        "Success Rate (%) | Total Atoms Extracted | Avg Atoms / 1k Tokens |"
    )
    print(
        "|-------------|--------------|----------------|---------------------|"
        "------------------|-----------------------|-----------------------|"
    )
    for r in runs:
        print(
            f"| {r.label:<11} | {r.chunks:<12} | {r.total_time_sec:<14.1f} | "
            f"{r.avg_latency_sec:<19.2f} | {r.success_rate_pct:<16.1f}% | "
            f"{r.total_atoms:<21} | {r.avg_atoms_per_1k:<21.2f} |"
        )


def _parse_sizes(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip().lower().replace("k", "000")
        if not part:
            continue
        out.append(int(part))
    return out or list(_DEFAULT_WINDOW_SIZES)


async def _amain(args: argparse.Namespace) -> int:
    text, source = _load_text(file=args.file, url=args.url, pdf=args.pdf)
    title = (
        args.title or Path(source).stem if not source.startswith("http") else source
    )[:200]
    total_tok = _estimate_tokens(text)
    print(f"source={source}")
    print(f"chars={len(text)} est_tokens≈{total_tok}")
    print(f"MAP system prompt chars={len(_MAP_SYSTEM)}")

    api_key = (args.api_key or GEMMA_API_KEY or "").strip()
    api_base = (args.api_base or GEMMA_API_BASE or "").strip()
    model = (args.model or GEMMA_PRIMARY_MODEL or "").strip()
    if not args.dry_run:
        if not api_key:
            raise SystemExit("GEMMA_API_KEY / GEMINI_API_KEY empty (or pass --api-key)")
        if not api_base:
            raise SystemExit("GEMMA_API_BASE empty")
        if not model:
            raise SystemExit("GEMMA_PRIMARY_MODEL empty")

    sizes = _parse_sizes(args.sizes)
    runs: list[RunSummary] = []
    for target in sizes:
        summary = await run_window_config(
            text=text,
            source=source,
            title=str(title),
            target_tokens=target,
            max_chunks=args.max_chunks,
            overlap_tokens=args.overlap,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_out=args.max_out or GEMMA_MAP_MAX_OUTPUT_TOKENS,
            timeout_sec=args.timeout,
            sleep_sec=args.sleep,
            dry_run=bool(args.dry_run),
        )
        runs.append(summary)
        # Pause between configurations to ease TPM.
        if not args.dry_run and args.config_sleep > 0:
            await asyncio.sleep(args.config_sleep)

    print_summary_table(runs)

    # Attention notes for the operator
    print()
    print("Watch:")
    print("  • Success Rate (%) — first window size that drops below 95%")
    print("  • Avg Atoms / 1k Tokens — density collapse on 8k–10k (lazy MAP)")
    print("  • Avg Latency / Chunk — linear vs super-linear growth")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        payload = {
            "source": source,
            "title": title,
            "est_tokens": total_tok,
            "model": model,
            "dry_run": bool(args.dry_run),
            "runs": [
                {
                    "window_target": r.window_target,
                    "label": r.label,
                    "chunks": r.chunks,
                    "total_time_sec": r.total_time_sec,
                    "avg_latency_sec": r.avg_latency_sec,
                    "success_rate_pct": r.success_rate_pct,
                    "total_atoms": r.total_atoms,
                    "avg_atoms_per_1k": r.avg_atoms_per_1k,
                    "calls": [
                        {
                            "chunk_index": c.chunk_index,
                            "in_tokens": c.in_tokens,
                            "latency_sec": c.latency_sec,
                            "success": c.success,
                            "atoms": c.atoms,
                            "error": c.error,
                            "validation_errors": c.validation_errors,
                            "raw_preview": c.raw_preview,
                        }
                        for c in r.calls
                    ],
                }
                for r in runs
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Benchmark Gemma MAP schema success vs chunk size"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", help="Path to article .txt / .md / .pdf")
    src.add_argument("--url", help="HTTP(S) URL to text or PDF")
    src.add_argument("--pdf", help="Local PDF path")
    p.add_argument("--title", default="", help="ARTICLE_TITLE override")
    p.add_argument(
        "--sizes",
        default="2000,4000,8000,10000",
        help="Comma-separated window token targets (default 2k,4k,8k,10k)",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap chunks per window-size run (cost control)",
    )
    p.add_argument(
        "--overlap", type=int, default=0, help="Overlap tokens between chunks"
    )
    p.add_argument("--model", default="", help="Override GEMMA_PRIMARY_MODEL")
    p.add_argument("--api-base", default="", help="Override GEMMA_API_BASE")
    p.add_argument("--api-key", default="", help="Override API key")
    p.add_argument(
        "--max-out",
        type=int,
        default=0,
        help="max_tokens for completion (default GEMMA_MAP_MAX_OUTPUT_TOKENS)",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout seconds")
    p.add_argument(
        "--sleep",
        type=float,
        default=1.25,
        help="Sleep between chunks inside one config (rate limit)",
    )
    p.add_argument(
        "--config-sleep",
        type=float,
        default=5.0,
        help="Sleep between window-size configurations",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Only split + estimate tokens"
    )
    p.add_argument("--json-out", default="", help="Write full results JSON")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
