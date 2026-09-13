"""Из TRACE: по каждой модели и каждой функции — INCOMING PROMPT + RAW OUTPUT (≤N примеров)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_STEP_BLOCK = re.compile(
    r">>> \[STEP (\d+): (.+?)\]\s*\n"
    r"--- INCOMING PROMPT ---\s*\n"
    r"(.*?)"
    r"\n--- RAW OUTPUT ---\s*\n"
    r"(.*?)"
    r"(?=\n={50}\n|\Z)",
    re.S,
)


def _model_key(step_title: str) -> str:
    t = step_title.strip()
    if t.startswith("Consensus"):
        return "consensus-playwright"
    parts = [p.strip() for p in t.split("/")]
    if parts and (parts[-1] == "ollama" or parts[-1].startswith("gemini-")):
        return parts[-1]
    return parts[-1] if parts else t


def _function_key(step_title: str) -> str:
    t = step_title.strip()
    if t.startswith("Consensus"):
        return "consensus/playwright_ui"
    parts = [p.strip() for p in t.split("/")]
    if parts and (parts[-1] == "ollama" or parts[-1].startswith("gemini-")):
        parts = parts[:-1]
    if parts and parts[0].startswith("summarizer"):
        return "summarizer/DocumentSummary"
    if any("chunker" in p for p in parts):
        return "v07 lite/chunker/ChunkExtractionResult"
    return "/".join(parts)


def extract_by_model_and_function(
    text: str,
    max_per_function: int = 2,
) -> tuple[str, list[tuple[str, str, int]]]:
    """Возвращает текст и список (model, function, count_included)."""
    seen: dict[tuple[str, str], int] = {}
    included: list[tuple[int, str, str, str, str]] = []

    for m in _STEP_BLOCK.finditer(text):
        step_n = int(m.group(1))
        title = m.group(2).strip()
        prompt = m.group(3).strip()
        raw = m.group(4).strip()
        model = _model_key(title)
        func = _function_key(title)
        key = (model, func)
        n = seen.get(key, 0)
        if n >= max_per_function:
            continue
        seen[key] = n + 1
        included.append((step_n, title, model, func, prompt, raw))

    # Сводка уникальных функций в логе (все STEP)
    all_funcs: dict[tuple[str, str], int] = {}
    for m in re.finditer(r">>> \[STEP (\d+): (.+?)\]", text):
        title = m.group(2).strip()
        k = (_model_key(title), _function_key(title))
        all_funcs[k] = all_funcs.get(k, 0) + 1

    lines: list[str] = [
        "LLM trace: INCOMING PROMPT + RAW OUTPUT по модели и функции.",
        f"В логе вызовов: {sum(all_funcs.values())} STEP | уникальных (модель×функция): {len(all_funcs)}",
        f"В этом файле: ≤{max_per_function} пример(ов) на каждую уникальную функцию.",
        "",
        "INDEX (модель | функция | вызовов в полном логе | примеров здесь):",
    ]
    summary: list[tuple[str, str, int]] = []
    for model, func in sorted(all_funcs.keys()):
        cnt = seen.get((model, func), 0)
        lines.append(
            f"  - {model} | {func} | full_log={all_funcs[(model, func)]} | sample={cnt}"
        )
        summary.append((model, func, cnt))
    lines.append("")

    # Группировка по модели для чтения
    by_model: dict[str, list[tuple[int, str, str, str, str]]] = {}
    for step_n, title, model, func, prompt, raw in included:
        by_model.setdefault(model, []).append((step_n, title, func, prompt, raw))

    for model in sorted(by_model.keys()):
        lines.append("=" * 72)
        lines.append(f"MODEL: {model}")
        lines.append("=" * 72)
        for step_n, title, func, prompt, raw in by_model[model]:
            lines.append("")
            lines.append(f"### STEP {step_n} | function: {func}")
            lines.append(f"### label: {title}")
            lines.append("")
            lines.append("--- INCOMING PROMPT ---")
            lines.append(prompt)
            lines.append("")
            lines.append("--- RAW OUTPUT ---")
            lines.append(raw)
            lines.append("")

    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract prompt+response samples per model and function"
    )
    parser.add_argument("log_file", type=Path, nargs="?", default=None)
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument(
        "--max-per-function",
        type=int,
        default=2,
        help="Макс. примеров на уникальную функцию (summarizer/chunker — несколько статей)",
    )
    args = parser.parse_args()
    default = Path("knowledge_engine/.runs/20260730-curriculum-search-first-TRACE.log")
    src = args.log_file or default
    if not src.is_file():
        print(f"Missing: {src}", file=__import__("sys").stderr)
        return 1
    text = src.read_text(encoding="utf-8")
    out_text, _ = extract_by_model_and_function(
        text, max_per_function=args.max_per_function
    )
    out = args.out or src.with_name(src.stem + "-BY-MODEL-FUNCTION.txt")
    out.write_text(out_text, encoding="utf-8")
    steps = out_text.count("### STEP ")
    print(f"written: {out} ({len(out_text)} chars, {steps} sample blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
