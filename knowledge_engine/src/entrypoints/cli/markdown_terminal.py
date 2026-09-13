"""Markdown в терминале: таблицы с горизонтальными линиями между строками."""

from __future__ import annotations

import re
from typing import List, Tuple

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_TABLE_LINE = re.compile(r"^\s*\|")
_SEP_CELL = re.compile(r"^:?-{3,}:?$")


def _split_segments(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    segments: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if _TABLE_LINE.match(lines[i]):
            start = i
            while i < len(lines) and _TABLE_LINE.match(lines[i]):
                i += 1
            segments.append(("table", "\n".join(lines[start:i])))
            continue
        start = i
        while i < len(lines) and not _TABLE_LINE.match(lines[i]):
            i += 1
        if start < i:
            segments.append(("text", "\n".join(lines[start:i])))
    return segments


def _parse_md_table(block: str) -> Tuple[List[str], List[List[str]]]:
    header: List[str] = []
    body: List[List[str]] = []
    for line in block.strip().splitlines():
        if not _TABLE_LINE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if all(_SEP_CELL.match(c.replace(" ", "")) for c in cells):
            continue
        if not header:
            header = cells
        else:
            body.append(cells)
    return header, body


def _rich_table_from_markdown(block: str) -> Table:
    header, body = _parse_md_table(block)
    table = Table(
        show_header=bool(header),
        header_style="bold cyan",
        show_lines=True,
        expand=True,
        pad_edge=True,
        padding=(0, 1),
    )
    if not header:
        return table
    for col in header:
        table.add_column(col, overflow="fold")
    for row in body:
        padded = list(row)
        while len(padded) < len(header):
            padded.append("")
        table.add_row(*padded[: len(header)])
    return table


def markdown_to_renderables(text: str) -> List[object]:
    """Текст + Rich Table (с show_lines) вместо плоского Markdown-таблицы."""
    out: List[object] = []
    for kind, block in _split_segments(text):
        if kind == "table":
            header, body = _parse_md_table(block)
            if header or body:
                out.append(_rich_table_from_markdown(block))
            continue
        stripped = block.strip()
        if stripped:
            out.append(Markdown(block))
    return out


def print_markdown_document(
    console: Console,
    text: str,
    *,
    panel_title: str | None = None,
    border_style: str = "green",
) -> None:
    renderables = markdown_to_renderables(text)
    if not renderables:
        return
    group = Group(*renderables)
    if panel_title:
        console.print(
            Panel(group, title=panel_title, border_style=border_style, padding=(1, 2))
        )
    else:
        console.print(group)


def unravel_panel(text: str, title: str) -> Panel:
    """Panel для unravel: таблицы failure modes с линиями между строками."""
    renderables = markdown_to_renderables(text)
    if not renderables:
        return Panel(Text("—"), title=title, border_style="green")
    return Panel(
        Group(*renderables),
        title=title,
        border_style="green",
        padding=(1, 2),
    )
