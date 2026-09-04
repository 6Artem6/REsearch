#!/usr/bin/env python3
"""Инспекция паспорта, атомов знаний и чанков из LanceDB.

Примеры:
  ./.venv/bin/python knowledge_engine/scripts/inspect_knowledge.py
  ./.venv/bin/python knowledge_engine/scripts/inspect_knowledge.py \\
    --doc-id ec72e70c03c9e40df44ed7b9
  ./.venv/bin/python knowledge_engine/scripts/inspect_knowledge.py \\
    --title \"Fault Tolerance\"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import lancedb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from knowledge_engine.config import LANCE_DB_PATH
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_CONTEXT_QUOTE,
)
from knowledge_engine.db.knowledge_atoms_schema import COL_DOC_ID as KA_DOC_ID
from knowledge_engine.db.knowledge_atoms_schema import (
    COL_SCOPE,
    COL_SOURCE_CHUNK_IDS,
    COL_STATEMENT,
    KNOWLEDGE_ATOMS_TABLE,
)
from knowledge_engine.db.rag_chunks_schema import (
    COL_CHUNK_ID,
    COL_CHUNK_INDEX,
    COL_CHUNK_TEXT,
    COL_DOC_ID,
    COL_TITLE,
    COL_URL,
    COL_WINDOW_SUMMARY,
    RAG_CHUNKS_TABLE,
)
from knowledge_engine.services.vector_store import (
    TABLE_NAME as DOCUMENT_SUMMARIES_TABLE,
)

console = Console()

SCOPE_COLORS = {
    "PRINCIPLE": "bold magenta",
    "MECHANIC": "bold cyan",
    "INSTANCE": "bold green",
}


def _sql_literal(value: str) -> str:
    return (value or "").replace("'", "''")


def _table_names(db: Any) -> list[str]:
    try:
        names = db.list_tables()
        if hasattr(names, "tables"):
            return list(names.tables)
        return list(names)
    except Exception:
        try:
            return list(db.table_names())
        except Exception:
            return []


def _rows_to_dicts(
    table: Any, *, where: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    """Read Lance rows without requiring pandas."""
    try:
        q = table.search()
        if where:
            q = q.where(where)
        q = q.limit(max(1, limit))
        try:
            return list(q.to_list())
        except Exception:
            pass
        arrow = q.to_arrow()
        return arrow.to_pylist()
    except Exception:
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            return []
        if not where:
            return rows[:limit]
        # Best-effort local filter for ``doc_id = '…'`` / title LIKE
        out: list[dict[str, Any]] = []
        for row in rows:
            if "doc_id =" in where:
                want = where.split("doc_id =", 1)[1].strip().strip("'")
                if str(row.get("doc_id") or "") == want:
                    out.append(row)
            elif "title LIKE" in where.lower():
                frag = where.split("%", 1)[1].rsplit("%", 1)[0].lower()
                if frag in str(row.get("title") or "").lower():
                    out.append(row)
            if len(out) >= limit:
                break
        return out


def get_db_connection(db_path: str | Path) -> Any:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        console.print(
            f"[bold red]LanceDB path does not exist:[/bold red] {path}\n"
            f"[dim]Hint: KE default is {LANCE_DB_PATH}[/dim]"
        )
        sys.exit(1)
    try:
        return lancedb.connect(str(path))
    except Exception as e:
        console.print(
            f"[bold red]Ошибка подключения к LanceDB по пути '{path}':[/bold red] {e}"
        )
        sys.exit(1)


def _pick_first_doc_id(db: Any) -> str | None:
    names = set(_table_names(db))
    if KNOWLEDGE_ATOMS_TABLE in names:
        rows = _rows_to_dicts(db.open_table(KNOWLEDGE_ATOMS_TABLE), limit=1)
        if rows:
            return str(rows[0].get(KA_DOC_ID) or "").strip() or None
    if RAG_CHUNKS_TABLE in names:
        rows = _rows_to_dicts(db.open_table(RAG_CHUNKS_TABLE), limit=1)
        if rows:
            return str(rows[0].get(COL_DOC_ID) or "").strip() or None
    return None


def _find_document_by_title(db: Any, title_search: str) -> dict[str, Any] | None:
    names = set(_table_names(db))
    frag = _sql_literal(title_search.strip())
    if not frag:
        return None
    if DOCUMENT_SUMMARIES_TABLE in names:
        rows = _rows_to_dicts(
            db.open_table(DOCUMENT_SUMMARIES_TABLE),
            where=f"title LIKE '%{frag}%'",
            limit=5,
        )
        if rows:
            return rows[0]
    if RAG_CHUNKS_TABLE in names:
        rows = _rows_to_dicts(
            db.open_table(RAG_CHUNKS_TABLE),
            where=f"title LIKE '%{frag}%'",
            limit=5,
        )
        if rows:
            return rows[0]
    return None


def _passport_for_doc(
    db: Any, doc_id: str, url_hint: str = ""
) -> dict[str, Any] | None:
    names = set(_table_names(db))
    if DOCUMENT_SUMMARIES_TABLE not in names:
        return None
    table = db.open_table(DOCUMENT_SUMMARIES_TABLE)
    rows = _rows_to_dicts(table, limit=5000)
    want_url = (url_hint or "").strip().rstrip("/").lower()
    # Prefer URL match when we know it from rag_chunks; else last row matching doc_id hash is N/A
    # document_summaries has url, not doc_id — match via VectorStore.doc_id_for_url.
    from knowledge_engine.services.vector_store import VectorStore

    found: dict[str, Any] | None = None
    for row in rows:
        url = str(row.get("url") or "").strip()
        if want_url and url.rstrip("/").lower() == want_url:
            found = row
            continue
        if VectorStore.doc_id_for_url(url) == doc_id:
            found = row
    return found


def inspect_document(
    db_path: str | Path,
    doc_id: str | None = None,
    title_search: str | None = None,
) -> None:
    db = get_db_connection(db_path)
    names = _table_names(db)
    console.print(
        f"[dim]LanceDB: {Path(db_path).resolve()} | tables={names or '∅'}[/dim]"
    )

    if not names:
        console.print(
            "[bold red]LanceDB пуста (нет таблиц).[/bold red]\n"
            f"[dim]Нужный путь обычно: {LANCE_DB_PATH}[/dim]\n"
            "[dim]Не путать с ./lancedb_data — это другой каталог.[/dim]"
        )
        return

    doc_id_found = (doc_id or "").strip() or None
    document_row: dict[str, Any] | None = None
    url_hint = ""

    if not doc_id_found and title_search:
        document_row = _find_document_by_title(db, title_search)
        if document_row:
            url_hint = str(document_row.get(COL_URL) or document_row.get("url") or "")
            from knowledge_engine.services.vector_store import VectorStore

            if document_row.get(COL_DOC_ID):
                doc_id_found = str(document_row.get(COL_DOC_ID)).strip()
            elif url_hint.startswith("http"):
                doc_id_found = VectorStore.doc_id_for_url(url_hint)
            console.print(
                f"[dim]Найден по title={title_search!r} → doc_id={doc_id_found}[/dim]"
            )
        else:
            console.print(
                f"[yellow]Документ с title LIKE %{title_search}% не найден.[/yellow]"
            )

    if not doc_id_found:
        doc_id_found = _pick_first_doc_id(db)
        if doc_id_found:
            console.print(
                f"[dim]doc_id не указан. Автоматически выбран: "
                f"[bold]{doc_id_found}[/bold][/dim]\n"
            )
        else:
            console.print(
                "[bold red]Нечего инспектировать: нет knowledge_atoms / rag_chunks "
                "с doc_id.[/bold red]"
            )
            if KNOWLEDGE_ATOMS_TABLE not in names:
                console.print(
                    f"[yellow]Таблица '{KNOWLEDGE_ATOMS_TABLE}' отсутствует в этой БД.[/yellow]\n"
                    f"[dim]Hint: откройте {LANCE_DB_PATH} "
                    f"(python …/inspect_knowledge.py без --db или "
                    f"--db {LANCE_DB_PATH})[/dim]"
                )
            return

    # Resolve URL from rag_chunks for passport lookup
    if RAG_CHUNKS_TABLE in names and not url_hint:
        chunk_rows = _rows_to_dicts(
            db.open_table(RAG_CHUNKS_TABLE),
            where=f"{COL_DOC_ID} = '{_sql_literal(doc_id_found)}'",
            limit=1,
        )
        if chunk_rows:
            url_hint = str(chunk_rows[0].get(COL_URL) or "").strip()

    if document_row is None:
        document_row = _passport_for_doc(db, doc_id_found, url_hint)

    console.print(
        Panel(
            f"[bold yellow]ИНСПЕКЦИЯ ДОКУМЕНТА:[/bold yellow] "
            f"[white]{doc_id_found}[/white]"
            + (f"\n[dim]{url_hint}[/dim]" if url_hint else ""),
            expand=False,
        )
    )

    # --- Passport (document_summaries) ---
    if document_row:
        passport_text = Text()
        passport_text.append("Title:\n", style="bold underline blue")
        passport_text.append(f"{document_row.get('title') or 'Н/Д'}\n\n")
        passport_text.append("URL:\n", style="bold underline blue")
        passport_text.append(f"{document_row.get('url') or url_hint or 'Н/Д'}\n\n")

        exec_sum = str(document_row.get("executive_summary") or "").strip()
        passport_text.append(
            "Executive Summary (passport):\n", style="bold underline green"
        )
        if exec_sum:
            passport_text.append(f"{exec_sum}\n\n")
        else:
            passport_text.append("(пусто)\n\n")

        takeaways = document_row.get("key_takeaways", [])
        if isinstance(takeaways, str):
            try:
                takeaways = json.loads(takeaways)
            except Exception:
                takeaways = [takeaways]
        passport_text.append(
            "Key Takeaways (passport):\n", style="bold underline green"
        )
        if takeaways:
            for item in takeaways:
                passport_text.append(f" • {item}\n")
        else:
            passport_text.append(" • (пусто)\n")

        console.print(
            Panel(passport_text, title="1. Паспорт Документа", border_style="blue")
        )
    else:
        console.print(
            "[yellow]1. Паспорт: строка в document_summaries не найдена "
            f"(doc_id={doc_id_found}).[/yellow]"
        )

    # --- Knowledge atoms ---
    if KNOWLEDGE_ATOMS_TABLE not in names:
        console.print(
            f"[bold red]Таблица '{KNOWLEDGE_ATOMS_TABLE}' отсутствует в этой БД.[/bold red]\n"
            f"[dim]Доступные таблицы: {names}[/dim]\n"
            f"[dim]Правильный путь: {LANCE_DB_PATH}[/dim]"
        )
    else:
        try:
            atoms = _rows_to_dicts(
                db.open_table(KNOWLEDGE_ATOMS_TABLE),
                where=f"{KA_DOC_ID} = '{_sql_literal(doc_id_found)}'",
                limit=200,
            )
            if not atoms:
                console.print(
                    f"[bold red]Атомы знаний для doc_id='{doc_id_found}' не "
                    "найдены.[/bold red]"
                )
                takeaways = (document_row or {}).get("key_takeaways") or []
                exec_sum = str((document_row or {}).get("executive_summary") or "")
                if takeaways or exec_sum.strip():
                    console.print(
                        "[dim]Паспорт есть, таблица knowledge_atoms пуста: "
                        "ingest записал REDUCE в document_summaries, но не "
                        "вызвал upsert_knowledge_atoms. Нужен повторный "
                        "ingest / backfill.[/dim]"
                    )
            else:
                table = Table(
                    title=f"2. Очищенные Атомы Знаний (Всего: {len(atoms)})",
                    show_lines=True,
                )
                table.add_column("Scope", style="bold", width=12)
                table.add_column("Утверждение (Statement / Fact)", style="white")
                table.add_column("Цитата (Context Quote)", style="dim italic", width=35)
                table.add_column(
                    "Связи (source_chunk_ids)", style="bold cyan", width=20
                )

                for row in atoms:
                    scope = str(row.get(COL_SCOPE, "UNKNOWN")).upper()
                    scope_color = SCOPE_COLORS.get(scope, "white")
                    chunk_ids = row.get(COL_SOURCE_CHUNK_IDS, [])
                    if isinstance(chunk_ids, str):
                        try:
                            chunk_ids = json.loads(chunk_ids)
                        except Exception:
                            chunk_ids = [chunk_ids]
                    if chunk_ids is None:
                        chunk_ids = []
                    chunks_str = ", ".join(str(c) for c in chunk_ids) or "—"
                    quote = row.get(COL_CONTEXT_QUOTE) or "—"
                    table.add_row(
                        f"[{scope_color}]{scope}[/{scope_color}]",
                        str(row.get(COL_STATEMENT) or ""),
                        str(quote),
                        chunks_str,
                    )
                console.print(table)
        except Exception as e:
            console.print(
                f"[bold red]Ошибка при чтении {KNOWLEDGE_ATOMS_TABLE}:[/bold red] {e}"
            )

    # --- rag_chunks + window_summary ---
    if RAG_CHUNKS_TABLE not in names:
        console.print(f"[yellow]Таблица '{RAG_CHUNKS_TABLE}' отсутствует.[/yellow]")
        return
    try:
        chunks = _rows_to_dicts(
            db.open_table(RAG_CHUNKS_TABLE),
            where=f"{COL_DOC_ID} = '{_sql_literal(doc_id_found)}'",
            limit=200,
        )
        chunks.sort(key=lambda r: int(r.get(COL_CHUNK_INDEX) or 0))
        if not chunks:
            console.print(
                f"[yellow]rag_chunks для doc_id='{doc_id_found}' пусты.[/yellow]"
            )
            return
        console.print(
            f"\n[bold yellow]3. Выжимки окон и сырые чанки "
            f"(Всего: {len(chunks)})[/bold yellow]"
        )
        for row in chunks:
            c_id = row.get(COL_CHUNK_ID) or row.get(COL_CHUNK_INDEX)
            win_raw = row.get(COL_WINDOW_SUMMARY)
            if win_raw is None or not str(win_raw).strip():
                win_summary = "[red]Window Summary отсутствует[/red]"
            else:
                win_summary = str(win_raw)
            raw_text = str(row.get(COL_CHUNK_TEXT) or "")[:200].replace("\n", " ")
            chunk_panel = Text()
            chunk_panel.append("Window Summary (Gemma):\n", style="bold green")
            chunk_panel.append(f"{win_summary}\n\n")
            chunk_panel.append("Сырой текст (Превью 200 символов):\n", style="dim")
            chunk_panel.append(f"{raw_text}...")
            title = str(row.get(COL_TITLE) or "").strip()
            console.print(
                Panel(
                    chunk_panel,
                    title=f"Chunk ID: {c_id}" + (f" | {title[:40]}" if title else ""),
                    border_style="dim white",
                )
            )
    except Exception as e:
        console.print(f"[yellow]Предупреждение при чтении rag_chunks:[/yellow] {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-инспектор базы знаний LanceDB")
    parser.add_argument(
        "--db",
        default=str(LANCE_DB_PATH),
        help=f"Путь к папке LanceDB (по умолчанию: {LANCE_DB_PATH})",
    )
    parser.add_argument("--doc-id", help="Точный doc_id документа для инспекции")
    parser.add_argument("--title", help="Поиск документа по фрагменту заголовка")
    args = parser.parse_args(argv)
    inspect_document(db_path=args.db, doc_id=args.doc_id, title_search=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
