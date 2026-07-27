"""LanceDB: DocumentSummary (legacy) + KnowledgeNode graph (v0.3)."""

from __future__ import annotations

import json
import uuid
from typing import Callable, List, Optional, TypeVar

import lancedb
from langchain_ollama import OllamaEmbeddings

from knowledge_engine.config import EMBED_MODEL, LANCE_DB_PATH, OLLAMA_BASE_URL
from knowledge_engine.schemas import DocumentSummary, KnowledgeNode
from knowledge_engine.services.lance_db_maintenance import (
    is_lance_format_error,
    reset_lance_directory,
)
from knowledge_engine.ui.run_log import trace

TABLE_NAME = "document_summaries"
NODES_TABLE = "knowledge_nodes"

T = TypeVar("T")


def _summary_document(summary: DocumentSummary) -> str:
    parts = [
        summary.title,
        summary.url,
        " ".join(summary.cs_concepts),
        " ".join(summary.key_takeaways),
        " ".join(summary.failure_modes),
        " ".join(summary.diagram_descriptions),
    ]
    return "\n".join(parts)


class VectorStore:
    def __init__(self) -> None:
        LANCE_DB_PATH.mkdir(parents=True, exist_ok=True)
        self._embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self._db = lancedb.connect(str(LANCE_DB_PATH))
        self._verify_lance_readable()

    def _connect(self) -> None:
        self._db = lancedb.connect(str(LANCE_DB_PATH))

    def _verify_lance_readable(self) -> None:
        for name in self._db.table_names():
            try:
                self._db.open_table(name).count_rows()
            except Exception as exc:
                if is_lance_format_error(exc):
                    reset_lance_directory(f"verify table {name}: {exc}")
                    self._connect()
                    return
                raise

    def _with_lance_recovery(self, fn: Callable[..., T], *args, **kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_lance_format_error(exc):
                raise
            reset_lance_directory(str(exc))
            self._connect()
            return fn(*args, **kwargs)

    def _table(self):
        if TABLE_NAME not in self._db.table_names():
            raise RuntimeError("LanceDB table empty — сначала save_summary")
        return self._db.open_table(TABLE_NAME)

    def save_summary(self, summary: DocumentSummary) -> None:
        document = _summary_document(summary)
        trace(f"EMBED ▶ {EMBED_MODEL} | LanceDB save {summary.url[:60]}")
        from knowledge_engine.ui.logger import set_phase, set_status

        set_phase(f"embed {EMBED_MODEL}")
        set_status(f"[LanceDB] embed → save {summary.title[:50]}…")
        vector = self._embeddings.embed_query(document)
        row = {
            "title": summary.title,
            "url": summary.url,
            "cs_concepts": json.dumps(summary.cs_concepts, ensure_ascii=False),
            "key_takeaways": json.dumps(summary.key_takeaways, ensure_ascii=False),
            "failure_modes": json.dumps(summary.failure_modes, ensure_ascii=False),
            "diagram_descriptions": json.dumps(
                summary.diagram_descriptions, ensure_ascii=False
            ),
            "document": document,
            "vector": vector,
        }
        if TABLE_NAME not in self._db.table_names():
            self._db.create_table(TABLE_NAME, data=[row])
            try:
                self._db.open_table(TABLE_NAME).create_fts_index("document")
            except Exception:
                pass
        else:
            self._table().add([row])

    def hybrid_search(self, query: str, limit: int = 3) -> List[DocumentSummary]:
        if TABLE_NAME not in self._db.table_names():
            return []

        table = self._table()
        if table.count_rows() == 0:
            return []

        query_vector = self._embeddings.embed_query(query)
        try:
            results = (
                table.search(query, query_type="hybrid")
                .vector(query_vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            results = table.search(query_vector).limit(limit).to_list()

        summaries: list[DocumentSummary] = []
        for row in results:
            summaries.append(
                DocumentSummary(
                    title=row.get("title") or "",
                    url=row.get("url") or "",
                    cs_concepts=json.loads(row.get("cs_concepts") or "[]"),
                    key_takeaways=json.loads(row.get("key_takeaways") or "[]"),
                    failure_modes=json.loads(row.get("failure_modes") or "[]"),
                    diagram_descriptions=json.loads(
                        row.get("diagram_descriptions") or "[]"
                    ),
                )
            )
        return summaries

    def _nodes_table(self):
        if NODES_TABLE not in self._db.table_names():
            return None
        return self._db.open_table(NODES_TABLE)

    def save_knowledge_node(
        self,
        level: str,
        content: str,
        parent_id: Optional[str] = None,
        source_url: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or str(uuid.uuid4())
        document = "\n".join(filter(None, [level, content, source_url or ""]))
        trace(f"KNODE save {level} | {nid[:8]}…")
        vector = self._embeddings.embed_query(document)
        row = {
            "id": nid,
            "level": level,
            "parent_id": parent_id or "",
            "content": content,
            "source_url": source_url or "",
            "document": document,
            "vector": vector,
        }
        if NODES_TABLE not in self._db.table_names():
            self._db.create_table(NODES_TABLE, data=[row])
            try:
                self._db.open_table(NODES_TABLE).create_fts_index("document")
            except Exception:
                pass
        else:
            self._with_lance_recovery(lambda: self._nodes_table().add([row]))
        return nid

    def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        table = self._nodes_table()
        if table is None or table.count_rows() == 0:
            return None
        try:
            rows = table.search().where(f"id = '{node_id}'").limit(1).to_list()
        except Exception:
            rows = []
            for row in table.to_arrow().to_pylist():
                if row.get("id") == node_id:
                    rows.append(row)
                    break
        if not rows:
            return None
        r = rows[0]
        return KnowledgeNode(
            id=r["id"],
            level=r.get("level") or "",
            parent_id=(r.get("parent_id") or None) or None,
            content=r.get("content") or "",
            source_url=(r.get("source_url") or None) or None,
        )

    def get_hierarchical_context(self, node_id: str) -> str:
        """L2 → L1 → L0 для передачи в stateless Gemini."""
        node = self.get_knowledge_node(node_id)
        if node is None:
            return ""

        lines: list[str] = []
        current: Optional[KnowledgeNode] = node
        chain: list[KnowledgeNode] = []
        while current is not None:
            chain.append(current)
            if not current.parent_id:
                break
            current = self.get_knowledge_node(current.parent_id)

        for n in reversed(chain):
            prefix = n.level
            url = f" ({n.source_url})" if n.source_url else ""
            lines.append(f"[{prefix}]{url}\n{n.content}")

        return "\n\n".join(lines)

    def hybrid_search_nodes(self, query: str, limit: int = 5) -> List[KnowledgeNode]:
        table = self._nodes_table()
        if table is None or table.count_rows() == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        try:
            results = (
                table.search(query, query_type="hybrid")
                .vector(query_vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            results = table.search(query_vector).limit(limit).to_list()

        nodes: list[KnowledgeNode] = []
        for row in results:
            nodes.append(
                KnowledgeNode(
                    id=row.get("id") or "",
                    level=row.get("level") or "",
                    parent_id=(row.get("parent_id") or None) or None,
                    content=row.get("content") or "",
                    source_url=(row.get("source_url") or None) or None,
                )
            )
        return nodes
