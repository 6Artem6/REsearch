"""Qdrant VectorStore adapter — async companion to vector_store.py (LanceDB).

Payload shape mirrors the existing `rag_chunks` contract from
`db/rag_chunks_schema.py` / `vector_store.py.search_rag_chunk_rows`: chunk_id,
doc_id, url, title, chunk_text, chunk_index, chunks_in_doc, source_type,
trust_score, embed_model. `qdrant_client` is imported lazily (optional dep,
not required for LanceDB-only deployments) so this module stays importable
without it installed.
"""

from __future__ import annotations

import uuid
from typing import Any

from knowledge_engine.config import QDRANT_API_KEY, QDRANT_URL
from knowledge_engine.ui.run_log import trace


def _is_missing_payload_index_error(exc: BaseException, *, field: str) -> bool:
    """True for Qdrant's 400 'Index required but not found for "<field>"' response.

    Matches loosely (no exact quoting) — UnexpectedResponse.__str__ embeds the raw
    HTTP body as a bytes repr, which double-escapes the JSON's internal quotes.
    """
    msg = str(exc).lower()
    return "index required" in msg and field.lower() in msg


def _is_closed_loop_error(exc: BaseException) -> bool:
    """True for httpx/AsyncQdrantClient errors from a client whose connections
    were bound to an asyncio event loop that has since closed (or a different
    running loop) — happens when a client outlives the `asyncio.run()` call it
    was created under and gets reused from a later, separate `asyncio.run()`."""
    msg = str(exc).lower()
    return "event loop is closed" in msg or "attached to a different loop" in msg


def _point_id(document: dict[str, Any]) -> str:
    """Deterministic Qdrant point id (must be UUID/int) from id, chunk_id, or url.

    `id` takes priority when present — e.g. knowledge_atoms rows share a url
    across many atoms of the same document, so chunk_id/url would collide.
    """
    key = str(
        document.get("id") or document.get("chunk_id") or document.get("url") or ""
    ).strip()
    if not key:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantVectorStore:
    """Async Qdrant adapter. Disabled (safe no-op fallbacks) when QDRANT_URL is unset."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
    ) -> None:
        self._url = (url or "").strip()
        self._api_key = (api_key or "").strip()
        self._client: Any | None = None
        self._client_loop: Any | None = None
        self.enabled = bool(self._url)
        if not self.enabled:
            trace(
                "QDRANT_VECTOR_STORE disabled ⚠ | QDRANT_URL not set — "
                "upsert/search/fetch will return safe fallbacks"
            )

    def _get_client(self) -> Any:
        import asyncio

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is not None and self._client_loop is not current_loop:
            # Stale client from a previous asyncio.run() — its connections were
            # bound to that (now closed) loop and can't be reused on this one.
            self._client = None

        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key or None,
            )
            self._client_loop = current_loop
        return self._client

    def _drop_client(self) -> None:
        self._client = None
        self._client_loop = None

    async def _run_resilient(
        self,
        *,
        collection_name: str,
        index_field: str | list[str] | None,
        make_call: Any,
    ) -> Any:
        """Run `make_call(client)`, self-healing two transient failure modes:
        a stale client bound to a closed/foreign event loop (drop + reconnect),
        and (if `index_field` given) a missing Qdrant payload index — for each
        field in `index_field`, create it once (a query filtering on two unindexed
        fields hits this error one field at a time). Anything else propagates."""
        reconnected = False
        fields = (
            [index_field] if isinstance(index_field, str) else list(index_field or [])
        )
        healed: set[str] = set()
        while True:
            client = self._get_client()
            try:
                return await make_call(client)
            except Exception as exc:
                if _is_closed_loop_error(exc) and not reconnected:
                    reconnected = True
                    trace(
                        "QDRANT_VECTOR_STORE stale client (closed event loop) — "
                        "reconnecting, retry"
                    )
                    self._drop_client()
                    continue
                matched_field = next(
                    (
                        f
                        for f in fields
                        if f not in healed
                        and _is_missing_payload_index_error(exc, field=f)
                    ),
                    None,
                )
                if matched_field:
                    healed.add(matched_field)
                    from qdrant_client import models

                    await client.create_payload_index(
                        collection_name=collection_name,
                        field_name=matched_field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    trace(
                        f"QDRANT_VECTOR_STORE | created missing '{matched_field}' "
                        f"keyword index on {collection_name}, retrying"
                    )
                    continue
                raise

    async def ensure_collection(self, collection_name: str, size: int) -> bool:
        """Create the collection (COSINE, `size`-dim) if it doesn't exist yet."""
        if not self.enabled:
            return False
        from qdrant_client import models

        async def _do(client: Any) -> bool:
            cols = await client.get_collections()
            names = {c.name for c in cols.collections}
            if collection_name in names:
                return True
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=size, distance=models.Distance.COSINE
                ),
            )
            trace(
                f"QDRANT_VECTOR_STORE collection created ✓ | {collection_name} "
                f"size={size}"
            )
            return True

        try:
            return await self._run_resilient(
                collection_name=collection_name, index_field=None, make_call=_do
            )
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE ensure_collection ✗ | {collection_name} | "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    async def upsert_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> bool:
        """Upsert chunk payloads + vectors. Payload keys mirror rag_chunks_schema.py."""
        if not self.enabled:
            return False
        if not documents or not vectors or len(documents) != len(vectors):
            trace(
                f"QDRANT_VECTOR_STORE upsert skip ⚠ | documents={len(documents)} "
                f"vectors={len(vectors)} (length mismatch or empty)"
            )
            return False
        from qdrant_client import models

        points = [
            models.PointStruct(id=_point_id(doc), vector=vec, payload=doc)
            for doc, vec in zip(documents, vectors)
        ]

        async def _do(client: Any) -> None:
            await client.upsert(collection_name=collection_name, points=points)

        try:
            await self._run_resilient(
                collection_name=collection_name, index_field=None, make_call=_do
            )
            trace(
                f"QDRANT_VECTOR_STORE upsert ✓ | collection={collection_name} "
                f"points={len(points)}"
            )
            return True
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE upsert ✗ | collection={collection_name} | "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        doc_id_filter: list[str] | None = None,
        exclude_ids_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search; returns structured chunk payload + `_score`/`_id`.

        `doc_id_filter` restricts results to `payload.doc_id` in that list
        (Qdrant-side filtered ANN — the curriculum whitelist scoping path).
        `exclude_ids_filter` drops points whose `payload.id` is in that list.
        """
        if not self.enabled or not query_vector:
            return []
        from qdrant_client import models

        must: list[Any] = []
        must_not: list[Any] = []
        index_fields: list[str] = []
        allow_ids = [d for d in (doc_id_filter or []) if (d or "").strip()]
        if allow_ids:
            must.append(
                models.FieldCondition(
                    key="doc_id", match=models.MatchAny(any=allow_ids)
                )
            )
            index_fields.append("doc_id")
        excl_ids = [d for d in (exclude_ids_filter or []) if (d or "").strip()]
        if excl_ids:
            must_not.append(
                models.FieldCondition(key="id", match=models.MatchAny(any=excl_ids))
            )
            index_fields.append("id")
        query_filter = (
            models.Filter(must=must, must_not=must_not) if (must or must_not) else None
        )

        async def _do(client: Any) -> Any:
            return await client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        try:
            response = await self._run_resilient(
                collection_name=collection_name,
                index_field=index_fields or None,
                make_call=_do,
            )
            points = getattr(response, "points", response) or []
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE search ✗ | collection={collection_name} | "
                f"{type(exc).__name__}: {exc}"
            )
            return []

        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            payload["_score"] = float(getattr(point, "score", 0.0) or 0.0)
            payload["_id"] = getattr(point, "id", None)
            results.append(payload)
        return results

    async def delete_by_field(
        self,
        collection_name: str,
        field: str,
        value: str,
    ) -> bool:
        """Delete all points where `payload[field] == value` (e.g. replace a doc's old chunks)."""
        if not self.enabled or not (value or "").strip():
            return False
        from qdrant_client import models

        points_filter = models.Filter(
            must=[
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            ]
        )

        async def _do(client: Any) -> None:
            await client.delete(
                collection_name=collection_name, points_selector=points_filter
            )

        try:
            await self._run_resilient(
                collection_name=collection_name, index_field=field, make_call=_do
            )
            return True
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE delete_by_field ✗ | collection={collection_name} | "
                f"{field}={value[:60]} | {type(exc).__name__}: {exc}"
            )
            return False

    async def count_by_field(self, collection_name: str, field: str, value: str) -> int:
        """Exact count of points where `payload[field] == value`."""
        if not self.enabled or not (value or "").strip():
            return 0
        from qdrant_client import models

        count_filter = models.Filter(
            must=[
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            ]
        )

        async def _do(client: Any) -> Any:
            return await client.count(
                collection_name=collection_name, count_filter=count_filter, exact=True
            )

        try:
            result = await self._run_resilient(
                collection_name=collection_name, index_field=field, make_call=_do
            )
            return int(getattr(result, "count", 0) or 0)
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE count_by_field ✗ | collection={collection_name} | "
                f"{field}={value[:60]} | {type(exc).__name__}: {exc}"
            )
            return 0

    async def distinct_field_values(
        self, collection_name: str, field: str, *, page_size: int = 256
    ) -> set[str]:
        """Scroll every point and collect the distinct values of `payload[field]`.

        No dedicated Qdrant "distinct" primitive assumed available — scrolls the
        full collection with a minimal payload projection. Fine at small/medium
        scale (knowledge_atoms-sized collections); not meant for huge ones.
        """
        if not self.enabled:
            return set()

        values: set[str] = set()
        offset: Any = None

        async def _do(client: Any) -> Any:
            return await client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_payload=[field],
                with_vectors=False,
            )

        try:
            while True:
                points, next_offset = await self._run_resilient(
                    collection_name=collection_name, index_field=None, make_call=_do
                )
                for point in points:
                    v = (getattr(point, "payload", None) or {}).get(field)
                    if v:
                        values.add(str(v))
                if not next_offset or not points:
                    break
                offset = next_offset
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE distinct_field_values ✗ | collection={collection_name} | "
                f"field={field} | {type(exc).__name__}: {exc}"
            )
        return values

    async def fetch_by_url(
        self,
        collection_name: str,
        url: str,
    ) -> dict[str, Any] | None:
        """Read-through point lookup by `payload.url` — no embedding/vector spend."""
        target = (url or "").strip()
        if not self.enabled or not target:
            return None
        from qdrant_client import models

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="url", match=models.MatchValue(value=target))
            ]
        )

        async def _do(client: Any) -> Any:
            return await client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

        try:
            points, _next_offset = await self._run_resilient(
                collection_name=collection_name, index_field="url", make_call=_do
            )
        except Exception as exc:
            trace(
                f"QDRANT_VECTOR_STORE fetch_by_url ✗ | collection={collection_name} | "
                f"{type(exc).__name__}: {exc}"
            )
            return None

        if not points:
            return None
        payload = dict(getattr(points[0], "payload", None) or {})
        payload["_id"] = getattr(points[0], "id", None)
        return payload
