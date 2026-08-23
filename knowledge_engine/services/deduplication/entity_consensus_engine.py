"""Entity-guided grouping, local bge-m3/reranker merge, token-bounded batches."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from knowledge_engine.models.consensus import ConsensusNode, RawFact, _unique_anchors
from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType

GENERAL_GROUP_KEY = "general"

EmbedBatchFn = Callable[[Sequence[str]], list[list[float]]]
RerankPairFn = Callable[[str, list[str]], list[float]]


def claim_dedup_is_enabled(mode: str | None = None) -> bool:
    from knowledge_engine import config as ke_config

    raw = (mode if mode is not None else ke_config.CLAIM_DEDUP_MODE) or "none"
    return raw.strip().lower() in ("exact", "entity_consensus")


def entity_group_key(fact: RawFact) -> str:
    etype = (fact.entity_type or "general").strip().lower() or "general"
    if etype == "general":
        return GENERAL_GROUP_KEY
    subject = (fact.subject or "").strip().lower()
    return f"{etype}|{subject}"


def group_facts_by_entity(facts: list[RawFact]) -> dict[str, list[RawFact]]:
    grouped: dict[str, list[RawFact]] = {}
    for fact in facts:
        grouped.setdefault(entity_group_key(fact), []).append(fact)
    return grouped


def apply_anti_bloat_anchors(
    anchors: Sequence[str],
    *,
    limit: int | None = None,
) -> list[str]:
    from knowledge_engine import config as ke_config

    cap = int(limit) if limit is not None else int(ke_config.MAX_PRIMARY_ANCHORS or 3)
    cap = max(1, cap)
    return _unique_anchors(list(anchors))[:cap]


def _exact_merge_group(facts: list[RawFact]) -> list[RawFact]:
    buckets: dict[str, list[RawFact]] = {}
    order: list[str] = []
    for fact in facts:
        key = fact.canonical_text.strip().lower()
        if key not in buckets:
            order.append(key)
        buckets.setdefault(key, []).append(fact)
    merged: list[RawFact] = []
    for key in order:
        cluster = buckets[key]
        head = cluster[0]
        anchors = _unique_anchors([a for f in cluster for a in f.merged_anchors()])
        merged.append(
            head.model_copy(
                update={
                    "anchor": anchors[0] if anchors else head.anchor,
                    "all_anchors": anchors,
                }
            )
        )
    return merged


class LocalFactDeduplicator:
    """Level-2 micro-dedup inside one entity group (bge-m3 + reranker)."""

    def __init__(
        self,
        *,
        embed_fn: EmbedBatchFn | None = None,
        rerank_fn: RerankPairFn | None = None,
        cluster_threshold: float | None = None,
        rerank_threshold: float | None = None,
    ) -> None:
        from knowledge_engine import config as ke_config

        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._cluster_threshold = (
            float(cluster_threshold)
            if cluster_threshold is not None
            else float(ke_config.SPO_CLUSTER_THRESHOLD)
        )
        self._rerank_threshold = (
            float(rerank_threshold)
            if rerank_threshold is not None
            else float(ke_config.SPO_RERANKER_DUPLICATE_THRESHOLD)
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        from knowledge_engine.services.search.bge_m3_embed import embed_texts_bge_m3

        return embed_texts_bge_m3(texts)

    def _rerank(self, query: str, docs: list[str]) -> list[float]:
        if self._rerank_fn is not None:
            return self._rerank_fn(query, docs)
        from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs

        return score_relevance_pairs(query, docs)

    def deduplicate_entity_group(self, facts: list[RawFact]) -> list[RawFact]:
        if len(facts) <= 1:
            return list(facts)
        texts = [f.canonical_text for f in facts]
        try:
            vecs = self._embed(texts)
        except Exception:
            return list(facts)
        if len(vecs) != len(facts):
            return list(facts)

        n = len(facts)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        import numpy as np

        mat = np.asarray(vecs, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
        sims = mat @ mat.T
        for i in range(n):
            for j in range(i + 1, n):
                if float(sims[i, j]) <= self._cluster_threshold:
                    continue
                try:
                    scores = self._rerank(texts[i], [texts[j]])
                    score = float(scores[0]) if scores else 0.0
                except Exception:
                    continue
                if score > self._rerank_threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            clusters.setdefault(find(idx), []).append(idx)
        out: list[RawFact] = []
        for root in sorted(clusters):
            members = [facts[i] for i in clusters[root]]
            head = members[0]
            anchors = _unique_anchors([a for f in members for a in f.merged_anchors()])
            out.append(
                head.model_copy(
                    update={
                        "anchor": anchors[0] if anchors else head.anchor,
                        "all_anchors": anchors,
                    }
                )
            )
        return out


def _count_gemma_tokens(text: str) -> int:
    """Count using the cached local Gemma/Gemini tokenizer vocabulary."""
    from knowledge_engine.config import GEMMA_PRIMARY_MODEL
    from knowledge_engine.src.utils.fast_tokenizer import token_counter

    return max(1, token_counter.count_tokens(text or "", GEMMA_PRIMARY_MODEL))


def _projected_consensus_response(facts: list[RawFact]) -> str:
    """Conservative full-node JSON used only to reserve completion tokens."""
    nodes: list[dict[str, object]] = []
    for fact in facts:
        anchors = fact.merged_anchors()
        nodes.append(
            {
                "node_id": f"n-{fact.fact_id}",
                "entity": fact.subject,
                "summary_text": fact.canonical_text,
                "primary_anchors": anchors[:3],
                "all_anchors": anchors,
                "status": "unique",
                "disputed_details": None,
            }
        )
    return json.dumps({"nodes": nodes}, ensure_ascii=False, separators=(",", ":"))


def consensus_batch_token_counts(
    facts: list[RawFact],
    *,
    system_prompt: str | None = None,
) -> tuple[int, int]:
    """Model-tokenizer counts for the complete request and projected response.

    This is the only consensus token gate: Gemma/Gemini vocabulary via
    ``token_counter`` + ``GEMMA_PRIMARY_MODEL``. Packing must not use
    Qwen/heuristic estimates.
    """
    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        consensus_aggregator_system_prompt,
        consensus_batch_user_prompt,
    )
    from knowledge_engine.services.llm.gemma_client import _gemma_user_content

    system = (
        system_prompt
        if system_prompt is not None
        else consensus_aggregator_system_prompt()
    )
    prompt = consensus_batch_user_prompt(facts)
    input_tokens = _count_gemma_tokens(f"{system}\n{_gemma_user_content(prompt)}")
    output_tokens = _count_gemma_tokens(_projected_consensus_response(facts))
    return input_tokens, output_tokens


def build_token_bounded_batches(
    grouped_facts: dict[str, list[RawFact]],
    max_tokens: int | None = None,
    *,
    system_prompt: str | None = None,
    max_output_tokens: int | None = None,
    output_utilization: float = 0.75,
    max_nodes: int | None = None,
) -> list[list[RawFact]]:
    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        consensus_batch_limits,
    )

    limits = consensus_batch_limits()
    input_budget = max(
        64,
        int(max_tokens if max_tokens is not None else limits.max_input_tokens),
    )
    output_cap = int(
        max_output_tokens if max_output_tokens is not None else limits.max_output_tokens
    )
    output_budget = max(
        64,
        int(output_cap * min(0.95, max(0.25, float(output_utilization)))),
    )
    node_cap = max(
        1,
        int(max_nodes) if max_nodes is not None else limits.max_nodes,
    )
    batches: list[list[RawFact]] = []
    current: list[RawFact] = []

    def flush() -> None:
        nonlocal current
        if current:
            batches.append(current)
            current = []

    for _key, group in grouped_facts.items():
        for fact in group:
            if current and len(current) >= node_cap:
                flush()
            candidate = [*current, fact]
            input_tokens, output_tokens = consensus_batch_token_counts(
                candidate,
                system_prompt=system_prompt,
            )
            if current and (
                input_tokens > input_budget or output_tokens > output_budget
            ):
                flush()
            current.append(fact)
            if len(current) >= node_cap:
                flush()
    flush()
    return batches


def raw_facts_from_atoms(
    atoms: list[KnowledgeAtom],
    *,
    index_map: dict[str, dict[str, Any]] | None = None,
) -> list[RawFact]:
    chunk_to_anchor: dict[str, str] = {}
    for key, meta in (index_map or {}).items():
        if not isinstance(meta, dict):
            continue
        cid = str(meta.get("chunk_id") or "").strip()
        if cid:
            chunk_to_anchor[cid] = str(key)
    facts: list[RawFact] = []
    for i, atom in enumerate(atoms):
        statement = (atom.statement or "").strip() or "unknown claim"
        obj = (atom.context_quote or statement).strip() or statement
        anchor = ""
        for cid in atom.source_chunk_ids or []:
            if cid in chunk_to_anchor:
                anchor = chunk_to_anchor[cid]
                break
        if not anchor and atom.source_chunk_ids:
            anchor = str(atom.source_chunk_ids[0])
        if not anchor:
            anchor = f"A{i + 1}"
        facts.append(
            RawFact(
                fact_id=f"f{i + 1}",
                subject=statement[:200],
                entity_type="general",
                predicate="states",
                obj=obj[:800],
                anchor=anchor,
                all_anchors=[anchor],
            )
        )
    return facts


def consensus_nodes_to_atoms(
    nodes: list[ConsensusNode],
    *,
    index_map: dict[str, dict[str, Any]] | None = None,
) -> list[KnowledgeAtom]:
    atoms: list[KnowledgeAtom] = []
    for node in nodes:
        chunk_ids: list[str] = []
        for tag in node.all_anchors or node.primary_anchors:
            meta = (index_map or {}).get(tag)
            if isinstance(meta, dict) and meta.get("chunk_id"):
                chunk_ids.append(str(meta["chunk_id"]))
            else:
                chunk_ids.append(tag)
        scope = (
            ScopeType.PRINCIPLE
            if node.status == "consensus"
            else ScopeType.INSTANCE
        )
        statement = (node.summary_text or "").strip()
        if len(statement) < 8:
            statement = f"{statement} — {node.entity}".strip()
        atoms.append(
            KnowledgeAtom(
                scope=scope,
                statement=statement[:2000],
                source_chunk_ids=chunk_ids,
            )
        )
    return atoms


def collapse_facts_locally(
    facts: list[RawFact],
    *,
    mode: str | None = None,
    deduplicator: LocalFactDeduplicator | None = None,
) -> dict[str, list[RawFact]]:
    from knowledge_engine import config as ke_config

    resolved = (mode if mode is not None else ke_config.CLAIM_DEDUP_MODE) or "none"
    resolved = resolved.strip().lower()
    grouped = group_facts_by_entity(facts)
    if resolved not in ("exact", "entity_consensus"):
        return grouped
    if resolved == "exact":
        return {key: _exact_merge_group(grp) for key, grp in grouped.items()}
    engine = deduplicator or LocalFactDeduplicator()
    return {key: engine.deduplicate_entity_group(grp) for key, grp in grouped.items()}


async def apply_entity_consensus_to_atoms(
    atoms: list[KnowledgeAtom],
    *,
    index_map: dict[str, dict[str, Any]] | None = None,
    http_client: Any | None = None,
    gemma_rl: Any | None = None,
    deduplicator: LocalFactDeduplicator | None = None,
) -> tuple[list[KnowledgeAtom], list[ConsensusNode]] | None:
    """None when CLAIM_DEDUP_MODE skips consensus (legacy REDUCE)."""
    from knowledge_engine import config as ke_config
    from knowledge_engine.services.deduplication.consensus_synthesizer import (
        synthesize_consensus_batches,
    )

    if not claim_dedup_is_enabled():
        return None
    facts = raw_facts_from_atoms(atoms, index_map=index_map)
    if not facts:
        return None
    collapsed = collapse_facts_locally(facts, deduplicator=deduplicator)
    batches = build_token_bounded_batches(collapsed)
    nodes = await synthesize_consensus_batches(
        batches,
        http_client=http_client,
        gemma_rl=gemma_rl,
        allow_cloud=ke_config.CLAIM_DEDUP_MODE.strip().lower() == "entity_consensus",
    )
    clipped = [
        node.model_copy(
            update={
                "all_anchors": _unique_anchors(node.all_anchors or node.primary_anchors),
                "primary_anchors": apply_anti_bloat_anchors(
                    node.primary_anchors or node.all_anchors
                ),
            }
        )
        for node in nodes
    ]
    return consensus_nodes_to_atoms(clipped, index_map=index_map), clipped
