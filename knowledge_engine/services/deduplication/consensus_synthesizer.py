"""Cloud Consensus Aggregator — cache-friendly English system prefix + batch JSON."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from knowledge_engine.config import MAX_CONCURRENT_MAP_REQUESTS
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.models.consensus import (
    ConsensusBatchResponse,
    ConsensusNode,
    RawFact,
    _unique_anchors,
)
from knowledge_engine.ui.run_log import trace

_CONSENSUS_SYSTEM = (
    "You are the Consensus Aggregator for a knowledge-ingest REDUCE pass.\n"
    "Read the SPO facts in this batch. Merge paraphrases of the SAME claim; "
    "keep genuine contradictions as disputed.\n"
    "status: consensus (several sources agree), disputed (sources conflict), "
    "unique (single source / no corroboration).\n"
    "Every node object MUST include status. If unsure, use consensus. "
    "Never omit status, including for the last node or for unique claims.\n"
    "Anti-Bloat Anchor Policy: primary_anchors MUST list at most THREE most "
    "informative citation tags (A1, A2, A3). Copy the FULL supporting set into "
    "all_anchors. Never drop unknown tags silently.\n"
    "Return ConsensusBatchResponse JSON only: nodes[]. "
    "node_id, entity, summary_text, primary_anchors, all_anchors, status, "
    "disputed_details.\n"
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "summary_text is Russian. Citation tags stay ASCII (A1, A2)."
)


def consensus_aggregator_system_prompt() -> str:
    """Static cacheable system prefix (no per-batch dynamics)."""
    return _CONSENSUS_SYSTEM


@dataclass(frozen=True)
class ConsensusBatchLimits:
    """Declared request budgets. Enforcement uses the model tokenizer."""

    max_nodes: int
    max_input_tokens: int
    max_output_tokens: int


def max_nodes_per_consensus_batch() -> int:
    from knowledge_engine.config import MAX_CONSENSUS_NODES_PER_BATCH

    return max(1, int(MAX_CONSENSUS_NODES_PER_BATCH or 10))


def consensus_max_output_tokens() -> int:
    """Completion cap for consensus JSON — same value packing uses.

    AUDIT FIX: this used to hardcode `min(2048, GEMMA_REDUCE_MAX_OUTPUT_TOKENS)`
    — a silent, undocumented HALVING of the real configured REDUCE output
    ceiling (4096 by default). build_token_bounded_batches() derives its
    output_budget straight from this value (`output_cap * 0.75`), so the
    2048 clamp capped every consensus batch's PROJECTED output at 1536
    tokens — confirmed live: batches flushed at facts=6-9 / ~150-160 output
    tokens per node, i.e. the packer hit this output ceiling long before the
    input window (3072) or the node cap (10) ever bound, leaving 50-63% of
    the input budget unused per call and forcing 9 sequential micro-batches
    instead of a handful of dense ones. Respect the real configured ceiling
    like MAP already does for its own output cap."""
    from knowledge_engine.config import GEMMA_REDUCE_MAX_OUTPUT_TOKENS

    return max(1, int(GEMMA_REDUCE_MAX_OUTPUT_TOKENS))


def consensus_batch_limits() -> ConsensusBatchLimits:
    """SSOT for node cap, input tokens, and output tokens (config only)."""
    from knowledge_engine.config import MAX_CONSENSUS_BATCH_TOKENS

    return ConsensusBatchLimits(
        max_nodes=max_nodes_per_consensus_batch(),
        max_input_tokens=max(64, int(MAX_CONSENSUS_BATCH_TOKENS or 3072)),
        max_output_tokens=consensus_max_output_tokens(),
    )


def _micro_batches(facts: list[RawFact]) -> list[list[RawFact]]:
    """Re-pack through the same model-token + node packer as REDUCE."""
    if not facts:
        return []
    from knowledge_engine.services.deduplication.entity_consensus_engine import (
        build_token_bounded_batches,
    )

    return build_token_bounded_batches({"batch": facts})


def nodes_from_facts(facts: list[RawFact]) -> list[ConsensusNode]:
    from knowledge_engine.services.deduplication.entity_consensus_engine import (
        apply_anti_bloat_anchors,
    )

    nodes: list[ConsensusNode] = []
    for i, fact in enumerate(facts, start=1):
        anchors = fact.merged_anchors()
        status: str = "unique" if len(anchors) <= 1 else "consensus"
        nodes.append(
            ConsensusNode(
                node_id=f"n-{fact.fact_id or i}",
                entity=fact.subject,
                summary_text=fact.canonical_text,
                primary_anchors=apply_anti_bloat_anchors(anchors),
                all_anchors=anchors,
                status=status,  # type: ignore[arg-type]
                disputed_details=None,
            )
        )
    return nodes


def consensus_batch_user_prompt(facts: list[RawFact]) -> str:
    lines = ["<facts>", "Merge duplicates; classify each remaining cluster."]
    for fact in facts:
        anchors = ",".join(fact.merged_anchors()) or fact.anchor
        lines.append(
            f"- id={fact.fact_id} entity_type={fact.entity_type} "
            f"subject={fact.subject} spo={fact.canonical_text} anchors=[{anchors}]"
        )
    lines.append("</facts>")
    return "\n".join(lines)


async def synthesize_consensus_batches(
    batches: list[list[RawFact]],
    *,
    http_client: Any | None = None,
    gemma_rl: Any | None = None,
    allow_cloud: bool = True,
) -> list[ConsensusNode]:
    from knowledge_engine.services.deduplication.entity_consensus_engine import (
        apply_anti_bloat_anchors,
        consensus_batch_token_counts,
    )
    from knowledge_engine.services.llm.gemma_client import GemmaCloudClient

    system = consensus_aggregator_system_prompt()
    work: list[list[RawFact]] = []
    for batch in batches:
        if batch:
            work.extend(_micro_batches(batch))
    if not work:
        return []
    if not allow_cloud:
        return [node for batch in work for node in nodes_from_facts(batch)]

    output_cap = consensus_max_output_tokens()
    limits = consensus_batch_limits()
    sem = asyncio.Semaphore(max(1, MAX_CONCURRENT_MAP_REQUESTS))
    total = len(work)

    def _finalize(
        batch: list[RawFact],
        parsed: ConsensusBatchResponse | None,
    ) -> list[ConsensusNode]:
        if parsed is None or not (parsed.nodes or []):
            return nodes_from_facts(batch)
        nodes: list[ConsensusNode] = []
        for node in parsed.nodes:
            all_a = _unique_anchors(node.all_anchors or node.primary_anchors)
            nodes.append(
                node.model_copy(
                    update={
                        "all_anchors": all_a,
                        "primary_anchors": apply_anti_bloat_anchors(
                            node.primary_anchors or all_a
                        ),
                    }
                )
            )
        return nodes

    async def _run_one(batch_index: int, batch: list[RawFact]) -> list[ConsensusNode]:
        async with sem:
            prompt = consensus_batch_user_prompt(batch)
            input_tokens, projected_output_tokens = consensus_batch_token_counts(
                batch,
                system_prompt=system,
            )
            trace(
                f"CONSENSUS batch ▶ | {batch_index}/{total} "
                f"facts={len(batch)} input_tokens={input_tokens}/"
                f"{limits.max_input_tokens} "
                f"projected_out={projected_output_tokens}/{output_cap}"
            )
            parsed: ConsensusBatchResponse | None = None
            try:
                if gemma_rl is not None:
                    parsed = await gemma_rl.post_structured(
                        system,
                        prompt,
                        ConsensusBatchResponse,
                        label="consensus_batch",
                        client=http_client,
                        max_tokens=output_cap,
                        # Part of the REDUCE critical path (blocks REDUCE_DONE) —
                        # ignore the TPM/RPM headroom reserved away from MAP.
                        priority=True,
                    )
                else:
                    # AUDIT FIX: this used to call complete_structured() with
                    # no limiter at all — a real, callable hole past the
                    # shared TPM/RPM counter for any caller that omits
                    # gemma_rl (dead in the current pipeline, since
                    # map_reduce_jobs_pooled_async always builds one, but
                    # prompt.txt asked for 100% coverage regardless of
                    # caller). complete_structured()'s own `limiter` kwarg
                    # only reconciles AFTER the call — it never blocks
                    # admission — so this path must acquire against the SAME
                    # registry entry build_gemma_model_slots() uses for the
                    # primary model BEFORE sending the request, exactly like
                    # RateLimitedLLMClient._pick_slot() does for the gated
                    # branch above.
                    from knowledge_engine.config import GEMMA_PRIMARY_MODEL
                    from knowledge_engine.services.llm.rate_limiter import (
                        get_gemma_rate_limiter,
                    )

                    fallback_limiter = get_gemma_rate_limiter(
                        slot=f"gemma:{GEMMA_PRIMARY_MODEL}"
                    )
                    await fallback_limiter.acquire(
                        input_tokens + projected_output_tokens,
                        model=GEMMA_PRIMARY_MODEL,
                        priority=True,
                    )
                    parsed = await GemmaCloudClient().complete_structured(
                        system,
                        prompt,
                        ConsensusBatchResponse,
                        label="consensus_batch",
                        client=http_client,
                        limiter=fallback_limiter,
                        max_tokens=output_cap,
                    )
            except Exception as exc:
                trace(
                    f"CONSENSUS batch ✗ | {batch_index}/{total} | "
                    f"{type(exc).__name__}: {exc}"
                )
                parsed = None
            return _finalize(batch, parsed)

    results = await asyncio.gather(
        *[_run_one(index, batch) for index, batch in enumerate(work, start=1)]
    )
    return [node for nodes in results for node in nodes]
