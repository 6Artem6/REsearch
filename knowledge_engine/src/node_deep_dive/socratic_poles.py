"""Socratic Poles (Repulsion / Attraction) for asterisk-question deep_analysis."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from knowledge_engine.config import EMBED_MODEL, LANCE_DB_PATH
from knowledge_engine.db.embed_model_guard import (
    drop_if_embed_space_mismatch,
    row_matches_embed_model,
)
from knowledge_engine.db.socratic_poles_schema import (
    COL_CLAIM,
    COL_CONCEPT_ID,
    COL_CURRICULUM_ID,
    COL_EMBED_MODEL,
    COL_FACT_LINE,
    COL_ID,
    COL_NODE_ID,
    COL_POLARITY,
    COL_UPDATED_AT,
    COL_VECTOR,
    SOCRATIC_POLES_TABLE,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
from knowledge_engine.ui.run_log import trace

Polarity = Literal["repulsion", "attraction"]

_MAX_LOCAL_EACH = 8
_MAX_CROSS_EACH = 4

# Static system rules (cache-friendly — invariant across asterisk-question turns).
SOCRATIC_POLES_STATIC_RULES = (
    "=== SOCRATIC POLES (STATIC RULES — HOST FACTS) ===\n"
    "User payload may include [SOCRATIC_POLES_STATE] with templated lines:\n"
    "  FACT_REPULSION: … — already mastered; DO NOT re-explain as intro.\n"
    "  FACT_ATTRACTION: … — gaps/bottlenecks; TARGET analysis around these.\n"
    "technical_explanation MUST orbit FACT_ATTRACTION (Problem → Edge cases → "
    "Trade-off), using THIS turn's [R*] when present, and MUST skirt "
    "FACT_REPULSION (no surface re-teach).\n"
    "Section headings are DYNAMIC from FACT_ATTRACTION — do not reuse a fixed "
    "## 1–5 skeleton every turn.\n"
    "follow_up_question MUST be Bloom Evaluate/Create (synthesis / design / "
    "trade-off) drawn from contradictions or open edges highlighted in "
    "technical_explanation — not a rehash of FACT_REPULSION.\n"
)


def _layers_claim(row: Any) -> str:
    parts: list[str] = []
    if bool(getattr(row, "why_passed", False)):
        parts.append("WHY")
    if bool(getattr(row, "how_passed", False)):
        parts.append("HOW")
    if bool(getattr(row, "mechanic_passed", False)):
        parts.append("MECH")
    label = (getattr(row, "label", "") or "").strip() or (getattr(row, "id", "") or "")
    if parts:
        return f"{label} layers={'+'.join(parts)}"
    return label or "verified"


def format_fact_repulsion(
    *,
    node: str,
    concept_id: str,
    claim: str,
) -> str:
    return (
        f'FACT_REPULSION: [Node: {node}] Concept "{concept_id}" is VERIFIED. '
        f"Focus: {(claim or '').strip()[:220]}"
    )


def format_fact_attraction(
    *,
    node: str,
    concept_id: str,
    focus_hint: str,
) -> str:
    return (
        f'FACT_ATTRACTION: [Node: {node}] Concept "{concept_id}" has '
        f"GAP/BOTTLENECK. Focus: {(focus_hint or '').strip()[:220]}"
    )


def _local_repulsion_facts(
    memory: SessionMemory,
    *,
    node_id: str,
    proven_skills: list[str] | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    nid = (node_id or "").strip() or "node"
    for sc in memory.sub_concepts or []:
        layers_ok = bool(sc.why_passed or sc.how_passed or sc.mechanic_passed)
        if sc.status != "verified" and not layers_ok:
            continue
        claim = (sc.evidence or "").strip() or _layers_claim(sc)
        cid = (sc.id or "").strip() or "concept"
        line = format_fact_repulsion(node=nid, concept_id=cid, claim=claim)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "repulsion",
                "claim": claim[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH:
            break
    # agreed_concepts from fact_manifest
    try:
        agreed = list(getattr(memory.fact_manifest, "agreed_concepts", None) or [])
    except Exception:
        agreed = []
    for raw in agreed:
        claim = str(raw or "").strip()
        if not claim:
            continue
        cid = f"agreed:{hashlib.sha256(claim.encode()).hexdigest()[:8]}"
        line = format_fact_repulsion(node=nid, concept_id=cid, claim=claim)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "repulsion",
                "claim": claim[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH:
            break
    for skill in proven_skills or []:
        claim = str(skill or "").strip()
        if not claim:
            continue
        cid = f"proven:{hashlib.sha256(claim.encode()).hexdigest()[:8]}"
        line = format_fact_repulsion(node=nid, concept_id=cid, claim=claim)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "repulsion",
                "claim": claim[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH + 2:
            break
    return out[: _MAX_LOCAL_EACH + 2]


def _local_attraction_facts(
    memory: SessionMemory,
    *,
    node_id: str,
    blind_spots: list[str] | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    nid = (node_id or "").strip() or "node"
    for sc in memory.sub_concepts or []:
        if sc.status not in ("gap", "partial"):
            # Still include unchecked with focus_hint
            if not (sc.focus_hint or "").strip():
                continue
        hint = (sc.focus_hint or "").strip() or (sc.evidence or "").strip()
        if not hint and sc.status not in ("gap", "partial"):
            continue
        if not hint:
            hint = f"status={sc.status}"
        cid = (sc.id or "").strip() or "concept"
        line = format_fact_attraction(node=nid, concept_id=cid, focus_hint=hint)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "attraction",
                "claim": hint[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH:
            break
    try:
        bottlenecks = list(
            getattr(memory.fact_manifest, "open_bottlenecks", None) or []
        )
    except Exception:
        bottlenecks = []
    for raw in bottlenecks:
        hint = str(raw or "").strip()
        if not hint:
            continue
        cid = f"bottleneck:{hashlib.sha256(hint.encode()).hexdigest()[:8]}"
        line = format_fact_attraction(node=nid, concept_id=cid, focus_hint=hint)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "attraction",
                "claim": hint[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH:
            break
    for spot in blind_spots or []:
        hint = str(spot or "").strip()
        if not hint:
            continue
        cid = f"blind:{hashlib.sha256(hint.encode()).hexdigest()[:8]}"
        line = format_fact_attraction(node=nid, concept_id=cid, focus_hint=hint)
        out.append(
            {
                "node_id": nid,
                "concept_id": cid,
                "polarity": "attraction",
                "claim": hint[:220],
                "fact_line": line,
            }
        )
        if len(out) >= _MAX_LOCAL_EACH + 2:
            break
    return out[: _MAX_LOCAL_EACH + 2]


def format_socratic_poles_state_block(
    repulsion: list[dict[str, str]],
    attraction: list[dict[str, str]],
) -> str:
    """Dynamic bottom-of-payload poles block (templated FACT_* lines)."""
    rep_lines = [
        (r.get("fact_line") or "").strip()
        for r in repulsion
        if (r.get("fact_line") or "").strip()
    ]
    att_lines = [
        (a.get("fact_line") or "").strip()
        for a in attraction
        if (a.get("fact_line") or "").strip()
    ]
    if not rep_lines and not att_lines:
        return (
            "[SOCRATIC_POLES_STATE]\n"
            "=== POLARITY: REPULSION (DO NOT RE-EXPLAIN) ===\n"
            "(none yet)\n\n"
            "=== POLARITY: ATTRACTION (TARGET FOR ANALYSIS) ===\n"
            "(none yet)\n"
        )
    rep_body = "\n".join(f"- {x}" for x in rep_lines) if rep_lines else "(none yet)"
    att_body = "\n".join(f"- {x}" for x in att_lines) if att_lines else "(none yet)"
    return (
        "[SOCRATIC_POLES_STATE]\n"
        "=== POLARITY: REPULSION (DO NOT RE-EXPLAIN) ===\n"
        f"{rep_body}\n\n"
        "=== POLARITY: ATTRACTION (TARGET FOR ANALYSIS) ===\n"
        f"{att_body}\n"
    )


def _load_competency_lists(
    curriculum_id: str,
    node: Any | None,
) -> tuple[list[str], list[str]]:
    if not (curriculum_id or "").strip():
        return [], []
    try:
        from knowledge_engine.src.node_deep_dive.user_mastery_profile import (
            filter_competency_for_node,
            get_curriculum_competency_profile,
        )

        profile = get_curriculum_competency_profile(curriculum_id)
        if node is None:
            return (
                list(profile.proven_skills or [])[:4],
                list(profile.blind_spots or [])[:3],
            )
        return filter_competency_for_node(profile, node, curriculum_id)
    except Exception as exc:
        trace(f"SOCRATIC_POLES competency skip | {exc}")
        return [], []


def build_socratic_poles_payload(
    memory: SessionMemory,
    node_id: str,
    *,
    curriculum_id: str = "",
    node: Any | None = None,
    topic_query: str = "",
    include_cross_node: bool = True,
) -> dict[str, Any]:
    """
    Host-built poles payload: templated FACT_* lines + optional cross-node RAG.
    """
    nid = (node_id or "").strip() or "node"
    proven, blind = _load_competency_lists(curriculum_id, node)
    repulsion = _local_repulsion_facts(memory, node_id=nid, proven_skills=proven)
    attraction = _local_attraction_facts(memory, node_id=nid, blind_spots=blind)
    cross_rep: list[dict[str, str]] = []
    cross_att: list[dict[str, str]] = []
    if include_cross_node and (curriculum_id or "").strip():
        q = (topic_query or "").strip() or nid
        try:
            cross_rep, cross_att = search_cross_node_poles(
                curriculum_id,
                q,
                exclude_node_id=nid,
            )
        except Exception as exc:
            trace(f"SOCRATIC_POLES cross-node search skip | {exc}")
    # Prefer local first; append cross-node without duplicate fact_line.
    seen = {r["fact_line"] for r in repulsion}
    for row in cross_rep:
        fl = row.get("fact_line") or ""
        if fl and fl not in seen:
            repulsion.append(row)
            seen.add(fl)
        if len(repulsion) >= _MAX_LOCAL_EACH + _MAX_CROSS_EACH:
            break
    seen_a = {a["fact_line"] for a in attraction}
    for row in cross_att:
        fl = row.get("fact_line") or ""
        if fl and fl not in seen_a:
            attraction.append(row)
            seen_a.add(fl)
        if len(attraction) >= _MAX_LOCAL_EACH + _MAX_CROSS_EACH:
            break
    block = format_socratic_poles_state_block(repulsion, attraction)
    payload = {
        "node_id": nid,
        "repulsion": repulsion,
        "attraction": attraction,
        "block": block,
    }
    # Cache on memory for commit upsert / prompt reuse.
    try:
        memory.socratic_poles_snapshot = {
            "node_id": nid,
            "repulsion": repulsion[:16],
            "attraction": attraction[:16],
        }
    except Exception:
        pass
    return payload


def _embeddings():
    from knowledge_engine.services.search.bge_m3_embed import BgeM3Embeddings

    return BgeM3Embeddings()


def upsert_socratic_poles_snapshot(
    curriculum_id: str,
    node_id: str,
    repulsion: list[dict[str, str]],
    attraction: list[dict[str, str]],
) -> int:
    """
    Persist compact local poles snapshot into LanceDB (mental-map vector store).

    Uses templated FACT_* lines as the compressed snapshot (no extra LLM call).
    """
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    if not cid or not nid:
        return 0
    rows_src = list(repulsion or []) + list(attraction or [])
    if not rows_src:
        return 0
    try:
        import lancedb
        import numpy as np

        emb = _embeddings()
        db = lancedb.connect(str(LANCE_DB_PATH))
        drop_if_embed_space_mismatch(db, SOCRATIC_POLES_TABLE)
        now = datetime.now(timezone.utc).isoformat()
        # Replace prior rows for this node (stable mental-map slice).
        if SOCRATIC_POLES_TABLE in db.table_names():
            try:
                tbl = db.open_table(SOCRATIC_POLES_TABLE)
                tbl.delete(
                    f"{COL_CURRICULUM_ID} = '{cid.replace(chr(39), chr(39)*2)}' "
                    f"AND {COL_NODE_ID} = '{nid.replace(chr(39), chr(39)*2)}'"
                )
            except Exception as exc:
                trace(f"SOCRATIC_POLES delete skip | {exc}")
        rows: list[dict[str, Any]] = []
        for item in rows_src[:24]:
            claim = (item.get("claim") or item.get("fact_line") or "").strip()
            if len(claim) < 8:
                continue
            polarity = (item.get("polarity") or "").strip().lower()
            if polarity not in ("repulsion", "attraction"):
                continue
            vec = np.asarray(emb.embed_query(claim[:2000]), dtype=np.float64).tolist()
            rows.append(
                {
                    COL_ID: str(uuid.uuid4()),
                    COL_CURRICULUM_ID: cid,
                    COL_NODE_ID: nid,
                    COL_CONCEPT_ID: (item.get("concept_id") or "")[:64],
                    COL_POLARITY: polarity,
                    COL_CLAIM: claim[:400],
                    COL_FACT_LINE: (item.get("fact_line") or claim)[:500],
                    COL_VECTOR: vec,
                    COL_EMBED_MODEL: EMBED_MODEL,
                    COL_UPDATED_AT: now,
                }
            )
        if not rows:
            return 0
        if SOCRATIC_POLES_TABLE in db.table_names():
            db.open_table(SOCRATIC_POLES_TABLE).add(rows)
        else:
            db.create_table(SOCRATIC_POLES_TABLE, data=rows)
        trace(f"SOCRATIC_POLES upsert | curriculum={cid} node={nid} n={len(rows)}")
        return len(rows)
    except Exception as exc:
        trace(f"SOCRATIC_POLES upsert skip | {exc}")
        return 0


def search_cross_node_poles(
    curriculum_id: str,
    topic_query: str,
    *,
    exclude_node_id: str = "",
    limit_each: int = _MAX_CROSS_EACH,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Vector search over mental-map poles from other nodes."""
    cid = (curriculum_id or "").strip()
    q = (topic_query or "").strip()
    if not cid or not q:
        return [], []
    try:
        import lancedb
        import numpy as np

        db = lancedb.connect(str(LANCE_DB_PATH))
        if SOCRATIC_POLES_TABLE not in db.table_names():
            return [], []
        table = db.open_table(SOCRATIC_POLES_TABLE)
        if table.count_rows() == 0:
            return [], []
        emb = _embeddings()
        qv = np.asarray(emb.embed_query(q[:2000]), dtype=np.float64)
        qn = float(np.linalg.norm(qv))
        if qn > 0:
            qv = qv / qn
        where = f"{COL_CURRICULUM_ID} = '{cid.replace(chr(39), chr(39)*2)}'"
        excl = (exclude_node_id or "").strip()
        if excl:
            where += f" AND {COL_NODE_ID} != '{excl.replace(chr(39), chr(39)*2)}'"
        try:
            builder = table.search(qv.tolist()).where(where, prefilter=True)
        except TypeError:
            builder = table.search(qv.tolist()).where(where)
        except Exception:
            builder = table.search(qv.tolist())
        hits = builder.limit(max(12, limit_each * 4)).to_list()
        repulsion: list[dict[str, str]] = []
        attraction: list[dict[str, str]] = []
        for row in hits:
            if not row_matches_embed_model(row):
                continue
            if excl and str(row.get(COL_NODE_ID) or "") == excl:
                continue
            if str(row.get(COL_CURRICULUM_ID) or "") != cid:
                continue
            polarity = str(row.get(COL_POLARITY) or "").strip().lower()
            item = {
                "node_id": str(row.get(COL_NODE_ID) or ""),
                "concept_id": str(row.get(COL_CONCEPT_ID) or ""),
                "polarity": polarity,
                "claim": str(row.get(COL_CLAIM) or "")[:220],
                "fact_line": str(row.get(COL_FACT_LINE) or "")[:500],
            }
            if not item["fact_line"]:
                continue
            if polarity == "repulsion" and len(repulsion) < limit_each:
                repulsion.append(item)
            elif polarity == "attraction" and len(attraction) < limit_each:
                attraction.append(item)
        return repulsion, attraction
    except Exception as exc:
        trace(f"SOCRATIC_POLES search skip | {exc}")
        return [], []


def persist_socratic_poles_on_commit(
    memory: SessionMemory,
    *,
    curriculum_id: str,
    node_id: str,
) -> None:
    """Commit hook: upsert current local poles into mental-map vector store."""
    snap = getattr(memory, "socratic_poles_snapshot", None) or {}
    if not isinstance(snap, dict) or not snap:
        # Rebuild local-only if missing.
        payload = build_socratic_poles_payload(
            memory,
            node_id,
            curriculum_id=curriculum_id,
            include_cross_node=False,
        )
        snap = payload
    upsert_socratic_poles_snapshot(
        curriculum_id,
        node_id,
        list(snap.get("repulsion") or []),
        list(snap.get("attraction") or []),
    )
