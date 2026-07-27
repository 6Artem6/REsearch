"""v0.4: Gemini multimodal L2 из отобранных секций / code / media."""

from __future__ import annotations

from typing import Any

import httpx

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas import (
    DocumentStructure,
    EngineGraphState,
    L2EvidenceExtraction,
    StructureFilterResult,
)
from knowledge_engine.services.gemini_stateless import (
    global_anchor_from_state,
    run_stateless_gemini,
    run_stateless_gemini_multimodal,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.logger import set_status
from knowledge_engine.ui.pipeline_phase import pipeline_phase
from knowledge_engine.ui.run_log import node_end, node_start, trace


def _pick_l1_parent(l1_ids: list[str], hint: str, store: VectorStore) -> str | None:
    if not l1_ids:
        return None
    hint_l = hint.lower()
    for lid in l1_ids:
        node = store.get_knowledge_node(lid)
        if node and hint_l and hint_l in node.content.lower():
            return lid
    return l1_ids[0]


def _fetch_image(url: str) -> tuple[bytes, str] | None:
    if url.endswith("#svg") or not url.startswith("http"):
        return None
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "image/png").split(";")[0]
            if "image" not in mime and "svg" not in mime:
                mime = "image/png"
            return resp.content[:4_000_000], mime
    except httpx.HTTPError as exc:
        trace(f"deep_extractor image skip {url[:60]} | {exc}")
        return None


def deep_extractor_node(state: EngineGraphState) -> dict[str, Any]:
    node_start("deep_extractor_node (Gemini multimodal)")
    pipeline_phase("Deep Multimodal Extractor (Gemini Vision/Code)")
    filt_raw = state.get("structure_filter") or {}
    filt = StructureFilterResult.model_validate(filt_raw)
    raw_doc = state.get("document_structure")
    url = state.get("current_page_url") or ""

    if not filt.is_relevant or not raw_doc:
        reason = filt.reject_reason or "not relevant"
        trace(f"deep_extractor skip | {reason}")
        node_end("deep_extractor_node", "skipped")
        return {}

    doc = DocumentStructure.model_validate(raw_doc)
    anchor = global_anchor_from_state(
        state.get("original_query") or state.get("user_problem") or "",
        state.get("constraints") or state.get("context_constraints") or "",
        state.get("l0_summary") or "",
    )

    section_parts: list[str] = []
    for key in filt.selected_section_keys:
        if key in doc.sections:
            section_parts.append(f"## {key}\n{doc.sections[key][:4000]}")
    if not section_parts and doc.abstract:
        section_parts.append(doc.abstract[:4000])

    code_parts: list[str] = []
    for idx in filt.selected_code_indices:
        if 0 <= idx < len(doc.code_artifacts):
            c = doc.code_artifacts[idx]
            code_parts.append(
                f"### {c.context} ({c.language})\n```\n{c.code[:6000]}\n```"
            )

    media_urls: list[str] = []
    for idx in filt.selected_media_indices:
        if 0 <= idx < len(doc.media_artifacts):
            media_urls.append(doc.media_artifacts[idx].url)

    image_parts: list[tuple[bytes, str]] = []
    for murl in media_urls[:4]:
        fetched = _fetch_image(murl)
        if fetched:
            image_parts.append(fetched)

    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Из отобранных секций, кода и диаграмм извлеки L2-факты, failure modes, метрики, "
        "архитектурные узлы и bottlenecks. JSON L2EvidenceExtraction."
    )
    user = (
        f"URL: {url or doc.source_url}\n\n"
        "СЕКЦИИ:\n" + "\n\n".join(section_parts) + "\n\n"
        "КОД:\n" + "\n\n".join(code_parts)
    )

    set_status("[deep_extractor] Gemini multimodal L2…")
    if image_parts:
        extraction = run_stateless_gemini_multimodal(
            system,
            user,
            anchor,
            image_parts,
            response_schema=L2EvidenceExtraction,
            label="deep_extractor / multimodal L2",
            rpm_pause=True,
        )
    else:
        extraction = run_stateless_gemini(
            system,
            user,
            anchor,
            response_schema=L2EvidenceExtraction,
            label="deep_extractor / L2",
            rpm_pause=True,
        )

    store = VectorStore()
    l1_ids = list(state.get("l1_node_ids") or [])
    parent_l1 = _pick_l1_parent(l1_ids, extraction.l1_title_hint, store)
    new_ids = list(state.get("knowledge_node_ids") or [])

    for item in extraction.evidences[:12]:
        chunk = (
            f"FACT: {item.fact}\nFAILURE: {item.failure_mode}\nMETRIC: {item.metric}"
        ).strip()
        nid = store.save_knowledge_node(
            "L2_EVIDENCE",
            chunk,
            parent_id=parent_l1,
            source_url=url or doc.source_url,
        )
        new_ids.append(nid)

    node_end(
        "deep_extractor_node", f"L2={len(extraction.evidences)} img={len(image_parts)}"
    )
    return {"knowledge_node_ids": new_ids}
