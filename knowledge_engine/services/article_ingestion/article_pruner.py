"""Детерминированная и LLM-подрезка секций документа."""

from __future__ import annotations

import json
import re

import httpx

from knowledge_engine.config import (
    BLOG_SPATIAL_NUM_CTX,
    BLOG_SPATIAL_NUM_PREDICT,
    BLOG_SPATIAL_SUMMARIZER_MODEL,
    BLOG_SPATIAL_TIMEOUT_SEC,
    OLLAMA_BASE_URL,
    SELECTION_PROMPTS_KEEP_ALIVE,
)
from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.services.article_ingestion.annotated_article_ops import (
    _norm_p,
    p_index_map,
    sorted_p_ids,
)
from knowledge_engine.services.article_ingestion.triage_schemas import (
    DocumentStructureTree,
    TOCNode,
    TriageDecisionResponse,
)
from knowledge_engine.services.parsers.html_annotator import AnnotatedArticle
from knowledge_engine.ui.run_log import trace

_PRUNE_TITLE_RE = re.compile(
    r"^(references|bibliography|список литературы|appendi[x|ces]|приложени[ея]|"
    r"acknowledgements?|благодарности|index|предметный указатель|"
    r"legal disclaimer|disclaimer|copyright)\s*$",
    re.I,
)
_HEADING_CLEAN_RE = re.compile(r"^\d+([\.\)]\s+|\s+)")


class ArticleSectionPruner:
    def prune(
        self,
        structure: DocumentStructureTree,
        annotated: AnnotatedArticle,
    ) -> TriageDecisionResponse:
        order = sorted_p_ids(annotated.paragraph_map)
        if not order:
            return TriageDecisionResponse(
                keep_paragraph_ranges=[],
                pruned_sections_reason=[],
            )

        pruned_reasons: list[str] = []
        pruned_starts: list[tuple[str, TOCNode]] = []

        for node in structure.nodes:
            title = _HEADING_CLEAN_RE.sub("", (node.title or "").strip())
            if _PRUNE_TITLE_RE.match(title):
                pruned_starts.append((node.start_p_id, node))
                end = node.end_p_id or order[-1]
                pruned_reasons.append(f"Pruned {title} at [{node.start_p_id}..{end}]")

        if pruned_starts:
            first_tail_idx = min(
                p_index_map(annotated.paragraph_map)[_norm_p(s[0])]
                for s in pruned_starts
                if _norm_p(s[0]) in p_index_map(annotated.paragraph_map)
            )
            keep_ranges = []
            if first_tail_idx > 0:
                keep_ranges.append((order[0], order[first_tail_idx - 1]))
            decision = TriageDecisionResponse(
                keep_paragraph_ranges=keep_ranges,
                pruned_sections_reason=pruned_reasons,
            )
            trace(
                f"DOC_TRIAGE prune ✓ | regex | kept_ranges={len(keep_ranges)} "
                f"pruned={len(pruned_reasons)}"
            )
            return decision

        llm_decision = self._llm_triage(structure, annotated)
        if llm_decision is not None:
            trace(
                f"DOC_TRIAGE prune ✓ | llm | kept_ranges="
                f"{len(llm_decision.keep_paragraph_ranges)}"
            )
            return llm_decision

        decision = TriageDecisionResponse(
            keep_paragraph_ranges=[(order[0], order[-1])],
            pruned_sections_reason=["No back-matter match; kept full body"],
        )
        trace("DOC_TRIAGE prune ✓ | keep-all | no prune signals")
        return decision

    def _llm_triage(
        self,
        structure: DocumentStructureTree,
        annotated: AnnotatedArticle,
    ) -> TriageDecisionResponse | None:
        compact = [
            {
                "title": n.title,
                "level": n.level,
                "start": n.start_p_id,
                "end": n.end_p_id,
                "page": n.page_number,
            }
            for n in structure.nodes[:80]
        ]
        if not compact:
            return None
        prompt = (
            "Document TOC (compact). Return JSON TriageDecisionResponse.\n"
            "keep_paragraph_ranges: pairs [start_P_id, end_P_id] for MAIN technical content "
            "(intro, core, architecture, benchmarks, conclusion).\n"
            "Exclude bibliography, appendices, index, legal boilerplate.\n\n"
            f"{json.dumps(compact, ensure_ascii=False)}"
        )
        system = (
            f"{RUSSIAN_OUTPUT_RULE}\n"
            "Structural triage only. Use only P_ids from the TOC."
        )
        api = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": BLOG_SPATIAL_SUMMARIZER_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": TriageDecisionResponse.model_json_schema(),
            "keep_alive": SELECTION_PROMPTS_KEEP_ALIVE,
            "options": {
                "temperature": 0.05,
                "num_ctx": min(BLOG_SPATIAL_NUM_CTX, 8192),
                "num_predict": min(BLOG_SPATIAL_NUM_PREDICT, 2048),
            },
        }
        try:
            timeout = httpx.Timeout(BLOG_SPATIAL_TIMEOUT_SEC)
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(api, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            trace(f"DOC_TRIAGE llm ✗ | {exc}")
            return None
        raw = str(data.get("response") or "").strip()
        try:
            parsed = TriageDecisionResponse.model_validate_json(raw)
        except Exception:
            try:
                parsed = TriageDecisionResponse.model_validate(json.loads(raw))
            except Exception:
                trace("DOC_TRIAGE llm ✗ | invalid JSON")
                return None
        if not parsed.keep_paragraph_ranges:
            return None
        return parsed
