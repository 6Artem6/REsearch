"""v0.8 contextual explainer (Gemini Lite)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from knowledge_engine.services.gemini_stateless import GeminiUnavailableError
from knowledge_engine.services.llm_markdown_service import llm_markdown_to_html
from knowledge_engine.services.v07_run_store import v07_run_store
from knowledge_engine.src.processors.explainer import (
    DEFAULT_EXPLAIN_QUESTION,
    run_contextual_explain,
)
from knowledge_engine.src.processors.selection_prompts import (
    suggest_selection_questions,
)
from knowledge_engine.ui.run_log import trace

router = APIRouter(prefix="/v08", tags=["v08-explainer"])


class ExplainRequest(BaseModel):
    run_id: str = Field(min_length=8, max_length=32)
    selected_text: str = Field(min_length=2, max_length=8000)
    user_question: str = Field(default="", max_length=2000)
    surrounding_paragraph: str = Field(default="", max_length=12000)


class ExplainSourceRefOut(BaseModel):
    title: str = ""
    url: str = ""
    source_id: str = ""


class ExplainResponse(BaseModel):
    explanation: str
    explanation_html: str = ""
    source_ref: ExplainSourceRefOut
    matched_chunk_id: str = ""
    default_question: str = DEFAULT_EXPLAIN_QUESTION


class SuggestQuestionsRequest(BaseModel):
    selected_text: str = Field(min_length=2, max_length=8000)
    paragraph_context: str = Field(default="", max_length=12000)
    topic: str = Field(default="", max_length=4000)


@router.post("/suggest-questions")
async def post_suggest_questions(body: SuggestQuestionsRequest) -> dict[str, Any]:
    trace("API ▶ POST /v08/suggest-questions")
    result = await suggest_selection_questions(
        body.selected_text,
        body.paragraph_context,
        body.topic,
    )
    return result.model_dump()


@router.post("/explain")
def post_contextual_explain(body: ExplainRequest) -> dict[str, Any]:
    trace(f"API ▶ POST /v08/explain run={body.run_id}")
    run_id = body.run_id.strip()
    run = v07_run_store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    question_text = (body.user_question or "").strip() or DEFAULT_EXPLAIN_QUESTION
    v07_run_store.append_question_log(
        run_id,
        {
            "type": "explain",
            "text": question_text,
            "snippet": body.selected_text.strip()[:240],
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        result = run_contextual_explain(
            run_id,
            body.selected_text,
            body.user_question,
            body.surrounding_paragraph,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        trace(f"API ✗ explain | {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    registry = (run.result or {}).get("source_registry") if run.result else []
    explanation_html = llm_markdown_to_html(result.explanation, registry)

    return ExplainResponse(
        explanation=result.explanation,
        explanation_html=explanation_html,
        source_ref=ExplainSourceRefOut(
            title=result.source_ref.title,
            url=result.source_ref.url,
            source_id=result.source_ref.source_id,
        ),
        matched_chunk_id=result.matched_chunk_id,
    ).model_dump()
