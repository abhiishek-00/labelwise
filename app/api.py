"""FastAPI application: query, feedback, comparison, and health."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from app import config, db, rag, retrieval
from app.models import (
    CompareRequest,
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger("labelwise.api")

app = FastAPI(
    title="LabelWise API",
    description=(
        "Grounded question answering over Open Food Facts product data. "
        "Answers are generated only from retrieved product records."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Service health: database state and configured backends.

    Never raises - a degraded dependency is reported in the payload so the
    container healthcheck can distinguish "process up" from "fully ready".
    """
    database = db.health()
    return {
        "status": "ok" if database.get("connected") else "degraded",
        "database": database,
        "llm": {
            "backend": config.LLM_BACKEND,
            "egress_policy": config.LLM_EGRESS_POLICY,
        },
        "embeddings": {
            "backend": config.EMBED_BACKEND,
            "dim": config.EMBED_DIM,
        },
        "retrieval": {
            "method": config.RETRIEVAL_METHOD,
            "alpha": config.HYBRID_ALPHA,
            "top_k": config.RETRIEVAL_TOP_K,
            "rerank": config.ENABLE_RERANK,
            "rewrite": config.ENABLE_QUERY_REWRITE,
        },
        "generation": {
            "prompt_strategy": config.PROMPT_STRATEGY,
            "online_judge": config.ENABLE_ONLINE_JUDGE,
        },
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question from the product knowledge base.

    Every call is logged for monitoring. A logging failure is recorded but does
    not fail the request - losing a metric is preferable to losing an answer.
    """
    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    cfg = retrieval.default_config()
    if request.top_k:
        cfg.top_k = request.top_k

    result = rag.answer_question(question, history=request.history, cfg=cfg)

    try:
        db.save_conversation(result.to_log_record())
    except Exception as exc: # noqa: BLE001
        logger.warning("Failed to log conversation %s: %s", result.conversation_id, exc)

    return QueryResponse(
        conversation_id=result.conversation_id,
        question=result.question,
        rewritten_question=result.rewritten_question,
        answer=result.answer,
        sources=result.to_sources_out(),
        retrieval_method=result.retrieval_method,
        model=result.model_used,
        llm_backend=result.llm_backend,
        total_tokens=result.total_tokens + result.eval_total_tokens,
        estimated_cost_usd=round(result.estimated_cost_usd, 6),
        response_time_ms=round(result.response_time_ms, 1),
        relevance=result.relevance,
        grounding_score=result.grounding_score,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Record a thumbs up (+1) or thumbs down (-1) for an answer."""
    if request.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1.")

    if not db.conversation_exists(request.conversation_id):
        raise HTTPException(
            status_code=404, detail=f"Unknown conversation_id: {request.conversation_id}"
        )

    db.save_feedback(request.conversation_id, request.feedback)
    return {"status": "recorded", "conversation_id": request.conversation_id,
            "feedback": request.feedback}


@app.post("/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    """Return nutrition for several products side by side.

    Values come straight from the database with no model involvement, so the
    comparison view cannot hallucinate.
    """
    products = retrieval.get_products(request.codes)
    if len(products) < 2:
        raise HTTPException(
            status_code=404,
            detail="At least two of the requested product codes must exist.",
        )
    return {
        "products": [
            {
                "code": p.code,
                "product_name": p.product_name,
                "brands": p.brands,
                "nutriscore_grade": p.nutriscore_grade,
                "serving_size": p.serving_size,
                "allergens": p.allergens,
                "nutrition": p.nutrition(),
            }
            for p in products
        ]
    }


@app.get("/products/search")
def product_search(q: str, limit: int = 20) -> dict[str, Any]:
    """Name/brand lookup used by the comparison interface."""
    products = retrieval.search_products_by_name(q, min(limit, 50))
    return {
        "products": [
            {
                "code": p.code,
                "product_name": p.product_name,
                "brands": p.brands,
                "categories": p.categories,
            }
            for p in products
        ]
    }