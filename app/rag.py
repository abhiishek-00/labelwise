"""RAG pipeline: retrieve, generate, judge, and ground-check.

Ties retrieval and generation together and produces the full record needed for
monitoring: token counts, latency broken down by stage, an estimated cost, an
optional online relevance verdict, and a numeric grounding score.

The grounding check is specific to this domain. Nutrition answers are dense with
numbers, and a plausible-looking wrong number is the most damaging failure mode
here - more damaging than a vague answer, because it looks authoritative. The
check extracts numbers from the answer and verifies each against the retrieved
context, giving a quantitative hallucination signal that complements the
LLM judge's qualitative verdict.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app import config, llm, prompts, retrieval
from app.models import Product, RetrievalConfig, SourceOut

logger = logging.getLogger(__name__)

# Numbers in the answer, ignoring those attached to a product id.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")

# Values that carry no factual claim and would otherwise inflate the score.
_TRIVIAL_NUMBERS = {"0", "1", "2", "100", "0.0", "1.0", "2.0", "100.0"}


@dataclass(slots=True)
class RagResult:
    """Everything one question produced, ready for logging and the API."""

    conversation_id: str
    question: str
    answer: str
    rewritten_question: str | None = None
    sources: list[Product] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)

    retrieval_method: str = ""
    rerank_enabled: bool = False
    rewrite_enabled: bool = False

    llm_backend: str = ""
    model_used: str = ""
    prompt_strategy: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    eval_total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    response_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0

    relevance: str | None = None
    relevance_explanation: str | None = None
    grounding_score: float | None = None

    # Provider-side failure detail, kept out of ``answer`` so it never reaches
    # the user or the conversations table.
    error: str | None = None

    def to_log_record(self) -> dict[str, Any]:
        return {
            "id": self.conversation_id,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "answer": self.answer,
            "retrieval_method": self.retrieval_method,
            "rerank_enabled": self.rerank_enabled,
            "rewrite_enabled": self.rewrite_enabled,
            "retrieved_codes": [p.code for p in self.sources],
            "prompt_strategy": self.prompt_strategy,
            "llm_backend": self.llm_backend,
            "model_used": self.model_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "eval_total_tokens": self.eval_total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "response_time_ms": self.response_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "relevance": self.relevance,
            "relevance_explanation": self.relevance_explanation,
            "grounding_score": self.grounding_score,
        }

    def to_sources_out(self) -> list[SourceOut]:
        return [
            SourceOut(
                code=p.code,
                product_name=p.product_name,
                brands=p.brands,
                categories=p.categories,
                nutriscore_grade=p.nutriscore_grade,
                score=round(self.source_scores.get(p.code, 0.0), 4),
                nutrition=p.nutrition(),
            )
            for p in self.sources
        ]


# ── Numeric grounding ─────────────────────────────────────────────────────────


def _context_numbers(products: list[Product]) -> set[float]:
    """Collect every value a grounded answer could legitimately cite.

    Includes declared values and simple derivations a correct answer may
    perform - differences between products for comparison questions - so that
    valid arithmetic is not scored as ungrounded. Rounding is handled by
    :func:`_is_grounded` rather than by pre-formatting, so the caller keeps the
    full-precision values.
    """
    values: set[float] = set()
    for product in products:
        for value in product.nutrition().values():
            if value is not None:
                values.add(float(value))

    # Pairwise differences support comparison answers ("12g more sugar").
    numeric = sorted(values)
    for i, a in enumerate(numeric):
        for b in numeric[i + 1 :]:
            values.add(round(abs(b - a), 2))

    return values


def _is_grounded(number: str, values: set[float]) -> bool:
    """Is ``number`` a faithful rendering of any value in the context?

    Matching on formatted strings alone is inconsistent about rounding. Open
    Food Facts stores unit-converted values like ``52.6315789473684`` (674 of
    the 5,000 products carry more than three decimal places), and an answer may
    reasonably quote that as 52.6316, 52.63 or 52.6. String matching accepted
    the first and third - one because ``%g`` happens to round to six
    significant figures, the other because a one-decimal form is added
    explicitly - while scoring 52.63 as ungrounded. Rounding to two decimals is
    not a hallucination, so it should not be scored as one.

    A number is grounded if any context value rounds to it at the precision the
    answer chose to state.
    """
    try:
        target = float(number)
    except ValueError:
        return False
    decimals = len(number.partition(".")[2])
    return any(round(value, decimals) == target for value in values)


def grounding_score(answer: str, products: list[Product]) -> float | None:
    """Fraction of substantive numbers in the answer traceable to the context.

    Returns ``None`` when the answer contains no substantive numbers, since a
    purely qualitative answer cannot be scored this way and defaulting to 1.0
    would inflate the metric.
    """
    if not products:
        return None

    found = [n for n in _NUMBER_RE.findall(answer) if n not in _TRIVIAL_NUMBERS]
    # Barcodes are quoted as evidence, not numeric claims. They are long, so a
    # length cut removes them - but it must not also discard a value copied
    # verbatim from the corpus, where "52.6315789473684" is 16 characters and
    # perfectly grounded. Exclude by matching the codes themselves, and apply
    # the length cut only to integers, which is what barcodes are.
    codes = {p.code for p in products}
    found = [
        n for n in found
        if n not in codes and ("." in n or len(n) < 8)
    ]
    if not found:
        return None

    allowed = _context_numbers(products)
    grounded = sum(1 for n in found if _is_grounded(n, allowed))
    return round(grounded / len(found), 3)


# ── Online judge ──────────────────────────────────────────────────────────────


def judge_answer(question: str, answer: str) -> tuple[str, str, int]:
    """Score answer relevance. Returns ``(relevance, explanation, tokens)``."""
    try:
        system, user = prompts.build_judge_prompt(question, answer)
        parsed, response = llm.chat_json(system, user)
        return (
            str(parsed.get("Relevance", "UNKNOWN")).upper(),
            str(parsed.get("Explanation", "")),
            response.total_tokens,
        )
    except Exception as exc: # noqa: BLE001 - judging must never break answering
        return "UNKNOWN", f"judge failed: {str(exc)[:150]}", 0


# ── Pipeline ──────────────────────────────────────────────────────────────────


def answer_question(
    question: str,
    *,
    history: list[str] | None = None,
    cfg: RetrievalConfig | None = None,
    prompt_strategy: str | None = None,
    judge: bool | None = None,
    conversation_id: str | None = None,
) -> RagResult:
    """Answer one question end to end."""
    started = time.perf_counter()
    cfg = cfg or retrieval.default_config()
    strategy = prompt_strategy or config.PROMPT_STRATEGY
    run_judge = config.ENABLE_ONLINE_JUDGE if judge is None else judge

    outcome = retrieval.search(question, cfg, history=history)
    products = [r.product for r in outcome.results]

    result = RagResult(
        conversation_id=conversation_id or str(uuid.uuid4()),
        question=question,
        answer="",
        rewritten_question=outcome.rewritten_query,
        sources=products,
        source_scores={r.code: r.score for r in outcome.results},
        retrieval_method=outcome.config_label,
        rerank_enabled=cfg.rerank,
        rewrite_enabled=cfg.rewrite,
        prompt_strategy=strategy,
        retrieval_time_ms=outcome.retrieval_time_ms,
    )

    if not products:
        result.answer = (
            "I could not find any products in the database matching that question. "
            "Try naming a specific product or brand."
        )
        result.response_time_ms = (time.perf_counter() - started) * 1000
        result.relevance = "UNKNOWN"
        return result

    system, user = prompts.build_prompt(outcome.effective_query, products, strategy)
    try:
        response = llm.chat(system, user)
    except llm.LLMError as exc:
        # The provider's error text is for operators, not users: it carries
        # request ids, backend names and provider-side policy messages, and it
        # was being returned as the answer and persisted to the conversations
        # table. Users get a plain message; the detail goes to the log.
        logger.warning(
            "LLM call failed for conversation %s: %s", result.conversation_id, exc
        )
        result.answer = (
            "I could not generate an answer just now because the language model "
            "was unavailable. The matching products are listed below, and "
            "retrying usually works."
        )
        result.error = str(exc)[:500]
        result.response_time_ms = (time.perf_counter() - started) * 1000
        result.relevance = "UNKNOWN"
        return result

    result.answer = response.content.strip()
    result.llm_backend = response.backend
    result.model_used = response.model
    result.prompt_tokens = response.prompt_tokens
    result.completion_tokens = response.completion_tokens
    result.total_tokens = response.total_tokens
    result.estimated_cost_usd = response.estimated_cost_usd
    result.grounding_score = grounding_score(result.answer, products)

    # Tokens spent on rewriting and re-ranking belong in the cost picture.
    auxiliary_tokens = outcome.rewrite_tokens + outcome.rerank_tokens
    result.eval_total_tokens = auxiliary_tokens

    if run_judge:
        relevance, explanation, judge_tokens = judge_answer(question, result.answer)
        result.relevance = relevance
        result.relevance_explanation = explanation
        result.eval_total_tokens += judge_tokens

    if result.eval_total_tokens:
        result.estimated_cost_usd += (
            result.eval_total_tokens * config.COST_COMPLETION_PER_1M
        ) / 1_000_000

    result.response_time_ms = (time.perf_counter() - started) * 1000
    return result