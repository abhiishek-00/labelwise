"""Retrieval strategies over the product knowledge base.

Five strategies, all reachable through :func:`search` so that evaluation and
production run the same code path:

**Lexical** - PostgreSQL full-text search with ``ts_rank_cd`` over a weighted
tsvector. Strong on exact product names, brands and allergen terms.

**Vector** - pgvector cosine similarity over document embeddings. Strong on
paraphrased and descriptive queries where wording differs from the label.

**Hybrid** - combines both. Two strategies are implemented and compared:
``weighted`` (min-max normalised scores blended by ``alpha``) and ``rrf``
(reciprocal rank fusion, which uses ranks and so needs no normalisation).

**Re-ranking** - an LLM scores the shortlist directly against the query.

**Query rewriting** - an LLM turns a conversational follow-up into a
standalone retrieval query.

Score normalisation matters: ``ts_rank_cd`` is unbounded and cosine similarity
is bounded to [-1, 1], so blending them raw would let the lexical side dominate
arbitrarily. Min-max normalisation is applied per result set before blending.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app import config, db, embeddings, llm
from app.models import Product, RetrievalConfig, RetrievalOutcome, SearchResult

PRODUCT_FIELDS = """
    code, product_name, brands, categories, ingredients_text, allergens,
    labels, countries, serving_size, nutriscore_grade,
    energy_kcal_100g, fat_100g, saturated_fat_100g, carbohydrates_100g,
    sugars_100g, fiber_100g, proteins_100g, salt_100g, doc_text
"""

# Reciprocal rank fusion constant. 60 is the value from the original paper and
# is the conventional default; it damps the influence of top ranks enough that
# a single strong-but-wrong hit cannot dominate the fusion.
RRF_K = 60


# ── Single-strategy retrieval ─────────────────────────────────────────────────


def lexical_search(query: str, limit: int = 20) -> list[SearchResult]:
    """Full-text search using the weighted tsvector index.

    The query is turned into an OR of its lexemes rather than passed through
    ``plainto_tsquery``, which ANDs every term. Real questions carry words the
    product record cannot contain - "What are the ingredients in Knorr's
    Mediterranean Vegetable Bouillon Cubes?" becomes
    ``ingredi & knorr & mediterranean & veget & bouillon & cube``, and requiring
    *ingredi* to appear in the document drops the match count to zero. Measured
    on the benchmark, AND semantics scored Hit Rate@5 of 0.026; the same query
    under OR semantics matches 702 documents.

    Ranking still rewards conjunction: ``ts_rank_cd`` scores documents higher
    when more query terms are present and closer together, so OR widens the
    candidate set without flattening the ordering.
    """
    with db.get_cursor() as cur:
        cur.execute(
            f"""
            WITH q AS (
                SELECT to_tsquery(
                    'english',
                    array_to_string(
                        tsvector_to_array(to_tsvector('english', %s)), ' | '
                    )
                ) AS tsq
            )
            SELECT {PRODUCT_FIELDS},
                ts_rank_cd(search_vector, q.tsq) AS score
            FROM products, q
            WHERE q.tsq IS NOT NULL AND search_vector @@ q.tsq
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, limit),
        )
        rows = cur.fetchall()

    return [
        SearchResult(
            product=Product.from_row(row),
            score=float(row["score"]),
            lexical_score=float(row["score"]),
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


def vector_search(
    query: str,
    limit: int = 20,
    query_vector: list[float] | None = None,
) -> list[SearchResult]:
    """Cosine similarity search over product embeddings.

    ``query_vector`` may be supplied to reuse a cached embedding, which is what
    keeps repeated evaluation sweeps offline and free.
    """
    vector = query_vector if query_vector is not None else embeddings.embed_query(query)
    payload = json.dumps(vector)

    with db.get_cursor() as cur:
        cur.execute(
            f"""
            SELECT {PRODUCT_FIELDS},
                1 - (embedding <=> %s::vector) AS score
            FROM products
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (payload, payload, limit),
        )
        rows = cur.fetchall()

    return [
        SearchResult(
            product=Product.from_row(row),
            score=float(row["score"]),
            vector_score=float(row["score"]),
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


# ── Hybrid fusion ─────────────────────────────────────────────────────────────


def _min_max(values: dict[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1] within a single result set.

    Without this, unbounded ``ts_rank_cd`` values and bounded cosine
    similarities cannot be meaningfully blended.
    """
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def hybrid_search(
    query: str,
    limit: int = 20,
    alpha: float = 0.5,
    strategy: str = "weighted",
    query_vector: list[float] | None = None,
    candidate_multiplier: int = 3,
) -> list[SearchResult]:
    """Combine lexical and vector retrieval.

    Each side retrieves more candidates than requested, because a document
    ranked poorly by one method may rank well by the other; fusing only the
    final top-N would discard exactly those complementary hits.

    Args:
        alpha: weight on the lexical score for the ``weighted`` strategy.
            ``alpha=1`` is pure lexical, ``alpha=0`` pure vector. Swept during
            evaluation rather than assumed.
        strategy: ``weighted`` or ``rrf``.
    """
    pool = limit * candidate_multiplier
    lexical = lexical_search(query, pool)
    vector = vector_search(query, pool, query_vector=query_vector)

    products: dict[str, Product] = {}
    lexical_scores: dict[str, float] = {}
    vector_scores: dict[str, float] = {}
    lexical_ranks: dict[str, int] = {}
    vector_ranks: dict[str, int] = {}

    for result in lexical:
        products[result.code] = result.product
        lexical_scores[result.code] = result.score
        lexical_ranks[result.code] = result.rank
    for result in vector:
        products[result.code] = result.product
        vector_scores[result.code] = result.score
        vector_ranks[result.code] = result.rank

    combined: dict[str, float] = {}
    if strategy == "rrf":
        for code in products:
            score = 0.0
            if code in lexical_ranks:
                score += 1.0 / (RRF_K + lexical_ranks[code])
            if code in vector_ranks:
                score += 1.0 / (RRF_K + vector_ranks[code])
            combined[code] = score
    else:
        normalised_lexical = _min_max(lexical_scores)
        normalised_vector = _min_max(vector_scores)
        for code in products:
            combined[code] = (
                alpha * normalised_lexical.get(code, 0.0)
                + (1.0 - alpha) * normalised_vector.get(code, 0.0)
            )

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [
        SearchResult(
            product=products[code],
            score=score,
            lexical_score=lexical_scores.get(code),
            vector_score=vector_scores.get(code),
            rank=index,
        )
        for index, (code, score) in enumerate(ranked, start=1)
    ]

# ── LLM re-ranking ────────────────────────────────────────────────────────────

RERANK_SYSTEM = """You are a search relevance expert for a food product database.
You rank candidate products by how well they answer the user's query.
Respond with valid JSON only.""".strip()

RERANK_USER = """Query: {query}

Candidate products:
{candidates}

Rank the candidates by how well each one answers the query.
Consider the product name, brand, category, ingredients, allergens and nutrition.

Return a json object with a single key "ranking" containing an array of candidate
numbers ordered from most to least relevant. Include every candidate number exactly
once. Example: {{"ranking": [3, 1, 4, 2]}}""".strip()


def _candidate_summary(result: SearchResult, index: int, max_chars: int = 300) -> str:
    product = result.product
    parts = [f"[{index}] {product.product_name}"]
    if product.brands:
        parts.append(f"brand: {product.brands}")
    if product.categories:
        parts.append(f"category: {product.categories[:80]}")
    if product.allergens:
        parts.append(f"allergens: {product.allergens[:60]}")
    nutrition = []
    for field, label in (("energy_kcal_100g", "kcal"), ("sugars_100g", "sugar"),
                        ("proteins_100g", "protein"), ("salt_100g", "salt")):
        value = getattr(product, field)
        if value is not None:
            nutrition.append(f"{label} {value:g}")
    if nutrition:
        parts.append("per 100g: " + ", ".join(nutrition))
    if product.ingredients_text:
        parts.append(f"ingredients: {product.ingredients_text[:max_chars]}")
    return " | ".join(parts)


def rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 5,
) -> tuple[list[SearchResult], int]:
    """Re-rank candidates with an LLM. Returns ``(results, tokens_used)``.

    Listwise re-ranking is used rather than a cross-encoder because the
    environment has no usable cross-encoder library; the LLM sees all candidates
    at once and returns an ordering.

    Any failure degrades to the original ranking rather than propagating: a
    re-ranking problem should cost quality, not availability.
    """
    if len(candidates) <= 1:
        return candidates[:top_k], 0

    summaries = "\n".join(
        _candidate_summary(result, index) for index, result in enumerate(candidates, start=1)
    )
    try:
        parsed, response = llm.chat_json(
            RERANK_SYSTEM,
            RERANK_USER.format(query=query, candidates=summaries),
        )
        order = parsed.get("ranking") or []
        if not isinstance(order, list):
            raise ValueError("ranking is not a list")

        seen: set[int] = set()
        reranked: list[SearchResult] = []
        for position in order:
            try:
                index = int(position) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and index not in seen:
                seen.add(index)
                result = candidates[index]
                result.rerank_score = 1.0 / (len(reranked) + 1)
                reranked.append(result)

        # Any candidate the model omitted keeps its original relative order.
        for index, result in enumerate(candidates):
            if index not in seen:
                reranked.append(result)

        for position, result in enumerate(reranked, start=1):
            result.rank = position
        return reranked[:top_k], response.total_tokens

    except Exception as exc: # noqa: BLE001 - degrade, never fail the request
        print(f" [rerank] falling back to original order: {str(exc)[:120]}", flush=True)
        return candidates[:top_k], 0


# ── Query rewriting ───────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You rewrite user questions into standalone search queries for a
food product database. Respond with valid JSON only.""".strip()

REWRITE_USER = """Conversation so far:
{history}

Current user message: {query}

Rewrite the current message as a standalone search query that will retrieve the
right products without needing the conversation for context. Resolve pronouns and
references such as "it", "this one" or "the same but ...". Keep product names,
brands and nutritional constraints. Do not invent details that were never mentioned.

If the message is already standalone, return it unchanged.

Return a json object with a single key "query" containing the rewritten string.""".strip()


def rewrite_query(query: str, history: list[str] | None = None) -> tuple[str, int]:
    """Rewrite a conversational query into a standalone one.

    Returns ``(query, tokens_used)``; on any failure the original query is
    returned unchanged.
    """
    context = "\n".join(f"- {turn}" for turn in (history or [])) or "(no previous messages)"
    try:
        parsed, response = llm.chat_json(
            REWRITE_SYSTEM, REWRITE_USER.format(history=context, query=query)
        )
        rewritten = str(parsed.get("query") or "").strip()
        if not rewritten:
            return query, response.total_tokens
        return rewritten, response.total_tokens
    except Exception as exc: # noqa: BLE001 - degrade to the original query
        print(f" [rewrite] falling back to original query: {str(exc)[:120]}", flush=True)
        return query, 0


# ── Unified entry point ───────────────────────────────────────────────────────


def search(
    query: str,
    cfg: RetrievalConfig | None = None,
    *,
    history: list[str] | None = None,
    query_vector: list[float] | None = None,
) -> RetrievalOutcome:
    """Run one retrieval configuration end to end.

    Evaluation and production share this function, so a configuration measured
    in the benchmark is exactly what runs in the application.
    """
    cfg = cfg or default_config()
    outcome = RetrievalOutcome(original_query=query, config_label=cfg.label())

    effective_query = query
    if cfg.rewrite:
        t0 = time.perf_counter()
        rewritten, tokens = rewrite_query(query, history)
        outcome.rewrite_time_ms = (time.perf_counter() - t0) * 1000
        outcome.rewrite_tokens = tokens
        if rewritten != query:
            outcome.rewritten_query = rewritten
            effective_query = rewritten
            # A rewritten query invalidates any embedding cached for the original.
            query_vector = None

    # Retrieve a wider candidate set when re-ranking will narrow it afterwards.
    limit = cfg.top_n if cfg.rerank else cfg.top_k

    t0 = time.perf_counter()
    if cfg.method == "lexical":
        results = lexical_search(effective_query, limit)
    elif cfg.method == "vector":
        results = vector_search(effective_query, limit, query_vector=query_vector)
    elif cfg.method == "hybrid":
        results = hybrid_search(
            effective_query,
            limit,
            alpha=cfg.alpha,
            strategy=cfg.hybrid_strategy,
            query_vector=query_vector,
        )
    else:
        raise ValueError(f"Unknown retrieval method: {cfg.method!r}")
    outcome.retrieval_time_ms = (time.perf_counter() - t0) * 1000

    if cfg.rerank and results:
        t0 = time.perf_counter()
        results, tokens = rerank(effective_query, results, cfg.top_k)
        outcome.rerank_time_ms = (time.perf_counter() - t0) * 1000
        outcome.rerank_tokens = tokens
    else:
        results = results[: cfg.top_k]

    outcome.results = results
    return outcome


def default_config() -> RetrievalConfig:
    """Production configuration, driven by settings chosen in evaluation."""
    return RetrievalConfig(
        method=config.RETRIEVAL_METHOD,
        alpha=config.HYBRID_ALPHA,
        top_n=config.RETRIEVAL_TOP_N,
        top_k=config.RETRIEVAL_TOP_K,
        rerank=config.ENABLE_RERANK,
        rewrite=config.ENABLE_QUERY_REWRITE,
    )


def get_products(codes: list[str]) -> list[Product]:
    """Fetch products by barcode, preserving the requested order."""
    if not codes:
        return []
    with db.get_cursor() as cur:
        cur.execute(f"SELECT {PRODUCT_FIELDS} FROM products WHERE code = ANY(%s)", (codes,))
        found = {row["code"]: Product.from_row(row) for row in cur.fetchall()}
    return [found[code] for code in codes if code in found]


def search_products_by_name(term: str, limit: int = 20) -> list[Product]:
    """Name/brand lookup used by the comparison interface."""
    with db.get_cursor() as cur:
        cur.execute(
            f"""
            SELECT {PRODUCT_FIELDS}
            FROM products
            WHERE product_name ILIKE %s OR brands ILIKE %s
            ORDER BY length(product_name)
            LIMIT %s
            """,
            (f"%{term}%", f"%{term}%", limit),
        )
        return [Product.from_row(row) for row in cur.fetchall()]