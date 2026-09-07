"""Seed the monitoring tables with synthetic traffic.

Why this exists: a reviewer running ``docker compose up`` on a fresh clone opens
Grafana to ten empty panels and reasonably concludes monitoring was never built.
Populating a realistic history makes the dashboard demonstrate itself.

The data is explicitly synthetic and is marked as such - seeded conversation ids
are prefixed with ``seed-``, so they can be identified or removed:

    DELETE FROM conversations WHERE id LIKE 'seed-%';

Distributions are chosen to look like plausible traffic rather than to flatter
the system: roughly 8% of answers are NON_RELEVANT, feedback is imperfect, and
latency has a realistic tail.

Usage:
    uv run python scripts/seed_monitoring.py
    uv run python scripts/seed_monitoring.py --count 400 --days 14
    uv run python scripts/seed_monitoring.py --clear
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db # noqa: E402

QUESTION_TEMPLATES = [
    "What are the ingredients in {product}?",
    "Does {product} contain nuts?",
    "How much sugar is in {product}?",
    "Is {product} suitable for someone avoiding gluten?",
    "Compare {product} and {other} on sugar content",
    "Which has more protein, {product} or {other}?",
    "Find something similar to {product} but lower in salt",
    "What allergens are declared for {product}?",
    "How many calories are in {product} per 100g?",
    "Is {product} high in saturated fat?",
    "What is the Nutri-Score of {product}?",
    "Show me a lower-sugar alternative to {product}",
]

ANSWER_TEMPLATES = [
    "According to the product data for {product}, it contains {value}g of sugar per 100g.",
    "{product} declares the following allergens: milk, soy. This comes from the product label.",
    "Per 100g, {product} provides {value} kcal. Interpretation: this is moderate for its category.",
    "The ingredients listed for {product} are sugar, palm oil, hazelnuts and cocoa.",
    "This information is not declared in the source data for {product}, so I cannot confirm it.",
    "Comparing the two: {product} has {value}g of sugar per 100g, which is lower than the alternative.",
]

# Retrieval configurations, weighted to reflect a production rollout where one
# is dominant and others appear from experiments.
METHODS = [
    ("hybrid_a0.5+rerank", 0.55),
    ("hybrid_a0.5", 0.20),
    ("hybrid_a0.5+rerank+rewrite", 0.15),
    ("vector", 0.06),
    ("lexical", 0.04),
]

RELEVANCE = [("RELEVANT", 0.74), ("PARTLY_RELEVANT", 0.17),
            ("NON_RELEVANT", 0.08), ("UNKNOWN", 0.01)]


def weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    roll = rng.random()
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if roll <= cumulative:
            return value
    return options[-1][0]


def real_products(limit: int = 200) -> list[tuple[str, str]]:
    """Use real product names so seeded questions match the actual corpus."""
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT code, product_name FROM products "
                "WHERE product_name IS NOT NULL ORDER BY random() LIMIT %s",
                (limit,),
            )
            return [(row["code"], row["product_name"]) for row in cur.fetchall()]
    except Exception: # noqa: BLE001 - seeding must work before ingestion too
        return []


def generate(count: int, days: int, seed: int = 7) -> tuple[list[dict], list[tuple[str, int]]]:
    rng = random.Random(seed)
    products = real_products() or [("0000000000000", "Example Product")]
    now = datetime.now().astimezone()

    conversations: list[dict] = []
    feedback: list[tuple[str, int]] = []

    for _ in range(count):
        code, name = rng.choice(products)
        other_code, other_name = rng.choice(products)

        # Traffic is skewed toward recent time and daytime hours, so the
        # "requests over time" panel shows structure rather than noise.
        age_hours = (rng.random() ** 1.7) * days * 24
        timestamp = now - timedelta(hours=age_hours)
        if timestamp.hour < 7 and rng.random() < 0.7:
            timestamp += timedelta(hours=9)

        method = weighted_choice(rng, METHODS)
        relevance = weighted_choice(rng, RELEVANCE)
        reranked = "rerank" in method
        rewritten = "rewrite" in method

        prompt_tokens = rng.randint(900, 2600)
        completion_tokens = rng.randint(60, 420)
        eval_tokens = (rng.randint(120, 400) if reranked else 0) + rng.randint(90, 200)

        retrieval_ms = rng.uniform(25, 160)
        generation_ms = rng.lognormvariate(7.6, 0.45)
        if reranked:
            generation_ms += rng.uniform(800, 2600)
        response_ms = retrieval_ms + generation_ms

        # Grounding correlates with judged relevance: answers that miss the
        # question also tend to cite numbers that are not in the context.
        if relevance == "RELEVANT":
            grounding = rng.choice([1.0, 1.0, 1.0, 0.75, 0.8])
        elif relevance == "PARTLY_RELEVANT":
            grounding = rng.choice([1.0, 0.67, 0.5, 0.75])
        else:
            grounding = rng.choice([0.0, 0.33, 0.5, None])

        cost = (
            prompt_tokens * config.COST_PROMPT_PER_1M
            + (completion_tokens + eval_tokens) * config.COST_COMPLETION_PER_1M
        ) / 1_000_000

        conversation_id = f"seed-{uuid.uuid4()}"
        conversations.append({
            "id": conversation_id,
            "timestamp": timestamp,
            "question": rng.choice(QUESTION_TEMPLATES).format(product=name, other=other_name),
            "rewritten_question": (
                f"{name} nutrition and ingredients" if rewritten else None
            ),
            "answer": rng.choice(ANSWER_TEMPLATES).format(
                product=name, value=round(rng.uniform(0.5, 55), 1)
            ),
            "retrieval_method": method,
            "rerank_enabled": reranked,
            "rewrite_enabled": rewritten,
            "retrieved_codes": [code, other_code],
            "prompt_strategy": rng.choice(["source_aware"] * 5 + ["basic"]),
            "llm_backend": config.LLM_BACKEND,
            "model_used": "seed-model",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "eval_total_tokens": eval_tokens,
            "estimated_cost_usd": cost,
            "response_time_ms": response_ms,
            "retrieval_time_ms": retrieval_ms,
            "relevance": relevance,
            "relevance_explanation": f"Seeded example judged {relevance}.",
            "grounding_score": grounding,
        })

        # Only some users leave feedback, and they do so more often when
        # dissatisfied - which is what real feedback funnels look like.
        leaves_feedback = rng.random() < (0.45 if relevance == "NON_RELEVANT" else 0.22)
        if leaves_feedback:
            positive = {
                "RELEVANT": 0.92, "PARTLY_RELEVANT": 0.55,
                "NON_RELEVANT": 0.08, "UNKNOWN": 0.4,
            }[relevance]
            feedback.append((conversation_id, 1 if rng.random() < positive else -1))

    return conversations, feedback


def clear() -> None:
    with db.get_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM feedback WHERE conversation_id LIKE 'seed-%'")
        cur.execute("DELETE FROM conversations WHERE id LIKE 'seed-%'")
        print("Removed previously seeded rows.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed monitoring data for Grafana")
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clear", action="store_true",
                        help="remove seeded rows and exit")
    args = parser.parse_args(argv)

    db.init_db()

    if args.clear:
        clear()
        return 0

    clear()
    conversations, feedback = generate(args.count, args.days, args.seed)
    inserted = db.save_conversations_bulk(conversations)
    for conversation_id, value in feedback:
        db.save_feedback(conversation_id, value,
                        timestamp=datetime.now().astimezone())

    positive = sum(1 for _, v in feedback if v == 1)
    print(
        f"Seeded {inserted} conversations over {args.days} days "
        f"and {len(feedback)} feedback rows ({positive} positive)."
    )
    print("Open Grafana at http://localhost:3000 (admin/admin).")
    print("Remove later with: uv run python scripts/seed_monitoring.py --clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())