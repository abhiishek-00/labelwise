"""Generate the ground-truth evaluation datasets.

Two datasets are produced:

``ground_truth.csv``
    ~900 questions, three per sampled product, spread across question types.
    Each row maps a question to the barcode of the product it was generated
    from, which is the relevance label for Hit Rate and MRR.

``ground_truth_conversational.csv``
    ~80 two-turn conversations where the second turn is deliberately vague
    ("any alternatives?", "the same but less sugar"). This exists because
    query rewriting *cannot be measured* on the main dataset: questions
    generated from a single product are already standalone and keyword-rich,
    so rewriting them is a no-op. Evaluating rewriting on standalone questions
    would produce a meaningless null result and misrepresent the technique.

Generation is a one-time cost. The outputs are committed so the retrieval
benchmark reproduces with no API calls at all.

Usage:
    uv run python -m evaluation.generate_ground_truth
    uv run python -m evaluation.generate_ground_truth --products 300 --per-product 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
from typing import Any

from app import config, db, llm, prompts
from app.models import Product
from app.retrieval import PRODUCT_FIELDS
from ingestion.sources import categorise

GT_PATH = config.EVALUATION_DIR / "ground_truth.csv"
CONVERSATIONAL_PATH = config.EVALUATION_DIR / "ground_truth_conversational.csv"

RANDOM_SEED = 42

# A question is used on its own against the whole catalogue, so it has to
# identify its product without conversational context. An earlier run produced
# 61% anaphoric questions ("how much protein is in this supplement?"), which no
# retrieval method can resolve to one barcode among 5,000. Those questions score
# near-random for every method, which compresses the differences between methods
# and makes the comparison table meaningless. Rejecting them here keeps a silent
# prompt regression from reaching the benchmark.
ANAPHORIC_RE = re.compile(
    r"\b(this|these|those|it|its|it's|the product|the item|the same)\b", re.IGNORECASE
)


def question_defect(text: str) -> str | None:
    """Return a reason string if the question is unusable, else None."""
    if len(text) < 12:
        return "too_short"
    if ANAPHORIC_RE.search(text):
        return "anaphoric"
    # Needs some content word to anchor on, not just a bare nutrient query.
    if len(text.split()) < 4:
        return "too_vague"
    return None


def sample_products(n: int, seed: int = RANDOM_SEED) -> list[Product]:
    """Sample products spread across category buckets.

    Sampling uniformly at random would over-represent whichever categories
    dominate the corpus, biasing the benchmark toward them.
    """
    with db.get_cursor() as cur:
        cur.execute(f"SELECT {PRODUCT_FIELDS} FROM products ORDER BY code")
        products = [Product.from_row(row) for row in cur.fetchall()]

    if len(products) <= n:
        return products

    buckets: dict[str, list[Product]] = {}
    for product in products:
        buckets.setdefault(categorise(product.to_dict()), []).append(product)

    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)

    selected: list[Product] = []
    index = 0
    ordered = sorted(buckets.items())
    while len(selected) < n:
        added = False
        for _, items in ordered:
            if index < len(items):
                selected.append(items[index])
                added = True
                if len(selected) >= n:
                    break
        if not added:
            break
        index += 1
    return selected


def product_for_prompt(product: Product) -> str:
    """Compact product rendering for the generator prompt."""
    lines = [f"Name: {product.product_name}"]
    if product.brands:
        lines.append(f"Brand: {product.brands}")
    if product.categories:
        lines.append(f"Category: {product.categories}")
    if product.ingredients_text:
        lines.append(f"Ingredients: {product.ingredients_text[:400]}")
    if product.allergens:
        lines.append(f"Allergens: {product.allergens}")
    nutrition = [
        f"{label} {value:g}{unit}"
        for field, label, unit in (
            ("energy_kcal_100g", "energy", "kcal"),
            ("sugars_100g", "sugars", "g"),
            ("fat_100g", "fat", "g"),
            ("proteins_100g", "protein", "g"),
            ("salt_100g", "salt", "g"),
        )
        if (value := getattr(product, field)) is not None
    ]
    if nutrition:
        lines.append("Nutrition per 100g: " + ", ".join(nutrition))
    return "\n".join(lines)


async def generate_questions(
    products: list[Product],
    per_product: int,
) -> list[dict[str, Any]]:
    """Generate questions for each product concurrently."""
    prompt_pairs = [
        prompts.build_gt_prompt(product_for_prompt(p), per_product) for p in products
    ]
    print(
        f"Generating {per_product} questions for {len(products)} products "
        f"(concurrency {config.LLM_MAX_CONCURRENCY})..."
    )
    responses = await llm.chat_many(prompt_pairs, json_mode=True, return_exceptions=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    rejected: dict[str, int] = {}
    seen: set[str] = set()
    for product, response in zip(products, responses):
        if isinstance(response, BaseException):
            failures += 1
            continue
        try:
            parsed = llm.parse_json_content(response.content)
            questions = parsed.get("questions") or []
        except llm.LLMError:
            failures += 1
            continue

        for item in questions:
            if not isinstance(item, dict):
                continue
            text = str(item.get("question") or "").strip()
            if (reason := question_defect(text)) is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            key = text.lower()
            if key in seen:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue
            seen.add(key)
            rows.append({
                "question": text,
                "code": product.code,
                "product_name": product.product_name,
                "category": str(item.get("category") or "unknown").strip().lower(),
            })

    print(f" generated {len(rows)} questions ({failures} products failed)")
    if rejected:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(rejected.items()))
        total = sum(rejected.values())
        print(f" rejected {total} questions ({detail})")
    return rows


async def generate_conversational(products: list[Product]) -> list[dict[str, Any]]:
    """Generate two-turn conversations for evaluating query rewriting."""
    prompt_pairs = [
        prompts.build_conversational_prompt(product_for_prompt(p)) for p in products
    ]
    print(f"Generating {len(products)} conversational examples...")
    responses = await llm.chat_many(prompt_pairs, json_mode=True, return_exceptions=True)

    rows: list[dict[str, Any]] = []
    for product, response in zip(products, responses):
        if isinstance(response, BaseException):
            continue
        try:
            parsed = llm.parse_json_content(response.content)
        except llm.LLMError:
            continue
        turn_1 = str(parsed.get("turn_1") or "").strip()
        turn_2 = str(parsed.get("turn_2") or "").strip()
        if len(turn_1) < 8 or len(turn_2) < 4:
            continue
        rows.append({
            "turn_1": turn_1,
            "turn_2": turn_2,
            "code": product.code,
            "product_name": product.product_name,
        })

    print(f" generated {len(rows)} conversations")
    return rows


def write_csv(path: Any, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f" wrote {len(rows)} rows -> {path}")


def summarise(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print("\nQuestion categories:")
    for category, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f" {category:<20s} {count:5d} ({100 * count / len(rows):.1f}%)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ground-truth evaluation data")
    parser.add_argument("--products", type=int, default=300,
                        help="products to sample for question generation")
    parser.add_argument("--per-product", type=int, default=3,
                        help="questions per product")
    parser.add_argument("--conversational", type=int, default=80,
                        help="two-turn conversations for rewriting evaluation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    total = db.product_count()
    if total == 0:
        print("No products in the database. Run ingestion first.")
        return 1
    print(f"Corpus: {total} products\n")

    products = sample_products(args.products, args.seed)
    print(f"Sampled {len(products)} products for question generation")

    rows = asyncio.run(generate_questions(products, args.per_product))
    if not rows:
        print("No questions generated.")
        return 1
    write_csv(GT_PATH, rows, ["question", "code", "product_name", "category"])
    summarise(rows)

    # Drawn from a different slice so the rewriting evaluation is not measured
    # on products the main question set already covers.
    rng = random.Random(args.seed + 1)
    pool = [p for p in sample_products(args.products * 2, args.seed) if p not in products]
    rng.shuffle(pool)
    conversational = asyncio.run(generate_conversational(pool[: args.conversational]))
    if conversational:
        write_csv(CONVERSATIONAL_PATH, conversational,
                ["turn_1", "turn_2", "code", "product_name"])

    print("\nDone. These files are committed so the benchmark reproduces offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())