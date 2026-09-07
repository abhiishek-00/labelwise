"""Prompt templates for generation, judging, and grounding.

Two generation strategies are defined and compared empirically in Phase 5,
rather than one being assumed better:

**basic** - a minimal grounded prompt: answer only from context, say so when
information is missing.

**source_aware** - additionally requires the answer to attribute facts to
specific products and to separate retrieved facts from interpretation.

The hypothesis is that the stricter prompt improves attribution and reduces
fabricated nutrition values, at the cost of verbosity. The evaluation decides.
"""

from __future__ import annotations

from app.models import Product

# ── Context rendering ─────────────────────────────────────────────────────────

CONTEXT_ENTRY = """[Product {index}] (id: {code})
Name: {name}
Brand: {brands}
Category: {categories}
Ingredients: {ingredients}
Allergens: {allergens}
Labels: {labels}
Serving size: {serving_size}
Nutri-Score: {nutriscore}
Nutrition per 100g:
{nutrition}"""

NUTRITION_ROWS = [
    ("energy_kcal_100g", "energy", "kcal"),
    ("fat_100g", "fat", "g"),
    ("saturated_fat_100g", "saturated fat", "g"),
    ("carbohydrates_100g", "carbohydrates", "g"),
    ("sugars_100g", "sugars", "g"),
    ("fiber_100g", "fiber", "g"),
    ("proteins_100g", "protein", "g"),
    ("salt_100g", "salt", "g"),
]

_MISSING = "not declared in the source data"


def render_product(product: Product, index: int) -> str:
    """Render one product for the prompt context.

    Absent values are labelled explicitly rather than omitted or zero-filled, so
    the model can distinguish "contains no sugar" from "sugar was not declared".
    Silently dropping a field invites the model to fill the gap itself.
    """
    nutrition_lines = []
    for field, label, unit in NUTRITION_ROWS:
        value = getattr(product, field)
        nutrition_lines.append(
            f" {label}: {value:g} {unit}" if value is not None else f" {label}: {_MISSING}"
        )

    return CONTEXT_ENTRY.format(
        index=index,
        code=product.code,
        name=product.product_name,
        brands=product.brands or _MISSING,
        categories=product.categories or _MISSING,
        ingredients=product.ingredients_text or _MISSING,
        allergens=product.allergens or "none declared",
        labels=product.labels or _MISSING,
        serving_size=product.serving_size or _MISSING,
        nutriscore=(product.nutriscore_grade or _MISSING).upper(),
        nutrition="\n".join(nutrition_lines),
    )


def build_context(products: list[Product]) -> str:
    return "\n\n".join(render_product(p, i) for i, p in enumerate(products, start=1))


# ── Strategy A: basic grounded prompt ─────────────────────────────────────────

BASIC_SYSTEM = """You are LabelWise, an assistant that answers questions about packaged
food products using a database of product labels.

Rules:
- Answer only using the facts in the provided CONTEXT.
- Never invent ingredients, allergens or nutrition values.
- If the context does not contain the answer, say the information is not available.
- Be concise and factual.""".strip()

BASIC_USER = """QUESTION: {question}

CONTEXT:
{context}"""


# ── Strategy B: source-aware prompt ───────────────────────────────────────────

SOURCE_AWARE_SYSTEM = """You are LabelWise, an assistant that answers questions about
packaged food products using a database of product labels.

Follow these rules exactly:

1. Answer the question directly and concisely, using only the CONTEXT provided.
2. Attribute every factual claim to a specific product by name.
3. Never invent or estimate ingredients, allergens or nutrition values. If a value
is marked as not declared, say so explicitly instead of guessing or assuming zero.
4. When comparing products, state the specific numbers you are comparing.
5. Clearly separate facts taken from the data from any interpretation you offer.
Prefix interpretation with "Interpretation:".
6. If the context does not answer the question, say so and state what is missing.
7. You are not a medical or dietary authority. Do not give medical advice.

Remember: reporting that data is unavailable is correct and useful. Guessing is not.""".strip()

SOURCE_AWARE_USER = """QUESTION: {question}

CONTEXT (products retrieved from the database):
{context}

Answer using only the products above, naming the products you rely on."""


PROMPT_STRATEGIES: dict[str, tuple[str, str]] = {
    "basic": (BASIC_SYSTEM, BASIC_USER),
    "source_aware": (SOURCE_AWARE_SYSTEM, SOURCE_AWARE_USER),
}


def build_prompt(
    question: str,
    products: list[Product],
    strategy: str = "source_aware",
) -> tuple[str, str]:
    """Return ``(system, user)`` messages for the requested strategy."""
    if strategy not in PROMPT_STRATEGIES:
        raise ValueError(
            f"Unknown prompt strategy {strategy!r}. Available: {sorted(PROMPT_STRATEGIES)}"
        )
    system, user_template = PROMPT_STRATEGIES[strategy]
    return system, user_template.format(question=question, context=build_context(products))


# ── LLM-as-a-judge ────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator for a retrieval-augmented question
answering system about food products. Respond with valid JSON only.""".strip()

JUDGE_USER = """Evaluate how well the generated answer addresses the question.

Question: {question}

Generated answer: {answer}

Classify the relevance as exactly one of:
- "RELEVANT": fully answers the question
- "PARTLY_RELEVANT": addresses the question but is incomplete or partly off-target
- "NON_RELEVANT": fails to address the question

Correctly stating that the required information is unavailable counts as RELEVANT
when the data genuinely does not contain it.

Return a json object with keys "Relevance" and "Explanation" (one sentence)."""


def build_judge_prompt(question: str, answer: str) -> tuple[str, str]:
    return JUDGE_SYSTEM, JUDGE_USER.format(question=question, answer=answer)


# ── Ground-truth question generation ──────────────────────────────────────────

GT_SYSTEM = """You generate evaluation questions for a food product search system.
Respond with valid JSON only.""".strip()

GT_USER = """Given the product record below, write {n} distinct questions that a real
user might ask, where this specific product is the correct answer.

{product}

Each question is used on its own to search a catalogue of thousands of products.
The reader sees ONLY the question — never this record, and never the other
questions. So every question must stand completely alone.

Hard requirements:
- STANDALONE: never write "this product", "this supplement", "these", "it" or
"the product". There is no earlier turn to refer back to. A question that only
makes sense after seeing this record is unusable.
- IDENTIFYING: each question must contain enough detail to single this product
out from similar ones - typically the brand, or the specific product type plus
a distinguishing attribute (flavour, format, dietary claim, key ingredient).
"How much protein is in this powder?" is wrong; "How much protein is in
Orgain's organic plant-based protein powder?" is right.
- Answerable from this product's data alone.
- Vary the question type across these categories: {categories}
- Write naturally, as a shopper typing into a search box. Do not quote the
record verbatim, and do not include the barcode.
- Vary phrasing: not every question needs the full product name, but each one
must still pick out this product unambiguously.

Return a json object with a key "questions" containing an array of objects, each with
"question" (string) and "category" (one of: {categories})."""

GT_CATEGORIES = [
    "product_lookup",
    "ingredient",
    "allergen",
    "nutrition",
    "comparison",
    "discovery",
    "constraint",
]


def build_gt_prompt(product_text: str, n: int = 3) -> tuple[str, str]:
    return GT_SYSTEM, GT_USER.format(
        n=n, product=product_text, categories=", ".join(GT_CATEGORIES)
    )


# ── Conversational ground truth (for evaluating query rewriting) ──────────────

CONVERSATIONAL_SYSTEM = """You generate multi-turn conversation examples for testing
query rewriting in a food product search system. Respond with valid JSON only.""".strip()

CONVERSATIONAL_USER = """Given this product:

{product}

Create a short two-turn conversation where:
- turn 1 is a normal, standalone question about this product
- turn 2 is a vague follow-up that is meaningless without turn 1, using references
like "it", "this one", "the same but ...", or "any alternatives?"

Turn 2 must still be answerable using this same product or a close alternative.

Return a json object with keys "turn_1" (string) and "turn_2" (string)."""


def build_conversational_prompt(product_text: str) -> tuple[str, str]:
    return CONVERSATIONAL_SYSTEM, CONVERSATIONAL_USER.format(product=product_text)