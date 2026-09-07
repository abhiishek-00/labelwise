"""Normalise raw Open Food Facts records into searchable product documents.

Open Food Facts is community-contributed, so record quality varies widely: many
entries have no nutrition data, some have placeholder names, and popular
products appear many times under near-identical entries.

Two of those matter for evaluation quality:

*Completeness* - a product with no sugar value cannot answer a sugar question,
so incomplete records add noise to retrieval without adding answerable content.

*Duplication* - Open Food Facts contains many near-identical entries for the
same product. Ground truth maps one question to one barcode, so retrieving a
correct-but-different duplicate scores as a miss and understates measured
performance. Deduplication is therefore a correctness concern for the
benchmark, not a cosmetic one.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

# Nutrition fields kept, mapped from the Open Food Facts `nutriments` keys.
NUTRIENT_FIELDS: dict[str, str] = {
    "energy_kcal_100g": "energy-kcal_100g",
    "fat_100g": "fat_100g",
    "saturated_fat_100g": "saturated-fat_100g",
    "carbohydrates_100g": "carbohydrates_100g",
    "sugars_100g": "sugars_100g",
    "fiber_100g": "fiber_100g",
    "proteins_100g": "proteins_100g",
    "salt_100g": "salt_100g",
}

# A record must carry these to be answerable for the core nutrition questions.
REQUIRED_NUTRIENTS = ("energy_kcal_100g", "sugars_100g", "proteins_100g", "fat_100g")

MIN_NAME_LENGTH = 3
MIN_INGREDIENTS_LENGTH = 10
MAX_FIELD_LENGTH = 2000

_WS_RE = re.compile(r"\s+")
_TAG_PREFIX_RE = re.compile(r"^[a-z]{2}:")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# Open Food Facts marks allergens inline with underscores: "_wheat_ flour".
_UNDERSCORE_RE = re.compile(r"_([^_]+)_")


def _clean_text(value: Any, max_length: int = MAX_FIELD_LENGTH) -> str | None:
    """Collapse whitespace and normalise unicode; return None when empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = _UNDERSCORE_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return None
    return text[:max_length]


def _clean_tags(tags: Any, limit: int = 12) -> str | None:
    """Turn an Open Food Facts tag list into readable comma-separated text.

    Tags arrive language-prefixed (``en:gluten``); the prefix is stripped and
    hyphens become spaces so the text is useful for both lexical search and
    embedding.
    """
    if not tags:
        return None
    if isinstance(tags, str):
        tags = [t for t in tags.split(",") if t.strip()]
    if not isinstance(tags, (list, tuple)):
        return None

    seen: list[str] = []
    for tag in tags:
        label = _TAG_PREFIX_RE.sub("", str(tag).strip().lower()).replace("-", " ").strip()
        if label and label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return ", ".join(seen) if seen else None


def _to_float(value: Any) -> float | None:
    """Coerce to float, rejecting non-finite and implausible values."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if number < 0:
        return None
    return number


def _plausible_nutrition(nutrition: dict[str, float | None]) -> bool:
    """Reject records whose declared nutrition is physically impossible.

    Per-100g macronutrients cannot exceed 100g, and energy above ~950 kcal/100g
    exceeds pure fat. Such values indicate a unit or data-entry error, and would
    otherwise surface as confidently wrong answers.
    """
    for field in ("fat_100g", "saturated_fat_100g", "carbohydrates_100g",
                "sugars_100g", "fiber_100g", "proteins_100g"):
        value = nutrition.get(field)
        if value is not None and value > 100.0:
            return False

    energy = nutrition.get("energy_kcal_100g")
    if energy is not None and energy > 950.0:
        return False

    # Salt above 100g/100g is impossible; above 50g indicates an error.
    salt = nutrition.get("salt_100g")
    if salt is not None and salt > 50.0:
        return False

    # Sugars are a subset of carbohydrates; allow a small tolerance for
    # independently rounded source values.
    sugars, carbs = nutrition.get("sugars_100g"), nutrition.get("carbohydrates_100g")
    if sugars is not None and carbs is not None and sugars > carbs + 1.0:
        return False

    # Saturated fat is a subset of total fat.
    sat, fat = nutrition.get("saturated_fat_100g"), nutrition.get("fat_100g")
    if sat is not None and fat is not None and sat > fat + 1.0:
        return False

    return True


def dedup_key(product: dict[str, Any]) -> str:
    """Identity used to collapse near-duplicate entries.

    Normalises brand and product name by lowercasing and stripping all
    non-alphanumeric characters, so "Nutella" / "nutella " / "NUTELLA-" collapse
    to one entry.
    """
    brand = (product.get("brands") or "").lower().split(",")[0]
    name = (product.get("product_name") or "").lower()
    return f"{_NON_ALNUM_RE.sub('', brand)}|{_NON_ALNUM_RE.sub('', name)}"


def build_doc_text(product: dict[str, Any]) -> str:
    """Render a product as the text used for embedding and lexical search.

    Nutrition is included in readable ``key: value unit`` form so that numeric
    facts are retrievable by semantic search, not only by structured filters.
    Absent values are omitted rather than written as zero - the distinction
    between "contains no sugar" and "sugar not declared" must survive into
    retrieval.
    """
    lines: list[str] = [f"Product: {product['product_name']}"]

    for label, key in (
        ("Brand", "brands"),
        ("Category", "categories"),
        ("Ingredients", "ingredients_text"),
        ("Allergens", "allergens"),
        ("Labels", "labels"),
        ("Serving size", "serving_size"),
        ("Countries", "countries"),
    ):
        value = product.get(key)
        if value:
            lines.append(f"{label}: {value}")

    if product.get("nutriscore_grade"):
        lines.append(f"Nutri-Score: {str(product['nutriscore_grade']).upper()}")

    nutrition_lines = []
    for field, label, unit in (
        ("energy_kcal_100g", "energy", "kcal"),
        ("fat_100g", "fat", "g"),
        ("saturated_fat_100g", "saturated fat", "g"),
        ("carbohydrates_100g", "carbohydrates", "g"),
        ("sugars_100g", "sugars", "g"),
        ("fiber_100g", "fiber", "g"),
        ("proteins_100g", "protein", "g"),
        ("salt_100g", "salt", "g"),
    ):
        value = product.get(field)
        if value is not None:
            nutrition_lines.append(f" {label}: {value:g} {unit}")

    if nutrition_lines:
        lines.append("Nutrition per 100g:")
        lines.extend(nutrition_lines)

    return "\n".join(lines)


def normalize_product(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one raw Open Food Facts record into a product document.

    Returns ``None`` when the record fails a quality gate, so callers can
    count rejections by reason.
    """
    code = _clean_text(raw.get("code"), 64)
    if not code or not code.isdigit():
        return None

    name = _clean_text(raw.get("product_name"), 300)
    if not name or len(name) < MIN_NAME_LENGTH:
        return None

    ingredients = _clean_text(raw.get("ingredients_text"))
    if not ingredients or len(ingredients) < MIN_INGREDIENTS_LENGTH:
        return None

    nutriments = raw.get("nutriments") or {}
    if not isinstance(nutriments, dict):
        return None

    nutrition: dict[str, float | None] = {
        field: _to_float(nutriments.get(source)) for field, source in NUTRIENT_FIELDS.items()
    }

    if any(nutrition.get(field) is None for field in REQUIRED_NUTRIENTS):
        return None
    if not _plausible_nutrition(nutrition):
        return None

    product: dict[str, Any] = {
        "code": code,
        "product_name": name,
        "brands": _clean_text(raw.get("brands"), 200),
        "categories": _clean_tags(raw.get("categories_tags_en") or raw.get("categories_tags")),
        "ingredients_text": ingredients,
        "allergens": _clean_tags(raw.get("allergens_tags")),
        "labels": _clean_tags(raw.get("labels_tags")),
        "countries": _clean_tags(raw.get("countries_tags"), limit=5),
        "serving_size": _clean_text(raw.get("serving_size"), 100),
        "nutriscore_grade": _clean_nutriscore(raw.get("nutriscore_grade")),
        **nutrition,
    }

    product["doc_text"] = build_doc_text(product)
    return product

def _clean_nutriscore(value: Any) -> str | None:
    """Return a Nutri-Score grade only when it is a real grade.

    Open Food Facts uses ``unknown`` and ``not-applicable`` as placeholders.
    Those must become NULL rather than being stored as text, otherwise the
    assistant would report "Nutri-Score: UNKNOWN" as though it were a grade.
    """
    text = _clean_text(value, 20)
    if not text:
        return None
    grade = text.strip().lower()
    return grade if grade in {"a", "b", "c", "d", "e"} else None


def normalize_csv_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one row of the Open Food Facts CSV export.

    The CSV export flattens the nested ``nutriments`` object into top-level
    columns (``sugars_100g``, ``energy-kcal_100g``, ...) using the same keys, so
    the row is reshaped into the JSON record layout and passed through the same
    quality gates. One normalisation path means the CSV and API sources cannot
    drift apart.

    The CSV also uses ``allergens`` where the API uses ``allergens_tags``.
    """
    nutriments = {source: row.get(source) for source in NUTRIENT_FIELDS.values()}

    return normalize_product({
        "code": row.get("code"),
        "product_name": row.get("product_name"),
        "brands": row.get("brands"),
        "categories_tags": row.get("categories_tags"),
        "ingredients_text": row.get("ingredients_text"),
        "allergens_tags": row.get("allergens_tags") or row.get("allergens"),
        "labels_tags": row.get("labels_tags"),
        "countries_tags": row.get("countries_tags"),
        "serving_size": row.get("serving_size"),
        "nutriscore_grade": row.get("nutriscore_grade"),
        "nutriments": nutriments,
    })


def normalize_many(
    raw_records: Iterable[dict[str, Any]],
    *,
    deduplicate: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalise and deduplicate a stream of records.

    Returns the accepted products and a stats dict describing what was
    rejected and why, so corpus quality is reported rather than assumed.
    """
    stats = {
        "seen": 0,
        "rejected_quality": 0,
        "duplicate_barcode": 0,
        "duplicate_product": 0,
        "accepted": 0,
    }
    seen_codes: set[str] = set()
    seen_keys: set[str] = set()
    accepted: list[dict[str, Any]] = []

    for raw in raw_records:
        stats["seen"] += 1

        product = normalize_product(raw)
        if product is None:
            stats["rejected_quality"] += 1
            continue

        if product["code"] in seen_codes:
            stats["duplicate_barcode"] += 1
            continue

        if deduplicate:
            key = dedup_key(product)
            if key in seen_keys:
                stats["duplicate_product"] += 1
                continue
            seen_keys.add(key)

        seen_codes.add(product["code"])
        accepted.append(product)
        stats["accepted"] += 1

    return accepted, stats


def completeness_report(products: list[dict[str, Any]]) -> dict[str, float]:
    """Percentage of products carrying each optional field.

    Reported after ingestion so corpus quality is a measured figure in the
    README rather than a claim.
    """
    if not products:
        return {}
    total = len(products)
    fields = [
        "brands", "categories", "allergens", "labels", "countries",
        "serving_size", "nutriscore_grade", *NUTRIENT_FIELDS,
    ]
    return {
        field: round(100.0 * sum(1 for p in products if p.get(field) is not None) / total, 1)
        for field in fields
    }