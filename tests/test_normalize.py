"""Tests for product normalisation and corpus quality gates.

These gates decide what the knowledge base contains, so they directly bound
every retrieval metric the project reports.
"""

from __future__ import annotations

import pytest

from ingestion import normalize


def make_raw(**overrides) -> dict:
    """A record that passes every quality gate, for targeted mutation."""
    raw = {
        "code": "3017620422003",
        "product_name": "Chocolate Hazelnut Spread",
        "brands": "Ferrero",
        "categories_tags": ["en:spreads", "en:sweet-spreads"],
        "ingredients_text": "sugar, palm oil, _hazelnuts_ 13%, cocoa",
        "allergens_tags": ["en:nuts", "en:milk"],
        "labels_tags": ["en:no-gluten"],
        "countries_tags": ["en:france"],
        "serving_size": "15 g",
        "nutriscore_grade": "e",
        "nutriments": {
            "energy-kcal_100g": 539, "fat_100g": 30.9, "saturated-fat_100g": 10.6,
            "carbohydrates_100g": 57.5, "sugars_100g": 56.3, "fiber_100g": 0,
            "proteins_100g": 6.3, "salt_100g": 0.107,
        },
    }
    raw.update(overrides)
    return raw


class TestQualityGates:
    def test_valid_record_is_accepted(self):
        product = normalize.normalize_product(make_raw())
        assert product is not None
        assert product["code"] == "3017620422003"
        assert product["sugars_100g"] == 56.3

    @pytest.mark.parametrize("overrides,reason", [
        ({"code": None}, "missing code"),
        ({"code": "abc123"}, "non-numeric barcode"),
        ({"product_name": "X"}, "name too short"),
        ({"product_name": None}, "no name"),
        ({"ingredients_text": "sugar"}, "ingredients too short"),
        ({"ingredients_text": None}, "no ingredients"),
    ])
    def test_rejects_unusable_records(self, overrides, reason):
        assert normalize.normalize_product(make_raw(**overrides)) is None, reason

    def test_requires_core_nutrition(self):
        """A product with no sugar value cannot answer a sugar question."""
        raw = make_raw()
        del raw["nutriments"]["sugars_100g"]
        assert normalize.normalize_product(raw) is None

    def test_missing_optional_nutrient_is_kept_as_none(self):
        """Absent is not zero - the distinction must survive normalisation."""
        raw = make_raw()
        del raw["nutriments"]["fiber_100g"]
        product = normalize.normalize_product(raw)
        assert product is not None
        assert product["fiber_100g"] is None


class TestPlausibility:
    """Impossible values indicate data-entry errors and would otherwise become
    confidently wrong answers."""

    @pytest.mark.parametrize("field,value", [
        ("fat_100g", 150.0),
        ("sugars_100g", 300.0),
        ("proteins_100g", 101.0),
    ])
    def test_rejects_macros_above_100g(self, field, value):
        raw = make_raw()
        raw["nutriments"][field.replace("_100g", "") + "_100g"] = value
        assert normalize.normalize_product(raw) is None

    def test_rejects_impossible_energy(self):
        raw = make_raw()
        raw["nutriments"]["energy-kcal_100g"] = 5000
        assert normalize.normalize_product(raw) is None

    def test_rejects_sugars_exceeding_carbohydrates(self):
        raw = make_raw()
        raw["nutriments"]["sugars_100g"] = 90.0
        raw["nutriments"]["carbohydrates_100g"] = 20.0
        assert normalize.normalize_product(raw) is None

    def test_allows_rounding_tolerance(self):
        """Independently rounded source values may disagree slightly."""
        raw = make_raw()
        raw["nutriments"]["carbohydrates_100g"] = 56.0
        raw["nutriments"]["sugars_100g"] = 56.3
        assert normalize.normalize_product(raw) is not None

    def test_rejects_saturated_fat_above_total_fat(self):
        raw = make_raw()
        raw["nutriments"]["saturated-fat_100g"] = 50.0
        raw["nutriments"]["fat_100g"] = 10.0
        assert normalize.normalize_product(raw) is None

    def test_rejects_negative_values(self):
        raw = make_raw()
        raw["nutriments"]["sugars_100g"] = -5.0
        assert normalize.normalize_product(raw) is None


class TestNutriScore:
    @pytest.mark.parametrize("value,expected", [
        ("a", "a"), ("E", "e"), (" b ", "b"),
        ("unknown", None), ("not-applicable", None), ("", None), (None, None),
    ])
    def test_only_real_grades_survive(self, value, expected):
        """'unknown' must become NULL, not the string 'unkn'."""
        assert normalize._clean_nutriscore(value) == expected


class TestTagCleaning:
    def test_strips_language_prefixes(self):
        assert normalize._clean_tags(["en:gluten", "en:milk"]) == "gluten, milk"

    def test_converts_hyphens_and_deduplicates(self):
        assert normalize._clean_tags(["en:no-gluten", "fr:no-gluten"]) == "no gluten"

    def test_handles_empty_input(self):
        assert normalize._clean_tags(None) is None
        assert normalize._clean_tags([]) is None

    def test_unwraps_allergen_underscores(self):
        """Open Food Facts marks allergens inline as _wheat_ flour."""
        assert normalize._clean_text("_wheat_ flour and _milk_") == "wheat flour and milk"


class TestDeduplication:
    def test_same_product_different_formatting_collapses(self):
        a = {"brands": "Ferrero", "product_name": "Nutella"}
        b = {"brands": "ferrero ", "product_name": "NUTELLA-"}
        assert normalize.dedup_key(a) == normalize.dedup_key(b)

    def test_different_products_stay_distinct(self):
        a = {"brands": "Ferrero", "product_name": "Nutella"}
        b = {"brands": "Ferrero", "product_name": "Rocher"}
        assert normalize.dedup_key(a) != normalize.dedup_key(b)

    def test_normalize_many_reports_duplicates(self):
        """Near-duplicates depress Hit Rate, so they are counted, not hidden."""
        records = [make_raw(), make_raw(code="1111111111111"), make_raw(code="2222222222222")]
        products, stats = normalize.normalize_many(records, deduplicate=True)
        assert len(products) == 1
        assert stats["duplicate_product"] == 2
        assert stats["accepted"] == 1

    def test_deduplication_can_be_disabled(self):
        records = [make_raw(), make_raw(code="1111111111111")]
        products, _ = normalize.normalize_many(records, deduplicate=False)
        assert len(products) == 2


class TestDocText:
    def test_includes_declared_nutrition(self):
        product = normalize.normalize_product(make_raw())
        text = product["doc_text"]
        assert "Chocolate Hazelnut Spread" in text
        assert "sugars: 56.3 g" in text
        assert "Allergens: nuts, milk" in text

    def test_omits_undeclared_values_rather_than_zeroing(self):
        """Writing 0 for an absent value would make the model assert a false fact."""
        raw = make_raw()
        del raw["nutriments"]["fiber_100g"]
        text = normalize.normalize_product(raw)["doc_text"]
        assert "fiber" not in text

    def test_is_deterministic(self):
        """The snapshot stores columns only and rebuilds doc_text on load."""
        a = normalize.normalize_product(make_raw())["doc_text"]
        b = normalize.normalize_product(make_raw())["doc_text"]
        assert a == b


class TestCsvAdapter:
    def test_flat_csv_row_matches_json_record(self):
        """Both sources must produce identical products, or they will drift."""
        csv_row = {
            "code": "3017620422003",
            "product_name": "Chocolate Hazelnut Spread",
            "brands": "Ferrero",
            "categories_tags": "en:spreads,en:sweet-spreads",
            "ingredients_text": "sugar, palm oil, _hazelnuts_ 13%, cocoa",
            "allergens": "en:nuts,en:milk",
            "labels_tags": "en:no-gluten",
            "countries_tags": "en:france",
            "serving_size": "15 g",
            "nutriscore_grade": "e",
            "energy-kcal_100g": "539", "fat_100g": "30.9",
            "saturated-fat_100g": "10.6", "carbohydrates_100g": "57.5",
            "sugars_100g": "56.3", "fiber_100g": "0",
            "proteins_100g": "6.3", "salt_100g": "0.107",
        }
        from_csv = normalize.normalize_csv_row(csv_row)
        from_json = normalize.normalize_product(make_raw())
        assert from_csv is not None
        assert from_csv["code"] == from_json["code"]
        assert from_csv["sugars_100g"] == from_json["sugars_100g"]
        assert from_csv["allergens"] == from_json["allergens"]
        assert from_csv["doc_text"] == from_json["doc_text"]

    def test_blank_csv_values_become_none(self):
        row = {
            "code": "1234567890123", "product_name": "Test Product",
            "ingredients_text": "water, sugar, salt", "brands": "",
            "energy-kcal_100g": "100", "fat_100g": "1",
            "sugars_100g": "2", "proteins_100g": "3",
            "fiber_100g": "", "salt_100g": "",
        }
        product = normalize.normalize_csv_row(row)
        assert product is not None
        assert product["brands"] is None
        assert product["fiber_100g"] is None

    def test_rejects_row_without_nutrition(self):
        assert normalize.normalize_csv_row({
            "code": "1234567890123", "product_name": "Test Product",
            "ingredients_text": "water, sugar, salt",
        }) is None


class TestCompletenessReport:
    def test_reports_percentage_per_field(self):
        products, _ = normalize.normalize_many(
            [make_raw(), make_raw(code="9999999999999", brands=None)],
            deduplicate=False,
        )
        report = normalize.completeness_report(products)
        assert report["brands"] == 50.0
        assert report["sugars_100g"] == 100.0

    def test_empty_input(self):
        assert normalize.completeness_report([]) == {}