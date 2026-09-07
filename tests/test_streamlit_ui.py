"""UI contract tests for the Streamlit app.

These run offline. ``httpx`` is stubbed at the module level, so no API, no
database and no API key are needed, and the tests stay in the default suite
rather than behind the ``live`` marker.

The behaviour under test is Streamlit's rerun model, which is where this UI
actually broke. Every interaction re-executes the whole script, so any state not
held under a stable widget key is lost between clicks. Selecting an example
question and pressing Ask sent no request at all, because the question was
passed through ``value=st.session_state.pop(...)``: the rerun that filled the
box consumed the stored value, and the rerun triggered by Ask rebuilt the box
empty. The blank-question guard then skipped the request silently. Typing by
hand worked, so the failure only appeared via the example buttons.

None of that is reachable from the backend tests, which is why it shipped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"

ANSWER_TEXT = "Muesli No Sugar Added declares 10 g of sugar per 100 g."
CONVERSATION_ID = "11111111-2222-3333-4444-555555555555"


def _search_results(term: str) -> list[dict[str, Any]]:
    """Two disjoint result sets, so a second search cannot offer the first's items.

    That is the case that matters: it forces every selection from the first
    search to be stale once the second runs.
    """
    if term.strip().lower().startswith("cereal"):
        return [
            {"product_name": "Muesli No Sugar Added", "brands": "Alpen", "code": "1111111111111"},
            {"product_name": "Corn Flakes", "brands": "Kellogg", "code": "2222222222222"},
        ]
    return [
        {"product_name": "Chocolate hazelnut spread", "brands": "Nutella", "code": "3333333333333"},
        {"product_name": "Peanut butter", "brands": "Whole Earth", "code": "4444444444444"},
    ]


def _compare_product(code: str) -> dict[str, Any]:
    return {
        "code": code,
        "product_name": f"Product {code[:4]}",
        "brands": "Test Brand",
        "nutriscore_grade": "c",
        "serving_size": "30 g",
        "allergens": "",
        "nutrition": {
            "energy_kcal_100g": 380.0,
            "fat_100g": 5.0,
            "saturated_fat_100g": 1.0,
            "carbohydrates_100g": 70.0,
            "sugars_100g": 10.0,
            "fiber_100g": 6.0,
            "proteins_100g": 9.0,
            "salt_100g": 0.4,
        },
    }


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` covering what the app uses."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _query_response() -> dict[str, Any]:
    """Mirror ``QueryResponse`` in app.models.

    Every field is present because the UI indexes several of them directly. A
    partial stub passes a shallow test and hides a real KeyError.
    """
    return {
        "conversation_id": CONVERSATION_ID,
        "question": "ignored",
        "rewritten_question": None,
        "answer": ANSWER_TEXT,
        "sources": [
            {
                "code": "1234567890123",
                "product_name": "Muesli No Sugar Added",
                "brands": "Test Brand",
                "score": 0.87,
            }
        ],
        "retrieval_method": "hybrid_a0.1",
        "model": "llama-3.3-70b-versatile",
        "llm_backend": "openai",
        "total_tokens": 1234,
        "estimated_cost_usd": 0.00021,
        "response_time_ms": 1500.0,
        "relevance": "RELEVANT",
        "grounding_score": 1.0,
    }


@pytest.fixture
def api_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Stub httpx and record every POST the app makes."""
    import httpx
    import streamlit as st

    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any] | None = None, **_: Any) -> _FakeResponse:
        path = url.split("8000", 1)[-1] if "8000" in url else url
        calls.append((path, json or {}))
        if path.endswith("/query"):
            return _FakeResponse(_query_response())
        if path.endswith("/feedback"):
            return _FakeResponse({"status": "recorded"})
        if path.endswith("/compare"):
            codes = (json or {}).get("codes", [])
            return _FakeResponse({"products": [_compare_product(c) for c in codes]})
        return _FakeResponse({"products": []})

    def fake_get(url: str, params: dict[str, Any] | None = None, **_: Any) -> _FakeResponse:
        if "/products/search" in url:
            term = (params or {}).get("q", "")
            return _FakeResponse({"products": _search_results(term)})
        return _FakeResponse(
            {
                "status": "ok",
                "database": {"connected": True, "products": 5000, "embedded": 5000,
                            "embedding_dim": 384},
                "llm": {"backend": "openai", "egress_policy": "open"},
                "embeddings": {"backend": "local", "dim": 384},
                "retrieval": {"method": "hybrid", "alpha": 0.1, "top_k": 5,
                            "rerank": False, "rewrite": True},
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    # get_health is memoised; a cached entry would leak between tests.
    st.cache_data.clear()
    return calls


def _run_app() -> Any:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


def _example_buttons(app: Any) -> list[Any]:
    return [b for b in app.button if b.key and b.key.startswith("ex_")]


def _ask_button(app: Any) -> Any:
    return next(b for b in app.button if b.label == "Ask")


def test_app_starts_without_exception(api_calls: list[Any]) -> None:
    app = _run_app()
    assert not app.exception
    assert app.text_input[0].value == ""


def test_example_question_populates_the_box(api_calls: list[Any]) -> None:
    app = _run_app()
    buttons = _example_buttons(app)
    assert buttons, "no example question buttons rendered"

    buttons[0].click().run()

    assert not app.exception
    assert app.text_input[0].value != "", "example click did not fill the question box"


def test_example_question_survives_the_ask_click(api_calls: list[Any]) -> None:
    """The regression test.

    Before the fix the question was consumed by the rerun that displayed it, so
    by the time Ask ran the box was empty and no request was sent.
    """
    app = _run_app()
    example = _example_buttons(app)[0]
    example.click().run()
    chosen = app.text_input[0].value

    _ask_button(app).click().run()

    assert not app.exception
    queries = [payload for path, payload in api_calls if path.endswith("/query")]
    assert queries, "clicking Ask after choosing an example sent no /query request"
    assert queries[0]["query"] == chosen
    assert any(ANSWER_TEXT in md.value for md in app.markdown)

def test_typed_question_is_sent(api_calls: list[Any]) -> None:
    app = _run_app()
    app.text_input[0].set_value("Does Knorr bouillon contain celery?").run()

    _ask_button(app).click().run()

    queries = [payload for path, payload in api_calls if path.endswith("/query")]
    assert queries and queries[0]["query"] == "Does Knorr bouillon contain celery?"


def test_ask_with_empty_box_warns_and_sends_nothing(api_calls: list[Any]) -> None:
    app = _run_app()

    _ask_button(app).click().run()

    assert not app.exception
    assert not [p for path, p in api_calls if path.endswith("/query")]
    assert app.warning, "an empty Ask should say so rather than doing nothing"


def test_clear_conversation_empties_the_box(api_calls: list[Any]) -> None:
    """Clearing must run as a callback.

    Assigning to a widget's session-state key after that widget has been created
    raises ``StreamlitAPIException``, and the question box is created before the
    Clear button.
    """
    app = _run_app()
    _example_buttons(app)[0].click().run()
    assert app.text_input[0].value != ""

    next(b for b in app.button if b.label == "Clear conversation").click().run()

    assert not app.exception
    assert app.text_input[0].value == ""


def test_history_grows_so_follow_ups_can_be_rewritten(api_calls: list[Any]) -> None:
    """Each answered question must enter the history the next /query carries.

    Query rewriting is the largest measured effect in the benchmark, and it only
    works if the UI actually forwards the previous turns.
    """
    app = _run_app()
    app.text_input[0].set_value("What is Knorr Mediterranean vegetable bouillon?").run()
    _ask_button(app).click().run()

    app.text_input[0].set_value("does it contain celery?").run()
    _ask_button(app).click().run()

    queries = [payload for path, payload in api_calls if path.endswith("/query")]
    assert len(queries) == 2
    assert queries[0]["history"] == []
    assert queries[1]["history"] == ["What is Knorr Mediterranean vegetable bouillon?"]


class TestCompareTab:
    """The Compare tab's search box had no way to reset it.

    The multiselect offers its own clear affordance; the text box did not, so a
    search could only be undone by selecting the text and deleting it.
    """

    @staticmethod
    def _search_box(app: Any) -> Any:
        return next(w for w in app.text_input if w.key == "compare_search")

    @staticmethod
    def _clear_button(app: Any) -> Any:
        return next(b for b in app.button if b.label == "Clear")

    def test_clear_button_is_disabled_until_there_is_something_to_clear(
        self, api_calls: list[Any]
    ) -> None:
        app = _run_app()
        assert self._clear_button(app).disabled

    def test_clear_empties_the_search_box(self, api_calls: list[Any]) -> None:
        app = _run_app()
        self._search_box(app).set_value("cereal").run()
        assert not self._clear_button(app).disabled

        self._clear_button(app).click().run()

        assert not app.exception
        assert self._search_box(app).value == ""

    def test_clear_also_drops_the_selection(self, api_calls: list[Any]) -> None:
        """Leaving the selection behind would keep results on screen after a clear."""
        app = _run_app()
        self._search_box(app).set_value("cereal").run()
        options = app.multiselect[0].options
        app.multiselect[0].set_value(options[:2]).run()
        assert len(app.multiselect[0].value) == 2

        self._clear_button(app).click().run()

        assert not app.exception
        assert app.session_state["compare_selection"] == []

    def test_changing_the_search_discards_stale_selections(
        self, api_calls: list[Any]
    ) -> None:
        """Pins behaviour this code relies on rather than implements.

        Streamlit itself drops selections the new option list no longer
        contains. Verified by removing an explicit prune and seeing no change,
        so the guard was deleted rather than kept as decoration - but the
        invariant is load-bearing, so a version upgrade that regressed it should
        fail here.
        """
        app = _run_app()
        self._search_box(app).set_value("cereal").run()
        app.multiselect[0].set_value(app.multiselect[0].options[:2]).run()

        self._search_box(app).set_value("spread").run()

        assert not app.exception
        assert app.multiselect[0].value == []
        assert all("spread" in o.lower() or "butter" in o.lower()
                for o in app.multiselect[0].options)

    def test_two_selected_products_are_compared(self, api_calls: list[Any]) -> None:
        app = _run_app()
        self._search_box(app).set_value("cereal").run()
        app.multiselect[0].set_value(app.multiselect[0].options[:2]).run()

        posts = [p for path, p in api_calls if path.endswith("/compare")]
        assert posts and len(posts[-1]["codes"]) == 2