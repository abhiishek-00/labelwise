"""LabelWise Streamlit interface.

Three views:

**Ask** - free-form questions, showing the answer, the products it was drawn
from, and thumbs up/down feedback.

**Compare** - side-by-side nutrition for selected products, read straight from
the database with no model involvement.

**About** - the data source, its limitations, and how the pipeline works.

The UI talks to the FastAPI service rather than importing the RAG pipeline
directly, so what it exercises is the same interface any other client would use.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 180.0

NUTRITION_LABELS = {
    "energy_kcal_100g": "Energy (kcal)",
    "fat_100g": "Fat (g)",
    "saturated_fat_100g": "Saturated fat (g)",
    "carbohydrates_100g": "Carbohydrates (g)",
    "sugars_100g": "Sugars (g)",
    "fiber_100g": "Fiber (g)",
    "proteins_100g": "Protein (g)",
    "salt_100g": "Salt (g)",
}

EXAMPLE_QUESTIONS = [
    "Which breakfast cereals are lowest in sugar?",
    "Does this chocolate spread contain hazelnuts?",
    "Compare the protein content of these yogurts",
    "Find a snack with less than 5g of sugar per 100g",
    "What allergens are declared in this biscuit?",
]

st.set_page_config(page_title="LabelWise", page_icon=":label:", layout="wide")


# ── API helpers ───────────────────────────────────────────────────────────────


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc: # noqa: BLE001
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
    return None


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc: # noqa: BLE001
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
    return None


@st.cache_data(ttl=30)
def get_health() -> dict[str, Any] | None:
    return api_get("/health")


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    st.sidebar.title(":label: LabelWise")
    st.sidebar.caption("Grounded answers about packaged food products")

    health = get_health()
    if not health:
        st.sidebar.error("API unreachable")
        return

    database = health.get("database", {})
    if health.get("status") == "ok":
        st.sidebar.success(f"{database.get('products', 0):,} products indexed")
    else:
        st.sidebar.warning("Service degraded")

    with st.sidebar.expander("Configuration", expanded=False):
        retrieval = health.get("retrieval", {})
        st.write(f"**Retrieval:** {retrieval.get('method')}")
        if retrieval.get("method") == "hybrid":
            st.write(f"**Alpha:** {retrieval.get('alpha')}")
        st.write(f"**Top-k:** {retrieval.get('top_k')}")
        st.write(f"**Re-ranking:** {'on' if retrieval.get('rerank') else 'off'}")
        st.write(f"**Query rewriting:** {'on' if retrieval.get('rewrite') else 'off'}")
        st.write(f"**Prompt:** {health.get('generation', {}).get('prompt_strategy')}")
        st.write(f"**Embeddings:** {health.get('embeddings', {}).get('backend')} "
                f"({health.get('embeddings', {}).get('dim')}d)")

    st.sidebar.divider()
    st.sidebar.caption(
        "Data from Open Food Facts, a community-contributed database. "
        "Records may be incomplete or inaccurate. "
        "**Not medical or dietary advice.**"
    )


# ── Ask ───────────────────────────────────────────────────────────────────────


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.markdown("##### Sources")
    for index, source in enumerate(sources, start=1):
        header = f"{index}. {source['product_name']}"
        if source.get("brands"):
            header += f" — {source['brands']}"
        with st.expander(f"{header} (score {source['score']:.3f})"):
            st.caption(f"Barcode: `{source['code']}`")
            if source.get("categories"):
                st.caption(f"Category: {source['categories']}")
            if source.get("nutriscore_grade"):
                st.caption(f"Nutri-Score: {source['nutriscore_grade'].upper()}")

            nutrition = source.get("nutrition") or {}
            rows = [
                {"Nutrient": label, "Per 100g": nutrition.get(key)}
                for key, label in NUTRITION_LABELS.items()
            ]
            frame = pd.DataFrame(rows)
            # Missing values are shown as "not declared" rather than 0, so the
            # UI preserves the same distinction the prompt does.
            frame["Per 100g"] = frame["Per 100g"].apply(
                lambda v: "not declared" if pd.isna(v) or v is None else f"{v:g}"
            )
            st.dataframe(frame, hide_index=True, width="stretch")


def submit_feedback(conversation_id: str, value: int) -> None:
    result = api_post("/feedback", {"conversation_id": conversation_id, "feedback": value})
    if result:
        st.session_state.feedback_given.add(conversation_id)
        st.toast("Thanks for the feedback." if value == 1 else "Thanks — we'll use this.")


def _clear_conversation() -> None:
    """Reset the conversation and empty the question box.

    Runs as a button callback so that it executes before the widgets are
    rebuilt; assigning to ``question_input`` after the text box exists would
    raise ``StreamlitAPIException``.
    """
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.question_input = ""


def render_ask() -> None:
    st.header("Ask about a product")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

    # The text box owns its value through session state under a stable key.
    # Passing the example through `value=` instead looks like it works and does
    # not: the value is consumed on the rerun that fills the box, so the next
    # rerun - the one triggered by clicking Ask - rebuilds the widget with an
    # empty default, the question reads as blank, and no request is ever sent.
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    with st.expander("Example questions", expanded=False):
        columns = st.columns(len(EXAMPLE_QUESTIONS))
        for column, example in zip(columns, EXAMPLE_QUESTIONS):
            if column.button(example, key=f"ex_{example[:20]}", width="stretch"):
                st.session_state.question_input = example
                st.rerun()

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. Which of these cereals has the least sugar?",
    )

    col_ask, col_clear = st.columns([1, 6])
    ask = col_ask.button("Ask", type="primary")
    # Clearing runs in a callback because Streamlit forbids writing to a
    # widget's session-state key after that widget has been created, and the
    # text box is created above this line.
    col_clear.button("Clear conversation", on_click=_clear_conversation)

    if st.session_state.history:
        st.caption(f"Conversation context: {len(st.session_state.history)} previous turn(s)")

    if ask and not question.strip():
        st.warning("Type a question first, or pick one from Example questions.")

    if ask and question.strip():
        with st.spinner("Retrieving products and generating a grounded answer..."):
            result = api_post(
                "/query", {"query": question, "history": st.session_state.history[-4:]}
            )
        if result:
            st.session_state.last_result = result
            st.session_state.history.append(question)

    result = st.session_state.last_result
    if not result:
        return

    st.divider()
    if result.get("rewritten_question"):
        st.caption(f"Interpreted as: *{result['rewritten_question']}*")

    st.markdown("### Answer")
    st.markdown(result["answer"])

    conversation_id = result["conversation_id"]
    already = conversation_id in st.session_state.feedback_given
    col_up, col_down, col_meta = st.columns([1, 1, 8])
    if col_up.button(":+1:", disabled=already, key=f"up_{conversation_id}"):
        submit_feedback(conversation_id, 1)
        st.rerun()
    if col_down.button(":-1:", disabled=already, key=f"down_{conversation_id}"):
        submit_feedback(conversation_id, -1)
        st.rerun()
    if already:
        col_meta.caption("Feedback recorded — thank you.")

    metrics = st.columns(5)
    metrics[0].metric("Response time", f"{result['response_time_ms'] / 1000:.1f}s")
    metrics[1].metric("Tokens", f"{result['total_tokens']:,}")
    metrics[2].metric("Est. cost", f"${result['estimated_cost_usd']:.5f}")
    metrics[3].metric("Relevance", result.get("relevance") or "—")
    grounding = result.get("grounding_score")
    metrics[4].metric(
        "Numeric grounding", f"{grounding:.0%}" if grounding is not None else "n/a"
    )

    if result.get("sources"):
        st.divider()
        render_sources(result["sources"])

# ── Compare ───────────────────────────────────────────────────────────────────


def _clear_compare() -> None:
    """Reset the product search and any selection made from it.

    A callback for the same reason as ``_clear_conversation``: the button sits
    below the widgets it resets, and assigning to a widget's session-state key
    after that widget exists raises ``StreamlitAPIException``.

    The selection is cleared alongside the search term because leaving it would
    keep products on screen that the now-empty search no longer offers.
    """
    st.session_state.compare_search = ""
    st.session_state.compare_selection = []


def render_compare() -> None:
    st.header("Compare products")
    st.caption(
        "Values are read directly from the database, with no model involvement."
    )

    if "compare_search" not in st.session_state:
        st.session_state.compare_search = ""
    if "compare_selection" not in st.session_state:
        st.session_state.compare_selection = []

    col_search, col_clear = st.columns([6, 1], vertical_alignment="bottom")
    term = col_search.text_input(
        "Search for products by name or brand",
        key="compare_search",
        placeholder="e.g. cereal",
    )
    col_clear.button(
        "Clear",
        on_click=_clear_compare,
        disabled=not term.strip(),
        width="stretch",
        help="Clear the search and the current selection",
    )

    if not term.strip():
        return

    found = api_get("/products/search", {"q": term, "limit": 40})
    if not found or not found.get("products"):
        st.info("No products matched that search.")
        return

    options = {
        f"{p['product_name']} — {p['brands'] or 'unknown brand'} ({p['code']})": p["code"]
        for p in found["products"]
    }
    # Streamlit drops selections that a new search no longer offers, which
    # `test_changing_the_search_discards_stale_selections` pins so a version
    # upgrade cannot regress it quietly.
    chosen = st.multiselect(
        "Select 2 to 5 products to compare",
        list(options),
        key="compare_selection",
        max_selections=5,
    )
    if len(chosen) < 2:
        st.info("Select at least two products.")
        return

    result = api_post("/compare", {"codes": [options[label] for label in chosen]})
    if not result:
        return

    products = result["products"]
    table: dict[str, list[Any]] = {"Nutrient": list(NUTRITION_LABELS.values())}
    for product in products:
        column = f"{product['product_name'][:28]}"
        nutrition = product["nutrition"]
        table[column] = [
            "not declared" if nutrition.get(key) is None else f"{nutrition[key]:g}"
            for key in NUTRITION_LABELS
        ]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    chart_rows = []
    for product in products:
        for key, label in NUTRITION_LABELS.items():
            value = product["nutrition"].get(key)
            if value is not None and key != "energy_kcal_100g":
                chart_rows.append(
                    {"Product": product["product_name"][:24], "Nutrient": label, "Per 100g": value}
                )
    if chart_rows:
        st.markdown("##### Macronutrients per 100g")
        frame = pd.DataFrame(chart_rows)
        st.bar_chart(frame, x="Nutrient", y="Per 100g", color="Product", stack=False)

    st.markdown("##### Declared allergens")
    for product in products:
        st.write(f"**{product['product_name']}**: {product.get('allergens') or 'none declared'}")


# ── About ─────────────────────────────────────────────────────────────────────


def render_about() -> None:
    st.header("About LabelWise")
    st.markdown(
        """
LabelWise answers questions about packaged food products using a retrieval-augmented
generation (RAG) pipeline over **Open Food Facts**, a public, community-contributed
product database.

**How a question is answered**

1. The question is optionally rewritten into a standalone search query.
2. Products are retrieved using hybrid search - PostgreSQL full-text search combined
with pgvector semantic similarity.
3. The shortlist is re-ranked by relevance to the question.
4. An LLM answers using **only** the retrieved product records.
5. The answer is scored for relevance and for whether its numbers trace back to the
retrieved data.

**Limitations**

- Open Food Facts is community-contributed. Records may be incomplete, outdated or
wrong, and LabelWise reports what the data says rather than independently verifying it.
- Fields that are not declared are reported as unavailable rather than assumed to be zero.
- This is a course project, not a medical or dietary authority. Do not rely on it for
allergen safety decisions.
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    render_sidebar()
    ask_tab, compare_tab, about_tab = st.tabs(["Ask", "Compare", "About"])
    with ask_tab:
        render_ask()
    with compare_tab:
        render_compare()
    with about_tab:
        render_about()


main()
