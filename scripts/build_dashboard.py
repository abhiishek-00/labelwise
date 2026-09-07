"""Generate the Grafana dashboard JSON.

The dashboard is generated rather than hand-written so the SQL behind each
panel is reviewable in one place and stays consistent with the schema.

Run after changing a panel query:

    uv run python scripts/build_dashboard.py

The output is committed and mounted read-only into the Grafana container, which
provisions it from disk at start - no API calls, no import step, no ordering
race with Postgres.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "grafana" / "dashboards" / "labelwise.json"

DATASOURCE = {"type": "postgres", "uid": "labelwise-postgres"}


def target(sql: str, fmt: str = "time_series") -> dict[str, Any]:
    return {
        "datasource": DATASOURCE,
        "format": fmt,
        "rawQuery": True,
        "rawSql": sql.strip(),
        "refId": "A",
    }


def panel(
    panel_id: int,
    title: str,
    kind: str,
    sql: str,
    grid: dict[str, int],
    fmt: str = "time_series",
    description: str = "",
    options: dict[str, Any] | None = None,
    field_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": kind,
        "datasource": DATASOURCE,
        "gridPos": grid,
        "targets": [target(sql, fmt)],
        "options": options or {},
        "fieldConfig": field_config or {"defaults": {}, "overrides": []},
    }


def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []

    # Row 1 - volume and latency ----------------------------------------------

    panels.append(panel(
        1, "Requests over time", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
count(*) AS "requests"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 8, "x": 0, "y": 0},
        description="Query volume over time.",
        field_config={
            "defaults": {"custom": {"fillOpacity": 20, "lineWidth": 2}, "unit": "short"},
            "overrides": [],
        },
    ))

    panels.append(panel(
        2, "Average response time", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
avg(response_time_ms) / 1000.0 AS "avg seconds"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 8, "x": 8, "y": 0},
        description="Mean end-to-end latency, including retrieval and generation.",
        field_config={
            "defaults": {"unit": "s", "custom": {"fillOpacity": 10, "lineWidth": 2}},
            "overrides": [],
        },
    ))

    # P95 matters more than the mean here: a slow tail is what users notice,
    # and re-ranking adds an extra LLM call on a subset of requests.
    panels.append(panel(
        3, "P95 response time", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
percentile_cont(0.95) WITHIN GROUP (ORDER BY response_time_ms) / 1000.0 AS "p95 seconds"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 8, "x": 16, "y": 0},
        description="95th percentile latency - the slow tail users actually feel.",
        field_config={
            "defaults": {"unit": "s", "custom": {"fillOpacity": 10, "lineWidth": 2}},
            "overrides": [],
        },
    ))

    # Row 2 - quality ---------------------------------------------------------

    panels.append(panel(
        4, "User feedback", "piechart",
        """
SELECT
CASE WHEN feedback = 1 THEN 'Helpful' ELSE 'Not helpful' END AS "metric",
count(*) AS "value"
FROM feedback
WHERE $__timeFilter(timestamp)
GROUP BY 1
        """,
        {"h": 8, "w": 6, "x": 0, "y": 8}, fmt="table",
        description="Thumbs up versus thumbs down.",
        options={
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "pieType": "donut",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True},
        },
    ))

    panels.append(panel(
        5, "Answer relevance (LLM judge)", "piechart",
        """
SELECT
coalesce(relevance, 'UNKNOWN') AS "metric",
count(*) AS "value"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
        """,
        {"h": 8, "w": 6, "x": 6, "y": 8}, fmt="table",
        description="Online LLM-as-a-judge verdict on each answer.",
        options={
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "pieType": "pie",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True},
        },
    ))

    # Grounding is the domain-specific quality signal: it detects numbers in an
    # answer that do not trace back to the retrieved product records.
    panels.append(panel(
        6, "Numeric grounding over time", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
avg(grounding_score) AS "mean grounding"
FROM conversations
WHERE $__timeFilter(timestamp) AND grounding_score IS NOT NULL
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 12, "x": 12, "y": 8},
        description=(
            "Fraction of numbers in each answer traceable to the retrieved products. "
            "A drop indicates fabricated nutrition values."
        ),
        field_config={
            "defaults": {
                "unit": "percentunit", "min": 0, "max": 1,
                "custom": {"fillOpacity": 15, "lineWidth": 2},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "orange", "value": 0.7},
                        {"color": "green", "value": 0.9},
                    ],
                },
            },
            "overrides": [],
        },
    ))

# Row 3 - cost and usage --------------------------------------------------

    panels.append(panel(
        7, "Token usage over time", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
sum(prompt_tokens)     AS "prompt",
sum(completion_tokens) AS "completion",
sum(eval_total_tokens) AS "rewrite/rerank/judge"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 8, "x": 0, "y": 16},
        description="Token consumption split by stage.",
        field_config={
            "defaults": {
                "unit": "short",
                "custom": {"fillOpacity": 40, "lineWidth": 1, "stacking": {"mode": "normal"}},
            },
            "overrides": [],
        },
    ))

    panels.append(panel(
        8, "Cumulative estimated cost", "timeseries",
        """
SELECT
$__timeGroupAlias(timestamp, $__interval),
sum(sum(estimated_cost_usd)) OVER (ORDER BY $__timeGroup(timestamp, $__interval))
    AS "cumulative USD"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY 1
ORDER BY 1
        """,
        {"h": 8, "w": 8, "x": 8, "y": 16},
        description=(
            "ESTIMATE ONLY: computed from real token counts at published list prices. "
            "The configured backend may be self-hosted and free."
        ),
        field_config={
            "defaults": {"unit": "currencyUSD", "custom": {"fillOpacity": 10, "lineWidth": 2}},
            "overrides": [],
        },
    ))

    panels.append(panel(
        9, "Retrieval method: usage and latency", "table",
        """
SELECT
retrieval_method                AS "Method",
count(*)                        AS "Requests",
round(avg(response_time_ms)::numeric, 0)   AS "Avg ms",
round(avg(retrieval_time_ms)::numeric, 1)  AS "Retrieval ms",
round(avg(total_tokens)::numeric, 0)       AS "Avg tokens",
round(100.0 * count(*) FILTER (WHERE relevance = 'RELEVANT') / count(*), 1)
                                AS "% relevant"
FROM conversations
WHERE $__timeFilter(timestamp)
GROUP BY retrieval_method
ORDER BY count(*) DESC
        """,
        {"h": 8, "w": 8, "x": 16, "y": 16}, fmt="table",
        description="Comparison of retrieval configurations seen in production traffic.",
    ))

    # Row 4 - operational detail ----------------------------------------------

    panels.append(panel(
        10, "Recent questions receiving negative feedback", "table",
        """
SELECT
c.timestamp AS "Time",
c.question  AS "Question",
c.relevance AS "Judge",
round(c.grounding_score::numeric, 2) AS "Grounding",
c.retrieval_method AS "Method"
FROM conversations c
JOIN feedback f ON f.conversation_id = c.id
WHERE $__timeFilter(c.timestamp) AND f.feedback = -1
ORDER BY c.timestamp DESC
LIMIT 20
        """,
        {"h": 8, "w": 24, "x": 0, "y": 24}, fmt="table",
        description="Where the system is failing users - the queue to work through.",
    ))

    return {
        "uid": "labelwise-main",
        "title": "LabelWise - RAG Monitoring",
        "description": (
            "Traffic, latency, answer quality, and cost for the LabelWise "
            "food-label assistant."
        ),
        "tags": ["labelwise", "rag", "llm"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-24h", "to": "now"},
        "editable": True,
        "panels": panels,
    }


def main() -> int:
    dashboard = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(dashboard['panels'])} panels -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())