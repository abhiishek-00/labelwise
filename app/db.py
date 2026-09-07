"""PostgreSQL access: connections, schema initialisation, and monitoring writes.

Uses a small connection pool rather than per-call connections so the API can
serve concurrent requests without exhausting Postgres.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import DictCursor, execute_values

from app import config

_pool: pg_pool.ThreadedConnectionPool | None = None


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Return the process-wide connection pool, created on first use."""
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def get_connection() -> Iterator[Any]:
    """Borrow a connection from the pool, committing on success."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(dict_rows: bool = True) -> Iterator[Any]:
    """Borrow a cursor. Rows are dict-like by default."""
    with get_connection() as conn:
        factory = DictCursor if dict_rows else None
        with conn.cursor(cursor_factory=factory) as cur:
            yield cur


# ── Schema ────────────────────────────────────────────────────────────────────


def init_db(drop_products: bool = False) -> None:
    """Create the schema, substituting the active embedding dimension.

    pgvector columns are fixed-width, so ``EMBED_DIM`` is baked into the DDL at
    creation time. Switching embedding backend therefore requires
    ``drop_products=True`` and a re-ingest.
    """
    schema_path = config.ROOT_DIR / "sql" / "schema.sql"
    ddl = schema_path.read_text(encoding="utf-8")
    ddl = ddl.replace("{{EMBED_DIM}}", str(config.EMBED_DIM))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if drop_products:
                cur.execute("DROP TABLE IF EXISTS products CASCADE")
            cur.execute(ddl)


def current_embedding_dim() -> int | None:
    """Return the embedding width the ``products`` table was created with.

    Used to detect a mismatch between the database and the configured backend
    before ingestion or retrieval fails in a less obvious way.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod AS dim
            FROM pg_attribute
            WHERE attrelid = 'products'::regclass
            AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
        return int(row["dim"]) if row and row["dim"] and row["dim"] > 0 else None


def assert_dim_matches() -> None:
    """Raise when the table width disagrees with the configured ``EMBED_DIM``."""
    actual = current_embedding_dim()
    if actual is not None and actual != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: the products table was created with "
            f"{actual} dimensions but EMBED_DIM={config.EMBED_DIM}.\n"
            f"Switching embedding backend requires re-ingestion:\n"
            f" uv run python -m ingestion.dlt_pipeline --reset"
        )


def product_count() -> int:
    with get_cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM products")
        return int(cur.fetchone()["n"])


def embedded_count() -> int:
    with get_cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM products WHERE embedding IS NOT NULL")
        return int(cur.fetchone()["n"])


def health() -> dict[str, Any]:
    """Lightweight status summary for the ``/health`` endpoint."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        return {
            "connected": True,
            "products": product_count(),
            "embedded": embedded_count(),
            "embedding_dim": current_embedding_dim(),
        }
    except Exception as exc: # noqa: BLE001 - health must never raise
        return {"connected": False, "error": str(exc)[:200]}


# ── Monitoring writes ─────────────────────────────────────────────────────────

_CONVERSATION_COLUMNS = (
    "id, timestamp, question, rewritten_question, answer, retrieval_method, "
    "rerank_enabled, rewrite_enabled, retrieved_codes, prompt_strategy, "
    "llm_backend, model_used, prompt_tokens, completion_tokens, total_tokens, "
    "eval_total_tokens, estimated_cost_usd, response_time_ms, retrieval_time_ms, "
    "relevance, relevance_explanation, grounding_score"
)


def save_conversation(record: dict[str, Any]) -> None:
    """Insert one conversation log row.

    ``record`` keys mirror the ``conversations`` columns; missing keys fall back
    to sensible defaults so partial failures still produce a monitoring row.
    """
    values = (
        record["id"],
        record.get("timestamp") or datetime.now().astimezone(),
        record["question"],
        record.get("rewritten_question"),
        record["answer"],
        record.get("retrieval_method", "unknown"),
        bool(record.get("rerank_enabled", False)),
        bool(record.get("rewrite_enabled", False)),
        record.get("retrieved_codes") or [],
        record.get("prompt_strategy", "unknown"),
        record.get("llm_backend", ""),
        record.get("model_used", ""),
        int(record.get("prompt_tokens", 0)),
        int(record.get("completion_tokens", 0)),
        int(record.get("total_tokens", 0)),
        int(record.get("eval_total_tokens", 0)),
        float(record.get("estimated_cost_usd", 0.0)),
        float(record.get("response_time_ms", 0.0)),
        float(record.get("retrieval_time_ms", 0.0)),
        record.get("relevance"),
        record.get("relevance_explanation"),
        record.get("grounding_score"),
    )
    placeholders = ", ".join(["%s"] * len(values))
    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            f"INSERT INTO conversations ({_CONVERSATION_COLUMNS}) "
            f"VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING",
            values,
        )


def save_conversations_bulk(records: list[dict[str, Any]]) -> int:
    """Bulk insert conversation rows. Used by the monitoring seed script."""
    if not records:
        return 0
    rows = []
    for r in records:
        rows.append(
            (
                r["id"],
                r.get("timestamp") or datetime.now().astimezone(),
                r["question"],
                r.get("rewritten_question"),
                r["answer"],
                r.get("retrieval_method", "unknown"),
                bool(r.get("rerank_enabled", False)),
                bool(r.get("rewrite_enabled", False)),
                r.get("retrieved_codes") or [],
                r.get("prompt_strategy", "unknown"),
                r.get("llm_backend", ""),
                r.get("model_used", ""),
                int(r.get("prompt_tokens", 0)),
                int(r.get("completion_tokens", 0)),
                int(r.get("total_tokens", 0)),
                int(r.get("eval_total_tokens", 0)),
                float(r.get("estimated_cost_usd", 0.0)),
                float(r.get("response_time_ms", 0.0)),
                float(r.get("retrieval_time_ms", 0.0)),
                r.get("relevance"),
                r.get("relevance_explanation"),
                r.get("grounding_score"),
            )
        )
    with get_cursor(dict_rows=False) as cur:
        execute_values(
            cur,
            f"INSERT INTO conversations ({_CONVERSATION_COLUMNS}) "
            f"VALUES %s ON CONFLICT (id) DO NOTHING",
            rows,
        )
    return len(rows)


def save_feedback(conversation_id: str, feedback: int, timestamp: datetime | None = None) -> None:
    """Record a thumbs up (+1) or thumbs down (-1) against a conversation."""
    if feedback not in (1, -1):
        raise ValueError(f"feedback must be 1 or -1, got {feedback!r}")
    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
            (conversation_id, feedback, timestamp or datetime.now().astimezone()),
        )


def conversation_exists(conversation_id: str) -> bool:
    with get_cursor(dict_rows=False) as cur:
        cur.execute("SELECT 1 FROM conversations WHERE id = %s", (conversation_id,))
        return cur.fetchone() is not None