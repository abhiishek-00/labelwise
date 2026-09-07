"""dlt ingestion pipeline: Open Food Facts -> PostgreSQL + pgvector.

Flow:

    extract Open Food Facts API (cached) or committed snapshot
    |
    normalise quality gates, plausibility checks, deduplication
    |
    load dlt -> staging table (schema inference, load tracking, merge)
    |
    promote staging -> products, populating the tsvector index
    |
    enrich embed documents and write vectors
    |
    index refresh HNSW / GIN statistics

dlt owns extraction and loading, giving schema inference, load-level lineage,
and merge semantics for free. Embeddings are applied afterwards as a separate
enrichment pass, because they are expensive, batched, and resumable
independently of the load.

Usage:
    uv run python -m ingestion.dlt_pipeline # snapshot or cache
    uv run python -m ingestion.dlt_pipeline --refresh # re-fetch from API
    uv run python -m ingestion.dlt_pipeline --reset # rebuild from empty
    uv run python -m ingestion.dlt_pipeline --limit 500 # quick smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Iterator

import dlt

from app import config, db, embeddings
from ingestion import sources
from ingestion.normalize import (
    completeness_report,
    dedup_key,
    normalize_csv_row,
    normalize_product,
)

STAGING_TABLE = "products_staging"

PRODUCT_COLUMNS = [
    "code", "product_name", "brands", "categories", "ingredients_text",
    "allergens", "labels", "countries", "serving_size", "nutriscore_grade",
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "doc_text",
]


# ── Extract + normalise ───────────────────────────────────────────────────────


def _dedup_stream(existing: list[dict[str, Any]] | None = None) -> Any:
    """Return a normaliser that rejects duplicates as records stream past.

    Deduplicating during the stream rather than afterwards keeps memory flat
    and means the target pool size refers to *usable* products.

    ``existing`` primes the seen-sets, so products added later (for example by
    the API top-up) are deduplicated against a pool loaded from cache.
    """
    seen_codes: set[str] = {p["code"] for p in (existing or [])}
    seen_keys: set[str] = {dedup_key(p) for p in (existing or [])}
    dropped = {"duplicate_barcode": 0, "duplicate_product": 0}

    def normalise(raw: dict[str, Any]) -> dict[str, Any] | None:
        product = normalize_product(raw)
        if product is None:
            return None
        if product["code"] in seen_codes:
            dropped["duplicate_barcode"] += 1
            return None
        key = dedup_key(product)
        if key in seen_keys:
            dropped["duplicate_product"] += 1
            return None
        seen_codes.add(product["code"])
        seen_keys.add(key)
        return product

    normalise.dropped = dropped # type: ignore[attr-defined]
    return normalise


def load_products(
    max_products: int,
    *,
    refresh: bool = False,
    use_snapshot: bool = True,
    csv_path: str | None = None,
    pool_factor: int = 3,
) -> list[dict[str, Any]]:
    """Return normalised products, in order of preference.

    1. The committed snapshot - reproduces the published corpus exactly, offline.
    2. A local CSV export - samples the whole corpus without bias or network use.
    3. The streamed bulk export - network fallback, biased toward the file head.

    Over-collecting into a pool is what makes stratification possible: taking
    exactly ``max_products`` in file order would skew the corpus toward whatever
    the source happens to list first.
    """
    if use_snapshot and not refresh and sources.snapshot_exists():
        products = sources.load_snapshot()
        print(f"Loaded {len(products)} products from the committed snapshot")
        return products[:max_products]

    local_csv = sources.find_local_csv(csv_path)
    if local_csv:
        # Cap per bucket well above the per-bucket target so stratification has
        # slack in categories that turn out to be scarce.
        per_bucket = max(400, (max_products * pool_factor) // len(sources.CATEGORY_BUCKETS))
        pool, stats = sources.sample_local_csv(local_csv, normalize_csv_row, per_bucket)
        print(
            f"\n rows read : {stats['rows']:,}\n"
            f" rejected (quality) : {stats['rejected']:,}\n"
            f" duplicates dropped : {stats['duplicate']:,}\n"
            f" usable pool : {len(pool):,}"
        )
    else:
        print(
            "No local CSV export found. Falling back to the network stream.\n"
            " For a faster, unbiased corpus, download:\n"
            " https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz\n"
            " -> data/raw/\n"
        )
        normalise = _dedup_stream()
        pool, stats = sources.load_pool(normalise, max_products * pool_factor, refresh=refresh)
        dropped = normalise.dropped # type: ignore[attr-defined]
        print(
            f"\n records seen : {stats.get('seen', 0):,}\n"
            f" duplicate barcode : {dropped['duplicate_barcode']:,}\n"
            f" duplicate product : {dropped['duplicate_product']:,}\n"
            f" usable pool : {len(pool):,}"
        )

        # The streamed head under-represents some categories; the category
        # -scoped API can fill those gaps. Not needed for the CSV path, which
        # already samples the whole corpus.
        normalise = _dedup_stream(pool)
        target_per_bucket = max(60, max_products // (len(sources.CATEGORY_BUCKETS) * 2))
        extra = sources.topup_thin_buckets(normalise, pool, target_per_bucket)
        if extra:
            pool.extend(extra)
            print(f" pool now {len(pool)} products (+{len(extra)} from API top-up)")

    products, distribution = sources.stratified_sample(pool, max_products)
    print(f"\n stratified sample : {len(products)} products across "
        f"{sum(1 for v in distribution.values() if v)} buckets")
    for bucket, count in distribution.items():
        if count:
            print(f" {bucket:<30s} {count:5d}")

    sources.write_snapshot(products)
    size_mb = sources.SNAPSHOT_PATH.stat().st_size / 1_048_576
    print(f"\n snapshot -> {sources.SNAPSHOT_PATH} ({size_mb:.1f} MB, {len(products)} products)")
    return products


@dlt.resource(name=STAGING_TABLE, write_disposition="merge", primary_key="code")
def products_resource(products: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """dlt resource yielding normalised products.

    ``merge`` on the barcode makes re-runs idempotent and supports incremental
    refresh without duplicating rows.
    """
    yield from products


def run_dlt_load(products: list[dict[str, Any]], reset: bool) -> Any:
    """Load products into the staging table via dlt."""
    pipeline = dlt.pipeline(
        pipeline_name="labelwise_off",
        destination=dlt.destinations.postgres(config.postgres_dsn()),
        dataset_name="public",
        progress="log" if len(products) > 2000 else None,
        dev_mode=False,
    )
    print(f"\nLoading {len(products)} products into '{STAGING_TABLE}' via dlt...")
    info = pipeline.run(
        products_resource(products),
        write_disposition="replace" if reset else "merge",
    )
    return info

# ── Promote + enrich ──────────────────────────────────────────────────────────


def promote_to_products() -> int:
    """Copy staged rows into the application table.

    ``products`` carries the pgvector column and the generated tsvector, which
    dlt does not manage, so promotion is an explicit SQL step.
    """
    columns = ", ".join(PRODUCT_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in PRODUCT_COLUMNS if c != "code")

    with db.get_cursor(dict_rows=False) as cur:
        cur.execute(
            f"""
            INSERT INTO products ({columns})
            SELECT {columns} FROM {STAGING_TABLE}
            ON CONFLICT (code) DO UPDATE SET {updates}
            """
        )
        return cur.rowcount


def embed_pending(batch_size: int | None = None) -> int:
    """Embed products lacking a vector, writing back in batches.

    Only rows with a NULL embedding are processed, so an interrupted run
    resumes instead of repeating completed work.
    """
    with db.get_cursor() as cur:
        cur.execute(
            "SELECT code, doc_text FROM products WHERE embedding IS NULL ORDER BY code"
        )
        pending = [(r["code"], r["doc_text"]) for r in cur.fetchall()]

    if not pending:
        print("All products already embedded.")
        return 0

    size = batch_size or config.EMBED_BATCH_SIZE
    backend = embeddings.get_backend()
    print(
        f"\nEmbedding {len(pending)} products "
        f"(backend={backend.name}, model={backend.model}, dim={config.EMBED_DIM}, "
        f"batch={size}, concurrency={config.EMBED_MAX_CONCURRENCY})..."
    )

    t0 = time.perf_counter()
    done = 0
    # Chunked so progress is durable: a failure loses at most one chunk.
    chunk = size * config.EMBED_MAX_CONCURRENCY * 4
    for start in range(0, len(pending), chunk):
        window = pending[start : start + chunk]
        texts = [text for _, text in window]
        vectors = asyncio.run(embeddings.aembed_texts(texts, batch_size=size))

        with db.get_cursor(dict_rows=False) as cur:
            cur.executemany(
                "UPDATE products SET embedding = %s::vector WHERE code = %s",
                [(json.dumps(v), code) for (code, _), v in zip(window, vectors)],
            )

        done += len(window)
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed else 0.0
        remaining = (len(pending) - done) / rate if rate else 0.0
        print(
            f" {done}/{len(pending)} embedded ({rate:.1f}/s, ~{remaining / 60:.1f} min left)",
            flush=True,
        )

    print(f"Embedded {done} products in {(time.perf_counter() - t0) / 60:.1f} min")
    return done


def refresh_indexes() -> None:
    """Update planner statistics after a bulk load."""
    with db.get_cursor(dict_rows=False) as cur:
        cur.execute("ANALYZE products")


# ── Verification ──────────────────────────────────────────────────────────────


def verify(products: list[dict[str, Any]]) -> bool:
    """Report corpus quality and confirm retrieval primitives work."""
    print("\n" + "=" * 68)
    print("INGESTION REPORT")
    print("=" * 68)

    total = db.product_count()
    embedded = db.embedded_count()
    print(f"products in database : {total}")
    print(f"with embeddings : {embedded}")
    print(f"embedding dimension : {db.current_embedding_dim()}")

    if products:
        print("\nfield completeness (% of products):")
        report = completeness_report(products)
        for field, pct in sorted(report.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(pct / 5)
            print(f" {field:<22s} {pct:5.1f}% {bar}")

    with db.get_cursor() as cur:
        cur.execute(
            """
            SELECT product_name, brands, sugars_100g
            FROM products
            WHERE search_vector @@ plainto_tsquery('english', 'chocolate spread')
            ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', 'chocolate spread')) DESC
            LIMIT 3
            """
        )
        lexical = cur.fetchall()

    print("\nlexical search probe - 'chocolate spread':")
    for row in lexical:
        print(f" {row['product_name'][:45]:<45s} {row['brands'] or '-':<18s} sugar={row['sugars_100g']}")

    vector_ok = False
    if embedded:
        probe = embeddings.embed_query("breakfast cereal low in sugar")
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT product_name, sugars_100g, 1 - (embedding <=> %s::vector) AS similarity
                FROM products WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT 3
                """,
                (json.dumps(probe), json.dumps(probe)),
            )
            rows = cur.fetchall()
        print("\nvector search probe - 'breakfast cereal low in sugar':")
        for row in rows:
            print(f" {row['product_name'][:45]:<45s} sugar={row['sugars_100g']:<6} sim={row['similarity']:.3f}")
        vector_ok = bool(rows)

    ok = total > 0 and embedded == total and bool(lexical) and vector_ok
    print("\n" + ("GATE PASSED" if ok else "GATE FAILED"))
    print("=" * 68)
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LabelWise ingestion pipeline")
    parser.add_argument("--limit", type=int, default=config.MAX_PRODUCTS,
                        help="maximum products to ingest")
    parser.add_argument("--csv", type=str, default=None,
                        help="path to the Open Food Facts CSV export "
                            "(default: data/raw/en.openfoodfacts.org.products.csv.gz)")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the snapshot from source, ignoring the committed copy")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate the products table first")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="load products without generating vectors")
    args = parser.parse_args(argv)

    print("=" * 68)
    print(f"LabelWise ingestion | limit={args.limit} reset={args.reset} "
        f"refresh={args.refresh}")
    print("=" * 68)

    db.init_db(drop_products=args.reset)
    if not args.reset:
        db.assert_dim_matches()

    products = load_products(args.limit, refresh=args.refresh, csv_path=args.csv)
    if not products:
        print("No products to ingest.", file=sys.stderr)
        return 1

    run_dlt_load(products, reset=args.reset)
    promoted = promote_to_products()
    print(f"Promoted {promoted} rows into 'products'")

    if not args.skip_embeddings:
        embed_pending()

    refresh_indexes()
    return 0 if verify(products) else 1


if __name__ == "__main__":
    raise SystemExit(main())