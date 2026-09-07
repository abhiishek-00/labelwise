"""Open Food Facts data sources.

Two acquisition paths:

**Bulk export (default).** Streams the official gzipped JSONL export, decoding
incrementally and stopping once enough products are collected. Open Food Facts
explicitly asks bulk consumers to use the exports rather than the API, and the
API is rate-limited to ~10 search requests/minute and returns 503 under
sustained use. The export is ~12.8 GB in full, but because it is line-delimited
gzip it can be read incrementally: collecting a few thousand usable products
needs only tens of megabytes.

**Search API (alternative).** Category-scoped queries, rate-limited and
retried. Retained because it is the natural path for incremental top-up of an
existing corpus, and it is what a reader would expect to see attempted.

Records fetched by either path are cached to disk, and the normalised result is
written to a committed snapshot, so re-running ingestion or cloning fresh needs
no network access and yields an identical corpus.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from app import config
from app.retry import RateLimiter, with_retry

# ── Local CSV export (preferred) ──────────────────────────────────────────────

# Default location for the manually downloaded Open Food Facts CSV export:
# https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
LOCAL_CSV_CANDIDATES = [
    config.DATA_DIR / "raw" / "en.openfoodfacts.org.products.csv.gz",
    config.DATA_DIR / "raw" / "en.openfoodfacts.org.products.csv",
]


def find_local_csv(explicit: str | None = None) -> Path | None:
    """Locate the CSV export, preferring an explicitly supplied path."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for candidate in LOCAL_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _open_csv(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "rt", encoding="utf-8", errors="replace", newline="")


def sample_local_csv(
    path: Path,
    normalizer: Callable[[dict[str, Any]], dict[str, Any] | None],
    per_bucket_cap: int = 1200,
    progress_every: int = 250_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample products across the entire CSV export using reservoir sampling.

    The export holds several million products - far more than needed and far
    more than fits comfortably in memory. Reservoir sampling keeps a bounded,
    uniformly random sample *per category bucket* in a single pass.

    This is what fixes the sampling bias of the streaming approach: reading the
    head of a barcode-ordered file over-represents whichever products happen to
    be listed first, whereas a reservoir over the whole file does not. Doing it
    per bucket also guarantees rare categories survive, which uniform sampling
    across the whole file would not.
    """
    # Some ingredient fields exceed the default field-size limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    reservoirs: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    seen_codes: set[str] = set()
    seen_keys: set[str] = set()
    rng = random.Random(RESERVOIR_SEED)

    stats = {"rows": 0, "rejected": 0, "duplicate": 0, "accepted": 0}
    size_mb = path.stat().st_size / (1 << 20)
    print(f"Reading {path.name} ({size_mb:.0f} MB), sampling up to {per_bucket_cap} per category")
    t0 = time.perf_counter()

    with _open_csv(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stats["rows"] += 1

            if stats["rows"] % progress_every == 0:
                pooled = sum(len(v) for v in reservoirs.values())
                print(
                    f" {stats['rows']:>9,} rows {stats['accepted']:>7,} usable "
                    f"{pooled:>6,} pooled ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )

            product = normalizer(row)
            if product is None:
                stats["rejected"] += 1
                continue

            code = product["code"]
            key = dedup_key_of(product)
            if code in seen_codes or key in seen_keys:
                stats["duplicate"] += 1
                continue
            seen_codes.add(code)
            seen_keys.add(key)
            stats["accepted"] += 1

            bucket = categorise(product)
            counts[bucket] = counts.get(bucket, 0) + 1
            reservoir = reservoirs.setdefault(bucket, [])

            if len(reservoir) < per_bucket_cap:
                reservoir.append(product)
            else:
                # Standard reservoir sampling: replace with probability
                # cap/n, giving every row an equal chance of selection.
                j = rng.randrange(counts[bucket])
                if j < per_bucket_cap:
                    reservoir[j] = product

    pool = [product for reservoir in reservoirs.values() for product in reservoir]
    elapsed = time.perf_counter() - t0
    print(
        f" done: {stats['rows']:,} rows in {elapsed:.0f}s -> "
        f"{stats['accepted']:,} usable, {len(pool):,} pooled across {len(reservoirs)} buckets"
    )
    return pool, stats


# ── Bulk export (network fallback) ────────────────────────────────────────────

BULK_EXPORT_URL = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"

# Safety valve: stop streaming after this much compressed data regardless of
# how many products have been accepted, so a bad filter cannot pull 12.8 GB.
MAX_DOWNLOAD_MB = 600

# Categories are hierarchical and noisy; these broad buckets are used to spread
# the corpus across food types rather than taking whatever the file order gives.
CATEGORY_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("breakfast cereals", ("breakfast cereal", "muesli", "granola")),
    ("biscuits and cakes", ("biscuit", "cake", "cookie", "pastr")),
    ("chocolate and confectionery", ("chocolate", "candy", "confection", "sweets")),
    ("spreads", ("spread", "jam", "honey", "butter")),
    ("dairy", ("yogurt", "yoghurt", "cheese", "milk", "cream", "dairy")),
    ("beverages", ("beverage", "soda", "juice", "water", "drink", "tea", "coffee")),
    ("bread and bakery", ("bread", "bakery", "viennoiserie", "toast")),
    ("snacks", ("crisp", "chip", "snack", "nut", "popcorn")),
    ("meals and prepared", ("meal", "pizza", "sandwich", "prepared", "soup")),
    ("meat and fish", ("meat", "fish", "seafood", "poultry", "sausage", "ham")),
    ("pasta rice grains", ("pasta", "rice", "grain", "noodle", "cereal grain", "flour")),
    ("sauces and condiments", ("sauce", "condiment", "mayonnaise", "ketchup", "mustard", "vinegar")),
    ("frozen", ("frozen", "ice cream", "sorbet")),
    ("fruits and vegetables", ("fruit", "vegetable", "legume", "salad")),
    ("baby and dietary", ("baby", "infant", "dietetic", "diet")),
]

OTHER_BUCKET = "other"

# Fixed so the sampled corpus is reproducible from the same export file.
RESERVOIR_SEED = 42


def dedup_key_of(product: dict[str, Any]) -> str:
    """Deduplication identity, re-exported so sources need not import normalize."""
    from ingestion.normalize import dedup_key

    return dedup_key(product)

def categorise(product: dict[str, Any]) -> str:
    """Assign a product to a broad bucket for stratified sampling."""
    text = (product.get("categories") or "").lower()
    if not text:
        return OTHER_BUCKET
    for bucket, keywords in CATEGORY_BUCKETS:
        if any(keyword in text for keyword in keywords):
            return bucket
    return OTHER_BUCKET


def stream_bulk_export(
    normalizer: Callable[[dict[str, Any]], dict[str, Any] | None],
    target_pool: int,
    *,
    url: str = BULK_EXPORT_URL,
    max_download_mb: int = MAX_DOWNLOAD_MB,
    progress_every_mb: int = 25,
    max_resumes: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stream the export, normalising on the fly, until ``target_pool`` products.

    Decompression is incremental, so the connection closes as soon as enough
    products are found and the remainder of the multi-gigabyte file is never
    transferred.

    Long-lived connections through an intercepting proxy are routinely severed
    mid-stream. A dropped connection is resumed with an HTTP ``Range`` request
    from the last received byte, and the gzip decompressor is carried across the
    resume so the logical stream is uninterrupted. Without this, a drop at 100 MB
    discards everything already parsed.

    Prefer :func:`sample_local_csv` where possible: it is faster, unbiased across
    the whole corpus, and needs no network at all.
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buffer = b""
    accepted: list[dict[str, Any]] = []
    stats = {"seen": 0, "parse_errors": 0, "rejected": 0, "accepted": 0,
            "bytes": 0, "resumes": 0}
    byte_cap = max_download_mb * (1 << 20)
    next_report = progress_every_mb * (1 << 20)
    t0 = time.perf_counter()

    print(
        f"Streaming Open Food Facts bulk export "
        f"(target pool {target_pool}, cap {max_download_mb} MB)"
    )

    def consume(chunk: bytes) -> None:
        nonlocal buffer
        stats["bytes"] += len(chunk)
        buffer += decompressor.decompress(chunk)
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            stats["seen"] += 1
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                stats["parse_errors"] += 1
                continue
            product = normalizer(raw)
            if product is None:
                stats["rejected"] += 1
                continue
            accepted.append(product)
            stats["accepted"] += 1

    done = False
    while not done and stats["resumes"] <= max_resumes:
        headers = {"User-Agent": config.OFF_USER_AGENT}
        if stats["bytes"]:
            # Resume exactly where the previous connection stopped.
            headers["Range"] = f"bytes={stats['bytes']}-"

        try:
            with httpx.stream(
                "GET", url, verify=config.OFF_VERIFY_SSL, follow_redirects=True,
                timeout=180.0, headers=headers,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1 << 20):
                    consume(chunk)
                    if stats["bytes"] >= next_report:
                        next_report += progress_every_mb * (1 << 20)
                        rate = stats["accepted"] / max(time.perf_counter() - t0, 1e-6)
                        print(
                            f" {stats['accepted']:6d} accepted / {stats['seen']:7d} seen "
                            f"({stats['bytes'] / (1 << 20):5.0f} MB, {rate:.0f}/s)",
                            flush=True,
                        )
                    if len(accepted) >= target_pool or stats["bytes"] >= byte_cap:
                        done = True
                        break
                else:
                    done = True # server closed cleanly
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.TimeoutException,
                httpx.ConnectError) as exc:
            if not stats["bytes"]:
                raise
            stats["resumes"] += 1
            print(
                f" connection dropped at {stats['bytes'] / (1 << 20):.0f} MB "
                f"({type(exc).__name__}); resuming "
                f"[{stats['resumes']}/{max_resumes}], {len(accepted)} products kept",
                flush=True,
            )
            time.sleep(min(30.0, 2.0 * stats["resumes"]))

    elapsed = time.perf_counter() - t0
    print(
        f" done: {stats['accepted']} products from {stats['seen']} records, "
        f"{stats['bytes'] / (1 << 20):.0f} MB in {elapsed:.0f}s "
        f"({stats['resumes']} resume(s))"
    )
    return accepted, stats


def stratified_sample(
    products: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select ``limit`` products spread across category buckets.

    The export is ordered by barcode, which correlates with country and
    manufacturer. Taking the first N would skew the corpus and weaken the
    comparison and constraint questions the assistant is meant to answer.
    Round-robin selection across buckets keeps category coverage broad.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        buckets[categorise(product)].append(product)

    # Round-robin across buckets, so scarce categories are not crowded out.
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]))
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for _, items in ordered:
            if index < len(items):
                selected.append(items[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1

    distribution = {name: 0 for name, _ in ordered}
    for product in selected:
        distribution[categorise(product)] += 1
    return selected, dict(sorted(distribution.items(), key=lambda kv: -kv[1]))


# ── Search API (alternative path) ─────────────────────────────────────────────

DEFAULT_CATEGORIES: list[str] = [
    "Breakfast cereals", "Biscuits and cakes", "Chocolates", "Spreads",
    "Yogurts", "Cheeses", "Breads", "Sodas", "Fruit juices", "Crisps",
    "Pasta", "Sauces", "Plant-based foods", "Snacks", "Canned foods",
]

FIELDS = ",".join([
    "code", "product_name", "brands", "categories_tags_en", "ingredients_text",
    "allergens_tags", "labels_tags", "countries_tags", "serving_size",
    "nutriscore_grade", "nutriments",
])

PAGE_SIZE = 100
# Open Food Facts documents 10 search requests/minute; stay comfortably under.
OFF_RATE_LIMIT_RPS = 1.0 / 7.0

# Committed to git: the exact corpus every published result was measured on.
# Plain CSV rather than compressed JSON so reviewers can inspect it on GitHub.
SNAPSHOT_PATH = config.SNAPSHOT_DIR / "products.csv"
RAW_CACHE_PATH = config.CACHE_DIR / "off_raw.jsonl.gz"
POOL_CACHE_PATH = config.CACHE_DIR / "pool.jsonl.gz"


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=config.OFF_BASE_URL,
        headers={"User-Agent": config.OFF_USER_AGENT, "Accept": "application/json"},
        timeout=60.0,
        follow_redirects=True,
        verify=config.OFF_VERIFY_SSL,
    )


def fetch_category(
    client: httpx.Client,
    category: str,
    max_products: int,
    limiter: RateLimiter,
) -> list[dict[str, Any]]:
    """Fetch up to ``max_products`` raw records for one category."""

    @with_retry(
        max_attempts=config.RETRY_MAX_ATTEMPTS,
        base_delay=max(config.RETRY_BASE_DELAY, 2.0),
        max_delay=config.RETRY_MAX_DELAY,
        on_retry=lambda a, d, e: print(
            f" retry {a} after {d:.1f}s "
            f"({getattr(getattr(e, 'response', None), 'status_code', type(e).__name__)})",
            flush=True,
        ),
    )
    def _page(page: int) -> dict[str, Any]:
        limiter.acquire()
        resp = client.get(
            "/api/v2/search",
            params={
                "categories_tags_en": category,
                "fields": FIELDS,
                "page_size": PAGE_SIZE,
                "page": page,
            },
        )
        resp.raise_for_status()
        return resp.json()

    collected: list[dict[str, Any]] = []
    page = 1
    while len(collected) < max_products:
        try:
            body = _page(page)
        except Exception as exc: # noqa: BLE001 - one bad category must not abort the run
            print(f" ! {category} page {page} failed: {str(exc)[:120]}", flush=True)
            break

        products = body.get("products") or []
        if not products:
            break
        collected.extend(products)
        if len(products) < PAGE_SIZE:
            break
        page += 1

    return collected[:max_products]

def fetch_from_api(
    max_products: int,
    categories: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream raw records from the Open Food Facts search API."""
    cats = categories or DEFAULT_CATEGORIES
    per_category = max(PAGE_SIZE, int((max_products * 2.5) / len(cats)) + 1)
    limiter = RateLimiter(OFF_RATE_LIMIT_RPS)

    print(
        f"Fetching from the Open Food Facts API: {len(cats)} categories, "
        f"~{per_category} each, {1 / OFF_RATE_LIMIT_RPS:.0f}s between requests"
    )

    with _client() as client:
        for index, category in enumerate(cats, start=1):
            t0 = time.perf_counter()
            records = fetch_category(client, category, per_category, limiter)
            print(
                f" [{index:2d}/{len(cats)}] {category:<24s} {len(records):4d} records "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
            yield from records


# ── Pool assembly ─────────────────────────────────────────────────────────────


def bucket_counts(products: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name, _ in CATEGORY_BUCKETS}
    counts[OTHER_BUCKET] = 0
    for product in products:
        counts[categorise(product)] += 1
    return counts


# Maps a bucket to the Open Food Facts category names used to top it up.
BUCKET_TO_OFF_CATEGORIES: dict[str, list[str]] = {
    "breakfast cereals": ["Breakfast cereals", "Mueslis"],
    "bread and bakery": ["Breads", "Sandwich breads"],
    "fruits and vegetables": ["Fruits", "Vegetables", "Legumes"],
    "pasta rice grains": ["Pastas", "Rices"],
    "baby and dietary": ["Baby foods"],
    "meat and fish": ["Fishes", "Hams"],
    "spreads": ["Spreads", "Jams"],
    "beverages": ["Fruit juices", "Sodas"],
    "dairy": ["Yogurts", "Cheeses"],
}


def topup_thin_buckets(
    normalizer: Callable[[dict[str, Any]], dict[str, Any] | None],
    pool: list[dict[str, Any]],
    target_per_bucket: int,
) -> list[dict[str, Any]]:
    """Fetch extra products for under-represented categories via the API.

    The bulk export is ordered by barcode and only its head is read, so some
    food categories are barely present. The API is category-scoped and fills
    those gaps precisely.

    Best-effort by design: the API is rate-limited and returns 503 under load,
    so any failure leaves that category thin rather than aborting ingestion.
    """
    counts = bucket_counts(pool)
    thin = {
        bucket: target_per_bucket - count
        for bucket, count in counts.items()
        if count < target_per_bucket and bucket in BUCKET_TO_OFF_CATEGORIES
    }
    if not thin:
        return []

    print(f"\nTopping up {len(thin)} under-represented categories via the API")
    limiter = RateLimiter(OFF_RATE_LIMIT_RPS)
    added: list[dict[str, Any]] = []

    with _client() as client:
        for bucket, shortfall in sorted(thin.items(), key=lambda kv: -kv[1]):
            categories = BUCKET_TO_OFF_CATEGORIES[bucket]
            # Over-fetch: roughly half of raw records fail the quality gates.
            per_category = min(500, max(PAGE_SIZE, int(shortfall * 2.5 / len(categories)) + 1))
            got = 0
            for category in categories:
                records = fetch_category(client, category, per_category, limiter)
                for raw in records:
                    product = normalizer(raw)
                    if product is not None:
                        added.append(product)
                        got += 1
            status = "ok" if got else "unavailable"
            print(f" {bucket:<26s} +{got:4d} (needed {shortfall}, {status})", flush=True)

    return added


def load_pool(
    normalizer: Callable[[dict[str, Any]], dict[str, Any] | None],
    target_pool: int,
    *,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return the normalised candidate pool, using a disk cache when available.

    Caching the pool separates the expensive download from the cheap sampling
    step, so the corpus can be re-stratified without re-fetching.
    """
    if POOL_CACHE_PATH.exists() and not refresh:
        pool = read_jsonl_gz(POOL_CACHE_PATH)
        print(f"Loaded pool of {len(pool)} products from cache: {POOL_CACHE_PATH.name}")
        return pool, {"seen": 0, "cached": len(pool)}

    pool, stats = stream_bulk_export(normalizer, target_pool)
    write_jsonl_gz(POOL_CACHE_PATH, pool)
    print(f" cached pool -> {POOL_CACHE_PATH}")
    return pool, stats


# ── Snapshot I/O ──────────────────────────────────────────────────────────────


def write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# doc_text is deliberately excluded: it is a deterministic function of these
# columns, so storing it would double the file size and allow the committed
# copy to drift from the builder.
SNAPSHOT_COLUMNS = [
    "code", "product_name", "brands", "categories", "ingredients_text",
    "allergens", "labels", "countries", "serving_size", "nutriscore_grade",
    "energy_kcal_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
]

_FLOAT_COLUMNS = {c for c in SNAPSHOT_COLUMNS if c.endswith("_100g")}


def write_snapshot(products: list[dict[str, Any]], path: Path = SNAPSHOT_PATH) -> Path:
    """Write the corpus as a committed CSV, sorted for a stable diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(products, key=lambda p: p["code"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for product in ordered:
            writer.writerow({c: product.get(c) for c in SNAPSHOT_COLUMNS})
    return path


def snapshot_exists() -> bool:
    return SNAPSHOT_PATH.exists()


def load_snapshot(path: Path = SNAPSHOT_PATH) -> list[dict[str, Any]]:
    """Load the committed product snapshot and rebuild derived fields.

    This is the reproducible path: a fresh clone ingests the exact corpus the
    published results were measured on, with no downloads at all.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No snapshot at {path}. Build one from the CSV export with:\n"
            f" uv run python -m ingestion.dlt_pipeline --csv data/raw/"
            f"en.openfoodfacts.org.products.csv.gz"
        )

    from ingestion.normalize import build_doc_text

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    products: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product: dict[str, Any] = {}
            for column in SNAPSHOT_COLUMNS:
                value = row.get(column)
                if value == "" or value is None:
                    product[column] = None
                elif column in _FLOAT_COLUMNS:
                    product[column] = float(value)
                else:
                    product[column] = value
            product["doc_text"] = build_doc_text(product)
            products.append(product)
    return products