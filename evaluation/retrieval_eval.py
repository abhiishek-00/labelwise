"""Retrieval evaluation: Hit Rate and MRR across retrieval strategies.

Method:

1. Split the ground truth into validation (70%) and test (30%), seeded.
2. Embed every unique query **once** and cache the vectors to disk. Retrieval
sweeps then run entirely offline, which makes the benchmark reproducible and
free to re-run despite a non-deterministic LLM backend.
3. Sweep the hybrid ``alpha`` on validation only.
4. Compare strategies on validation, select the best, and report it once on the
held-out test set.

Tuning and selection happen on validation; the test set is touched only for the
final number. Selecting on the test set would report an optimistically biased
result.

Re-ranking and rewriting are evaluated on subsets, because each costs an LLM
call per query and the ranking signal saturates well before the full set.

Usage:
    uv run python -m evaluation.retrieval_eval
    uv run python -m evaluation.retrieval_eval --rerank-sample 150
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from app import config, db, embeddings, retrieval
from app.models import RetrievalConfig

GT_PATH = config.EVALUATION_DIR / "ground_truth.csv"
CONVERSATIONAL_PATH = config.EVALUATION_DIR / "ground_truth_conversational.csv"
QUERY_VECTOR_CACHE = config.EVALUATION_DIR / "query_vectors.jsonl.gz"

RESULTS_PATH = config.EVALUATION_DIR / "retrieval_results.csv"
ALPHA_SWEEP_PATH = config.EVALUATION_DIR / "retrieval_alpha_sweep.csv"
REWRITE_PATH = config.EVALUATION_DIR / "retrieval_rewrite_results.csv"

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.7


# ── Metrics ───────────────────────────────────────────────────────────────────


def hit_rate(relevance: list[list[bool]], k: int) -> float:
    """Fraction of queries with at least one relevant document in the top k."""
    if not relevance:
        return 0.0
    return sum(1 for row in relevance if any(row[:k])) / len(relevance)


def mrr(relevance: list[list[bool]], k: int) -> float:
    """Mean reciprocal rank of the first relevant document within the top k."""
    if not relevance:
        return 0.0
    total = 0.0
    for row in relevance:
        for rank, is_relevant in enumerate(row[:k], start=1):
            if is_relevant:
                total += 1.0 / rank
                break
    return total / len(relevance)


def score(relevance: list[list[bool]]) -> dict[str, float]:
    return {
        "hit_rate@5": round(hit_rate(relevance, 5), 4),
        "hit_rate@10": round(hit_rate(relevance, 10), 4),
        "mrr@5": round(mrr(relevance, 5), 4),
        "mrr@10": round(mrr(relevance, 10), 4),
    }


# ── Data ──────────────────────────────────────────────────────────────────────


def load_ground_truth(path: Path = GT_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No ground truth at {path}. Generate it with:\n"
            f" uv run python -m evaluation.generate_ground_truth"
        )
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split(
    rows: list[dict[str, str]], seed: int = RANDOM_SEED
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Shuffle and split into validation and test.

    Split by row rather than by product: questions from the same product may
    land in both halves, which is acceptable because the retrieval target is
    the question, not the product.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * VALIDATION_FRACTION)
    return shuffled[:cut], shuffled[cut:]


def build_query_cache(queries: list[str], refresh: bool = False) -> dict[str, list[float]]:
    """Embed unique queries once and cache them to disk.

    This is what makes the sweep cheap: every strategy that needs a query vector
    reuses these, so a full comparison costs zero embedding calls after the
    first run.
    """
    cache: dict[str, list[float]] = {}
    if QUERY_VECTOR_CACHE.exists() and not refresh:
        with gzip.open(QUERY_VECTOR_CACHE, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                cache[record["q"]] = record["v"]
        print(f"Loaded {len(cache)} cached query vectors")

    missing = [q for q in dict.fromkeys(queries) if q not in cache]
    if missing:
        print(f"Embedding {len(missing)} new queries...")
        vectors = asyncio.run(embeddings.aembed_texts(missing))
        cache.update(dict(zip(missing, vectors)))
        QUERY_VECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(QUERY_VECTOR_CACHE, "wt", encoding="utf-8") as fh:
            for query, vector in cache.items():
                fh.write(json.dumps({"q": query, "v": vector}) + "\n")
        print(f" cached {len(cache)} query vectors -> {QUERY_VECTOR_CACHE.name}")

    return cache


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    rows: list[dict[str, str]],
    search_fn: Callable[[dict[str, str]], list[str]],
    depth: int = 10,
) -> tuple[dict[str, float], list[list[bool]]]:
    """Run a retrieval function over the rows and score the rankings."""
    relevance: list[list[bool]] = []
    for row in rows:
        codes = search_fn(row)[:depth]
        relevance.append([code == row["code"] for code in codes])
    return score(relevance), relevance


def make_search_fn(
    cfg: RetrievalConfig,
    cache: dict[str, list[float]],
    query_field: str = "question",
) -> Callable[[dict[str, str]], list[str]]:
    def search_fn(row: dict[str, str]) -> list[str]:
        query = row[query_field]
        outcome = retrieval.search(query, cfg, query_vector=cache.get(query))
        return outcome.codes

    return search_fn


def run_alpha_sweep(
    rows: list[dict[str, str]],
    cache: dict[str, list[float]],
    alphas: list[float],
) -> list[dict[str, Any]]:
    """Sweep the lexical/vector blend weight on validation data."""
    print(f"\nSweeping hybrid alpha on {len(rows)} validation questions")
    print(f"{'alpha':>7} {'hit@5':>8} {'hit@10':>8} {'mrr@5':>8} {'mrr@10':>8}")
    results = []
    for alpha in alphas:
        cfg = RetrievalConfig(method="hybrid", alpha=alpha, top_k=10, rerank=False)
        metrics, _ = evaluate(rows, make_search_fn(cfg, cache))
        results.append({"alpha": alpha, **metrics})
        print(
            f"{alpha:7.2f} {metrics['hit_rate@5']:8.4f} {metrics['hit_rate@10']:8.4f} "
            f"{metrics['mrr@5']:8.4f} {metrics['mrr@10']:8.4f}"
        )
    return results


def run_rewrite_eval(
    cfg: RetrievalConfig,
    cache: dict[str, list[float]],
    limit: int,
) -> list[dict[str, Any]]:
    """Measure query rewriting on the conversational slice.

    Rewriting cannot be measured on the main ground truth: those questions are
    standalone and keyword-rich by construction (D18), so rewriting them is a
    no-op and would produce a meaningless null result. The conversational set
    exists precisely for this - turn 2 ("does it have natural flavors too?") is
    unresolvable without turn 1.

    Three conditions are compared on identical rows:
    ``turn_2_raw`` the vague follow-up alone - the failure case
    ``turn_2_rewritten`` the follow-up rewritten using turn 1 as history
    ``turn_1`` the standalone question, as an upper reference
    """
    if not CONVERSATIONAL_PATH.exists():
        print(f"\nNo conversational ground truth at {CONVERSATIONAL_PATH}; skipping.")
        return []

    with open(CONVERSATIONAL_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:limit]
    if not rows:
        return []

    print(f"\nEvaluating query rewriting on {len(rows)} conversational examples")

    rewrite_cfg = replace(cfg, rewrite=True)

    def raw_fn(row: dict[str, str]) -> list[str]:
        return retrieval.search(row["turn_2"], cfg).codes

    def turn1_fn(row: dict[str, str]) -> list[str]:
        return retrieval.search(row["turn_1"], cfg, query_vector=cache.get(row["turn_1"])).codes

    def rewritten_search(row: dict[str, str]) -> list[str]:
        return retrieval.search(row["turn_2"], rewrite_cfg, history=[row["turn_1"]]).codes

    conditions = [
        ("turn_2_raw", raw_fn),
        ("turn_2_rewritten", rewritten_search),
        ("turn_1_reference", turn1_fn),
    ]

    results: list[dict[str, Any]] = []
    print(f"{'condition':<20} {'hit@5':>8} {'hit@10':>8} {'mrr@5':>8} {'mrr@10':>8}")
    for name, fn in conditions:
        t0 = time.perf_counter()
        metrics, _ = evaluate(rows, fn)
        elapsed = time.perf_counter() - t0
        results.append({"condition": name, "n": len(rows), **metrics,
                        "seconds": round(elapsed, 1)})
        print(f"{name:<20} {metrics['hit_rate@5']:8.4f} {metrics['hit_rate@10']:8.4f} "
            f"{metrics['mrr@5']:8.4f} {metrics['mrr@10']:8.4f}")
    return results

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f" wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies")
    parser.add_argument("--rerank-sample", type=int, default=150,
                        help="questions used for re-ranking evaluation (LLM call each)")
    parser.add_argument("--rewrite-sample", type=int, default=70,
                        help="conversational examples used for query-rewrite evaluation")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="re-embed all queries")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    if db.product_count() == 0:
        print("No products in the database. Run ingestion first.")
        return 1

    ground_truth = load_ground_truth()
    validation, test = split(ground_truth, args.seed)
    print(f"Ground truth: {len(ground_truth)} questions "
        f"({len(validation)} validation / {len(test)} test)")

    cache = build_query_cache([row["question"] for row in ground_truth], args.refresh_cache)

    # 1. Alpha sweep on validation.
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep = run_alpha_sweep(validation, cache, alphas)
    write_csv(ALPHA_SWEEP_PATH, sweep)
    best_alpha = max(sweep, key=lambda r: r["mrr@5"])["alpha"]
    print(f"\nBest alpha on validation: {best_alpha}")

    # 2. Compare strategies on validation.
    strategies: list[tuple[str, RetrievalConfig]] = [
        ("lexical", RetrievalConfig(method="lexical", top_k=10)),
        ("vector", RetrievalConfig(method="vector", top_k=10)),
        ("hybrid_weighted", RetrievalConfig(method="hybrid", alpha=best_alpha, top_k=10)),
        ("hybrid_rrf", RetrievalConfig(method="hybrid", hybrid_strategy="rrf", top_k=10)),
    ]

    print(f"\nComparing strategies on {len(validation)} validation questions")
    print(f"{'strategy':<22} {'hit@5':>8} {'hit@10':>8} {'mrr@5':>8} {'mrr@10':>8} {'sec':>7}")
    rows: list[dict[str, Any]] = []
    for name, cfg in strategies:
        t0 = time.perf_counter()
        metrics, _ = evaluate(validation, make_search_fn(cfg, cache))
        elapsed = time.perf_counter() - t0
        rows.append({"strategy": name, "split": "validation", "n": len(validation),
                    **metrics, "seconds": round(elapsed, 1)})
        print(f"{name:<22} {metrics['hit_rate@5']:8.4f} {metrics['hit_rate@10']:8.4f} "
            f"{metrics['mrr@5']:8.4f} {metrics['mrr@10']:8.4f} {elapsed:7.1f}")

    # 3. Re-ranking on a subset - one LLM call per query.
    sample = validation[: args.rerank_sample]
    print(f"\nEvaluating re-ranking on {len(sample)} questions (LLM call per query)")
    rerank_cfg = RetrievalConfig(method="hybrid", alpha=best_alpha, top_n=20, top_k=10,
                                rerank=True)
    t0 = time.perf_counter()
    rerank_metrics, _ = evaluate(sample, make_search_fn(rerank_cfg, cache))
    rerank_elapsed = time.perf_counter() - t0

    baseline_cfg = RetrievalConfig(method="hybrid", alpha=best_alpha, top_k=10)
    baseline_metrics, _ = evaluate(sample, make_search_fn(baseline_cfg, cache))

    print(f" hybrid (same subset) : mrr@5={baseline_metrics['mrr@5']:.4f} "
        f"hit@5={baseline_metrics['hit_rate@5']:.4f}")
    print(f" hybrid + rerank : mrr@5={rerank_metrics['mrr@5']:.4f} "
        f"hit@5={rerank_metrics['hit_rate@5']:.4f} ({rerank_elapsed:.0f}s)")

    rows.append({"strategy": "hybrid_weighted_subset", "split": "validation",
                "n": len(sample), **baseline_metrics, "seconds": 0.0})
    rows.append({"strategy": "hybrid_weighted+rerank", "split": "validation",
                "n": len(sample), **rerank_metrics, "seconds": round(rerank_elapsed, 1)})

    # 4. Final held-out test, using the configuration chosen on validation.
    use_rerank = rerank_metrics["mrr@5"] > baseline_metrics["mrr@5"]
    final_cfg = RetrievalConfig(method="hybrid", alpha=best_alpha, top_n=20, top_k=10,
                                rerank=use_rerank)
    print(f"\nSelected configuration: {final_cfg.label()}")
    print(f"Evaluating on {len(test)} held-out test questions")

    test_rows = test if not use_rerank else test[: args.rerank_sample]
    final_metrics, _ = evaluate(test_rows, make_search_fn(final_cfg, cache))
    rows.append({"strategy": f"FINAL {final_cfg.label()}", "split": "test",
                "n": len(test_rows), **final_metrics, "seconds": 0.0})

    # Baselines on the same test split, for a like-for-like comparison.
    for name, cfg in strategies:
        metrics, _ = evaluate(test, make_search_fn(cfg, cache))
        rows.append({"strategy": name, "split": "test", "n": len(test),
                    **metrics, "seconds": 0.0})

    # Re-ranking is capped at --rerank-sample because every query costs an LLM
    # call, so the FINAL row covers fewer questions than the baselines above.
    # Comparing them directly would be comparing two different test sets, so the
    # winning baseline is repeated on exactly the rows the FINAL row used.
    if use_rerank and len(test_rows) < len(test):
        no_rerank_cfg = replace(final_cfg, rerank=False)
        subset_metrics, _ = evaluate(test_rows, make_search_fn(no_rerank_cfg, cache))
        rows.append({"strategy": f"{no_rerank_cfg.label()} (same rows as FINAL)",
                    "split": "test", "n": len(test_rows), **subset_metrics,
                    "seconds": 0.0})

    rewrite_rows = run_rewrite_eval(
        RetrievalConfig(method="hybrid", alpha=best_alpha, top_k=10),
        cache,
        args.rewrite_sample,
    )
    if rewrite_rows:
        write_csv(REWRITE_PATH, rewrite_rows)

    write_csv(RESULTS_PATH, rows)

    print("\n" + "=" * 74)
    print("HELD-OUT TEST RESULTS")
    print("=" * 74)
    print(f"{'strategy':<32} {'n':>5} {'hit@5':>8} {'hit@10':>8} {'mrr@5':>8} {'mrr@10':>8}")
    for row in rows:
        if row["split"] == "test":
            print(f"{row['strategy']:<32} {row['n']:5d} {row['hit_rate@5']:8.4f} "
                f"{row['hit_rate@10']:8.4f} {row['mrr@5']:8.4f} {row['mrr@10']:8.4f}")
    print("=" * 74)
    print(f"\nSet these in .env:\n HYBRID_ALPHA={best_alpha}\n ENABLE_RERANK={str(use_rerank).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())