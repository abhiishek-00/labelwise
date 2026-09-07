"""Generation evaluation: compare prompt strategies with an LLM judge.

Compares the two prompt strategies from :mod:`app.prompts` on the same
questions, using the same retrieval configuration, so the only variable is the
prompt.

Two complementary measures:

**LLM-as-a-judge relevance** - RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.
Captures whether the answer addresses the question, but a judge cannot reliably
detect a fabricated nutrition value; it sees only the question and the answer.

**Numeric grounding** - the fraction of numbers in the answer traceable to the
retrieved context. This is the measure that catches hallucinated nutrition
figures, which is the failure mode that matters most in this domain: a wrong
number reads as authoritative in a way a vague answer does not.

Reporting only judge relevance would miss confidently-wrong numeric answers
entirely, so both are reported.

Usage:
    uv run python -m evaluation.rag_eval
    uv run python -m evaluation.rag_eval --sample 200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app import config, db, llm, prompts, rag, retrieval
from app.models import RetrievalConfig
from evaluation.retrieval_eval import load_ground_truth, split

RESULTS_PATH = config.EVALUATION_DIR / "rag_results.csv"
SUMMARY_PATH = config.EVALUATION_DIR / "rag_summary.csv"

RELEVANCE_LABELS = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT", "UNKNOWN"]


async def generate_answers(
    questions: list[dict[str, str]],
    strategy: str,
    cfg: RetrievalConfig,
) -> list[dict[str, Any]]:
    """Retrieve and generate an answer for each question under one strategy.

    Retrieval runs synchronously (it is database-bound and fast); generation is
    batched concurrently, since the LLM round-trip dominates.
    """
    print(f"\n[{strategy}] retrieving context for {len(questions)} questions...")
    contexts: list[list[Any]] = []
    for row in questions:
        outcome = retrieval.search(row["question"], cfg)
        contexts.append([r.product for r in outcome.results])

    prompt_pairs = [
        prompts.build_prompt(row["question"], products, strategy)
        for row, products in zip(questions, contexts)
    ]

    print(f"[{strategy}] generating {len(prompt_pairs)} answers...")
    t0 = time.perf_counter()
    responses = await llm.chat_many(prompt_pairs, return_exceptions=True)
    print(f"[{strategy}] generation took {time.perf_counter() - t0:.0f}s")

    records: list[dict[str, Any]] = []
    for row, products, response in zip(questions, contexts, responses):
        if isinstance(response, BaseException):
            records.append({
                "question": row["question"], "code": row["code"], "strategy": strategy,
                "answer": "", "error": str(response)[:200], "total_tokens": 0,
                "grounding_score": None, "retrieved_codes": "",
                "hit": False, "answer_chars": 0,
            })
            continue

        answer = response.content.strip()
        codes = [p.code for p in products]
        records.append({
            "question": row["question"],
            "code": row["code"],
            "strategy": strategy,
            "answer": answer,
            "error": "",
            "total_tokens": response.total_tokens,
            "grounding_score": rag.grounding_score(answer, products),
            "retrieved_codes": "|".join(codes),
            "hit": row["code"] in codes,
            "answer_chars": len(answer),
        })
    return records


async def judge_answers(records: list[dict[str, Any]]) -> None:
    """Attach an LLM relevance verdict to each record, in place."""
    judgeable = [r for r in records if r["answer"]]
    if not judgeable:
        return

    prompt_pairs = [
        prompts.build_judge_prompt(r["question"], r["answer"]) for r in judgeable
    ]
    print(f" judging {len(prompt_pairs)} answers...")
    responses = await llm.chat_many(prompt_pairs, json_mode=True, return_exceptions=True)

    for record, response in zip(judgeable, responses):
        if isinstance(response, BaseException):
            record["relevance"] = "UNKNOWN"
            record["relevance_explanation"] = str(response)[:150]
            continue
        try:
            parsed = llm.parse_json_content(response.content)
            relevance = str(parsed.get("Relevance", "UNKNOWN")).upper()
            record["relevance"] = relevance if relevance in RELEVANCE_LABELS else "UNKNOWN"
            record["relevance_explanation"] = str(parsed.get("Explanation", ""))[:300]
        except llm.LLMError:
            record["relevance"] = "UNKNOWN"
            record["relevance_explanation"] = "unparsable judge output"

    for record in records:
        record.setdefault("relevance", "UNKNOWN")
        record.setdefault("relevance_explanation", "")


def summarise(records: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    """Aggregate one strategy's records into a reportable row.

    Relevance is reported over answers that actually received a verdict, not
    over every attempted question. Two kinds of infrastructure failure are
    excluded and counted separately:

    * an empty answer, where generation itself failed (proxy 500s)
    * an ``UNKNOWN`` verdict, where the judge call failed

    Dividing by the attempted total instead conflates prompt quality with
    proxy reliability. On the first run it did exactly that: ``basic`` showed
    76.0% relevant against ``source_aware``'s 64.7%, a gap that looks decisive
    but was almost entirely a difference in failure rate (29 vs 37 empty
    answers, 3 vs 13 judge failures). Scored over valid verdicts the two are
    96.6% and 97.0% - indistinguishable. ``failure_pct`` is kept in the output
    so a high loss rate stays visible rather than being silently absorbed into
    the quality number.
    """
    subset = [r for r in records if r["strategy"] == strategy]
    if not subset:
        return {}

    total = len(subset)
    counts = Counter(r.get("relevance", "UNKNOWN") for r in subset)
    judged = [
        r for r in subset
        if r["answer"] and r.get("relevance") not in (None, "", "UNKNOWN")
    ]
    scored = len(judged)
    verdicts = Counter(r["relevance"] for r in judged)
    grounded = [r["grounding_score"] for r in subset if r["grounding_score"] is not None]
    tokens = [r["total_tokens"] for r in subset if r["total_tokens"]]
    lengths = [r["answer_chars"] for r in subset if r["answer_chars"]]

    def pct(count: int) -> float | None:
        return round(100 * count / scored, 1) if scored else None

    return {
        "strategy": strategy,
        "attempted": total,
        "scored": scored,
        "relevant_pct": pct(verdicts["RELEVANT"]),
        "partly_relevant_pct": pct(verdicts["PARTLY_RELEVANT"]),
        "non_relevant_pct": pct(verdicts["NON_RELEVANT"]),
        "generation_failed": sum(1 for r in subset if not r["answer"]),
        "judge_failed": counts["UNKNOWN"],
        "failure_pct": round(100 * (total - scored) / total, 1) if total else None,
        "mean_grounding": round(sum(grounded) / len(grounded), 3) if grounded else None,
        "fully_grounded_pct": (
            round(100 * sum(1 for g in grounded if g >= 0.999) / len(grounded), 1)
            if grounded else None
        ),
        "scored_numeric_answers": len(grounded),
        "mean_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
        "mean_answer_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
    }

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})
    print(f" wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate generation strategies")
    parser.add_argument("--sample", type=int, default=150,
                        help="questions per strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", nargs="+",
                        default=list(prompts.PROMPT_STRATEGIES),
                        help="prompt strategies to compare")
    args = parser.parse_args(argv)

    if db.product_count() == 0:
        print("No products in the database. Run ingestion first.")
        return 1

    ground_truth = load_ground_truth()
    validation, _ = split(ground_truth, args.seed)
    rng = random.Random(args.seed)
    questions = rng.sample(validation, min(args.sample, len(validation)))

    cfg = retrieval.default_config()
    # Rewriting is irrelevant here: these questions are already standalone, and
    # holding retrieval fixed isolates the prompt as the only variable.
    cfg.rewrite = False

    print(f"Evaluating {len(args.strategies)} strategies on {len(questions)} questions")
    print(f"Retrieval held constant at: {cfg.label()}")

    all_records: list[dict[str, Any]] = []
    for strategy in args.strategies:
        records = asyncio.run(generate_answers(questions, strategy, cfg))
        asyncio.run(judge_answers(records))
        all_records.extend(records)

    write_csv(RESULTS_PATH, all_records)

    summaries = [s for s in (summarise(all_records, st) for st in args.strategies) if s]
    write_csv(SUMMARY_PATH, summaries)

    print("\n" + "=" * 104)
    print("GENERATION EVALUATION (percentages over answers that received a verdict)")
    print("=" * 104)
    header = (
        f"{'strategy':<16} {'scored':>7} {'RELEVANT':>9} {'PARTLY':>8} {'NON':>7} "
        f"{'grounding':>10} {'fully_gr':>9} {'lost':>6} {'tokens':>7} {'chars':>6}"
    )
    print(header)
    print("-" * 104)
    for row in summaries:
        grounding = f"{row['mean_grounding']:.3f}" if row["mean_grounding"] is not None else "n/a"
        fully = f"{row['fully_grounded_pct']:.0f}%" if row["fully_grounded_pct"] is not None else "n/a"
        print(
            f"{row['strategy']:<16} {row['scored']:7d} {row['relevant_pct']:8.1f}% "
            f"{row['partly_relevant_pct']:7.1f}% {row['non_relevant_pct']:6.1f}% "
            f"{grounding:>10} {fully:>9} {row['failure_pct']:5.1f}% "
            f"{row['mean_tokens']:7d} {row['mean_answer_chars']:6d}"
        )
    print("=" * 104)
    for row in summaries:
        print(
            f" {row['strategy']}: {row['attempted'] - row['scored']} of "
            f"{row['attempted']} lost ({row['generation_failed']} generation, "
            f"{row['judge_failed']} judge)"
        )

    if summaries:
        # Rank by relevance first, then grounding: an answer must address the
        # question before its numeric fidelity is meaningful.
        best = max(
            summaries,
            key=lambda r: (r["relevant_pct"] or 0.0, r["mean_grounding"] or 0.0),
        )
        # A relevance gap smaller than this is not resolvable at n~100, so
        # falling back to cost is honest where declaring a quality winner is not.
        TIE_THRESHOLD_PCT = 2.0
        contenders = [
            r for r in summaries
            if (best["relevant_pct"] or 0.0) - (r["relevant_pct"] or 0.0) <= TIE_THRESHOLD_PCT
        ]
        if len(contenders) > 1:
            best = min(contenders, key=lambda r: r["mean_tokens"] or 0)
            names = ", ".join(
                f"{r['strategy']} {r['relevant_pct']:.1f}%" for r in contenders
            )
            print(
                f"\nWithin {TIE_THRESHOLD_PCT:.0f} points on relevance ({names}) - "
                f"too close to separate on quality at this sample size."
            )
            print(f"Chosen on cost instead: {best['strategy']} "
                f"({best['mean_tokens']} mean tokens)")
        else:
            print(f"\nBest strategy: {best['strategy']}")
        print(f"Set this in .env:\n PROMPT_STRATEGY={best['strategy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())