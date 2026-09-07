# LabelWise

A grounded question-answering assistant over packaged-food label data. Ask what
is in a product, whether it declares an allergen, how much sugar it contains, or
how two products compare — and get an answer built from retrieved product
records, with the sources shown.

Built as the final project for the DataTalksClub LLM Zoomcamp.

## Evaluation criteria — where to look

| Criterion | Where it is satisfied |
|---|---|
| Problem description | [Problem](#problem) |
| Retrieval flow | [Architecture](#architecture) — pgvector knowledge base + LLM |
| Retrieval evaluation | [Retrieval evaluation](#retrieval-evaluation) — 4 methods compared, best selected on validation and confirmed on held-out test |
| LLM evaluation | [LLM evaluation](#llm-evaluation) — 2 prompt strategies, LLM judge, grounding check |
| Interface | [Quickstart](#quickstart) — Streamlit UI and FastAPI |
| Ingestion pipeline | [Data and ingestion](#data-and-ingestion) — automated with dlt |
| Monitoring | [Monitoring](#monitoring) — user feedback plus a 10-panel Grafana dashboard |
| Containerization | `docker-compose.yml` — all four services |
| Reproducibility | [Reproducibility](#reproducibility) — committed corpus and ground truth, pinned lockfile and images |
| Hybrid search | [Retrieval evaluation](#retrieval-evaluation) — lexical + vector, alpha swept |
| Document re-ranking | [Re-ranking earns its place](#re-ranking-earns-its-place) — implemented, evaluated, enabled |
| Query rewriting | [Query rewriting](#query-rewriting-recovers-unanswerable-follow-ups) |

---

## Problem

Food label data is public but hostile to read. Open Food Facts holds millions of
products, and the fields that matter — the ingredients list, the declared
allergens, the per-100g nutrition — are free text and sparse numbers scattered
across a very wide row. Answering "does this bouillon contain celery?" means
finding the right product among millions, then reading a semicolon-delimited
ingredients string.

A general-purpose language model answers this kind of question fluently and
sometimes wrongly, because it is recalling a plausible product rather than
reading a specific one. For allergens that is not a cosmetic failure.

LabelWise constrains the model to retrieved records. Every answer is generated
from product data pulled out of the knowledge base for that question, the
sources are returned alongside the answer, and numbers in the answer are checked
back against the source values. When the data does not contain the answer, the
intended behaviour is to say so rather than to fill the gap.

*A caveat worth stating up front:* Open Food Facts is crowd-sourced. Fields are
missing, inconsistent, and occasionally wrong, and unit conversion leaves values
like `52.6315789473684 g` of protein. LabelWise is faithful to its source, which
means it is exactly as correct as the source is. It is a demonstration of
grounded retrieval, not a food-safety tool.

---

## What it does

 *Ask* a question in natural language and get an answer with its sources.
 *Follow up* conversationally — "does it contain celery?" resolves against
  the previous turn.
 *Compare* products side by side on calories, sugar, protein, fat and salt.
 *Rate* answers thumbs up or down; the feedback lands in Postgres and shows
  up on the dashboard.

---

## Architecture

 ```mermaid
 flowchart LR
     subgraph Ingestion["Ingestion (dlt, run once)"]
         CSV["Open Food Facts<br/>CSV export"] --> SAMPLE["Stratified<br/>reservoir sample"]
         SAMPLE --> SNAP["Committed snapshot<br/>5,000 products"]
         SNAP --> EMBED["Embed"]
     end
 
     subgraph Store["Postgres + pgvector"]
         PRODUCTS[("products<br/>text + tsvector + vector")]
         CONV[("conversations")]
         FB[("feedback")]
     end
 
     subgraph Serving["Serving"]
         UI["Streamlit"] --> API["FastAPI"]
         API --> RAG["RAG pipeline"]
         RAG --> RETR["Retrieval<br/>lexical + vector + hybrid + rerank"]
         RAG --> LLM["OpenAI"]
     end
 
     EMBED --> PRODUCTS
     RETR <--> PRODUCTS
     RAG --> CONV
     API --> FB
     CONV --> GRAF["Grafana"]
     FB --> GRAF
```


Postgres is the only datastore. It holds the documents, the full-text index, the
vectors, the conversation log and the feedback, and it is the Grafana data
source. A dedicated vector database would add a service without adding a
capability at this scale.

*Stack:* Python 3.14, FastAPI, Streamlit, Postgres 16 + pgvector, dlt for
ingestion, Grafana for monitoring, OpenAI for generation and embeddings, all in
Docker Compose.

---

## Quickstart

You need Docker and an OpenAI API key. A complete run — ingestion plus both
evaluations — costs well under a dollar.

 ```bash
 git clone <your-repo-url> && cd labelwise
 cp .env.example .env
```

Set your key in .env:

```
 OPENAI_API_KEY=sk-...
```

Then:

``` bash
 docker compose up -d --build
```

Four services start: Postgres, the API, Streamlit and Grafana. `/health` will
report `degraded` until the database has a schema, which ingestion creates:

``` bash
 docker compose exec api python -m ingestion.dlt_pipeline
```

This loads the *committed 5,000-product snapshot* — no download of the 1 GB
Open Food Facts export — and embeds it with `text-embedding-3-small`. A few
minutes, almost all of it embedding.

Optionally seed the dashboard so its panels are populated before you have asked
anything:

``` bash
 docker compose exec api python -m scripts.seed_monitoring
```

Then open:

| | |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin / admin) |

Verify:

``` bash
 curl -s localhost:8000/health
```

`products` and `embedded` should both read 5000, and `embedding_dim` 1536.

> *If you change the code, rebuild.* Compose bakes the source into the image
> rather than mounting it, so `docker compose up` alone keeps serving the old
> build. Use `docker compose up -d --build`.

---

## Configuration

All settings live in `.env`; `.env.example` documents every option. The ones
that matter:

| Variable | Value | Why |
|---|---|---|
| LLM_EGRESS_POLICY | `open` | Defaults to `proxy_only`, which refuses external backends outright |
| LLM_BACKEND | `openai` | |
| OPENAI_BASE_URL | `https://api.openai.com/v1` | Its default points elsewhere — set it explicitly |
| OPENAI_MODEL | `gpt-4o-mini` | |
| EMBED_BACKEND | `openai` | One key covers generation and embeddings |
| EMBED_DIM | `1536` | *Must* match text-embedding-3-small; pgvector columns are fixed-width |
| HYBRID_ALPHA | `0.1` | Chosen by sweep — see below |
| ENABLE_RERANK | `true` | Measured improvement — see below |
| ENABLE_QUERY_REWRITE | `true` | Largest single effect measured |
| PROMPT_STRATEGY | `basic` | Tied on quality, cheaper in tokens |

Every one of these is the outcome of a measurement recorded below and
reproducible from the committed artifacts in `data/evaluation/`.

---

## Data and ingestion

The corpus is 5,000 products sampled from the Open Food Facts CSV export
(~4.5 M rows).

Ingestion is a *dlt* pipeline: normalise records, load to a staging table,
promote into `products`, generate embeddings, build indexes, then verify against
a gate that fails the run rather than reporting success on a broken corpus.

Two decisions shaped the corpus:

*Quality gates reject roughly 86% of source rows,* almost entirely for missing
ingredients text. A record without ingredients cannot answer the questions this
project is about.

*Sampling is stratified and reservoir-based over the whole file.* The export is
ordered by barcode, so reading its head yields a corpus skewed toward whatever
is listed first. Per-bucket reservoir sampling over every row in a single pass
spreads the corpus across 16 category buckets evenly.

The sampled corpus is committed to `data/snapshot/products.csv`. Open Food Facts
regenerates its export daily, so without a pinned snapshot a later run would
draw different barcodes — and the ground truth, which is barcode-keyed, would
break.

To rebuild from source instead:

``` bash
 # expects the export at data/raw/en.openfoodfacts.org.products.csv.gz
 docker compose exec api python -m ingestion.dlt_pipeline --refresh --reset
```

---

## Retrieval evaluation

1,066 questions were generated from 340 sampled products with an LLM, each
labelled with the barcode of the product it came from, then split 70/30 into
validation and held-out test. Configuration was chosen on validation alone; the
table below is the held-out test set.

*Held-out test, 320 questions:*

| Method | Hit Rate@5 | Hit Rate@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Lexical (Postgres FTS) | 0.8219 | 0.8812 | 0.6873 | 0.6949 |
| Vector (pgvector cosine) | 0.9625 | 0.9844 | 0.9286 | 0.9314 |
| *Hybrid (α = 0.1)* | *0.9688* | *0.9875* | *0.9310* | *0.9333* |
| Hybrid RRF | 0.9281 | 0.9563 | 0.8796 | 0.8838 |

The alpha sweep on validation is monotonic — vector-leaning wins, and pure
lexical collapses:

| α | 0.0 | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| MRR@5 | 0.9361 | *0.9372* | 0.9292 | 0.8804 | 0.7662 | 0.6330 |

Hybrid beats both of its parts, but the margin over pure vector is small
(+0.0024 MRR@5 on test). The corpus is semantically searchable and the questions
are brand-rich, so vector search alone is already close to the ceiling. Lexical
search is the weakest single method and yet still contributes: weighting it at
10% is better than excluding it.

### Re-ranking earns its place

LLM listwise re-ranking is evaluated on 150 questions, capped because every query
costs an LLM call. To keep the comparison honest, the baseline is re-run on
*exactly those 150 rows* rather than compared against the 320-row figures
above — otherwise two different test sets would be compared.

| | n | Hit Rate@5 | MRR@5 |
|---|---:|---:|---:|
| Hybrid α = 0.1 | 150 | 0.9533 | 0.9224 |
| *Hybrid α = 0.1 + rerank* | 150 | *0.9800* | *0.9717* |

*+0.0493 MRR@5 and +0.0267 Hit Rate@5*, for 1.37 s of added latency per query.
The gain held from validation (0.9347 → 0.9900) through to held-out data, so
re-ranking ships *enabled*.

### Query rewriting recovers unanswerable follow-ups

Rewriting cannot be measured on the main ground truth, because those questions
are standalone by construction and rewriting them is a no-op. It is evaluated on
a separate slice of 70 two-turn conversations where the second turn is
deliberately vague ("does it have natural flavors too?"):

| Condition | Hit Rate@5 | MRR@5 |
|---|---:|---:|
| Vague follow-up, raw | 0.0286 | 0.0064 |
| *Vague follow-up, rewritten* | *1.0000* | *1.0000* |
| Turn 1 standalone (reference) | 1.0000 | 1.0000 |

Without rewriting, vague follow-ups are effectively unretrievable — 3 hits in
70. Rewritten, they match the standalone reference exactly. This is the largest
single effect measured anywhere in the project.

Reproduce:

``` bash
 docker compose exec api python -m evaluation.retrieval_eval
```

---

## LLM evaluation

Two prompt strategies compared on 150 questions with retrieval held constant, so
the prompt is the only variable. An LLM judge rates each answer
relevant / partly relevant / non-relevant. Numeric grounding is computed
separately by checking every number in the answer against the source records.

| Strategy | Scored | Relevant | Partly | Non-relevant | Grounding | Fully grounded | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic | 150/150 | 90.7% | 6.7% | 2.7% | 0.915 | 86% | 1,446 |
| source_aware | 150/150 | 92.7% | 7.3% | 0.0% | 0.904 | 86% | 1,612 |

*Selected: `basic`, on cost rather than quality.*

The 2.0-point relevance gap is three answers out of 150 — inside sampling noise
at this size, so the harness falls back to token cost rather than declaring a
quality winner it cannot support. `basic` is 10% cheaper per answer and produces
shorter answers (148 vs 263 characters) at equivalent grounding.

`source_aware` has one genuine edge worth noting: *zero non-relevant answers*,
against 2.7% for `basic`. On a larger sample that might separate the two. It is
recorded here rather than buried, and switching is a one-line change to
`PROMPT_STRATEGY`.

Percentages are over answers that received a verdict. *No calls were lost* —
0 generation failures and 0 judge failures across all 300 answers.

Reproduce:

``` bash
 docker compose exec api python -m evaluation.rag_eval
```

---

## Monitoring

Grafana reads directly from Postgres. Ten panels, provisioned from files so they
exist on first boot with no manual setup:

request volume, average and P95 response time, user feedback, answer relevance,
numeric grounding over time, token usage, cumulative estimated cost, retrieval
method usage and latency, and recent questions that received negative feedback.

Every question is logged to `conversations` with its retrieval method, token
counts, latency breakdown, relevance verdict and grounding score. Thumbs
up/down go to `feedback`.

`scripts/seed_monitoring.py` writes synthetic history so the dashboard is
populated on a fresh boot rather than showing empty panels.

---

## What it costs

Measured against gpt-4o-mini and text-embedding-3-small at list prices.

| Step | Approx. cost |
|---|---:|
| Ingestion — embed 5,000 products | $0.02 |
| Ground truth — 420 calls | $0.10 |
| Retrieval evaluation — re-ranking and rewriting | $0.15 |
| RAG evaluation — 300 generations + 300 judgements | $0.30 |
| *Full rebuild from scratch* | *~$0.60* |

Serving a single user question costs roughly $0.0005 with re-ranking and the
online judge both enabled. Set `COST_PROMPT_PER_1M` and
`COST_COMPLETION_PER_1M` to match your model, or Grafana's cost panel will
report a confident but wrong number.

---

## Reproducibility

 The *5,000-product corpus* is committed, so everyone indexes identical data.
 The *ground truth* (1,066 questions plus 80 conversational examples) is
  committed, so retrieval metrics recompute without regenerating questions.
 All *evaluation outputs* are committed under `data/evaluation/`.
 Dependencies are pinned in `uv.lock`; Docker images are pinned by tag.
 Retrieval evaluation after the first run is pure SQL plus a cached
  query-vector file — no API calls for the lexical, vector, hybrid and
  alpha-sweep comparisons.
 The published numbers were measured with `text-embedding-3-small` at 1536
  dimensions, which is what the documented configuration gives you, so they
  should reproduce closely.

What will *not* reproduce exactly: temperature is not pinned, so
LLM-dependent results — judge verdicts, re-ranking, rewriting — vary slightly
run to run. Every raw artifact is committed so the published numbers can be
inspected even where they cannot be reproduced bit-for-bit.

---

## Development

``` bash
 uv sync
 docker compose up -d postgres
 pytest tests -q # 120 tests, no API calls, no key required
```

Live backend tests are deselected by default and need real credentials:

``` bash
 pytest -m live
```

```
 app/ config, db, llm, embeddings, retrieval, rag, prompts, api
 ingestion/ dlt pipeline, sampling, normalisation
 evaluation/ ground-truth generation, retrieval eval, RAG eval
 streamlit_app/ UI
 scripts/ dashboard builder, monitoring seeder
 grafana/ provisioned datasource + dashboard
 sql/ schema
 tests/ contract, retry, normalisation and UI tests
```

---

## Known limitations

 *The benchmark sits near its ceiling.* Hit Rate@5 around 0.96 leaves little
  room to separate methods. Ground-truth questions are generated from the
  product they label, so most share tokens with its name, making the task easier
  than real user queries. The alpha sweep, re-ranking and rewriting slices all
  still separate cleanly, so the evaluation retains discriminating power where
  it matters.
 *Source data quality varies.* Missing fields, inconsistent units and
  artefacts like `52.6315789473684 g`. Answers inherit this, faithfully.
 *Question categories are unevenly distributed.* Nutrition, ingredient and
  allergen questions dominate; comparison questions are only 1.1% of the set,
  because requiring each question to identify a single product makes comparisons
  hard to generate. A handful of questions were also labelled with categories
  outside the requested list.
 *Re-ranking is measured on 150 of 320 test questions,* since each costs an
  LLM call. The comparison is like-for-like on those rows, but the sample is
  smaller than the headline table.
 *Prompt strategies are separated by three answers.* `basic` ships on cost;
  the quality difference is not established at this sample size.

---

## Licence

MIT — see LICENSE. Product data is from
Open Food Facts, licensed under the
Open Database License.