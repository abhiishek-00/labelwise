-- LabelWise database schema.
--
-- {{EMBED_DIM}} is substituted at runtime by app.db.init_db() with the value of
-- the EMBED_DIM setting, because pgvector columns are fixed-width and the
-- active embedding backend determines the width:
-- local BAAI/bge-small-en-v1.5 -> 384
-- pulse ada-2 -> 1536
-- openai text-embedding-3-small -> 1536

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Knowledge base ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS products (
    code TEXT PRIMARY KEY, -- barcode; stable ground-truth id
    product_name TEXT NOT NULL,
    brands TEXT,
    categories TEXT,
    ingredients_text TEXT,
    allergens TEXT,
    labels TEXT,
    countries TEXT,
    serving_size TEXT,
    nutriscore_grade TEXT,

    -- Nutrition per 100g. NULL means "not declared in the source data", which
    -- the assistant must report as unavailable rather than infer.
    energy_kcal_100g DOUBLE PRECISION,
    fat_100g DOUBLE PRECISION,
    saturated_fat_100g DOUBLE PRECISION,
    carbohydrates_100g DOUBLE PRECISION,
    sugars_100g DOUBLE PRECISION,
    fiber_100g DOUBLE PRECISION,
    proteins_100g DOUBLE PRECISION,
    salt_100g DOUBLE PRECISION,

    -- Flattened text used for embedding and lexical search.
    doc_text TEXT NOT NULL,
    embedding vector({{EMBED_DIM}}),

    -- Generated tsvector, weighted so name/brand outrank the ingredient blob.
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(product_name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(brands, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(categories, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(allergens, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(ingredients_text, '')), 'C')
    ) STORED,

    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS products_search_idx ON products USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS products_brands_idx ON products (lower(brands));

-- HNSW for approximate nearest-neighbour cosine search.
CREATE INDEX IF NOT EXISTS products_embedding_idx
    ON products USING hnsw (embedding vector_cosine_ops);

-- ── Monitoring ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),

    question TEXT NOT NULL,
    rewritten_question TEXT,
    answer TEXT NOT NULL,

    retrieval_method TEXT NOT NULL,
    rerank_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rewrite_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    retrieved_codes TEXT[],
    prompt_strategy TEXT NOT NULL,

    llm_backend TEXT NOT NULL,
    model_used TEXT NOT NULL,

    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    eval_total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,

    response_time_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    retrieval_time_ms DOUBLE PRECISION NOT NULL DEFAULT 0,

    -- Online LLM-as-a-judge verdict: RELEVANT | PARTLY_RELEVANT | NON_RELEVANT | UNKNOWN
    relevance TEXT,
    relevance_explanation TEXT,

    -- Numeric grounding: fraction of numbers in the answer traceable to context.
    grounding_score DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS conversations_ts_idx ON conversations (timestamp DESC);
CREATE INDEX IF NOT EXISTS conversations_relevance_idx ON conversations (relevance);
CREATE INDEX IF NOT EXISTS conversations_method_idx ON conversations (retrieval_method);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    feedback SMALLINT NOT NULL CHECK (feedback IN (-1, 1)),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_conversation_idx ON feedback (conversation_id);
CREATE INDEX IF NOT EXISTS feedback_ts_idx ON feedback (timestamp DESC);