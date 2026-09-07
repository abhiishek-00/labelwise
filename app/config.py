"""Central configuration, loaded from environment variables.

Every value has a default so the module imports cleanly in tests and CI
without a populated ``.env``. Nothing here contains hardcoded infrastructure
endpoints - all URLs come from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshot"
EVALUATION_DIR = DATA_DIR / "evaluation"
CACHE_DIR = DATA_DIR / "cache"


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# ── Egress policy ─────────────────────────────────────────────────────────────
# Fail-closed control over which AI backends may be contacted.
#
# "proxy_only" (default)
# Only the Pulse proxy may be used for LLM and embedding traffic.
# Selecting any backend that would reach an external AI service
# (OpenAI, Groq, HuggingFace model downloads) raises immediately.
#
# "open"
# Any configured backend is permitted. Required for running outside a
# restricted network - see .env.example.
#
# The default is deliberately restrictive: a misconfiguration should fail
# loudly rather than silently send data to a third-party service.

LLM_EGRESS_POLICY = _env("LLM_EGRESS_POLICY", "proxy_only").lower()

# Backends that reach an external AI service.
EXTERNAL_LLM_BACKENDS = {"openai"}
EXTERNAL_EMBED_BACKENDS = {"openai", "local"} # "local" downloads model weights


def assert_egress_allowed(backend: str, kind: str) -> None:
    """Raise unless ``backend`` is permitted by the active egress policy.

    Args:
        backend: Backend name being requested, e.g. ``"openai"``.
        kind: Either ``"llm"`` or ``"embedding"``, used for the message.
    """
    if LLM_EGRESS_POLICY == "open":
        return
    external = EXTERNAL_LLM_BACKENDS if kind == "llm" else EXTERNAL_EMBED_BACKENDS
    if backend in external:
        raise PermissionError(
            f"Egress blocked: {kind} backend {backend!r} would contact an external "
            f"AI service, but LLM_EGRESS_POLICY={LLM_EGRESS_POLICY!r}.\n"
            f" - Inside a restricted network: use the proxy backend instead "
            f"(LLM_BACKEND=pulse / EMBED_BACKEND=pulse).\n"
            f" - Outside it: set LLM_EGRESS_POLICY=open in .env (see .env.example)."
        )


# ── LLM backend ───────────────────────────────────────────────────────────────
# "openai" -> any OpenAI-compatible endpoint (OpenAI, Groq, Together, ...)
# "pulse" -> a custom proxy exposing /v1/chat/completions with a flat
# {"content": ..., "usage": {...}} response shape.

LLM_BACKEND = _env("LLM_BACKEND", "pulse").lower()

# OpenAI-compatible settings. Point OPENAI_BASE_URL at Groq to use Groq.
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "llama-3.3-70b-versatile")

# Proxy backend settings (URL supplied entirely via environment).
PULSE_PROXY_URL = _env("PULSE_PROXY_URL")
PULSE_API_KEY = _env("PULSE_API_KEY")
PULSE_MODEL = _env("PULSE_MODEL", "gpt-4o")
# Corporate TLS interception / self-signed chains.
PULSE_VERIFY_SSL = _env_bool("PULSE_VERIFY_SSL", False)
# Corporate HTTP(S)_PROXY env vars break direct calls to internal hosts.
PULSE_TRUST_ENV = _env_bool("PULSE_TRUST_ENV", False)

LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 90.0)
LLM_MAX_CONCURRENCY = _env_int("LLM_MAX_CONCURRENCY", 8)

# ── Retry / rate limiting ─────────────────────────────────────────────────────
# Bulk ingestion and evaluation issue thousands of requests. Every backend call
# retries with exponential backoff and jitter; the rate limiters put a hard
# ceiling on requests per second, which a concurrency cap alone cannot do.

RETRY_MAX_ATTEMPTS = _env_int("RETRY_MAX_ATTEMPTS", 5)
RETRY_BASE_DELAY = _env_float("RETRY_BASE_DELAY", 1.0)
RETRY_MAX_DELAY = _env_float("RETRY_MAX_DELAY", 60.0)

# Requests per second; 0 disables limiting.
LLM_RATE_LIMIT_RPS = _env_float("LLM_RATE_LIMIT_RPS", 0.0)
EMBED_RATE_LIMIT_RPS = _env_float("EMBED_RATE_LIMIT_RPS", 0.0)

# ── Embedding backend ─────────────────────────────────────────────────────────
# "openai" -> OpenAI-compatible /v1/embeddings
# "pulse" -> proxy /v1/embeddings, flat {"embedding"|"embeddings"} shape
# "local" -> fastembed, runs in-process, no API key
# (requires: uv pip install -r requirements-local-embed.txt)

EMBED_BACKEND = _env("EMBED_BACKEND", "pulse").lower()

# MUST match the active backend's output width - pgvector columns are
# fixed-width, so changing this requires re-running ingestion.
# local BAAI/bge-small-en-v1.5 -> 384
# pulse ada-2 -> 1536
# openai text-embedding-3-small -> 1536
EMBED_DIM = _env_int("EMBED_DIM", 1536)

EMBED_OPENAI_BASE_URL = _env("EMBED_OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBED_OPENAI_API_KEY = _env("EMBED_OPENAI_API_KEY") or OPENAI_API_KEY
EMBED_OPENAI_MODEL = _env("EMBED_OPENAI_MODEL", "text-embedding-3-small")

EMBED_PULSE_MODEL = _env("EMBED_PULSE_MODEL", "text-embedding-3-small")
EMBED_LOCAL_MODEL = _env("EMBED_LOCAL_MODEL", "BAAI/bge-small-en-v1.5")

EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 50)
# Measured safe on the proxy; deliberately below the observed ceiling so that
# ingestion leaves headroom for other consumers.
EMBED_MAX_CONCURRENCY = _env_int("EMBED_MAX_CONCURRENCY", 4)

# Expected dimensions per backend/model, used to fail fast on a mismatch
# between EMBED_DIM and what the backend actually returns.
EXPECTED_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "ada-2": 1536,
    "text-embedding-ada-002": 1536,
}

# ── Database ──────────────────────────────────────────────────────────────────

POSTGRES_HOST = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _env_int("POSTGRES_PORT", 5432)
POSTGRES_DB = _env("POSTGRES_DB", "labelwise")
POSTGRES_USER = _env("POSTGRES_USER", "labelwise")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", "labelwise")


def postgres_dsn() -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

MAX_PRODUCTS = _env_int("MAX_PRODUCTS", 5000)
OFF_USER_AGENT = _env("OFF_USER_AGENT", "LabelWise/0.1 (llm-zoomcamp project)")
OFF_BASE_URL = _env("OFF_BASE_URL", "https://world.openfoodfacts.org")
# Defaults to true (correct for normal networks). Set false only where a TLS
# interception proxy presents a certificate chain Python cannot verify.
OFF_VERIFY_SSL = _env_bool("OFF_VERIFY_SSL", True)

# ── Retrieval (production defaults; final values set by Phase 4 evaluation) ────

RETRIEVAL_METHOD = _env("RETRIEVAL_METHOD", "hybrid")
HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.5)
RETRIEVAL_TOP_N = _env_int("RETRIEVAL_TOP_N", 20) # candidates before reranking
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5) # documents sent to the LLM
ENABLE_RERANK = _env_bool("ENABLE_RERANK", True)
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", True)

# ── Generation / monitoring ───────────────────────────────────────────────────

PROMPT_STRATEGY = _env("PROMPT_STRATEGY", "source_aware")
ENABLE_ONLINE_JUDGE = _env_bool("ENABLE_ONLINE_JUDGE", True)

# Estimated USD per 1M tokens, used for the monitoring cost panel. The proxy
# is internally hosted and free, so cost is always an ESTIMATE derived from
# real token counts at published list prices.
COST_PROMPT_PER_1M = _env_float("COST_PROMPT_PER_1M", 2.50)
COST_COMPLETION_PER_1M = _env_float("COST_COMPLETION_PER_1M", 10.00)

API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
API_BASE_URL = _env("API_BASE_URL", "http://localhost:8000")