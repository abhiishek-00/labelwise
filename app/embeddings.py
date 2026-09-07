"""Embedding layer with pluggable, swappable backends.

Mirrors :mod:`app.llm`: one interface, three interchangeable backends selected
by the ``EMBED_BACKEND`` environment variable.

``EMBED_BACKEND=pulse``
    Proxy ``/v1/embeddings``. Returns ``{"embedding": [...]}`` for a single
    string and ``{"embeddings": [[...], ...]}`` for a list. 1536 dimensions.

``EMBED_BACKEND=openai``
    Any OpenAI-compatible ``/v1/embeddings`` endpoint. 1536 dimensions for
    ``text-embedding-3-small``.

``EMBED_BACKEND=local``
    ``fastembed`` in-process, no API key and no per-query network hop.
    384 dimensions for ``BAAI/bge-small-en-v1.5``. Downloads model weights from
    HuggingFace on first use, so it requires outbound internet access.
    Install with ``uv pip install -r requirements-local-embed.txt``.

Because pgvector columns are fixed-width, switching backend between different
dimensionalities requires re-running ingestion. ``EMBED_DIM`` must match the
active backend; a mismatch is detected on the first call rather than surfacing
later as an opaque database error.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from app import config
from app.retry import AsyncRateLimiter, RateLimiter, with_async_retry, with_retry

Vector = list[float]

# Module-level so the ceiling applies across all concurrent workers.
_rate_limiter = RateLimiter(config.EMBED_RATE_LIMIT_RPS)
_arate_limiter = AsyncRateLimiter(config.EMBED_RATE_LIMIT_RPS)


def _log_retry(attempt: int, delay: float, exc: BaseException) -> None:
    reason = getattr(getattr(exc, "response", None), "status_code", type(exc).__name__)
    print(f" [embeddings] retry {attempt} after {delay:.1f}s (reason: {reason})", flush=True)


_RETRY_KW = dict(
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    base_delay=config.RETRY_BASE_DELAY,
    max_delay=config.RETRY_MAX_DELAY,
    on_retry=_log_retry,
)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails or returns an unusable payload."""


@dataclass(slots=True)
class EmbeddingStats:
    """Timing information for a batch embedding call."""

    count: int = 0
    latency_ms: float = 0.0
    backend: str = ""
    model: str = ""


# ── Backends ──────────────────────────────────────────────────────────────────


class _EmbedBackend:
    name = "base"
    model = ""

    def embed(self, texts: list[str]) -> list[Vector]:
        raise NotImplementedError

    async def aembed(self, texts: list[str]) -> list[Vector]:
        # Default: run the blocking implementation off the event loop.
        return await asyncio.to_thread(self.embed, texts)


class PulseEmbedBackend(_EmbedBackend):
    """Proxy ``/v1/embeddings``.

    Response shape differs by input type: a single string yields ``embedding``
    (singular), a list yields ``embeddings`` (plural). Both are handled.
    """

    name = "pulse"

    def __init__(self) -> None:
        if not config.PULSE_PROXY_URL:
            raise EmbeddingError(
                "PULSE_PROXY_URL is not set. Required for EMBED_BACKEND=pulse. "
                "Set it in .env (never commit the value)."
            )
        self.model = config.EMBED_PULSE_MODEL
        self._url = config.PULSE_PROXY_URL.rstrip("/") + "/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if config.PULSE_API_KEY:
            headers["Authorization"] = f"Bearer {config.PULSE_API_KEY}"
        self._client_kwargs: dict[str, Any] = {
            "headers": headers,
            "verify": config.PULSE_VERIFY_SSL,
            "trust_env": config.PULSE_TRUST_ENV,
            "timeout": config.LLM_TIMEOUT_SECONDS,
        }

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {
            "request_id": str(uuid.uuid4()),
            "model": self.model,
            "input": texts,
            "metadata": {},
        }

    @staticmethod
    def _extract(body: dict[str, Any]) -> list[Vector]:
        if "embeddings" in body:
            return body["embeddings"]
        if "embedding" in body:
            return [body["embedding"]]
        raise EmbeddingError(f"Unexpected embeddings response keys: {sorted(body)}")

    def embed(self, texts: list[str]) -> list[Vector]:
        @with_retry(**_RETRY_KW)
        def _call() -> list[Vector]:
            _rate_limiter.acquire()
            with httpx.Client(**self._client_kwargs) as client:
                resp = client.post(self._url, json=self._payload(texts))
                resp.raise_for_status()
                return self._extract(resp.json())

        try:
            return _call()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"pulse embeddings HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except EmbeddingError:
            raise
        except Exception as exc: # noqa: BLE001
            raise EmbeddingError(f"pulse embeddings call failed: {exc}") from exc

    async def aembed(self, texts: list[str]) -> list[Vector]:
        @with_async_retry(**_RETRY_KW)
        async def _call() -> list[Vector]:
            await _arate_limiter.acquire()
            async with httpx.AsyncClient(**self._client_kwargs) as client:
                resp = await client.post(self._url, json=self._payload(texts))
                resp.raise_for_status()
                return self._extract(resp.json())

        try:
            return await _call()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"pulse embeddings HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except EmbeddingError:
            raise
        except Exception as exc: # noqa: BLE001
            raise EmbeddingError(f"pulse embeddings call failed: {exc}") from exc


class OpenAIEmbedBackend(_EmbedBackend):
    """OpenAI SDK against any OpenAI-compatible embeddings endpoint.

    NOTE: not exercised during development under a proxy-only egress policy.
    """

    name = "openai"

    def __init__(self) -> None:
        if not config.EMBED_OPENAI_API_KEY:
            raise EmbeddingError(
                "EMBED_OPENAI_API_KEY (or OPENAI_API_KEY) is not set. "
                "Required for EMBED_BACKEND=openai."
            )
        from openai import AsyncOpenAI, OpenAI

        self.model = config.EMBED_OPENAI_MODEL
        self._client = OpenAI(
            api_key=config.EMBED_OPENAI_API_KEY,
            base_url=config.EMBED_OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        self._aclient = AsyncOpenAI(
            api_key=config.EMBED_OPENAI_API_KEY,
            base_url=config.EMBED_OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )

    def embed(self, texts: list[str]) -> list[Vector]:
        @with_retry(**_RETRY_KW)
        def _call() -> Any:
            _rate_limiter.acquire()
            return self._client.embeddings.create(model=self.model, input=texts)

        try:
            resp = _call()
        except Exception as exc: # noqa: BLE001
            raise EmbeddingError(f"openai embeddings call failed: {exc}") from exc
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def aembed(self, texts: list[str]) -> list[Vector]:
        @with_async_retry(**_RETRY_KW)
        async def _call() -> Any:
            await _arate_limiter.acquire()
            return await self._aclient.embeddings.create(model=self.model, input=texts)

        try:
            resp = await _call()
        except Exception as exc: # noqa: BLE001
            raise EmbeddingError(f"openai embeddings call failed: {exc}") from exc
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


class LocalEmbedBackend(_EmbedBackend):
    """In-process ``fastembed``. No API key, no per-query network hop.

    NOTE: not exercised during development under a proxy-only egress policy;
    downloading model weights from HuggingFace counts as external egress.
    """

    name = "local"

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc: # pragma: no cover - depends on optional install
            raise EmbeddingError(
                "EMBED_BACKEND=local requires fastembed, which is not installed.\n"
                " uv pip install -r requirements-local-embed.txt"
            ) from exc

        self.model = config.EMBED_LOCAL_MODEL
        self._model = TextEmbedding(self.model)

    def embed(self, texts: list[str]) -> list[Vector]:
        try:
            return [v.tolist() for v in self._model.embed(texts)]
        except Exception as exc: # noqa: BLE001
            raise EmbeddingError(f"local embeddings failed: {exc}") from exc


# ── Factory ───────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[_EmbedBackend]] = {
    "pulse": PulseEmbedBackend,
    "openai": OpenAIEmbedBackend,
    "local": LocalEmbedBackend,
}

_instance: _EmbedBackend | None = None
_dim_verified = False


def get_backend(name: str | None = None) -> _EmbedBackend:
    """Return the active embedding backend, constructed lazily and cached.

    Enforces the egress policy before any client is constructed or any model
    weights are downloaded.
    """
    global _instance
    requested = (name or config.EMBED_BACKEND).lower()
    if requested not in _BACKENDS:
        raise EmbeddingError(
            f"Unknown EMBED_BACKEND={requested!r}. Valid options: {sorted(_BACKENDS)}"
        )
    config.assert_egress_allowed(requested, "embedding")
    if _instance is not None and _instance.name == requested:
        return _instance
    _instance = _BACKENDS[requested]()
    return _instance


def reset_backend() -> None:
    """Drop the cached backend (used by tests that flip configuration)."""
    global _instance, _dim_verified
    _instance = None
    _dim_verified = False


def _verify_dim(vectors: list[Vector]) -> None:
    """Fail fast when the backend's width disagrees with ``EMBED_DIM``.

    Caught here, this is a one-line config fix. Caught later, it surfaces as an
    opaque pgvector insert error after a long ingestion run.
    """
    global _dim_verified
    if _dim_verified or not vectors:
        return
    actual = len(vectors[0])
    if actual != config.EMBED_DIM:
        backend = get_backend()
        raise EmbeddingError(
            f"Embedding dimension mismatch: backend {backend.name!r} "
            f"(model {backend.model!r}) returned {actual}-dim vectors but "
            f"EMBED_DIM={config.EMBED_DIM}.\n"
            f"Set EMBED_DIM={actual} in .env and re-run ingestion "
            f"(pgvector columns are fixed-width)."
        )
    _dim_verified = True


def _batches(texts: list[str], size: int) -> list[list[str]]:
    return [texts[i : i + size] for i in range(0, len(texts), size)]


# ── Public API ────────────────────────────────────────────────────────────────


def embed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> list[Vector]:
    """Embed a list of texts, batched. Returns vectors in input order."""
    if not texts:
        return []
    backend = get_backend()
    size = batch_size or config.EMBED_BATCH_SIZE
    chunks = _batches(texts, size)

    iterator: Any = chunks
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(chunks, desc=f"embedding ({backend.name})", unit="batch")

    out: list[Vector] = []
    for chunk in iterator:
        vectors = backend.embed(chunk)
        if len(vectors) != len(chunk):
            raise EmbeddingError(
                f"Backend returned {len(vectors)} vectors for {len(chunk)} inputs."
            )
        _verify_dim(vectors)
        out.extend(vectors)
    return out


def embed_query(text: str) -> Vector:
    """Embed a single query string."""
    return embed_texts([text])[0]


async def aembed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    concurrency: int | None = None,
) -> list[Vector]:
    """Embed a list of texts with concurrent batches, preserving input order.

    Used by ingestion, where thousands of documents make sequential batching
    the dominant cost.
    """
    if not texts:
        return []
    backend = get_backend()
    size = batch_size or config.EMBED_BATCH_SIZE
    limit = concurrency or config.EMBED_MAX_CONCURRENCY
    semaphore = asyncio.Semaphore(limit)
    chunks = _batches(texts, size)

    async def _one(chunk: list[str]) -> list[Vector]:
        async with semaphore:
            vectors = await backend.aembed(chunk)
            if len(vectors) != len(chunk):
                raise EmbeddingError(
                    f"Backend returned {len(vectors)} vectors for {len(chunk)} inputs."
                )
            return vectors

    results = await asyncio.gather(*(_one(c) for c in chunks))
    out: list[Vector] = [v for chunk_vectors in results for v in chunk_vectors]
    _verify_dim(out)
    return out


def probe() -> EmbeddingStats:
    """Single round-trip against the active backend, for health checks."""
    backend = get_backend()
    t0 = time.perf_counter()
    vectors = embed_texts(["LabelWise embedding backend probe."])
    _verify_dim(vectors)
    return EmbeddingStats(
        count=len(vectors),
        latency_ms=(time.perf_counter() - t0) * 1000,
        backend=backend.name,
        model=backend.model,
    )