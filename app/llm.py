"""LLM access layer with pluggable, swappable backends.

Two backends expose one identical interface, so switching between them is a
single environment variable and no application code changes:

``LLM_BACKEND=openai``
    Any OpenAI-compatible chat-completions endpoint. Point ``OPENAI_BASE_URL``
    at OpenAI, Groq, Together, etc. Uses the official ``openai`` SDK.

``LLM_BACKEND=pulse``
    A proxy that accepts an OpenAI-shaped request but returns a flat
    ``{"content": ..., "usage": {...}}`` body instead of ``choices[0].message``.
    The endpoint is supplied entirely via ``PULSE_PROXY_URL``.

Both backends return a normalised :class:`LLMResponse`, so token accounting,
latency and cost estimation work identically regardless of provider.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from app import config
from app.retry import AsyncRateLimiter, RateLimiter, with_async_retry, with_retry

# ── Normalised response ───────────────────────────────────────────────────────


@dataclass(slots=True)
class LLMResponse:
    """Provider-agnostic completion result."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost from real token counts at configured list prices.

        The proxy backend is internally hosted and free; this is therefore an
        *estimate* for monitoring comparability, never a billed amount.
        """
        return (
            self.prompt_tokens * config.COST_PROMPT_PER_1M
            + self.completion_tokens * config.COST_COMPLETION_PER_1M
        ) / 1_000_000


class LLMError(RuntimeError):
    """Raised when a backend call fails or returns an unusable payload."""


# Module-level so the ceiling applies across all concurrent workers.
_rate_limiter = RateLimiter(config.LLM_RATE_LIMIT_RPS)
_arate_limiter = AsyncRateLimiter(config.LLM_RATE_LIMIT_RPS)


def _log_retry(attempt: int, delay: float, exc: BaseException) -> None:
    reason = getattr(getattr(exc, "response", None), "status_code", type(exc).__name__)
    print(f" [llm] retry {attempt} after {delay:.1f}s (reason: {reason})", flush=True)


_RETRY_KW = dict(
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    base_delay=config.RETRY_BASE_DELAY,
    max_delay=config.RETRY_MAX_DELAY,
    on_retry=_log_retry,
)


# ── JSON-mode guard ───────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _assert_json_hint(system: str, user: str) -> None:
    """Fail fast when JSON mode is requested without the word 'json' present.

    OpenAI-compatible endpoints reject ``response_format={"type":"json_object"}``
    unless the literal token "json" appears somewhere in the messages. Without
    this guard the failure surfaces as an opaque HTTP 400/500 from the provider,
    which is slow to diagnose.
    """
    if "json" not in f"{system} {user}".lower():
        raise LLMError(
            "JSON mode requires the word 'json' to appear in the system or user "
            "prompt (OpenAI-compatible API constraint). Add an instruction such "
            "as 'Respond with valid JSON only.' to the system prompt."
        )


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating stray prose or fences.

    Native JSON mode makes this unnecessary, but it is retained so the same
    calling code works against providers or models where JSON mode is
    unavailable and the model wraps output in a ```json fence.
    """
    text = _JSON_FENCE_RE.sub("", content.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise LLMError(f"No JSON object found in model output: {content[:200]!r}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Malformed JSON in model output: {content[:200]!r}") from exc


# ── Backends ──────────────────────────────────────────────────────────────────


class _Backend:
    name = "base"

    def complete(self, system: str, user: str, *, json_mode: bool, model: str | None) -> LLMResponse:
        raise NotImplementedError

    async def acomplete(
        self, system: str, user: str, *, json_mode: bool, model: str | None
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleBackend(_Backend):
    """OpenAI SDK against any OpenAI-compatible base URL (OpenAI, Groq, ...)."""

    name = "openai"

    def __init__(self) -> None:
        if not config.OPENAI_API_KEY:
            raise LLMError(
                "OPENAI_API_KEY is not set. Required for LLM_BACKEND=openai "
                f"(base_url={config.OPENAI_BASE_URL})."
            )
        from openai import AsyncOpenAI, OpenAI

        self._model = config.OPENAI_MODEL
        self._client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        self._aclient = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )

    def _kwargs(self, system: str, user: str, json_mode: bool, model: str | None) -> dict[str, Any]:
        if json_mode:
            _assert_json_hint(system, user)
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _normalise(self, resp: Any, elapsed_ms: float) -> LLMResponse:
        # `choices` is the one field here that cannot be defaulted away. An
        # OpenAI-compatible provider can return an empty list - content
        # filtering, or a completion truncated before any token was emitted -
        # and indexing it raised IndexError. That is not an LLMError, so the
        # graceful handler in rag.py did not catch it and the request became a
        # 500 with the retrieved products thrown away. Raising LLMError instead
        # degrades to the user-facing message with the sources still shown, and
        # it is deliberately not in the retryable set: an empty completion is a
        # verdict from the provider, not a transient fault.
        choices = getattr(resp, "choices", None) or []
        if not choices:
            finish = getattr(resp, "id", "") or ""
            raise LLMError(
                f"{self.name} backend returned no completion choices"
                + (f" (response id {finish})" if finish else "")
            )

        message = getattr(choices[0], "message", None)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=getattr(message, "content", None) or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            model=getattr(resp, "model", "") or "",
            backend=self.name,
            latency_ms=elapsed_ms,
            cached=False,
        )

    def complete(self, system: str, user: str, *, json_mode: bool, model: str | None) -> LLMResponse:
        kwargs = self._kwargs(system, user, json_mode, model)

        @with_retry(**_RETRY_KW)
        def _call() -> Any:
            _rate_limiter.acquire()
            return self._client.chat.completions.create(**kwargs)

        t0 = time.perf_counter()
        try:
            resp = _call()
        except Exception as exc: # noqa: BLE001 - normalise all provider errors
            raise LLMError(f"openai backend call failed: {exc}") from exc
        return self._normalise(resp, (time.perf_counter() - t0) * 1000)

    async def acomplete(
        self, system: str, user: str, *, json_mode: bool, model: str | None
    ) -> LLMResponse:
        kwargs = self._kwargs(system, user, json_mode, model)

        @with_async_retry(**_RETRY_KW)
        async def _call() -> Any:
            await _arate_limiter.acquire()
            return await self._aclient.chat.completions.create(**kwargs)

        t0 = time.perf_counter()
        try:
            resp = await _call()
        except Exception as exc: # noqa: BLE001
            raise LLMError(f"openai backend call failed: {exc}") from exc
        return self._normalise(resp, (time.perf_counter() - t0) * 1000)


class PulseProxyBackend(_Backend):
    """Proxy returning a flat ``{"content", "usage"}`` body.

    Note: this proxy ignores the ``model`` field and always serves a single
    configured backing model, so model comparison is not possible here.
    """

    name = "pulse"

    def __init__(self) -> None:
        if not config.PULSE_PROXY_URL:
            raise LLMError(
                "PULSE_PROXY_URL is not set. Required for LLM_BACKEND=pulse. "
                "Set it in .env (never commit the value)."
            )
        self._url = config.PULSE_PROXY_URL.rstrip("/") + "/v1/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if config.PULSE_API_KEY:
            self._headers["Authorization"] = f"Bearer {config.PULSE_API_KEY}"
        # trust_env=False bypasses corporate HTTP(S)_PROXY vars that would
        # otherwise break direct calls to an internal host.
        self._client_kwargs: dict[str, Any] = {
            "headers": self._headers,
            "verify": config.PULSE_VERIFY_SSL,
            "trust_env": config.PULSE_TRUST_ENV,
            "timeout": config.LLM_TIMEOUT_SECONDS,
        }

    def _payload(self, system: str, user: str, json_mode: bool, model: str | None) -> dict[str, Any]:
        if json_mode:
            _assert_json_hint(system, user)
        payload: dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "model": model or config.PULSE_MODEL,
            "stream": False,
            "metadata": {},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _normalise(self, body: dict[str, Any], elapsed_ms: float) -> LLMResponse:
        usage = body.get("usage") or {}
        return LLMResponse(
            content=body.get("content", "") or "",
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
            model=body.get("model", "") or "",
            backend=self.name,
            # Prefer the proxy's own timing when present.
            latency_ms=float(body.get("latency_ms") or elapsed_ms),
            cached=bool(body.get("cached", False)),
            raw=body,
        )

    def complete(self, system: str, user: str, *, json_mode: bool, model: str | None) -> LLMResponse:
        payload = self._payload(system, user, json_mode, model)

        @with_retry(**_RETRY_KW)
        def _call() -> dict[str, Any]:
            _rate_limiter.acquire()
            with httpx.Client(**self._client_kwargs) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                return resp.json()

        t0 = time.perf_counter()
        try:
            body = _call()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"pulse backend HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except Exception as exc: # noqa: BLE001
            raise LLMError(f"pulse backend call failed: {exc}") from exc
        return self._normalise(body, (time.perf_counter() - t0) * 1000)

    async def acomplete(
        self, system: str, user: str, *, json_mode: bool, model: str | None
    ) -> LLMResponse:
        payload = self._payload(system, user, json_mode, model)

        @with_async_retry(**_RETRY_KW)
        async def _call() -> dict[str, Any]:
            await _arate_limiter.acquire()
            async with httpx.AsyncClient(**self._client_kwargs) as client:
                resp = await client.post(self._url, json=payload)
                resp.raise_for_status()
                return resp.json()

        t0 = time.perf_counter()
        try:
            body = await _call()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"pulse backend HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except Exception as exc: # noqa: BLE001
            raise LLMError(f"pulse backend call failed: {exc}") from exc
        return self._normalise(body, (time.perf_counter() - t0) * 1000)


# ── Factory + public API ──────────────────────────────────────────────────────

_BACKENDS: dict[str, type[_Backend]] = {
    "openai": OpenAICompatibleBackend,
    "pulse": PulseProxyBackend,
}

_instance: _Backend | None = None


def get_backend(name: str | None = None) -> _Backend:
    """Return the active backend, constructed lazily and cached.

    Enforces the egress policy before any client is constructed, so a
    misconfiguration cannot reach an external service.
    """
    global _instance
    requested = (name or config.LLM_BACKEND).lower()
    if requested not in _BACKENDS:
        raise LLMError(
            f"Unknown LLM_BACKEND={requested!r}. Valid options: {sorted(_BACKENDS)}"
        )
    config.assert_egress_allowed(requested, "llm")
    if _instance is not None and _instance.name == requested:
        return _instance
    _instance = _BACKENDS[requested]()
    return _instance


def reset_backend() -> None:
    """Drop the cached backend (used by tests that flip configuration)."""
    global _instance
    _instance = None


def chat(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    model: str | None = None,
) -> LLMResponse:
    """Single completion against the configured backend."""
    return get_backend().complete(system, user, json_mode=json_mode, model=model)


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
) -> tuple[dict[str, Any], LLMResponse]:
    """Completion parsed into a dict. Returns ``(parsed, response)``."""
    response = chat(system, user, json_mode=True, model=model)
    return parse_json_content(response.content), response


async def achat(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    model: str | None = None,
) -> LLMResponse:
    """Async single completion against the configured backend."""
    return await get_backend().acomplete(system, user, json_mode=json_mode, model=model)


async def chat_many(
    prompts: list[tuple[str, str]],
    *,
    json_mode: bool = False,
    model: str | None = None,
    concurrency: int | None = None,
    return_exceptions: bool = False,
) -> list[LLMResponse | BaseException]:
    """Run many ``(system, user)`` prompts concurrently, preserving input order.

    Bulk evaluation is the dominant cost in this project; capped concurrency
    keeps it fast without tripping provider rate limits.
    """
    limit = concurrency or config.LLM_MAX_CONCURRENCY
    semaphore = asyncio.Semaphore(limit)

    async def _one(system: str, user: str) -> LLMResponse:
        async with semaphore:
            return await achat(system, user, json_mode=json_mode, model=model)

    return await asyncio.gather(
        *(_one(system, user) for system, user in prompts),
        return_exceptions=return_exceptions,
    )
