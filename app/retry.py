"""Retry helpers for backend calls.

Bulk ingestion and evaluation issue thousands of requests, so transient
failures are a certainty rather than an edge case. Every backend call is
wrapped in exponential backoff with jitter.

Retries cover rate limiting (429), server-side faults (5xx), and transport
errors (timeouts, connection resets). Client errors such as 400 or 401 are
raised immediately - retrying a malformed request only wastes time.
"""

from __future__ import annotations

import asyncio
import functools
import random
import ssl
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

T = TypeVar("T")

# Transport-level failures worth retrying.
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0


def _status_of(exc: BaseException) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _retry_after(exc: BaseException) -> float | None:
    """Honour a server-supplied ``Retry-After`` header when present."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _is_tls_failure(exc: BaseException) -> bool:
    """Detect certificate/TLS failures anywhere in the exception chain.

    These surface as ``httpx.ConnectError``, which is normally retryable, but a
    certificate failure is a configuration problem: retrying burns the full
    backoff schedule and then reports the same error several seconds later.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        if "CERTIFICATE_VERIFY_FAILED" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_retryable(exc: BaseException) -> bool:
    if _is_tls_failure(exc):
        return False
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    status = _status_of(exc)
    return status is not None and status in _RETRYABLE_STATUS


def compute_delay(attempt: int, base: float, cap: float, exc: BaseException | None = None) -> float:
    """Exponential backoff with equal jitter, capped.

    Uses *equal* jitter - half the window fixed, half random - rather than full
    jitter. Full jitter can emit a near-zero delay on an early attempt, which
    hammers a service that is already failing; equal jitter still decorrelates
    concurrent workers but guarantees the delay grows with each attempt.
    """
    if exc is not None:
        server_hint = _retry_after(exc)
        if server_hint is not None:
            return min(server_hint, cap)
    window = min(cap, base * (2**attempt))
    half = window / 2.0
    return half + random.uniform(0.0, half)


def with_retry(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a synchronous function with retry and backoff."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc: # noqa: BLE001
                    last = exc
                    if attempt == max_attempts - 1 or not is_retryable(exc):
                        raise
                    delay = compute_delay(attempt, base_delay, max_delay, exc)
                    if on_retry:
                        on_retry(attempt + 1, delay, exc)
                    time.sleep(delay)
            raise last # pragma: no cover - loop always returns or raises

        return wrapper

    return decorator


def with_async_retry(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an async function with retry and backoff."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc: # noqa: BLE001
                    last = exc
                    if attempt == max_attempts - 1 or not is_retryable(exc):
                        raise
                    delay = compute_delay(attempt, base_delay, max_delay, exc)
                    if on_retry:
                        on_retry(attempt + 1, delay, exc)
                    await asyncio.sleep(delay)
            raise last # pragma: no cover

        return wrapper

    return decorator


class AsyncRateLimiter:
    """Token-bucket limiter enforcing a minimum interval between calls.

    Concurrency caps alone bound in-flight requests but not the request *rate*;
    fast responses can still produce a burst. This provides a hard ceiling on
    requests per second regardless of how quickly they complete.
    """

    def __init__(self, rate_per_second: float | None) -> None:
        self._interval = 1.0 / rate_per_second if rate_per_second and rate_per_second > 0 else 0.0
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._next_allowed = 0.0

    def _current_lock(self) -> asyncio.Lock:
        """Return a lock bound to the running event loop.

        ``asyncio.Lock`` binds to the loop that first awaits it, and this
        limiter is a module-level singleton shared across calls. Callers invoke
        ``asyncio.run`` repeatedly - once per ingestion chunk, once per
        evaluation stage - and every call builds a fresh loop, so reusing the
        original lock fails with "bound to a different event loop". That is a
        crash rather than a slowdown, and it only appears when rate limiting is
        switched on, which is why it survived a full evaluation run against a
        backend configured with no limit.

        Those loops are strictly sequential, never concurrent, so rebuilding the
        lock per loop preserves the mutual exclusion that actually matters:
        ordering between coroutines inside one loop. ``_next_allowed`` is
        deliberately not reset, so pacing carries across loops and a new chunk
        cannot burst past the ceiling.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._current_lock():
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._interval


class RateLimiter:
    """Synchronous counterpart to :class:`AsyncRateLimiter`."""

    def __init__(self, rate_per_second: float | None) -> None:
        self._interval = 1.0 / rate_per_second if rate_per_second and rate_per_second > 0 else 0.0
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        wait = self._next_allowed - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        self._next_allowed = max(now, self._next_allowed) + self._interval