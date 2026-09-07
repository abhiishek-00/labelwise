"""Tests for retry, backoff, and rate limiting.

These guard the behaviour that keeps bulk ingestion and evaluation from
tripping provider rate limits.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app import retry


def _http_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://example.test/v1/x")
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestRetryClassification:
    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
    def test_retryable_statuses(self, status):
        assert retry.is_retryable(_http_error(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retried(self, status):
        """Retrying a malformed or unauthorised request only wastes time."""
        assert not retry.is_retryable(_http_error(status))

    def test_transport_errors_are_retryable(self):
        assert retry.is_retryable(httpx.TimeoutException("timeout"))
        assert retry.is_retryable(httpx.ConnectError("refused"))

    def test_tls_failures_are_not_retried(self):
        """A certificate failure is configuration, not transience."""
        import ssl
        wrapped = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed cert")
        assert not retry.is_retryable(wrapped)
        chained = httpx.ConnectError("connect failed")
        chained.__cause__ = ssl.SSLError("bad cert")
        assert not retry.is_retryable(chained)

    def test_unrelated_exceptions_are_not_retryable(self):
        assert not retry.is_retryable(ValueError("nope"))


class TestBackoff:
    def test_delay_grows_and_is_capped(self):
        for attempt in range(8):
            delay = retry.compute_delay(attempt, base=1.0, cap=10.0)
            assert 0.0 <= delay <= 10.0

    def test_equal_jitter_keeps_delay_above_half_the_window(self):
        """Full jitter can emit ~0s and hammer an already-failing service."""
        for attempt in range(4):
            window = min(10.0, 1.0 * (2**attempt))
            for _ in range(20):
                assert retry.compute_delay(attempt, 1.0, 10.0) >= window / 2.0

    def test_delay_increases_with_attempt(self):
        first = [retry.compute_delay(0, 1.0, 60.0) for _ in range(20)]
        later = [retry.compute_delay(4, 1.0, 60.0) for _ in range(20)]
        assert min(later) > max(first)

    def test_jitter_produces_varied_delays(self):
        """Fixed backoff would make concurrent workers retry in lockstep."""
        delays = {retry.compute_delay(5, 1.0, 60.0) for _ in range(50)}
        assert len(delays) > 1

    def test_retry_after_header_is_honoured(self):
        delay = retry.compute_delay(0, 1.0, 60.0, _http_error(429, retry_after="7"))
        assert delay == 7.0

    def test_retry_after_is_capped(self):
        delay = retry.compute_delay(0, 1.0, 5.0, _http_error(429, retry_after="900"))
        assert delay == 5.0

    def test_malformed_retry_after_falls_back_to_backoff(self):
        delay = retry.compute_delay(0, 1.0, 10.0, _http_error(429, retry_after="soon"))
        assert 0.0 <= delay <= 10.0


class TestSyncRetry:
    def test_succeeds_after_transient_failures(self):
        calls = {"n": 0}

        @retry.with_retry(max_attempts=5, base_delay=0.001, max_delay=0.01)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _http_error(429)
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        @retry.with_retry(max_attempts=3, base_delay=0.001, max_delay=0.01)
        def always_429():
            calls["n"] += 1
            raise _http_error(429)

        with pytest.raises(httpx.HTTPStatusError):
            always_429()
        assert calls["n"] == 3

    def test_non_retryable_fails_immediately(self):
        calls = {"n": 0}

        @retry.with_retry(max_attempts=5, base_delay=0.001)
        def bad_request():
            calls["n"] += 1
            raise _http_error(400)

        with pytest.raises(httpx.HTTPStatusError):
            bad_request()
        assert calls["n"] == 1

    def test_on_retry_callback_receives_context(self):
        seen: list[tuple[int, BaseException]] = []

        @retry.with_retry(
            max_attempts=3,
            base_delay=0.001,
            max_delay=0.01,
            on_retry=lambda a, d, e: seen.append((a, e)),
        )
        def flaky():
            raise _http_error(503)

        with pytest.raises(httpx.HTTPStatusError):
            flaky()
        assert [a for a, _ in seen] == [1, 2]


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failures(self):
        calls = {"n": 0}

        @retry.with_async_retry(max_attempts=5, base_delay=0.001, max_delay=0.01)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.TimeoutException("slow")
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_non_retryable_fails_immediately(self):
        calls = {"n": 0}

        @retry.with_async_retry(max_attempts=5, base_delay=0.001)
        async def bad():
            calls["n"] += 1
            raise _http_error(401)

        with pytest.raises(httpx.HTTPStatusError):
            await bad()
        assert calls["n"] == 1


class TestRateLimiter:
    def test_sync_limiter_enforces_minimum_interval(self):
        limiter = retry.RateLimiter(rate_per_second=20.0) # 50ms apart
        t0 = time.monotonic()
        for _ in range(3):
            limiter.acquire()
        assert time.monotonic() - t0 >= 0.09 # first is free, then 2 x 50ms

    def test_zero_rate_disables_limiting(self):
        limiter = retry.RateLimiter(rate_per_second=0)
        t0 = time.monotonic()
        for _ in range(100):
            limiter.acquire()
        assert time.monotonic() - t0 < 0.05

    @pytest.mark.asyncio
    async def test_async_limiter_paces_concurrent_workers(self):
        """A concurrency cap bounds in-flight calls but not the request rate."""
        limiter = retry.AsyncRateLimiter(rate_per_second=20.0)
        t0 = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(4)))
        assert time.monotonic() - t0 >= 0.14 # 3 x 50ms after the first


class TestAsyncRateLimiterAcrossEventLoops:
    """The limiter is a module-level singleton used from repeated asyncio.run calls.

    ``embed_pending`` runs one ``asyncio.run`` per chunk, ground-truth
    generation runs two, and the RAG evaluation runs two per strategy. An
    ``asyncio.Lock`` created in ``__init__`` binds to the first loop and then
    raises "bound to a different event loop" on the second call. The failure is
    invisible unless rate limiting is enabled, because ``acquire`` returns
    early when the interval is zero.
    """

    def test_survives_sequential_event_loops_under_contention(self):
        """Contention is required to reproduce this.

        ``asyncio.Lock.acquire`` has an uncontended fast path that sets a flag
        and returns without ever asking for the running loop. Only a waiter -
        which needs a second coroutine already holding the lock - creates a
        future, and that is where the loop mismatch is raised. A sequential
        version of this test passes against the broken implementation, so it
        gathers instead, mirroring EMBED_MAX_CONCURRENCY > 1 in production.
        """
        limiter = retry.AsyncRateLimiter(50.0)

        async def contended():
            await asyncio.gather(*(limiter.acquire() for _ in range(4)))

        for _ in range(3):
            asyncio.run(contended())

    def test_disabled_limiter_never_touches_a_lock(self):
        limiter = retry.AsyncRateLimiter(0)

        async def use():
            await limiter.acquire()
            return True

        for _ in range(3):
            assert asyncio.run(use()) is True

    def test_pacing_carries_across_loops(self):
        """A new loop must not be allowed to burst past the ceiling."""
        limiter = retry.AsyncRateLimiter(20.0) # 50 ms apart

        async def one():
            await limiter.acquire()

        start = time.monotonic()
        for _ in range(4):
            asyncio.run(one())
        elapsed = time.monotonic() - start
        assert elapsed >= 0.10, f"four calls at 20/s finished too fast: {elapsed:.3f}s"

    def test_concurrent_workers_inside_one_loop_are_still_serialised(self):
        limiter = retry.AsyncRateLimiter(50.0)
        order: list[int] = []

        async def worker(i):
            await limiter.acquire()
            order.append(i)

        async def main():
            await asyncio.gather(*(worker(i) for i in range(5)))

        asyncio.run(main())
        assert sorted(order) == [0, 1, 2, 3, 4]