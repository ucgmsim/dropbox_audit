"""One rate limiter for the whole process, because Dropbox's limit is account-wide.

Measured against the real account on 2026-08-13: 8 concurrent recursive listings ran
for two minutes with zero 429s; 16 started drawing them; rclone at ``--checkers 32``
earned a ``retry_after: 300`` penalty. So the defaults are deliberately modest, and
the response to a 429 is to park *every* worker for the full ``Retry-After`` --
letting the other workers keep firing during a five-minute penalty is what turns a
brief stall into a long outage.

Concurrency follows additive-increase/additive-decrease: one worker off per 429, one
back after a clean interval.

The request rate follows additive-increase/multiplicative-decrease, as TCP does. It
starts at ``rps``, the rate measured safe, and rises by ``rate_step`` after each clean
interval in which it held requests back, up to ``max_rps``; a 429 halves it, never below
``min_rps``. It climbs only while it is the bottleneck, so it cannot drift up unused and
then burst, and it climbs slowly because a 429 can cost every worker five minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("dbaudit.limiter")


class AdaptiveLimiter:
    def __init__(
        self,
        rps: float = 5.0,
        max_concurrency: int = 8,
        min_concurrency: int = 1,
        recover_after: float = 300.0,
        burst: float = 1.0,
        clock=time.monotonic,
        sleep=time.sleep,
        max_rps: float | None = None,
        min_rps: float = 1.0,
        rate_step: float = 0.5,
    ):
        self.rps = float(rps)
        # Without a ceiling above the starting rate, the rate stays fixed.
        self.max_rps = max(self.rps, float(max_rps)) if max_rps is not None else self.rps
        self.min_rps = min(float(min_rps), self.rps)
        self.rate_step = float(rate_step)
        self.max_concurrency = int(max_concurrency)
        self.min_concurrency = int(min_concurrency)
        self.recover_after = float(recover_after)
        self.burst = float(burst)
        self.clock = clock
        self.sleep = sleep

        self._lock = threading.Lock()
        self._concurrency = int(max_concurrency)
        self._in_flight = 0
        self._tokens = float(burst)
        self._last_refill = clock()
        self._paused_until = 0.0
        self._last_429 = float("-inf")
        self._held_back = False         # the rate made a request wait since it last changed
        self._rate_changed_at = clock()
        self.rate_limit_events = 0

    # ---- observable state ---------------------------------------------

    @property
    def concurrency(self) -> int:
        with self._lock:
            return self._concurrency

    @property
    def paused_until(self) -> float:
        with self._lock:
            return self._paused_until

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    # ---- internals -----------------------------------------------------

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rps)
            self._last_refill = now

    def _try_acquire(self) -> float:
        """Take a slot, or return how long to wait before trying again."""
        with self._lock:
            now = self.clock()
            self._refill(now)
            if now < self._paused_until:
                return self._paused_until - now
            if self._in_flight >= self._concurrency:
                return 0.05  # a peer will finish; poll rather than sleep on a condvar
            if self._tokens < 1.0:
                self._held_back = True
                return max((1.0 - self._tokens) / self.rps, 0.001)
            self._tokens -= 1.0
            self._in_flight += 1
            return 0.0

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1

    # ---- public API ----------------------------------------------------

    @contextmanager
    def slot(self):
        """Block until it is this worker's turn to make one API request."""
        while True:
            wait = self._try_acquire()
            if wait == 0.0:
                break
            self.sleep(wait)
        try:
            yield
        finally:
            self._release()

    def on_rate_limited(self, retry_after: float) -> None:
        with self._lock:
            now = self.clock()
            # Requests in flight when Dropbox objects all come back 429 together: one
            # overload, so the rate is halved once per pause rather than once per reply.
            halve = now >= self._paused_until
            self._paused_until = max(self._paused_until, now + float(retry_after))
            self._last_429 = now
            self._concurrency = max(self.min_concurrency, self._concurrency - 1)
            self.rate_limit_events += 1
            if halve:
                self.rps = max(self.min_rps, self.rps / 2)
            self._rate_changed_at = now
            self._held_back = False
            rate = self.rps
        if halve:
            log.warning("rate limited by Dropbox: request rate down to %.2f/s", rate)

    def on_success(self) -> None:
        raised = None
        with self._lock:
            now = self.clock()
            if (self._held_back and self.rps < self.max_rps
                    and now - self._rate_changed_at > self.recover_after):
                self.rps = min(self.max_rps, self.rps + self.rate_step)
                self._rate_changed_at = now
                self._held_back = False
                raised = self.rps
            if (self._concurrency < self.max_concurrency
                    and now - self._last_429 > self.recover_after):
                self._concurrency += 1
                # Restart the clean-interval window so we step up gradually rather
                # than jumping straight back to full concurrency.
                self._last_429 = now
        if raised is not None:
            log.info("request rate up to %.2f/s", raised)
