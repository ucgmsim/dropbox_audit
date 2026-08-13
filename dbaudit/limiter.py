"""One rate limiter for the whole process, because Dropbox's limit is account-wide.

Measured against the real account on 2026-08-13: 8 concurrent recursive listings ran
for two minutes with zero 429s; 16 started drawing them; rclone at ``--checkers 32``
earned a ``retry_after: 300`` penalty. So the defaults are deliberately modest, and
the response to a 429 is to park *every* worker for the full ``Retry-After`` --
letting the other workers keep firing during a five-minute penalty is what turns a
brief stall into a long outage.

Concurrency follows additive-increase/additive-decrease: one worker off per 429, one
back after a clean interval.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


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
    ):
        self.rps = float(rps)
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
            self._paused_until = max(self._paused_until, now + float(retry_after))
            self._last_429 = now
            self._concurrency = max(self.min_concurrency, self._concurrency - 1)
            self.rate_limit_events += 1

    def on_success(self) -> None:
        with self._lock:
            if self._concurrency >= self.max_concurrency:
                return
            if self.clock() - self._last_429 > self.recover_after:
                self._concurrency += 1
                # Restart the clean-interval window so we step up gradually rather
                # than jumping straight back to full concurrency.
                self._last_429 = self.clock()
