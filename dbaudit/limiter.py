"""One rate limiter for the whole process, because Dropbox's limit is account-wide.

Measured against the real account on 2026-08-13: 8 concurrent recursive listings ran
for two minutes with zero 429s; 16 started drawing them; rclone at ``--checkers 32``
earned a ``retry_after: 300`` penalty. So the defaults are deliberately modest, and
the response to a 429 is to park *every* worker for the full ``Retry-After`` --
letting the other workers keep firing during a five-minute penalty is what turns a
brief stall into a long outage.

Concurrency follows additive-increase/additive-decrease: one worker off per pause, one
back after a clean interval.

The request rate follows additive-increase/multiplicative-decrease, as TCP does. It
starts at ``rps``, the rate measured safe. After each clean interval in which requests
went out at no less than 90% of the rate -- so the rate, not latency or the number of
workers, was what held them back -- it rises by 0.5/s, up to ``max_rps``. A 429 halves
it, once per pause, never below 1/s. Clean intervals are counted from the end of a pause:
waiting one out says nothing about what Dropbox will take. The rate climbs slowly
because a 429 can cost every worker five minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("dbaudit.limiter")

RATE_STEP = 0.5       # requests/s added after an interval the rate was the bottleneck of
MIN_RPS = 1.0         # a 429 never halves the rate below this
SATURATED = 0.9       # share of the rate an interval's requests must reach to count


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
        start_concurrency: int | None = None,
    ):
        # Without a ceiling above the starting rate, the rate never climbs past where
        # it started; a 429 still halves it.
        self.max_rps = float(max_rps) if max_rps is not None else float(rps)
        self.rps = min(float(rps), self.max_rps)
        self.min_rps = min(MIN_RPS, self.rps)
        self.max_concurrency = int(max_concurrency)
        self.min_concurrency = int(min_concurrency)
        self.recover_after = float(recover_after)
        self.burst = float(burst)
        self.clock = clock
        self.sleep = sleep

        self._lock = threading.Lock()
        # Below the ceiling, the climb waits a full clean interval from the start: the
        # first request is no evidence that Dropbox will take more.
        start = self.max_concurrency if start_concurrency is None else int(start_concurrency)
        if not self.min_concurrency <= start <= self.max_concurrency:
            raise ValueError(f"start_concurrency {start} is outside "
                             f"[{self.min_concurrency}, {self.max_concurrency}]")
        self._concurrency = start
        self._in_flight = 0
        self._tokens = float(burst)
        self._last_refill = clock()
        self._paused_until = 0.0
        self._last_429 = clock() if start < self.max_concurrency else float("-inf")
        self._interval_start = clock()  # the clean interval the rate is judged over
        self._granted = 0               # requests let through since it began
        self.rate_limit_events = 0
        self._granted_total = 0          # every request let through, for progress

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
    def granted_total(self) -> int:
        with self._lock:
            return self._granted_total

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
            self._granted += 1
            self._granted_total += 1
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
            # overload, so one halving of the rate and one worker fewer per pause.
            first = now >= self._paused_until
            self._paused_until = max(self._paused_until, now + float(retry_after))
            self.rate_limit_events += 1
            if first:
                self.rps = max(self.min_rps, self.rps / 2)
                self._concurrency = max(self.min_concurrency, self._concurrency - 1)
            self._last_429 = self._interval_start = self._paused_until
            self._granted = 0
            rate, concurrency = self.rps, self._concurrency
        if first:
            log.warning("rate limited by Dropbox: concurrency down to %d, request rate "
                        "down to %.2f/s", concurrency, rate)

    def on_success(self) -> None:
        raised = stepped = None
        with self._lock:
            now = self.clock()
            elapsed = now - self._interval_start
            if elapsed > self.recover_after:
                if (self.rps < self.max_rps
                        and self._granted >= SATURATED * self.rps * elapsed):
                    self.rps = min(self.max_rps, self.rps + RATE_STEP)
                    raised = self.rps
                self._interval_start, self._granted = now, 0
            if (self._concurrency < self.max_concurrency
                    and now - self._last_429 > self.recover_after):
                self._concurrency += 1
                stepped = self._concurrency
                # Restart the clean-interval window so we step up gradually rather
                # than jumping straight back to full concurrency.
                self._last_429 = now
        if raised is not None:
            log.info("request rate up to %.2f/s", raised)
        if stepped is not None:
            log.info("concurrency up to %d", stepped)
