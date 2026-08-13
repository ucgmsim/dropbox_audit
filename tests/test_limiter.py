import threading

from dbaudit.limiter import AdaptiveLimiter


class FakeClock:
    """Deterministic time. ``sleep`` is the only thing that advances it."""

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, d):
        self.slept.append(d)
        self.t += d


def make(**kw):
    clock = FakeClock()
    kw.setdefault("rps", 100.0)
    kw.setdefault("max_concurrency", 8)
    return clock, AdaptiveLimiter(clock=clock.now, sleep=clock.sleep, **kw)


def test_rate_limit_pauses_globally_for_retry_after():
    clock, lim = make()
    lim.on_rate_limited(300)
    assert lim.paused_until == 300.0
    with lim.slot():
        pass
    assert sum(clock.slept) >= 300


def test_rate_limit_reduces_concurrency():
    _, lim = make(max_concurrency=8)
    lim.on_rate_limited(5)
    assert lim.concurrency == 7
    lim.on_rate_limited(5)
    assert lim.concurrency == 6


def test_concurrency_never_below_minimum():
    _, lim = make(max_concurrency=2, min_concurrency=1)
    for _ in range(5):
        lim.on_rate_limited(1)
    assert lim.concurrency == 1


def test_concurrency_recovers_after_clean_interval():
    clock, lim = make(max_concurrency=8, recover_after=300)
    lim.on_rate_limited(1)
    assert lim.concurrency == 7
    clock.t += 301
    lim.on_success()
    assert lim.concurrency == 8


def test_concurrency_does_not_recover_too_soon():
    clock, lim = make(max_concurrency=8, recover_after=300)
    lim.on_rate_limited(1)
    clock.t += 100
    lim.on_success()
    assert lim.concurrency == 7


def test_recovery_never_exceeds_maximum():
    clock, lim = make(max_concurrency=4, recover_after=10)
    for _ in range(5):
        clock.t += 11
        lim.on_success()
    assert lim.concurrency == 4


def test_token_bucket_limits_request_rate():
    clock, lim = make(rps=2.0)
    for _ in range(4):
        with lim.slot():
            pass
    assert clock.t >= 1.5


def test_slot_releases_capacity_even_when_body_raises():
    _, lim = make(rps=1000.0, max_concurrency=1)
    try:
        with lim.slot():
            raise ValueError("boom")
    except ValueError:
        pass
    with lim.slot():  # would hang forever if capacity leaked
        pass


def test_concurrency_cap_is_enforced_across_threads():
    """Never allow more simultaneous requests than the current concurrency."""
    lim = AdaptiveLimiter(rps=1e6, max_concurrency=3)
    peak = 0
    live = 0
    guard = threading.Lock()
    barrier_done = threading.Event()

    def worker():
        nonlocal peak, live
        for _ in range(20):
            with lim.slot():
                with guard:
                    live += 1
                    peak = max(peak, live)
                barrier_done.wait(0.001)
                with guard:
                    live -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    barrier_done.set()
    for t in threads:
        t.join()
    assert peak <= 3
