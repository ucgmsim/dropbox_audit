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


def hold_back(clock, lim, n=4):
    """Take n slots back to back, so the token bucket makes callers wait."""
    for _ in range(n):
        with lim.slot():
            pass


def test_the_rate_climbs_while_it_holds_requests_back_and_stops_at_the_ceiling():
    clock, lim = make(rps=2.0, max_rps=2.8, recover_after=10)
    seen = []
    for _ in range(3):
        hold_back(clock, lim)
        clock.t += 11
        lim.on_success()
        seen.append(lim.rps)
    assert seen == [2.5, 2.8, 2.8]


def test_the_rate_does_not_climb_unless_it_held_requests_back():
    clock, lim = make(rps=2.0, max_rps=10.0, recover_after=10)
    for _ in range(30):
        clock.t += 1.0          # slower than the rate allows: never waits on a token
        with lim.slot():
            pass
        lim.on_success()
    assert lim.rps == 2.0


def test_a_429_halves_the_rate_but_not_below_the_minimum():
    clock, lim = make(rps=8.0, max_rps=20.0, min_rps=1.5)
    rates = []
    for _ in range(4):
        lim.on_rate_limited(1)
        rates.append(lim.rps)
        clock.t += 2             # past the pause, so each 429 counts
    assert rates == [4.0, 2.0, 1.5, 1.5]


def test_429s_during_one_pause_halve_the_rate_once():
    """Several requests in flight can all come back 429 for the same overload."""
    _, lim = make(rps=8.0, max_rps=20.0)
    for _ in range(5):
        lim.on_rate_limited(60)
    assert lim.rps == 4.0


def test_after_a_429_the_rate_waits_a_clean_interval_before_climbing():
    clock, lim = make(rps=4.0, max_rps=20.0, recover_after=10)
    clock.t += 100             # long after the limiter started, so only the 429 restarts it
    lim.on_rate_limited(1)
    assert lim.rps == 2.0
    hold_back(clock, lim)
    clock.t += 5
    lim.on_success()
    assert lim.rps == 2.0
    hold_back(clock, lim)
    clock.t += 11
    lim.on_success()
    assert lim.rps == 2.5


def test_waiting_out_a_pause_does_not_count_as_the_rate_holding_requests_back():
    clock, lim = make(rps=4.0, max_rps=20.0, recover_after=10)
    lim.on_rate_limited(30)
    with lim.slot():            # waits for the pause to end, never for a token
        pass
    clock.t += 50
    lim.on_success()
    assert lim.rps == 2.0
