import logging
import threading

import pytest

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
    clock, lim = make(max_concurrency=8)
    lim.on_rate_limited(5)
    assert lim.concurrency == 7
    clock.t += 6
    lim.on_rate_limited(5)
    assert lim.concurrency == 6


def test_concurrency_never_below_minimum():
    clock, lim = make(max_concurrency=2, min_concurrency=1)
    for _ in range(5):
        lim.on_rate_limited(1)
        clock.t += 2
    assert lim.concurrency == 1


def test_concurrency_recovers_after_clean_interval():
    """The interval is counted from the end of the 1 s pause."""
    clock, lim = make(max_concurrency=8, recover_after=300)
    lim.on_rate_limited(1)
    assert lim.concurrency == 7
    clock.t += 302
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


def saturate(clock, lim, seconds):
    """Requests back to back for `seconds`: as many as the rate allows."""
    end = clock.t + seconds
    while clock.t < end:
        with lim.slot():
            pass
        lim.on_success()


def test_the_rate_climbs_once_per_interval_it_was_the_bottleneck_and_stops_at_the_ceiling():
    """Climbs at about t=10.5; not again until a whole interval later, about t=21."""
    clock, lim = make(rps=2.0, max_rps=2.8, recover_after=10)
    seen = []
    for seconds in (11, 7, 4, 11):
        saturate(clock, lim, seconds)
        seen.append(lim.rps)
    assert seen == [2.5, 2.5, 2.8, 2.8]


def test_the_rate_does_not_climb_when_requests_go_out_slower_than_it_allows():
    """Pairs of requests 2 s apart: the bucket holds back the second of each pair, but
    1 request/s on average is the workers' pace, not the rate's."""
    clock, lim = make(rps=2.0, max_rps=20.0, recover_after=10)
    for _ in range(40):
        for _ in range(2):
            with lim.slot():
                pass
            lim.on_success()
        clock.t += 2
    assert lim.rps == 2.0


def test_a_clean_interval_starts_when_the_pause_ends():
    clock, lim = make(rps=4.0, max_rps=20.0, recover_after=10)
    lim.on_rate_limited(5)              # rate 2.0; everything waits until t=5
    saturate(clock, lim, 14)            # the pause, then 9 s at the full rate
    assert lim.rps == 2.0
    saturate(clock, lim, 2)             # a whole clean interval since the pause ended
    assert lim.rps == 2.5


def test_concurrency_does_not_recover_as_a_pause_ends():
    clock, lim = make(max_concurrency=8, recover_after=300)
    lim.on_rate_limited(300)
    clock.t += 301                      # the pause is just over: no clean interval yet
    lim.on_success()
    assert lim.concurrency == 7


def test_429s_during_one_pause_halve_the_rate_and_drop_a_worker_once():
    """Several requests in flight can all come back 429 for the same overload."""
    _, lim = make(rps=8.0, max_rps=20.0, max_concurrency=8)
    for _ in range(5):
        lim.on_rate_limited(60)
    assert (lim.rps, lim.concurrency) == (4.0, 7)


def test_429s_that_come_back_together_are_one_overload_even_at_retry_after_zero():
    """Seen live: three 429s that said retry after 0 came back within 0.16 s, and each
    halved the rate. Requests in flight when Dropbox objects are one overload."""
    clock, lim = make(rps=8.0, max_rps=20.0, max_concurrency=24)
    for _ in range(3):
        lim.on_rate_limited(0)
        clock.t += 0.08
    assert (lim.rps, lim.concurrency) == (4.0, 23)
    clock.t += 5                        # Dropbox objecting again later is a new overload
    lim.on_rate_limited(0)
    assert (lim.rps, lim.concurrency) == (2.0, 22)


def test_429s_that_keep_coming_still_halve_the_rate_once_a_second():
    """The window runs from an overload's first 429, not its latest: a throttle answering
    "retry after 0" every 0.6 s for as long as it lasts is a new overload each second."""
    clock, lim = make(rps=8.0, max_rps=20.0, max_concurrency=24)
    for _ in range(3):
        lim.on_rate_limited(0)
        clock.t += 0.6
    assert lim.rps == 2.0


def test_a_429_halves_the_rate_but_not_below_one_request_a_second():
    clock, lim = make(rps=8.0, max_rps=20.0)
    rates = []
    for _ in range(5):
        lim.on_rate_limited(1)
        rates.append(lim.rps)
        clock.t += 2             # past the pause, so each 429 counts
    assert rates == [4.0, 2.0, 1.0, 1.0, 1.0]


def test_a_ceiling_below_the_starting_rate_lowers_the_start():
    _, lim = make(rps=5.0, max_rps=3.0)
    assert lim.rps == 3.0


def test_a_limiter_can_start_below_its_ceiling_and_climbs_a_stream_per_clean_interval():
    """`archive run` starts at 12 streams and earns its way to 24: the first step comes a
    full clean interval after the start, not on the first request."""
    clock, lim = make(max_concurrency=24, start_concurrency=12, recover_after=300)
    assert lim.concurrency == 12
    clock.t += 100
    lim.on_success()
    assert lim.concurrency == 12
    clock.t += 201
    lim.on_success()
    assert lim.concurrency == 13
    clock.t += 301
    lim.on_success()
    assert lim.concurrency == 14


def test_the_starting_concurrency_lies_between_the_floor_and_the_ceiling():
    for start in (0, 25):
        with pytest.raises(ValueError):
            make(max_concurrency=24, min_concurrency=1, start_concurrency=start)


def test_the_limiter_counts_every_request_it_lets_through():
    clock, lim = make()
    for _ in range(5):
        with lim.slot():
            pass
    assert lim.granted_total == 5


def test_every_change_in_concurrency_is_logged(caplog):
    clock, lim = make(max_concurrency=24, start_concurrency=12, recover_after=300)
    with caplog.at_level(logging.INFO, logger="dbaudit.limiter"):
        clock.t += 301
        lim.on_success()
        lim.on_rate_limited(5)
    messages = [r.getMessage() for r in caplog.records]
    assert "concurrency up to 13" in messages
    assert any("concurrency down to 12" in m for m in messages)
