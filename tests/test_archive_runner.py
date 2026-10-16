"""The scheduler behind `archive run`, with walks and a limiter that are fakes."""

import logging
import threading
import time

import pytest

from dbaudit.archive.runner import Done, Job, Runner


class Limiter:
    """The two things the scheduler reads from an AdaptiveLimiter."""

    def __init__(self, concurrency):
        self.concurrency = concurrency
        self.granted_total = 0


def jobs(*sizes, chains=1):
    return [Job(key=f"k{i}", label=f"a{i}.tar", size=size, chains=chains)
            for i, size in enumerate(sizes)]


def run(walk, todo, concurrency=4, **kw):
    return Runner(walk, Limiter(concurrency), threading.Event(), **kw).run(todo)


def test_the_largest_archive_starts_first():
    order = []
    report = run(lambda job: order.append(job.label) or Done("complete"),
                 jobs(5, 50, 20), concurrency=1, margin=0)
    assert order == ["a1.tar", "a2.tar", "a0.tar"]
    assert report.outcomes == {"complete": 3} and report.started == 3


def test_archives_start_until_their_chains_reach_the_streams_plus_the_margin():
    """Four streams and a margin of two: three archives of two chains, never a fourth."""
    lock, flying, most = threading.Lock(), [0], [0]
    full, release = threading.Event(), threading.Event()

    def walk(job):
        with lock:
            flying[0] += job.chains
            most[0] = max(most[0], flying[0])
            if flying[0] >= 6:
                full.set()
        release.wait(timeout=5)
        with lock:
            flying[0] -= job.chains
        return Done("complete")

    def releaser():
        full.wait(timeout=5)
        time.sleep(0.3)                  # time enough for a fourth archive to start
        release.set()

    threading.Thread(target=releaser).start()
    report = run(walk, jobs(*range(40, 0, -1), chains=2), concurrency=4, margin=2)
    assert report.outcomes == {"complete": 40}
    assert most[0] == 6


def test_an_archive_wanting_more_chains_than_allowed_still_starts_alone():
    report = run(lambda job: Done("complete"), jobs(10, chains=8), concurrency=2, margin=0)
    assert report.outcomes == {"complete": 1}


def test_more_archives_start_as_the_limiter_earns_streams():
    limiter, seen = Limiter(1), []
    gate = threading.Event()

    def walk(job):
        seen.append(job.label)
        if job.label == "a0.tar":
            limiter.concurrency = 3           # the limiter earned two more streams
            gate.wait(timeout=5)
        else:
            gate.set()
        return Done("complete")

    report = Runner(walk, limiter, threading.Event(), margin=0).run(jobs(30, 20, 10))
    assert report.outcomes == {"complete": 3}
    assert seen[0] == "a0.tar" and set(seen) == {"a0.tar", "a1.tar", "a2.tar"}


def test_an_archive_that_ends_in_error_is_tried_once_more_at_the_end():
    tries = {}

    def walk(job):
        tries[job.label] = tries.get(job.label, 0) + 1
        if job.label == "a0.tar":
            return Done("error")              # twice: stays in error
        if job.label == "a1.tar" and tries[job.label] == 1:
            return Done("error")              # once: the second try completes
        return Done("complete")

    report = run(walk, jobs(30, 20, 10), concurrency=1, margin=0)
    assert tries == {"a0.tar": 2, "a1.tar": 2, "a2.tar": 1}
    assert report.outcomes == {"error": 1, "complete": 2}


def test_the_third_unexpected_failure_stops_the_run():
    recorded = []

    def walk(job):
        if job.size >= 70:
            raise RuntimeError(f"bug in {job.label}")
        return Done("complete")

    stop = threading.Event()
    report = Runner(walk, Limiter(1), stop, margin=0,
                    on_bug=lambda job, exc: recorded.append((job.label, str(exc)))
                    ).run(jobs(100, 90, 80, 70, 60, 50))
    assert recorded == [("a0.tar", "bug in a0.tar"), ("a1.tar", "bug in a1.tar"),
                        ("a2.tar", "bug in a2.tar")]
    assert report.failed and not report.stopped and stop.is_set()
    assert report.outcomes == {"error": 3} and report.started == 3


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_walk_that_dies_of_more_than_an_exception_still_reports_back():
    """SystemExit, say: unreported, the archive would stay in flight and the run would wait
    for it for ever."""
    walking = []

    def walk(job):
        walking.append(threading.current_thread())
        raise SystemExit(1)

    result = {}
    waiting = threading.Thread(target=lambda: result.update(report=run(walk, jobs(10))),
                               daemon=True)
    waiting.start()
    waiting.join(timeout=10)
    assert not waiting.is_alive()
    assert result["report"].outcomes == {"error": 1}
    walking[0].join(timeout=10)             # it dies here, with its warning, not in the next test


def test_an_archive_tried_again_counts_as_its_estimate_until_its_walk_reports():
    """Its last walk ended reporting one chain; the second begins wanting its estimate."""
    seen = []

    def walk(job):
        seen.append(job.chains)
        job.chains = 1                      # as a walk reports while its chains finish
        return Done("error" if len(seen) == 1 else "complete")

    run(walk, jobs(10, chains=3), concurrency=8, margin=0)
    assert seen == [3, 3]


def test_a_walk_that_reports_a_bug_counts_as_one_and_is_not_retried():
    tries = []
    report = run(lambda job: tries.append(job.label) or Done("error", bug=True),
                 jobs(10), concurrency=1)
    assert tries == ["a0.tar"]
    assert report.outcomes == {"error": 1} and not report.failed


def test_a_limit_caps_the_archives_started_but_not_their_second_tries():
    tries = []

    def walk(job):
        tries.append(job.label)
        return Done("error" if len(tries) == 1 else "complete")

    report = run(walk, jobs(30, 20, 10), concurrency=1, margin=0, limit=2)
    assert tries == ["a0.tar", "a1.tar", "a0.tar"]
    assert report.started == 2 and report.outcomes == {"complete": 2}


def test_a_limit_holds_even_when_the_streams_would_start_more_at_once():
    report = run(lambda job: Done("complete"), jobs(30, 20, 10), concurrency=4, margin=0,
                 limit=2)
    assert report.started == 2 and report.outcomes == {"complete": 2}


def test_a_stop_starts_nothing_new_and_lets_the_walks_in_flight_finish(caplog):
    stop, times = threading.Event(), iter(range(0, 10_000, 100))

    def walk(job):
        stop.set()
        return Done("stopped")

    with caplog.at_level(logging.INFO, logger="dbaudit"):
        report = Runner(walk, Limiter(1), stop, margin=0, summary_every=1,
                        clock=lambda: next(times)).run(jobs(30, 20, 10))
    assert report.outcomes == {"stopped": 1} and report.started == 1
    assert report.stopped and not report.failed
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("progress:")]
    assert "1 done, 0 in flight, 2 queued | 0 B of 30 B" in line   # stopped: not done, not left


def test_progress_is_logged_with_the_streams_and_the_request_rate(caplog):
    limiter = Limiter(3)
    times = iter(range(0, 10_000, 100))

    def walk(job):
        limiter.granted_total += 50
        return Done("complete", members=7, requests=50)

    with caplog.at_level(logging.INFO, logger="dbaudit"):
        Runner(walk, limiter, threading.Event(), margin=0, summary_every=1,
               clock=lambda: next(times)).run(jobs(2 << 40, 1 << 40))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("progress:")]
    assert lines and "3 streams" in lines[-1] and "3.0 TiB of 3.0 TiB" in lines[-1]


def test_each_archive_logs_its_time_and_the_etas_divide_what_is_left_by_the_pace(caplog):
    """One archive an hour, 3 TiB an hour: with two archives and 3 TiB left, the ETAs are
    two hours by count and one by size."""
    now = [0.0]

    def walk(job):
        now[0] += 3600
        return Done("complete", members=7, requests=50)

    with caplog.at_level(logging.INFO, logger="dbaudit"):
        Runner(walk, Limiter(1), threading.Event(), margin=0, summary_every=1,
               clock=lambda: now[0]).run(jobs(3 << 40, 2 << 40, 1 << 40))
    messages = [r.getMessage() for r in caplog.records]
    assert "a0.tar: complete (7 members, 50 requests, 1h 0m)" in messages
    first = next(m for m in messages if m.startswith("progress:"))
    assert "1 done, 0 in flight, 2 queued | 3.0 TiB of 6.0 TiB" in first
    assert "eta 2h 0m by count, 1h 0m by size" in first


def test_only_archives_walked_count_towards_what_is_done(caplog):
    """An archive that had gone, had lost a part, or held content walked before was never
    walked here: its size is neither done nor left, and the pace leaves it out."""
    now, ends = [0.0], iter(["gone", "refused", "done", "complete"])

    def walk(job):
        now[0] += 60
        return Done(next(ends))

    with caplog.at_level(logging.INFO, logger="dbaudit"):
        Runner(walk, Limiter(1), threading.Event(), margin=0, summary_every=1,
               clock=lambda: now[0]).run(jobs(4 << 40, 3 << 40, 2 << 40, 1 << 40))
    last = [r.getMessage() for r in caplog.records if r.getMessage().startswith("progress:")][-1]
    assert "4 done, 0 in flight, 0 queued | 1.0 TiB of 1.0 TiB" in last
