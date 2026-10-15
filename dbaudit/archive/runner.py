"""Walk many archives at once: largest first, as many as Dropbox's streams allow.

A single-part archive is one chain of headers, so its walk is one stream of requests, one
after another; walking archives one at a time leaves every other stream idle. Here each
archive in flight wants some chains -- one for a single-part archive, one per unfinished
part of a split one, up to a cap -- and the next archive starts whenever the chains wanted
fall short of what the limiter currently allows, plus a margin, so that a stream that
frees always has a chain waiting for it. The limiter is shared by every walk, so when
Dropbox pushes back, every walk slows together, and as the limiter earns more streams,
more archives start.

The walk itself is the caller's: `Runner` only schedules, retries, counts and reports.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

from dbaudit.units import human_bytes, human_duration

log = logging.getLogger("dbaudit")


@dataclass
class Job:
    """One archive to walk. ``chains`` is how many it wants at once."""
    key: str
    label: str
    size: int
    chains: int
    folder: str = ""
    name: str = ""
    archive_id: int | None = None
    retried: bool = False
    #: (folder, name) of each copy of the same content, tried in turn if this one has gone.
    copies: tuple = ()


@dataclass
class Done:
    """What one walk came to. ``state`` is the archive's state afterwards, or `stopped`
    (resumable), `gone` (no longer where the listing said) or `done` (walked before, or
    being walked by another job)."""
    state: str
    members: int = 0
    requests: int = 0
    bug: bool = False


@dataclass
class Report:
    outcomes: Counter
    started: int
    stopped: bool
    failed: bool


class Runner:
    """Schedules ``walk(job) -> Done`` over a list of jobs, each on its own thread.

    ``margin`` is how many chains beyond the limiter's concurrency may be waiting for a
    stream. An archive whose walk ends in `error` goes to the back of the queue once. A
    walk that reports a bug, or raises, counts towards ``max_bugs``; reaching it sets
    ``stop``, and the run fails. ``limit`` caps how many archives the run starts.
    """

    def __init__(self, walk, limiter, stop, *, margin=2, max_bugs=3, limit=0,
                 summary_every=600.0, on_bug=None, clock=time.monotonic):
        self.walk, self.limiter, self.stop = walk, limiter, stop
        self.margin, self.max_bugs, self.limit = margin, max_bugs, limit
        self.summary_every, self.on_bug, self.clock = summary_every, on_bug, clock

    def run(self, jobs) -> Report:
        waiting = deque(sorted(jobs, key=lambda job: -job.size))
        flying = {}                         # job key -> (job, when it started)
        finished = queue.Queue()
        outcomes, bugs, started = Counter(), 0, 0
        done_size, recent = 0, deque()      # (when, size) of each walk that finished
        granted0, started_at = self.limiter.granted_total, self.clock()
        last_summary, failed = started_at, False

        def work(job):
            try:
                done = self.walk(job)
            except Exception as exc:
                log.exception("%s: unexpected failure", job.label)
                if self.on_bug is not None:
                    try:
                        self.on_bug(job, exc)
                    except Exception:
                        log.exception("%s: could not record the failure", job.label)
                done = Done("error", bug=True)
            finished.put((job, done))

        while waiting or flying:
            if self.limit and started >= self.limit:
                # Only second tries are left to start: archives this run already began.
                waiting = deque(job for job in waiting if job.retried)
            if not self.stop.is_set():
                capacity = self.limiter.concurrency + self.margin
                wanted = sum(job.chains for job, _ in flying.values())
                while waiting and wanted < capacity:
                    if self.limit and started >= self.limit and not waiting[0].retried:
                        break
                    job = waiting.popleft()
                    started += not job.retried
                    flying[job.key] = (job, self.clock())
                    wanted += job.chains
                    threading.Thread(target=work, args=(job,), daemon=True,
                                     name=f"archive {job.label}").start()
            if not flying:
                break                       # stopping, or the limit is reached
            try:
                job, done = finished.get(timeout=1.0)
            except queue.Empty:
                job = None
            while job is not None:
                took = human_duration(self.clock() - flying.pop(job.key)[1])
                if done.bug:
                    bugs += 1
                    if bugs >= self.max_bugs and not failed:
                        failed = True
                        log.error("stopping: %d walks failed unexpectedly", bugs)
                        self.stop.set()
                if done.state == "error" and not job.retried and not done.bug:
                    job.retried = True
                    waiting.append(job)
                    log.info("%s: error after %s; trying it once more at the end of the "
                             "queue", job.label, took)
                else:
                    outcomes[done.state] += 1
                    if done.state != "stopped":
                        done_size += job.size
                        recent.append((self.clock(), job.size))
                    log.info("%s: %s (%s members, %s requests, %s)", job.label, done.state,
                             f"{done.members:,}", f"{done.requests:,}", took)
                try:
                    job, done = finished.get_nowait()
                except queue.Empty:
                    job = None
            now = self.clock()
            if now - last_summary >= self.summary_every:
                granted = self.limiter.granted_total
                rate = (granted - granted0) / (now - last_summary) if now > last_summary else 0.0
                granted0, last_summary = granted, now
                while recent and recent[0][0] < now - 3600:
                    recent.popleft()
                self._summary(outcomes, flying, waiting, done_size, recent,
                              now - started_at, rate)
        return Report(outcomes, started, self.stop.is_set() and not failed, failed)

    def _summary(self, outcomes, flying, waiting, done_size, recent, elapsed, rate):
        """What is left, in flight or queued, at the last hour's pace: by count and by
        size. The largest archives go first, so the truth lies between the two."""
        window = min(elapsed, 3600.0)
        left = sum(job.size for job, _ in flying.values()) + sum(job.size for job in waiting)
        recent_bytes = sum(size for _, size in recent)
        by_count = ((len(flying) + len(waiting)) * window / len(recent) if recent
                    else float("inf"))
        by_size = left * window / recent_bytes if recent_bytes else float("inf")
        log.info("progress: %s done, %d in flight, %d queued | %s of %s | "
                 "%d streams, %.1f requests/s | eta %s by count, %s by size",
                 f"{sum(outcomes.values()):,}", len(flying), len(waiting),
                 human_bytes(done_size), human_bytes(done_size + left),
                 self.limiter.concurrency, rate, human_duration(by_count),
                 human_duration(by_size))
