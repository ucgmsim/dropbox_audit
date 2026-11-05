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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from dbaudit.units import human_bytes, human_duration

log = logging.getLogger("dbaudit")

#: How a walk ends having read its archive: only these count towards the TiB done, and
#: the pace. Gone, refused and done archives were not walked here; stopped and error ones
#: are left for the next run.
WALKED = frozenset({"complete", "truncated", "corrupt", "unsupported", "stale"})


class LimiterView(Protocol):
    """What the scheduler reads of the limiter every walk shares.

    `AdaptiveLimiter` satisfies it; so does any object with these two attributes.
    """

    @property
    def concurrency(self) -> int:
        """The number of requests allowed in flight now.

        Returns
        -------
        int
            The current limit.
        """
        ...

    @property
    def granted_total(self) -> int:
        """Every request let through so far.

        Returns
        -------
        int
            The count.
        """
        ...


@dataclass
class Job:
    """One archive to walk.

    ``chains`` is how many it wants at once: an estimate until its walk begins, then the
    number its walk reports as chains finish.

    Parameters
    ----------
    key : str
        Identifies the archive's content; unique among the jobs of a run.
    label : str
        What the logs call this job.
    size : int
        The archive's total size, in bytes; the largest are started first.
    chains : int
        How many chains the job wants at once, now.
    folder : str
        The Dropbox folder the archive was listed in.
    name : str
        The archive's file name.
    archive_id : int | None
        Its row in the index once registered; None before that.
    retried : bool
        Whether this is already the job's second try.
    copies : tuple[tuple[str, str], ...]
        ``(folder, name)`` of each copy of the same content.
    """

    key: str
    label: str
    size: int
    chains: int
    folder: str = ""
    name: str = ""
    archive_id: int | None = None
    retried: bool = False
    #: (folder, name) of each copy of the same content, tried in turn if this one has gone.
    copies: tuple[tuple[str, str], ...] = ()


@dataclass
class Done:
    """What one walk came to.

    ``state`` is the archive's state afterwards, or `stopped` (resumable), `gone` (no
    longer where the listing said) or `done` (walked before, or being walked by another
    job).

    Parameters
    ----------
    state : str
        The archive's state after the walk, or `stopped`, `gone` or `done`.
    members : int
        How many members the walk recorded.
    requests : int
        How many reads the walk made.
    bug : bool
        Whether the walk failed in a way that is a defect, not a fact about the archive.
    """

    state: str
    members: int = 0
    requests: int = 0
    bug: bool = False


@dataclass
class Report:
    """What a run came to.

    Parameters
    ----------
    outcomes : Counter[str]
        How many archives ended in each `Done.state`.
    started : int
        How many archives the run started, not counting second tries.
    stopped : bool
        Whether the run was stopped from outside before it finished.
    failed : bool
        Whether the run stopped itself after too many unexpected failures.
    """

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

    Parameters
    ----------
    walk : Callable[[Job], Done]
        Walks one archive; called on its own thread, once per start.
    limiter : LimiterView
        The limiter every walk shares; read for its concurrency and its grants.
    stop : threading.Event
        Set to stop starting archives; the run then waits for those in flight.
    margin : int
        Chains beyond the limiter's concurrency that may wait for a stream.
    max_bugs : int
        How many walks may report a bug or raise before the run stops itself.
    limit : int
        The most archives to start; 0 for no limit.
    summary_every : float
        Seconds between progress summaries.
    on_bug : Callable[[Job, Exception], None] | None
        Called with the job and the exception when a walk raises; None to only log it.
    clock : Callable[[], float]
        Monotonic seconds; replaced in tests.
    """

    def __init__(  # noqa: D107
        self,
        walk: Callable[[Job], Done],
        limiter: LimiterView,
        stop: threading.Event,
        *,
        margin: int = 2,
        max_bugs: int = 3,
        limit: int = 0,
        summary_every: float = 600.0,
        on_bug: Callable[[Job, Exception], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.walk, self.limiter, self.stop = walk, limiter, stop
        self.margin, self.max_bugs, self.limit = margin, max_bugs, limit
        self.summary_every, self.on_bug, self.clock = summary_every, on_bug, clock

    def run(self, jobs: Iterable[Job]) -> Report:
        """Walk every job, as the limiter allows, and report how it went.

        Parameters
        ----------
        jobs : Iterable[Job]
            The archives to walk; started largest first.

        Returns
        -------
        Report
            Outcomes by state, how many archives started, and whether the run stopped or failed.
        """
        waiting = deque(sorted(jobs, key=lambda job: -job.size))
        estimate = {job.key: job.chains for job in waiting}  # a second try starts here
        flying: dict[str, tuple[Job, float]] = {}  # job key -> (job, when it started)
        finished = queue.Queue()
        outcomes, bugs, started = Counter(), 0, 0
        done_size, recent = 0, deque()  # (when, size) of each walk that finished
        granted0, started_at = self.limiter.granted_total, self.clock()
        last_summary, failed = started_at, False

        def work(job: Job) -> None:
            """Run ``walk`` on one job, and hand its `Done` back through the queue.

            Parameters
            ----------
            job : Job
                The job to walk.
            """
            done = Done(
                "error", bug=True
            )  # unless the walk returns: even SystemExit reports
            try:
                done = self.walk(job)
            except Exception as exc:
                log.exception("%s: unexpected failure", job.label)
                if self.on_bug is not None:
                    try:
                        self.on_bug(job, exc)
                    except Exception:
                        log.exception("%s: could not record the failure", job.label)
            finally:
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
                    threading.Thread(
                        target=work,
                        args=(job,),
                        daemon=True,
                        name=f"archive {job.label}",
                    ).start()
            if not flying:
                break  # stopping, or the limit is reached
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
                    job.retried, job.chains = True, estimate[job.key]
                    waiting.append(job)
                    log.info(
                        "%s: error after %s; trying it once more at the end of the "
                        "queue",
                        job.label,
                        took,
                    )
                else:
                    outcomes[done.state] += 1
                    if done.state in WALKED:
                        done_size += job.size
                        recent.append((self.clock(), job.size))
                    log.info(
                        "%s: %s (%s members, %s requests, %s)",
                        job.label,
                        done.state,
                        f"{done.members:,}",
                        f"{done.requests:,}",
                        took,
                    )
                try:
                    job, done = finished.get_nowait()
                except queue.Empty:
                    job = None
            now = self.clock()
            if now - last_summary >= self.summary_every:
                granted = self.limiter.granted_total
                rate = (
                    (granted - granted0) / (now - last_summary)
                    if now > last_summary
                    else 0.0
                )
                granted0, last_summary = granted, now
                while recent and recent[0][0] < now - 3600:
                    recent.popleft()
                self._summary(
                    outcomes, flying, waiting, done_size, recent, now - started_at, rate
                )
        return Report(outcomes, started, self.stop.is_set() and not failed, failed)

    def _summary(
        self,
        outcomes: Counter[str],
        flying: dict[str, tuple[Job, float]],
        waiting: deque[Job],
        done_size: int,
        recent: deque[tuple[float, int]],
        elapsed: float,
        rate: float,
    ) -> None:
        """Log what is left, in flight or queued, at the last hour's pace.

        Estimated by count and by size. The largest archives go first, so the truth lies
        between the two.

        Parameters
        ----------
        outcomes : Counter[str]
            How many archives ended in each state so far.
        flying : dict[str, tuple[Job, float]]
            The jobs in flight, with when each started (clock seconds).
        waiting : deque[Job]
            The jobs not yet started.
        done_size : int
            Bytes of archive walked so far.
        recent : deque[tuple[float, int]]
            ``(clock seconds, bytes)`` of each walk that finished in the last hour.
        elapsed : float
            Seconds since the run began.
        rate : float
            Requests granted per second since the last summary.
        """
        window = min(elapsed, 3600.0)
        left = sum(job.size for job, _ in flying.values()) + sum(
            job.size for job in waiting
        )
        recent_bytes = sum(size for _, size in recent)
        by_count = (
            (len(flying) + len(waiting)) * window / len(recent)
            if recent
            else float("inf")
        )
        by_size = left * window / recent_bytes if recent_bytes else float("inf")
        log.info(
            "progress: %s done, %d in flight, %d queued | %s of %s | "
            "%d streams, %.1f requests/s | eta %s by count, %s by size",
            f"{sum(outcomes.values()):,}",
            len(flying),
            len(waiting),
            human_bytes(done_size),
            human_bytes(done_size + left),
            self.limiter.concurrency,
            rate,
            human_duration(by_count),
            human_duration(by_size),
        )
