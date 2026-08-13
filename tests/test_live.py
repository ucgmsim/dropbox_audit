"""End-to-end tests against the real Dropbox account.

Opt in with ``DBAUDIT_LIVE=1``; they are skipped otherwise so the normal suite stays
offline and fast. They consume real API quota, so keep them small.

    DBAUDIT_LIVE=1 .venv/bin/python -m pytest tests/test_live.py -q
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DBAUDIT_LIVE") != "1",
    reason="set DBAUDIT_LIVE=1 to run against the real Dropbox account",
)

SUBTREE = os.environ.get("DBAUDIT_LIVE_ROOT", "/TeamSpace/arr65")
PYTHON = sys.executable


def dbaudit(*args, expect=0):
    proc = subprocess.run([PYTHON, "-m", "dbaudit", *args], capture_output=True, text=True)
    assert proc.returncode == expect, f"{args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout + proc.stderr


def test_full_pass_matches_an_independent_rclone_walk(tmp_path):
    db = str(tmp_path / "live.db")
    dbaudit("init", "--db", db, "--root", SUBTREE, "--min-free-gb", "1")
    dbaudit("run", "--db", db, "--workers", "4")

    status = dbaudit("status", "--db", db)
    assert "0 pending, 0 running, 0 error" in status

    out = dbaudit("verify", "--db", db, SUBTREE)
    assert "0 differences" in out, out


def test_interrupted_crawl_resumes_to_the_same_totals(tmp_path):
    """Kill a crawl outright, restart it, and land on identical numbers."""
    db = str(tmp_path / "resume.db")
    dbaudit("init", "--db", db, "--root", SUBTREE, "--min-free-gb", "1")

    killed = subprocess.Popen([PYTHON, "-m", "dbaudit", "run", "--db", db, "--workers", "2"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        killed.wait(timeout=4)
    except subprocess.TimeoutExpired:
        killed.kill()          # SIGKILL: no chance to clean up
        killed.wait()

    dbaudit("run", "--db", db, "--workers", "4")

    import sqlite3

    conn = sqlite3.connect(db)
    files, total = conn.execute("SELECT COUNT(*), SUM(size) FROM files").fetchone()
    duplicates = conn.execute(
        "SELECT COUNT(*) FROM (SELECT dbx_id FROM files GROUP BY dbx_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert duplicates == 0

    reference = str(tmp_path / "ref.db")
    dbaudit("init", "--db", reference, "--root", SUBTREE, "--min-free-gb", "1")
    dbaudit("run", "--db", reference, "--workers", "4")
    ref_files, ref_bytes = sqlite3.connect(reference).execute(
        "SELECT COUNT(*), SUM(size) FROM files"
    ).fetchone()

    assert (files, total) == (ref_files, ref_bytes)


def test_incremental_pass_costs_a_single_call(tmp_path):
    db = str(tmp_path / "inc.db")
    dbaudit("init", "--db", db, "--root", SUBTREE, "--min-free-gb", "1")
    dbaudit("run", "--db", db, "--workers", "4")

    import sqlite3

    before = sqlite3.connect(db).execute("SELECT COUNT(*), SUM(size) FROM files").fetchone()
    dbaudit("run", "--db", db, "--incremental")

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*), SUM(size) FROM files").fetchone() == before
    assert conn.execute("SELECT pages FROM shards WHERE path='<delta>'").fetchone()[0] == 1


def test_second_instance_is_refused(tmp_path):
    from dbaudit.lock import InstanceLock

    db = str(tmp_path / "lock.db")
    dbaudit("init", "--db", db, "--root", SUBTREE, "--min-free-gb", "1")
    with InstanceLock(f"{db}.lock"):
        dbaudit("run", "--db", db, expect=3)


def test_report_runs_on_real_data(tmp_path):
    db = str(tmp_path / "report.db")
    dbaudit("init", "--db", db, "--root", SUBTREE, "--min-free-gb", "1")
    dbaudit("run", "--db", db, "--workers", "4")
    out = dbaudit("report", "--db", db, "--top", "5")
    for section in ("Reconciliation", "Reclaimable duplicates", "Cold data",
                    "Largest directories", "By file type"):
        assert section in out, out
