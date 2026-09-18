"""Walks one small real archive on Dropbox. Opt in with DBAUDIT_LIVE=1.

FaultSZ03_Source.tar is 320 KB -- the smallest real tarball in the TeamSpace tree --
so this costs a handful of API calls.

Everything else in the suite walks tars the tests built themselves, which cannot say
whether Dropbox honours a Range the way the design assumes, whether a real GNU tar
parses the way a generated one does, or whether the content hash computed here is the
one Dropbox computed for the same bytes. This is the only place those get a real
answer, which is why it is worth the calls.

The archive is registered and downloaded once for the whole module: each rclone call is
a transfer the operator has to approve, and each listing spends the account's shared
API budget.
"""

import os
import subprocess
import sys
import tarfile

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DBAUDIT_LIVE") != "1",
    reason="set DBAUDIT_LIVE=1 to run against the real Dropbox account")

FOLDER = "/TeamSpace/Public/Simulations/v01p0/FaultSZ03"
NAME = "FaultSZ03_Source.tar"
PYTHON = sys.executable


def dbaudit(*args, expect=0):
    proc = subprocess.run([PYTHON, "-m", "dbaudit", *args], capture_output=True, text=True)
    assert proc.returncode == expect, f"{args}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Register the archive and fetch one local copy, once."""
    root = tmp_path_factory.mktemp("live")
    db = str(root / "archives.db")
    dbaudit("archive", "register", "--db", db, "--folder", FOLDER, "--name", NAME)
    local = root / NAME
    subprocess.run(["rclone", "copyto", f"dropbox:{FOLDER}/{NAME}", str(local)], check=True)
    return db, local


def test_a_real_archive_matches_a_downloaded_copy(live):
    db, local = live
    dbaudit("archive", "index", "--db", db, "--archive", NAME)
    assert "complete" in dbaudit("archive", "status", "--db", db)

    with tarfile.open(local, mode="r:") as reference:
        expected = sorted((m.name, m.size, m.offset) for m in reference)
    # Two empty listings compare equal, so the comparison below is only evidence if
    # the reference actually found members. This archive holds 109.
    assert len(expected) > 100, f"reference listing looks wrong: {len(expected)} members"

    from dbaudit.archive.store import ArchiveStore

    store = ArchiveStore(db)
    archive_id = store.get(NAME)["id"]
    rows = store.query_members(archive_id)
    got = sorted(((f"{r['dir']}/{r['name']}" if r["dir"] else r["name"]),
                  r["size"], r["hdr_offset"]) for r in rows)
    assert got == expected


def test_our_content_hash_is_the_one_dropbox_computed(live):
    """The only non-circular check of `_local_hash`.

    A local archive and a Dropbox one are meant to share an identity, so that
    registering parts from disk and re-registering them from the folder they were
    uploaded to updates one row rather than creating a second archive. Every offline
    test of that compares our hash against our own hash, so a wrong constant would be
    self-consistent and invisible. Dropbox's `content_hash`, recorded at registration,
    is the outside opinion.

    Note the limit: this archive is smaller than one 4 MiB block, so this pins the
    digest-of-digests construction but not the block boundary. Walking a multi-block
    archive would settle that, and is not worth a larger download here.
    """
    db, local = live

    from dbaudit.archive.store import ArchiveStore
    from dbaudit.cli import DROPBOX_HASH_BLOCK, _local_hash

    store = ArchiveStore(db)
    part = store.parts_of(store.get(NAME)["id"])[0]
    assert _local_hash(local) == part.content_hash
    assert part.size == local.stat().st_size
    assert local.stat().st_size < DROPBOX_HASH_BLOCK, (
        "this archive now spans several hash blocks; tighten this test to pin the "
        "block boundary as well as the construction")
