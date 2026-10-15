"""Random archives walked through servers that get some requests wrong every time.

Opt-in and slow: `DBAUDIT_FUZZ=300 pytest tests/test_confirming_reads_fuzz.py` runs 300
seeds, each its own test, replayable by its id. Each builds a tar -- GNU long names, small
stored tarballs, one part or several -- and walks it through one of three faulty servers:

- ``request``: the first request to bring a chosen header to the walk -- not one that
  covers it only in the bytes it throws away -- is answered with another member's header
  there, and so is every request asked exactly like it;
- ``class``: every request that starts where that first one did, modulo 64 KiB or 1 MiB,
  is answered with bytes from further back;
- ``block``: Dropbox's own, every request that starts on a block's last byte answered with
  the block from its start, met by a first pass asked from the byte before a header that
  starts a block.

Every walk must end `complete` with every member true, or raise UnsettledRead: a server
that keeps contradicting itself is a reason to stop, never to record something false.
The faults stay within what the walk alone is built to catch: no copied header that lands
on another real one (the audit's to find, or no one's), no request pinned at both ends of
a small part (which shapes cannot move).
"""

import io
import os
import random
import tarfile

import pytest

from dbaudit.archive.parts import ArchiveSet
from dbaudit.archive.reader import ConcatFile
from dbaudit.archive.tarwalk import Member, UnsettledRead, walk
from tests.archive_fakes import build_tar, suffix

TRIALS = int(os.environ.get("DBAUDIT_FUZZ", "0"))
#: No part smaller than the largest window and two leads, so no request can run from a
#: part's first byte to its last: shapes cannot move such a request (test_archive_reader).
MIN_PART = (64 << 10) + 2 * (64 << 10)
pytestmark = pytest.mark.skipif(not TRIALS, reason="slow: set DBAUDIT_FUZZ=<seeds> to run")


class _Server:
    """Parts held in memory, answered through ``fault(part, offset, length, true)``."""

    def __init__(self, parts, fault):
        self.parts, self.fault = parts, fault
        self.requests = self.bytes_fetched = 0

    def read_range(self, part_idx, offset, length):
        true = self.parts[part_idx][offset:offset + length]
        assert len(true) == length
        self.requests += 1
        self.bytes_fetched += length
        return self.fault(part_idx, offset, length, true)


def _members(rng, filler=0):
    members = [("run/filler.bin", b"f" * filler)] if filler else []
    for i in range(rng.randint(5, 30)):
        roll = rng.random()
        if roll < 0.15:
            inner = build_tar([(f"in/{j}.dat", bytes([65 + j]) * rng.randint(1, 900))
                               for j in range(rng.randint(1, 3))])
            members.append((f"run/s{i:02d}.tar", inner))
            continue
        name = (f"run/{'d' * rng.randint(100, 180)}/f{i:02d}.bin" if roll < 0.3
                else f"run/f{i:02d}.bin")
        size = rng.choice([rng.randint(0, 3000), rng.randint(3000, 20_000),
                           rng.randint(70_000, 300_000)])
        members.append((name, bytes([1 + i % 250]) * size))
    return members + [("run/zz.bin", b"z" * MIN_PART)]


def _split(rng, data, whole):
    """One part, or several of MIN_PART or more."""
    size = len(data) if whole else rng.randint(MIN_PART, 4 * MIN_PART)
    if 0 < len(data) % size < MIN_PART:
        size = len(data)
    chunks = [data[at:at + size] for at in range(0, len(data), size)]
    archive = ArchiveSet.from_entries(
        [{"name": f"a.tar.{suffix(i)}", "size": len(c), "id": f"id:{i}",
          "content_hash": f"{i:064x}"} for i, c in enumerate(chunks)], "a.tar")
    return archive, chunks


@pytest.mark.parametrize("seed", range(TRIALS))
def test_a_walk_is_never_fooled_by_a_bad_request_shape(seed):
    rng = random.Random(seed)
    kind = rng.choice(["request", "class", "block"])
    block = rng.choice([1 << 16, 1 << 20]) if kind == "block" else None
    data = build_tar(_members(rng, filler=block - 512 if block else 0),
                     format=tarfile.GNU_FORMAT)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        infos = list(tar)
    truth = {m.offset: Member.from_tarinfo(m) for m in infos}
    archive, chunks = _split(rng, data, whole=kind == "block")
    # Where a copied header must not send the walk: onto any header, outer or stored.
    headers = {m.offset for m in infos} | {m.offset_data - 512 for m in infos}
    stored = [(m.offset_data, m.offset_data + m.size) for m in infos
              if m.name.endswith(".tar")]
    at = rng.choice(infos).offset
    state = {"bad": None}

    def request(part_idx, offset, length, true):
        start = archive.parts[part_idx].offset + offset
        if state["bad"] is None and start <= at < start + length:
            state["bad"] = (part_idx, offset, length)
        if (part_idx, offset, length) != state["bad"]:
            return true
        state["answered"] = True
        cut = at - start
        return true[:cut] + donor[:length - cut] + true[cut + 512:]

    def shifted(part_idx, offset, length, back):
        source = max(offset - back, 0)
        return chunks[part_idx][source:source + length]

    def klass(part_idx, offset, length, true):
        start = archive.parts[part_idx].offset + offset
        if state["bad"] is None and start <= at < start + length:
            # Not the class of a part's first byte, where every request near it starts
            # once its lead is cut: shapes cannot move those.
            state["bad"] = offset % modulus or -1
        if offset % modulus != state["bad"]:
            return true
        return shifted(part_idx, offset, length, back)

    def edge(part_idx, offset, length, true):
        if offset % block != block - 1:
            return true
        return shifted(part_idx, offset, length, block - 1)

    shape = None
    if kind == "request":
        plain = [m for m in infos if m.offset_data - m.offset == 512 and m.offset != at]
        size_at = next(m.size for m in infos if m.offset == at)

        def lands(m):
            landing = at + 512 + -(-m.size // 512) * 512
            return (landing in headers or any(a <= landing < b for a, b in stored)
                    or -(-m.size // 512) == -(-size_at // 512))

        plain = [m for m in plain if not lands(m)]
        if not plain:
            pytest.skip("no header to copy that the walk alone must catch")
        copied = rng.choice(plain).offset
        donor = data[copied:copied + 512]
        fault = request
    elif kind == "class":
        modulus = rng.choice([1 << 16, 1 << 20])
        back = rng.randrange(1, 1 << 16, 2)         # odd: never a header's alignment
        fault = klass
    else:
        fault = edge
        shape = 0                                    # the first pass asks from the byte before
    class Kept(ConcatFile):
        """Notes whether a bad answer reached the bytes a fill keeps. One that fell where
        the fill throws bytes away was never read, so the fault waits for the next request
        to cover the header."""

        def _fill(self, pos, need):
            super()._fill(pos, need)
            start, end = self.cached()
            if state.pop("answered", False):
                if start <= at < end:
                    state["landed"] = True
                elif not state.get("landed"):
                    state["bad"] = None

    windows = rng.choice([(4096, 4096), (4096, 65536), (16384, 16384)])
    rows = {}
    try:
        result = walk(Kept(archive, _Server(chunks, fault), window_min=windows[0],
                           window_max=windows[1], shape=shape), 0,
                      lambda batch, _next: rows.update((m.hdr_offset, m) for m in batch))
    except UnsettledRead:
        result = None                                # stopped: never a false verdict
    if kind == "request":
        assert state.get("landed"), "the bad answer never reached bytes the walk kept"
    if result is not None:
        assert result.state == "complete", f"{kind}: {result}"
        assert rows == truth, f"{kind}: rows differ"
