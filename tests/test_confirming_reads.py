"""A fault tied to how a read is asked does not survive being asked again.

Dropbox answers some requests wrongly every time they are asked (reader.DROPBOX_BLOCK).
Every read made to confirm another -- the walk's confirming pass, `--recheck`, the audit, a
tie-break, a long-name or pax re-read, `cat` -- now asks differently from the read it checks
(ConcatFile.drop_cache), so such a fault shows up as a disagreement instead of being
confirmed. Each test here fails under `pytest --unshaped`, which asks as dbaudit did before.
"""

import io
import random
import tarfile

import dbaudit.archive.tarwalk as tarwalk
import dbaudit.cli as cli
from dbaudit.archive.reader import ConcatFile, LocalRangeReader
from dbaudit.archive.store import ArchiveStore
from dbaudit.archive.tarwalk import walk
from dbaudit.cli import main
from tests.archive_fakes import build_tar, write_parts

#: Members this big are jumped over, so no request made before one reaches past it.
BIG = 256 * 1024
#: One window size throughout, so that a request repeats exactly when it is asked the same way.
SMALL = ["--workers", "1", "--window-min", "4096", "--window-max", "4096"]


class _BadShape:
    """A server that answers one request wrongly every time it is asked exactly that: the
    first request to cover ``at`` while ``armed()`` holds. The answer carries ``wrong`` at
    ``at`` -- real bytes from elsewhere in the archive, as Dropbox's were -- and is true
    everywhere else; the same bytes asked for any other way come back true.

    ``first``, if given, is what the very first request to cover ``at`` carries there
    instead, once: an ordinary bad read, so that there is something to read again."""

    def __init__(self, directory, at, wrong, first=None, armed=lambda: True):
        self.directory, self.at, self.wrong, self.first = directory, at, wrong, first
        self.armed = armed
        self.shape = None           # the request it gets wrong: (part, offset, length)
        self.lied = 0               # how many times that request was asked

    def reader(self, archive):
        server = self

        class _Reader(LocalRangeReader):
            def read_range(self, part_idx, offset, length):
                data = super().read_range(part_idx, offset, length)
                return server.answer(self.archive, part_idx, offset, length, data)

        return _Reader(self.directory, archive)

    def reader_for(self, row, archive_set, tokens=None, limiter=None):
        return self.reader(archive_set)

    def answer(self, archive, part_idx, offset, length, data):
        start = archive.parts[part_idx].offset + offset
        if not start <= self.at < start + length:
            return data
        if self.first is not None:
            fill, self.first = self.first, None
        else:
            if self.shape is None and self.armed():
                self.shape = (part_idx, offset, length)
            if (part_idx, offset, length) != self.shape:
                return data
            self.lied += 1
            fill = self.wrong
        cut = self.at - start
        return data[:cut] + fill[:length - cut] + data[cut + len(fill):]


def _offsets(data):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        return {m.name: m.offset for m in tar}


def _jumped():
    """A big member, then ten small ones of different sizes: the walk jumps to the first
    small one's header, so the first request to cover it is the walk's own read of it."""
    members = [("run/big.bin", b"b" * BIG)] + [
        (f"run/m{i:02d}.bin", b"x" * (600 + 700 * i)) for i in range(10)]
    data = build_tar(members)
    return members, data, _offsets(data)


def _walk(tmp_path, data, server):
    archive = write_parts(tmp_path, data, part_size=len(data))
    rows = {}
    result = walk(ConcatFile(archive, server.reader(archive), window_min=4096,
                             window_max=4096), 0,
                  lambda batch, _next: rows.update((m.hdr_offset, m) for m in batch))
    return result, [f"{rows[k].dir}/{rows[k].name}" for k in sorted(rows)]


def _indexed(tmp_path, data):
    source = tmp_path / "parts"
    source.mkdir()
    write_parts(source, data, part_size=len(data))
    db = str(tmp_path / "archives.db")
    main(["archive", "register", "--db", db, "--local-dir", str(source), "--name", "a.tar"])
    return db, source


def _paths(db):
    store = ArchiveStore(db)
    return [f"{r['dir']}/{r['name']}" for r in store.connect().execute(
        "SELECT dir, name FROM members WHERE archive_id=? ORDER BY hdr_offset",
        (store.get("a.tar")["id"],))]


def _while(module, name, flag):
    """``module.name`` wrapped so that ``flag`` is non-empty while it runs."""
    original = getattr(module, name)

    def wrapped(*args, **kwargs):
        flag.append(name)
        try:
            return original(*args, **kwargs)
        finally:
            flag.pop()

    return wrapped


def test_the_confirming_pass_does_not_repeat_a_bad_request(tmp_path):
    """Review finding 1. The walk's read of member 0's header -- from the byte before it,
    as tarfile reads -- is answered with member 5's header, which sends the chain into
    member data. The confirming pass used to ask the same and agree: `corrupt`."""
    members, data, at = _jumped()
    wrong = data[at["run/m05.bin"]:at["run/m05.bin"] + 512]
    server = _BadShape(tmp_path, at["run/m00.bin"], wrong)

    result, paths = _walk(tmp_path, data, server)

    assert result.state == "complete", result.detail
    assert paths == [name for name, _ in members]
    assert f"re-read at {at['run/m00.bin']}" in result.detail
    assert server.lied == 1, "the bad request was asked again"


def test_recheck_does_not_repeat_the_request_that_condemned_an_archive(tmp_path,
                                                                     monkeypatch):
    """Review finding 1, `--recheck`. A walk that asked as before shapes confirmed the bad
    read and left the archive `corrupt`, as October's walks did. --recheck walks the chain
    again in a new shape, never asks that request, and finds the archive complete."""
    members, data, at = _jumped()
    wrong = data[at["run/m05.bin"]:at["run/m05.bin"] + 512]
    db, source = _indexed(tmp_path, data)
    server = _BadShape(source, at["run/m00.bin"], wrong)
    monkeypatch.setattr("dbaudit.cli._reader_for", server.reader_for)
    with monkeypatch.context() as patch:
        patch.setattr("dbaudit.archive.reader.LEAD_STEP", 0)
        assert main(["archive", "index", "--db", db, "--archive", "a.tar", *SMALL]) == 1
    assert ArchiveStore(db).get("a.tar")["state"] == "corrupt" and server.lied >= 2
    lied = server.lied

    assert main(["archive", "index", "--db", db, "--archive", "a.tar", "--recheck",
                 *SMALL]) == 0
    assert ArchiveStore(db).get("a.tar")["state"] == "complete"
    assert _paths(db) == [name for name, _ in members]
    assert server.lied == lied, "--recheck asked the bad request again"


def test_the_audits_second_read_does_not_repeat_its_first(tmp_path, monkeypatch):
    """Review finding 2. The archive holds one path twice, as `tar -r` leaves it, so the
    audit reads both rows again. Its read of the newer one is answered with member 7's
    header. The audit's second read used to ask the same and agree, so the segment was
    walked again, round after round, until the run gave up."""
    members, _, _ = _jumped()
    members = members + [("run/m03.bin", b"y" * 999)]
    data = build_tar(members)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        infos = list(tar)
    newer, donor = infos[-1].offset, infos[8].offset           # run/m03.bin again; m07
    db, source = _indexed(tmp_path, data)
    auditing = []
    server = _BadShape(source, newer, data[donor:donor + 512], armed=lambda: bool(auditing))
    monkeypatch.setattr("dbaudit.cli._reader_for", server.reader_for)
    monkeypatch.setattr("dbaudit.cli._audit_repeated_paths",
                        _while(cli, "_audit_repeated_paths", auditing))

    assert main(["archive", "index", "--db", db, "--archive", "a.tar", *SMALL]) == 0
    store = ArchiveStore(db)
    events = store.connect().execute(
        "SELECT kind, detail FROM events WHERE archive_id=?",
        (store.get("a.tar")["id"],)).fetchall()
    assert [d for k, d in events if k == "audit"] == [
        "read 2 rows under repeated paths again; 0 disagreed with the index"]
    assert not [k for k, _ in events if k == "audit_repair"]
    assert _paths(db) == [name for name, _ in members]
    assert server.lied == 1, "the audit asked the bad request again"


def test_a_tie_break_is_not_won_by_one_bad_request(tmp_path, monkeypatch):
    """Review finding 3, tie-breaks. The walk first reads member 0's header as member 5's;
    the confirming pass reads it truly; the tie-break's first read is answered with member
    7's. Its later reads used to ask the same, so member 7 had two votes and won."""
    members, data, at = _jumped()
    first = data[at["run/m05.bin"]:at["run/m05.bin"] + 512]
    wrong = data[at["run/m07.bin"]:at["run/m07.bin"] + 512]
    voting = []
    server = _BadShape(tmp_path, at["run/m00.bin"], wrong, first=first,
                       armed=lambda: bool(voting))
    monkeypatch.setattr(tarwalk, "_tie_break", _while(tarwalk, "_tie_break", voting))

    result, paths = _walk(tmp_path, data, server)

    assert result.state == "complete", result.detail
    assert paths == [name for name, _ in members]
    assert server.lied == 1, "a tie-break vote asked the bad request again"


def _pax(name):
    """An extended header and the member it applies to: 1,536 bytes of pax, as a stored
    pax tarball would carry them."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo(name)
        info.pax_headers = {"mtime": "1700000000.5"}
        tar.addfile(info, io.BytesIO(b""))
    block = buf.getvalue()[:1536]
    assert block[156:157] == b"x"
    return block


def test_a_fake_pax_header_is_not_confirmed_by_one_bad_request(tmp_path, monkeypatch):
    """Review finding 3, pax. The walk first reads a pax header where member 0's belongs;
    its re-read is answered with another pax header. The next re-read used to ask the same
    and agree, and pax confirmed twice made the archive `unsupported`."""
    members, data, at = _jumped()
    settling = []
    server = _BadShape(tmp_path, at["run/m00.bin"], _pax("elsewhere/two.dat"),
                       first=_pax("elsewhere/one.dat"), armed=lambda: bool(settling))
    monkeypatch.setattr(tarwalk, "_settled", _while(tarwalk, "_settled", settling))

    result, paths = _walk(tmp_path, data, server)

    assert result.state == "complete", result.detail
    assert paths == [name for name, _ in members]
    assert server.lied == 1, "a pax re-read asked the bad request again"


def test_a_garbled_long_name_is_not_confirmed_by_one_bad_request(tmp_path, monkeypatch):
    """Review finding 3, long names. The walk first reads member 0's long name garbled; its
    re-read is answered with another garbling. The next re-read used to ask the same and
    agree, and the garbled name was recorded."""
    long_name = "run/" + "d" * 150 + "/deep.bin"
    members = [("run/big.bin", b"b" * BIG), (long_name, b"x" * 700),
               ("run/after.bin", b"a" * 900)]
    data = build_tar(members, format=tarfile.GNU_FORMAT)
    name_block = 512 + BIG + 512            # the long name's data, after its own header

    def garbled(tag):
        return (b"run/garbled-%s/" % tag).ljust(512, b"\0")

    settling = []
    server = _BadShape(tmp_path, name_block, garbled(b"two"), first=garbled(b"one"),
                       armed=lambda: bool(settling))
    monkeypatch.setattr(tarwalk, "_settled", _while(tarwalk, "_settled", settling))

    result, paths = _walk(tmp_path, data, server)

    assert result.state == "complete", result.detail
    assert paths == [name for name, _ in members]
    assert server.lied == 1, "a long-name re-read asked the bad request again"


def test_cat_rereads_a_bad_body_in_a_new_shape(tmp_path, monkeypatch):
    """`cat` re-reads whatever does not line up. A body wrong for one request -- here the
    member's own header, replaced by another member's -- used to come back the same on
    every re-read, until cat refused the member. (Big members either side keep its reads
    clear of the part's ends, where shapes cannot move a request.)"""
    body = random.Random(20261015).randbytes(40_000)
    data = build_tar([("run/big.bin", b"b" * BIG), ("run/data.bin", body),
                      ("run/after.bin", b"a" * BIG)])
    at = _offsets(data)
    db, source = _indexed(tmp_path, data)
    assert main(["archive", "index", "--db", db, "--archive", "a.tar"]) == 0
    server = _BadShape(source, at["run/data.bin"],
                       data[at["run/after.bin"]:at["run/after.bin"] + 512])
    monkeypatch.setattr("dbaudit.cli._reader_for", server.reader_for)
    out = tmp_path / "data.bin"

    assert main(["archive", "cat", "--db", db, "--archive", "a.tar", "--member",
                 "run/data.bin", "--out", str(out)]) == 0
    assert out.read_bytes() == body
    assert server.lied == 1, "cat asked the bad request again"
