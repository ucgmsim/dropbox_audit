import calendar
import hashlib
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pytest

from dbaudit.kinds import (
    BY_NAME,
    LF_SPLIT,
    LOOSE,
    MEMBERS,
    YEAR,
    KindsError,
    Rules,
    WalkRow,
    archive_index,
    tally,
    walk,
)
from dbaudit.store import Store
from tests.listing_fakes import LISTED, archive, dir_of, file, listing
from tests.listing_fakes import archive_index as index_file

T = {y: calendar.timegm((y, 1, 1, 0, 0, 0)) for y in range(2019, 2027)}


_RULES = Rules()


def rows(
    store: Store,
    rules: Rules = _RULES,
    index: str | None = None,
    under: str | None = None,
):
    return sorted(walk(store, rules, archive_index(index) if index else None, under))


def by_kind(found: Iterable[WalkRow]):
    out = {}
    for _, kind, size, how, _ in found:
        out[kind, how] = out.get((kind, how), 0) + size
    return out


def test_a_loose_file_gets_its_kind_folder_and_own_date(tmp_path: Path):
    store = listing(
        tmp_path, file("/R/v01p0/vp3dfile.p", 100, modified="2024-01-01T00:00:00Z")
    )
    assert rows(store) == [(dir_of(store, "/R/v01p0"), "vm", 100, LOOSE, T[2024])]


def test_a_copy_takes_the_date_of_its_oldest_copy(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/a/x.txt", 5, "x", "2020-01-01T00:00:00Z"),
        file("/R/b/x.txt", 5, "x", "2025-01-01T00:00:00Z"),
    )
    assert [date for *_, date in rows(store)] == [T[2020], T[2020]]


def test_a_tarball_is_shared_over_its_listed_members_by_size(tmp_path: Path):
    store = listing(tmp_path, file("/R/runs/a.tar", 900, "A"))
    index = index_file(
        tmp_path,
        archive(
            "a.tar",
            "/R/runs",
            [("a.tar", 900, "A")],
            [
                ("run", "faultsz03_bb.bin", 600, T[2023]),
                ("run/v01p0/vm", "notes.txt", 300, T[2021]),
            ],
        ),
    )
    d = dir_of(store, "/R/runs")
    assert rows(store, index=index) == [
        (d, "vm", 300, MEMBERS, T[2021]),
        (d, "wf_bin_bb", 600, MEMBERS, T[2023]),
    ]


def test_each_part_and_copy_of_a_tarball_shares_out_its_own_bytes(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/t/a.tar.aa", 600, "P1"),
        file("/R/t/a.tar.ab", 400, "P2"),
        file("/R/copy/a.tar.aa", 600, "P1"),
    )
    index = index_file(
        tmp_path,
        archive(
            "a.tar",
            "/R/t",
            [("a.tar.aa", 600, "P1"), ("a.tar.ab", 400, "P2")],
            [("", "bb.bin", 900, T[2022]), ("", "x.srf", 100, T[2022])],
        ),
    )
    assert by_kind(rows(store, index=index)) == pytest.approx(
        {("wf_bin_bb", MEMBERS): 1440, ("src", MEMBERS): 160}
    )


@pytest.mark.parametrize(
    "state,listed,how",
    [
        ("walking", 400, BY_NAME),
        ("walking", 500, MEMBERS),
        ("stale", 499, BY_NAME),
        ("truncated", 100, MEMBERS),
        ("complete", 100, MEMBERS),
    ],
)
def test_a_listing_is_trusted_when_finished_or_at_least_half_the_archive(
    tmp_path: Path, state: str, listed: int, how: str
):
    store = listing(
        tmp_path,
        file("/R/t/faultsz03_bb.tar", 1000, "A", modified="2025-01-01T00:00:00Z"),
    )
    index = index_file(
        tmp_path,
        archive(
            "faultsz03_bb.tar",
            "/R/t",
            [("faultsz03_bb.tar", 1000, "A")],
            [("", "vp3dfile.p", listed, T[2021])],
            state=state,
        ),
    )
    (got,) = {row[3] for row in rows(store, index=index)}
    assert got == how
    if how == BY_NAME:
        assert rows(store, index=index) == [
            (dir_of(store, "/R/t"), "wf_bin_bb", 1000, BY_NAME, T[2025])
        ]


def test_an_archive_with_no_file_members_is_judged_by_name(tmp_path: Path):
    store = listing(tmp_path, file("/R/t/faultsz03_bb.tar", 1000, "A"))
    index = index_file(
        tmp_path,
        archive(
            "faultsz03_bb.tar",
            "/R/t",
            [("faultsz03_bb.tar", 1000, "A")],
            [("", "run", 0, T[2021], "5")],
        ),
    )
    assert by_kind(rows(store, index=index)) == {("wf_bin_bb", BY_NAME): 1000}


def test_a_trusted_row_beats_an_untrusted_one_and_the_latest_trusted_row_wins(
    tmp_path: Path,
):
    store = listing(tmp_path, file("/R/t/a.tar", 1000, "A"))
    part = [("a.tar", 1000, "A")]
    index = index_file(
        tmp_path,
        archive("a.tar", "/R/t", part, [("", "vp3dfile.p", 1000, T[2021])]),
        archive("a.tar", "/R/t", part, [("", "x.srf", 1000, T[2021])]),
        archive("a.tar", "/R/t", part, [("", "bb.bin", 10, T[2021])], state="walking"),
    )
    assert by_kind(rows(store, index=index)) == {("src", MEMBERS): 1000}


def test_a_finished_listing_beats_a_stale_one_registered_later(tmp_path: Path):
    # A stale row lists the archive as it was, before a part went: the finished walk of the
    # parts still there shows what they hold.
    store = listing(tmp_path, file("/R/t/a.tar.aa", 1000, "A"))
    index = index_file(
        tmp_path,
        archive(
            "a.tar",
            "/R/t",
            [("a.tar.aa", 1000, "A")],
            [("", "x.srf", 1000, T[2021])],
            state="truncated",
        ),
        archive(
            "a.tar",
            "/R/t",
            [("a.tar.aa", 1000, "A"), ("a.tar.ab", 500, "B")],
            [("", "vp3dfile.p", 1500, T[2021])],
            state="stale",
        ),
    )
    assert by_kind(rows(store, index=index)) == {("src", MEMBERS): 1000}


def test_a_tarball_the_index_does_not_know_is_judged_by_name(tmp_path: Path):
    store = listing(tmp_path, file("/R/t/faultsz03_lf.tar", 1000))
    assert by_kind(rows(store)) == pytest.approx(
        {(kind, BY_NAME): 1000 * w for kind, w in LF_SPLIT.items()}
    )


def test_archives_the_index_does_not_know_are_dated_by_their_own_time(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/t/big.tar.aa", 5, "A", "2025-01-01T00:00:00Z"),
        file("/R/u/big.tar.aa", 5, "A", "2019-01-01T00:00:00Z"),
        file("/R/t/x.tar.zst", 7, modified="2025-01-01T00:00:00Z"),
    )
    assert sorted((size, how, date) for _, _, size, how, date in rows(store)) == [
        (5, BY_NAME, T[2019]),
        (5, BY_NAME, T[2025]),
        (7, BY_NAME, T[2025]),
    ]


def test_split_parts_the_index_does_not_know_count_as_tarred(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/t/big.tar.partaa", 1),
        file("/R/t/data.partab", 2),
        file("/R/t/x_part_ac", 4),
        file("/R/t/notes.txt", 8),
    )
    assert by_kind(rows(store)) == {
        ("unknown_archive", BY_NAME): 1,
        ("other", BY_NAME): 6,
        ("other", LOOSE): 8,
    }


def test_kept_velocity_models_are_labelled_apart(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/Scratch/Proj00001/v01p0/vp3dfile.p", 3),
        file("/R/other/vp3dfile.p", 5, "other"),
    )
    rules = Rules(keep=("/r/scratch/proj00001/",))
    assert by_kind(rows(store, rules)) == {("vm_kept", LOOSE): 3, ("vm", LOOSE): 5}


def test_a_subtree_walks_only_the_files_below_it_but_dates_by_copies_anywhere(
    tmp_path: Path,
):
    store = listing(
        tmp_path,
        file("/R/a/x.txt", 1, "x", "2025-01-01T00:00:00Z"),
        file("/R/ab/y.txt", 2),
        file("/R/z/x.txt", 1, "x", "2019-01-01T00:00:00Z"),
    )
    assert [(size, date) for _, _, size, _, date in rows(store, under="/R/a")] == [
        (1, T[2019])
    ]


def test_an_archive_index_must_be_one(tmp_path: Path):
    store = listing(tmp_path)
    with pytest.raises(KindsError, match="holds no archive index"):
        archive_index(store.path)
    with pytest.raises(KindsError, match="holds no archive index"):
        archive_index(str(tmp_path / "missing.db"))


def test_an_archive_index_may_sit_at_a_path_with_odd_characters(tmp_path: Path):
    store = listing(tmp_path, file("/R/t/a.tar", 10, "A"))
    index = index_file(
        tmp_path,
        archive("a.tar", "/R/t", [("a.tar", 10, "A")], [("", "x.srf", 10, T[2021])]),
        name="odd ?#% name.db",
    )
    assert by_kind(rows(store, index=index)) == {("src", MEMBERS): 10}


def test_a_walk_leaves_both_databases_as_they_were(tmp_path: Path):
    store = listing(tmp_path, file("/R/t/a.tar", 10, "A"), file("/R/x.txt", 1))
    index = index_file(
        tmp_path,
        archive("a.tar", "/R/t", [("a.tar", 10, "A")], [("", "x.srf", 10, T[2021])]),
    )
    store.connect().execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def state():
        # A write lands in the write-ahead log first, so the logs count as much as the files.
        # A reader may leave an empty log behind, which holds nothing.
        files = (store.path, store.path + "-wal", index, index + "-wal")
        return (
            [
                hashlib.sha256(open(f, "rb").read()).hexdigest()
                if os.path.exists(f) and os.path.getsize(f)
                else None
                for f in files
            ],
            store.connect().total_changes,
        )

    before = state()
    rows(store, index=index)
    assert state() == before


def test_tally_adds_bytes_by_kind_month_and_age_back_from_the_listing():
    found = [
        (1, "vm", 10, LOOSE, LISTED - 0.4 * YEAR),
        (1, "vm", 20, MEMBERS, LISTED - 1.5 * YEAR),
        (2, "xyts", 40, BY_NAME, LISTED - 6 * YEAR),
        (2, "xyts", 80, LOOSE, None),
    ]
    t = tally(found, LISTED)
    assert t.kinds == {"vm": [10, 20], "xyts": [80, 40]}
    assert (t.members, t.by_name, t.undated, t.total()) == (20, 40, 80, 150)
    assert t.older["vm", 0.5] == [0, 20] and t.older["vm", 1] == [0, 20]
    assert t.older["vm", 2] == [0, 0] and t.older["xyts", 5] == [0, 40]
    assert sum(sum(v) for v in t.months.values()) == 70
    assert t.groups() == {"velocity models": [10, 20], "xyts": [80, 40]}
    assert t.group_older(1) == {"velocity models": [0, 20], "xyts": [0, 40]}


def test_months_are_utc_and_odd_dates_still_count():
    late = calendar.timegm((2025, 12, 31, 23, 30, 0))
    t = tally(
        [
            (1, "vm", 1, LOOSE, late),
            (1, "vm", 2, LOOSE, LISTED + YEAR),
            (1, "vm", 4, MEMBERS, 0),
        ],
        LISTED,
    )
    assert t.months["vm", "2025-12"] == [1, 0] and t.months["vm", "1970-01"] == [0, 4]
    assert t.older["vm", 10] == [0, 4]  # mtime 0 is older than everything
    assert t.months["vm", "2027-10"] == [
        2,
        0,
    ]  # a date after the listing is older than nothing
    assert t.group_years()["velocity models"] == {1970: 4, 2025: 1, 2027: 2}


def test_files_with_no_bytes_leave_no_mark():
    t = tally([(1, "vm", 0, LOOSE, T[2019]), (1, "vm", 5, LOOSE, T[2020])], LISTED)
    assert t.group_years() == {"velocity models": {2020: 5}}
    assert list(t.months) == [("vm", "2020-01")]


def test_a_part_hash_that_is_not_hex_is_passed_over(tmp_path: Path):
    store = listing(tmp_path, file("/R/t/faultsz03_bb.tar", 10, "A"))
    index = index_file(
        tmp_path,
        archive(
            "faultsz03_bb.tar",
            "/R/t",
            [("faultsz03_bb.tar", 10, "A")],
            [("", "x.srf", 10, T[2021])],
        ),
    )
    conn = sqlite3.connect(index)
    conn.execute("UPDATE parts SET content_hash = 'not a hash'")
    conn.commit()
    conn.close()
    assert by_kind(rows(store, index=index)) == {("wf_bin_bb", BY_NAME): 10}
