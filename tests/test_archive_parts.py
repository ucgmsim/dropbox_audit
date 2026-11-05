from dataclasses import replace

import pytest

from dbaudit.archive.parts import ArchiveSet, ArchiveSetError, Part


def entries(*specs):
    return [
        {
            "name": n,
            "size": s,
            "path_display": f"/d/{n}",
            "id": f"id:{n}",
            "rev": "r",
            "content_hash": "ab" * 32,
        }
        for n, s in specs
    ]


def test_orders_split_suffixes_and_maps_offsets():
    s = ArchiveSet.from_entries(entries(("a.tar.ab", 100), ("a.tar.aa", 300)), "a.tar")
    assert [p.name for p in s.parts] == ["a.tar.aa", "a.tar.ab"]
    assert [p.offset for p in s.parts] == [0, 300]
    assert s.total_size == 400
    assert s.locate(0) == (0, 0)
    assert s.locate(299) == (0, 299)
    assert s.locate(300) == (1, 0)


def test_a_whole_tar_is_a_one_part_set():
    """A plain .tar must not be read as a part whose suffix happens to be 'tar'."""
    s = ArchiveSet.from_entries(entries(("a.tar", 512)), "a.tar")
    assert len(s.parts) == 1 and s.total_size == 512


def test_reads_that_straddle_a_boundary_split_per_part():
    s = ArchiveSet.from_entries(
        entries(("a.tar.aa", 10), ("a.tar.ab", 10), ("a.tar.ac", 10)), "a.tar"
    )
    assert s.slices(8, 6) == [(0, 8, 2), (1, 0, 4)]
    assert s.slices(5, 20) == [(0, 5, 5), (1, 0, 10), (2, 0, 5)]
    assert s.slices(25, 99) == [(2, 5, 5)]  # clipped at the end of the archive


def test_a_missing_part_is_refused_and_named():
    with pytest.raises(ArchiveSetError, match="ab"):
        ArchiveSet.from_entries(entries(("a.tar.aa", 1), ("a.tar.ac", 1)), "a.tar")


def test_numeric_and_part_dash_suffixes_are_understood():
    s = ArchiveSet.from_entries(entries(("a.tar.001", 1), ("a.tar.000", 1)), "a.tar")
    assert [p.name for p in s.parts] == ["a.tar.000", "a.tar.001"]
    s2 = ArchiveSet.from_entries(
        entries(("a.tar.part-01", 1), ("a.tar.part-00", 1)), "a.tar"
    )
    assert [p.name for p in s2.parts] == ["a.tar.part-00", "a.tar.part-01"]


def test_a_foreign_name_is_refused():
    with pytest.raises(ArchiveSetError):
        ArchiveSet.from_entries(entries(("a.tar.aa", 1), ("b.tar.ab", 1)), "a.tar")


def test_set_hash_survives_a_move_but_not_a_content_change():
    plain = entries(("a.tar.aa", 5))
    moved = entries(("a.tar.aa", 5))
    moved[0]["path_display"] = "/somewhere/else/a.tar.aa"
    changed = entries(("a.tar.aa", 5))
    changed[0]["content_hash"] = "cd" * 32
    base = ArchiveSet.from_entries(plain, "a.tar").set_hash()
    assert ArchiveSet.from_entries(moved, "a.tar").set_hash() == base
    assert ArchiveSet.from_entries(changed, "a.tar").set_hash() != base


@pytest.mark.parametrize(
    "names, missing",
    [
        (("a.tar.ab", "a.tar.ac"), "aa"),
        (("a.tar.aab", "a.tar.aac"), "aaa"),
        (("a.tar.02", "a.tar.03"), "01"),
    ],
)
def test_a_set_missing_its_first_part_is_refused_and_the_part_named(
    names: tuple[str, ...],
    missing: str,
):
    """Only gaps *between* parts used to be caught, so with the first part gone the
    second quietly became part 0 -- and `index` then called the archive corrupt at
    offset 0, blaming the data for a missing file, a verdict it refuses to revisit.
    These parts have moved between folders before, so a listing taken mid-move is
    exactly how this would happen."""
    with pytest.raises(ArchiveSetError, match=missing):
        ArchiveSet.from_entries(entries(*((n, 1) for n in names)), "a.tar")


@pytest.mark.parametrize(
    "names",
    [
        ("a.tar.00", "a.tar.01"),  # split -d
        ("a.tar.01", "a.tar.02"),  # tools that count from one
        ("a.tar.aa",),
    ],
)
def test_a_sequence_may_start_where_split_starts_it(names: tuple[str, ...]):
    s = ArchiveSet.from_entries(entries(*((n, 1) for n in names)), "a.tar")
    assert [p.name for p in s.parts] == list(names)


def test_no_files_at_all_is_refused_naming_the_archive():
    with pytest.raises(ArchiveSetError, match="no files given for 'a.tar'"):
        ArchiveSet.from_entries([], "a.tar")


def test_a_whole_file_beside_other_files_is_refused():
    with pytest.raises(ArchiveSetError, match="exists alongside 1 other files"):
        ArchiveSet.from_entries(entries(("a.tar", 5), ("a.tar.aa", 5)), "a.tar")


def test_parts_whose_suffixes_differ_in_width_are_refused():
    with pytest.raises(ArchiveSetError, match="mixed suffix widths"):
        ArchiveSet.from_entries(entries(("a.tar.aa", 1), ("a.tar.abc", 1)), "a.tar")


def test_parts_numbered_in_some_places_and_lettered_in_others_are_refused():
    with pytest.raises(ArchiveSetError, match="mixed suffix styles"):
        ArchiveSet.from_entries(entries(("a.tar.aa", 1), ("a.tar.01", 1)), "a.tar")


@pytest.mark.parametrize("offset", [-1, 20])
def test_an_offset_outside_the_archive_is_refused(offset: int):
    s = ArchiveSet.from_entries(entries(("a.tar.aa", 10), ("a.tar.ab", 10)), "a.tar")
    with pytest.raises(ArchiveSetError, match=f"offset {offset} outside archive of 20"):
        s.locate(offset)


def test_a_set_whose_parts_overlap_cannot_place_an_offset_past_them():
    """Parts that do not run on from one another are not an archive; locate says so
    rather than return a position in the wrong part."""
    part = Part(0, "a.tar.aa", 10, 0, "/d/a.tar.aa", "id:aa", "r", "ab" * 32)
    overlapping = ArchiveSet([part, replace(part, idx=1, name="a.tar.ab")])
    with pytest.raises(ArchiveSetError, match="offset 15 outside archive of 20"):
        overlapping.locate(15)
