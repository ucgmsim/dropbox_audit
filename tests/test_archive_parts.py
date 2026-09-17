import pytest

from dbaudit.archive.parts import ArchiveSet, ArchiveSetError


def entries(*specs):
    return [{"name": n, "size": s, "path_display": f"/d/{n}", "id": f"id:{n}",
             "rev": "r", "content_hash": "ab" * 32} for n, s in specs]


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
        entries(("a.tar.aa", 10), ("a.tar.ab", 10), ("a.tar.ac", 10)), "a.tar")
    assert s.slices(8, 6) == [(0, 8, 2), (1, 0, 4)]
    assert s.slices(5, 20) == [(0, 5, 5), (1, 0, 10), (2, 0, 5)]
    assert s.slices(25, 99) == [(2, 5, 5)]      # clipped at the end of the archive


def test_a_missing_part_is_refused_and_named():
    with pytest.raises(ArchiveSetError, match="ab"):
        ArchiveSet.from_entries(entries(("a.tar.aa", 1), ("a.tar.ac", 1)), "a.tar")


def test_numeric_and_part_dash_suffixes_are_understood():
    s = ArchiveSet.from_entries(entries(("a.tar.001", 1), ("a.tar.000", 1)), "a.tar")
    assert [p.name for p in s.parts] == ["a.tar.000", "a.tar.001"]
    s2 = ArchiveSet.from_entries(entries(("a.tar.part-01", 1), ("a.tar.part-00", 1)), "a.tar")
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
