import csv
from pathlib import Path

import pytest

from dbaudit.cli import main
from tests.listing_fakes import archive, file, listing
from tests.listing_fakes import archive_index as index_file

T2021 = 1609459200


def sample(tmp_path: Path):
    store = listing(
        tmp_path,
        file("/R/a/vp3dfile.p", 300),
        file("/R/t/faultsz03_bb.tar", 700, "A", modified="2021-01-01T00:00:00Z"),
    )
    index = index_file(
        tmp_path,
        archive(
            "faultsz03_bb.tar",
            "/R/t",
            [("faultsz03_bb.tar", 700, "A")],
            [("", "bb.bin", 700, T2021)],
        ),
    )
    return store.path, index


def test_kinds_prints_each_group_with_its_loose_and_tarred_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db, index = sample(tmp_path)
    assert main(["kinds", "--db", db, "--archive-db", index]) == 0
    out = capsys.readouterr().out
    assert "listed 2026-10-01" in out and "## By kind" in out and "## By age" in out
    assert "## By year" in out
    line = next(
        l for l in out.splitlines() if l.strip().startswith("simulated waveforms")
    )
    assert line.split()[2:] == ["700", "B", "70.0%", "0", "B", "700", "B"]
    assert "shared out over the members the archive index lists" in out
    assert "Undated" not in out


def test_kinds_says_how_much_it_could_not_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, file("/R/t/a.tar", 100, "A"))
    index = index_file(
        tmp_path,
        archive("a.tar", "/R/t", [("a.tar", 100, "A")], [("", "x.srf", 100, None)]),
    )
    out = tmp_path / "csv"
    assert (
        main(["kinds", "--db", store.path, "--archive-db", index, "--out", str(out)])
        == 0
    )
    assert "Undated: 100 B" in capsys.readouterr().out
    with open(out / "kinds_listing.csv") as f:
        (about,) = list(csv.DictReader(f))
    assert about["undated"] == "100"


def test_without_an_index_tarballs_are_judged_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db, _ = sample(tmp_path)
    assert main(["kinds", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "no --archive-db" in out and "give --archive-db to look inside them" in out


def test_a_group_shows_its_kinds_when_it_has_more_than_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path, file("/R/vp3dfile.p", 3), file("/R/model.pertb", 4))
    assert main(["kinds", "--db", store.path]) == 0
    out = capsys.readouterr().out
    assert "velocity-model files" in out and "velocity-model perturbations" in out


def test_kinds_writes_four_csv_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db, index = sample(tmp_path)
    out = tmp_path / "csv"
    assert main(["kinds", "--db", db, "--archive-db", index, "--out", str(out)]) == 0
    with open(out / "kinds.csv") as f:
        table = list(csv.DictReader(f))
    assert {(r["kind"], r["bytes"], r["loose"], r["tarred"]) for r in table} == {
        ("vm", "300", "300", "0"),
        ("wf_bin_bb", "700", "0", "700"),
    }
    with open(out / "kinds_listing.csv") as f:
        (about,) = list(csv.DictReader(f))
    assert (about["root"], about["listed"], about["finished"]) == (
        "/R",
        "2026-10-01T00:00:00+00:00",
        "true",
    )
    for name in ("kinds_by_month.csv", "kinds_older_than.csv"):
        assert (out / name).exists()


def test_kinds_on_an_empty_listing_prints_zeros(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path)
    assert main(["kinds", "--db", store.path]) == 0
    assert "0 B" in capsys.readouterr().out


def test_kinds_warns_when_the_crawl_is_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    store = listing(tmp_path)
    store.add_shard("/R/more", 1)
    assert main(["kinds", "--db", store.path]) == 0
    assert "WARNING: the crawl is incomplete" in capsys.readouterr().out


def test_kinds_covers_only_the_subtree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db, _ = sample(tmp_path)
    assert main(["kinds", "--db", db, "--subtree", "/R/a"]) == 0
    out = capsys.readouterr().out
    assert "Kinds of data in /R/a" in out and "simulated waveforms" not in out


@pytest.mark.parametrize(
    "extra,says",
    [
        (["--rules", "missing.toml"], "missing.toml"),
        (["--archive-db", "AUDIT"], "holds no archive index"),
        (["--subtree", "/R/nowhere"], "is not in"),
        (["--out", "FILE"], "is not a folder"),
    ],
)
def test_kinds_refuses_what_it_cannot_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra, says
):
    db, _ = sample(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    extra = [{"AUDIT": db, "FILE": str(tmp_path / "f.txt")}.get(a, a) for a in extra]
    assert main(["kinds", "--db", db, *extra]) == 2
    captured = capsys.readouterr()
    assert says in captured.err and captured.out == ""
    assert (tmp_path / "f.txt").read_text() == "x"


def test_kinds_on_an_uninitialised_database_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["kinds", "--db", str(tmp_path / "missing.db")]) == 2
    assert "is not initialised" in capsys.readouterr().err


def offline(monkeypatch: pytest.MonkeyPatch):
    def no_network(remote):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("dbaudit.cli._space_usage", no_network)


def test_report_ends_with_a_by_kind_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    offline(monkeypatch)
    db, index = sample(tmp_path)
    assert main(["report", "--db", db, "--archive-db", index]) == 0
    out = capsys.readouterr().out
    assert out.index("## By kind") > out.index("## By file type")
    section = out[out.index("## By kind") :]
    assert "> 10 years" in section and "simulated waveforms" in section
    assert "shared out over the members the archive index lists" in section


def test_report_refuses_a_bad_rules_file_before_printing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    offline(monkeypatch)
    db, _ = sample(tmp_path)
    assert main(["report", "--db", db, "--rules", str(tmp_path / "missing.toml")]) == 2
    captured = capsys.readouterr()
    assert "missing.toml" in captured.err and captured.out == ""
