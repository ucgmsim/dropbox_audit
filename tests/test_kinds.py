import pytest

from dbaudit.kinds import (
    GROUPS,
    KINDS,
    LF_SPLIT,
    SHORT,
    KindsError,
    Rules,
    by_archive_name,
    load_rules,
    shares,
)

CASES = [  # (folder, name, rules, kind): name rules first, in order, then folders
    ("/r/vm", "faultsz03_xyts.e3d", {}, "xyts"),
    ("/r/a", "faultsz03.xyzts.e3d", {}, "xyts"),
    ("/r/a", "station.mseed", {}, "wf_recorded"),
    ("/r/a", "event.v1a", {}, "wf_recorded"),
    ("/r/observed_db/2020", "event.zip", {"recorded": ("Observed_DB",)}, "wf_recorded"),
    ("/r/observed_db/2020", "sta.090", {"recorded": ("observed_db",)}, "wf_recorded"),
    ("/r/a", "event.zip", {}, "other"),
    ("/r/sources", "vp3dfile.p", {}, "vm"),
    ("/r/a", "in_basin_mask.b", {}, "vm"),
    ("/r/a", "velocity_model.h5", {}, "vm"),
    ("/r/a", "model.pertb", {}, "vm_pert"),
    ("/r/a", "perturbation_01.bin", {}, "vm_pert"),
    ("/r/a", "faultsz03.bb", {}, "wf_nc_bb"),
    ("/r/a", "r12_hf.h5", {}, "wf_h5_hf"),
    ("/r/a", "broadband.h5", {}, "wf_h5_bb"),
    ("/r/a", "seis_lf.nc", {}, "wf_seis_nc"),
    ("/r/a", "faultsz03_seis-00001.e3d", {}, "wf_seis"),
    (
        "/r/lf_netcdf/a",
        "station.nc",
        {"lf_converted": ("lf_netcdf",)},
        "wf_lf_converted",
    ),
    ("/r/a", "station.nc", {}, "other"),
    ("/r/a", "faultsz03_hf.bin", {}, "wf_bin_hf"),
    ("/r/a", "bb.bin", {}, "wf_bin_bb"),
    ("/r/a", "sta.000", {}, "wf_ascii"),
    ("/r/a", "faultsz03_bb.tar", {}, "wf_bin_bb"),
    ("/r/a", "faultsz03_lf.tar", {}, "lf_mix"),
    ("/r/a", "data.tar", {}, "unknown_archive"),
    ("/r/vm/a", "readme.txt", {}, "vm"),
    ("/r/v01p0_vm", "notes.txt", {}, "vm"),
    ("/r/velmodels/a", "notes.txt", {"vm_folders": ("velmodels",)}, "vm"),
    ("/r/velmodels/a", "notes.txt", {}, "other"),
    ("/r/a", "faultsz03.srf", {}, "src"),
    ("/r/srf/a", "notes.txt", {}, "src"),
    ("/r/a/sources", "notes.txt", {}, "src"),
    ("/r/a", "results.csv", {}, "im"),
    ("/r/im_results/a", "notes.txt", {"im_folders": ("im_results",)}, "im"),
    ("/r/a", "notes.txt", {}, "other"),
]


@pytest.mark.parametrize("folder,name,rules,kind", CASES)
def test_a_file_gets_the_kind_its_folder_and_name_say(folder, name, rules, kind):
    assert Rules(**rules).classify(folder, name) == kind


@pytest.mark.parametrize(
    "folder,name,kind",
    [
        ("/r/a", "x_hf_3.tar", "wf_bin_hf"),
        ("/r/runs_bb", "a.tar", "wf_bin_bb"),
        ("/r/a", "outbin.tar", "lf_mix"),
        ("/r/a", "x_im.tar", "im"),
        ("/r/a", "sources.tar", "src"),
        ("/r/a", "v01p0_vms.tar", "vm"),
        ("/r/a", "data.tar", "unknown_archive"),
    ],
)
def test_a_tarball_nobody_listed_is_judged_by_its_name(folder, name, kind):
    assert by_archive_name(folder, name) == kind


def test_keep_matches_path_prefixes_without_case():
    rules = Rules(keep=("/TeamSpace/Scratch/Proj00001/",))
    assert rules.kept("/teamspace/scratch/proj00001/v01p0")
    assert not rules.kept("/teamspace/scratch/proj00002/v01p0")


def test_every_kind_has_a_label_and_a_group_with_a_short_name():
    made = {kind for *_, kind in CASES} - {"lf_mix"}
    assert made | set(LF_SPLIT) <= set(KINDS)
    assert {group for _, group in KINDS.values()} == set(GROUPS) == set(SHORT)


def test_an_unlisted_lf_tarball_is_shared_out_like_the_listed_ones():
    assert shares("lf_mix") == LF_SPLIT and sum(LF_SPLIT.values()) == pytest.approx(1)
    assert shares("vm") == {"vm": 1.0}


def test_a_rules_file_gives_its_words(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text(
        'recorded = ["observed_db"]\nim_folders = ["im_results", "hazard"]\n'
        'lf_converted = ["lf_netcdf"]\nvm_folders = ["velmodels"]\n'
        'keep = ["/TeamSpace/scratch/proj00001/"]\n'
    )
    assert load_rules(str(path)) == Rules(
        ("observed_db",),
        ("im_results", "hazard"),
        ("lf_netcdf",),
        ("velmodels",),
        ("/teamspace/scratch/proj00001/",),
    )
    assert load_rules(None) == Rules()


@pytest.mark.parametrize(
    "text,named",
    [
        ('recordd = ["x"]', "recordd"),
        ('keep = "/x/"', "keep"),
        ("recorded = [1]", "recorded"),
        ("recorded = [", "TOML"),
    ],
)
def test_a_rules_file_that_cannot_be_used_says_why(tmp_path, text, named):
    path = tmp_path / "site.toml"
    path.write_text(text)
    with pytest.raises(KindsError, match=named):
        load_rules(str(path))


def test_a_missing_rules_file_is_named(tmp_path):
    with pytest.raises(KindsError, match="nope.toml"):
        load_rules(str(tmp_path / "nope.toml"))
