"""Kinds of data: what each byte of a listing is, judged by its folder and file name.

A kind is a fine sort of file (LF binaries, velocity-model files, xyts, rupture sources and
so on), and each kind belongs to one group, the coarse sorts a reader is shown. The name
rules follow the outputs of EMOD3D-based ground-motion simulations. They run in order, so a
file named like an xyts or a velocity model wins over the folder it sits in. Every byte gets
a kind, but some kinds rest on naming conventions only: it is a first cut.

Folder names that only mean something on one Dropbox are not in the code. They come from a
TOML file of site rules (`load_rules`).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields

GROUPS = ("simulated waveforms", "recorded waveforms", "velocity models",
          "kept velocity models", "xyts", "rupture sources", "IM and hazard results", "other")
SHORT = {"simulated waveforms": "sim wf", "recorded waveforms": "rec wf",
         "velocity models": "vm", "kept velocity models": "vm kept", "xyts": "xyts",
         "rupture sources": "src", "IM and hazard results": "im", "other": "other"}
KINDS = {
    "wf_bin_bb": ("BB binary (.bin)", "simulated waveforms"),
    "wf_bin_hf": ("HF binary (.bin)", "simulated waveforms"),
    "wf_bin_lf": ("LF binary (.bin)", "simulated waveforms"),
    "wf_nc_bb": ("BB (.bb)", "simulated waveforms"),
    "wf_nc_hf": ("HF (.hf)", "simulated waveforms"),
    "wf_nc_lf": ("LF (.lf)", "simulated waveforms"),
    "wf_h5_bb": ("BB (HDF5)", "simulated waveforms"),
    "wf_h5_hf": ("HF (HDF5)", "simulated waveforms"),
    "wf_h5_lf": ("LF (HDF5)", "simulated waveforms"),
    "wf_seis": ("seis (.e3d)", "simulated waveforms"),
    "wf_seis_nc": ("seis (netCDF)", "simulated waveforms"),
    "wf_seis_tar": ("seis in unlisted LF tarballs", "simulated waveforms"),
    "wf_lf_converted": ("LF converted to netCDF", "simulated waveforms"),
    "wf_ascii": ("ASCII waveforms (.000, .090, .ver)", "simulated waveforms"),
    "wf_recorded": ("recorded waveforms", "recorded waveforms"),
    "vm": ("velocity-model files", "velocity models"),
    "vm_pert": ("velocity-model perturbations", "velocity models"),
    "vm_kept": ("kept velocity models", "kept velocity models"),
    "xyts": ("xyts", "xyts"),
    "src": ("rupture sources", "rupture sources"),
    "im": ("IM and hazard results", "IM and hazard results"),
    "other": ("other files", "other"),
    "unknown_archive": ("unlisted tarballs of unknown content", "other"),
}

VM_NAME = re.compile(r"^(vp3dfile|vs3dfile|rho3dfile|in_basin_mask)")
WF_BIN = re.compile(r"(^|_)(bb|hf|lf)\.bin$")
WF_H5 = re.compile(r"^r\d+_(lf|hf|bb)\.h5$")
BB_HF_TAR = re.compile(r"(^|_)(bb|hf)(_\d+)?\.tar")
VM_FOLDERS = ("vm", "vms", "velocity_model", "velocity_models", "velocitymodel", "velocitymodels")
# What the LF tarballs the index could list held, by bytes, in October 2026. An LF tarball it
# could not list (`lf_mix`) is shared out the same way.
LF_SPLIT = {"wf_seis_tar": 0.303, "xyts": 0.684, "other": 0.013}


class KindsError(ValueError):
    """A rules file or archive index that kinds cannot use. The message says why."""


@dataclass(frozen=True)
class Rules:
    """The classifier, given the folder words that only mean something on one Dropbox."""

    recorded: tuple[str, ...] = ()       # words in a folder path that mark recorded waveforms
    im_folders: tuple[str, ...] = ()     # folders that hold IM and hazard results
    lf_converted: tuple[str, ...] = ()   # words that mark LF converted to netCDF
    vm_folders: tuple[str, ...] = ()     # more folder names that hold velocity models
    keep: tuple[str, ...] = ()           # path prefixes whose velocity models are kept
    _im_dirs: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _vm_dir: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in RULE_KEYS:
            object.__setattr__(self, name, tuple(w.lower() for w in getattr(self, name)))
        object.__setattr__(self, "_im_dirs", tuple("/" + w.strip("/") for w in self.im_folders))
        words = VM_FOLDERS + self.vm_folders
        object.__setattr__(self, "_vm_dir", re.compile(
            r"/(" + "|".join(map(re.escape, words)) + r")(/|$)|_vm(/|$)"))

    def classify(self, folder: str, name: str) -> str:
        """The kind of a file, both arguments lowercased. `folder` is a folder path, or an
        archive's path, "::" and a member's folder."""
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if "xyts" in name or "xyzts" in name:
            return "xyts"
        if ext in ("mseed", "miniseed", "sac", "v1a", "v2a"):
            return "wf_recorded"
        if ext in ("zip", "000", "090", "ver") and any(w in folder for w in self.recorded):
            return "wf_recorded"
        if VM_NAME.match(name) or "velocity_model" in name:
            return "vm"
        if ".pertb" in name or name.startswith("perturbation_"):
            return "vm_pert"
        if ext in ("lf", "hf", "bb"):
            return "wf_nc_" + ext
        match = WF_H5.match(name)
        if match:
            return "wf_h5_" + match.group(1)
        if name == "broadband.h5":
            return "wf_h5_bb"
        if "seis" in name and ext == "nc":
            return "wf_seis_nc"
        if "seis" in name and ext == "e3d":
            return "wf_seis"
        if ext == "nc" and any(w in folder for w in self.lf_converted):
            return "wf_lf_converted"
        match = WF_BIN.search(name)
        if match:
            return "wf_bin_" + match.group(2)
        if ext in ("000", "090", "ver"):
            return "wf_ascii"
        if ext == "tar" or ".tar." in name:
            return by_archive_name(folder, name)
        if self._vm_dir.search(folder):
            return "vm"
        if (ext in ("srf", "stoch", "srfinfo") or "_source" in name or "/srf" in folder
                or "/sources/" in folder + "/"):
            return "src"
        if (ext in ("db", "csv", "parquet", "feather") or "intensity_measures" in name
                or any(d in folder for d in self._im_dirs)):
            return "im"
        return "other"

    def kept(self, folder: str) -> bool:
        """Is this folder (lowercased) under a `keep` prefix?"""
        return any(folder.startswith(prefix) for prefix in self.keep)


RULE_KEYS = tuple(f.name for f in fields(Rules) if f.init)


def load_rules(path: str | None) -> Rules:
    """The rules in a TOML file of site words, or the built-in rules alone without one."""
    if path is None:
        return Rules()
    try:
        with open(path, "rb") as handle:
            table = tomllib.load(handle)
    except OSError as exc:
        raise KindsError(f"cannot read the rules file {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise KindsError(f"{path} is not TOML: {exc}") from None
    for key, value in table.items():
        if key not in RULE_KEYS:
            raise KindsError(f"{path}: unknown key {key!r} (the keys are "
                             f"{', '.join(RULE_KEYS)})")
        if not isinstance(value, list) or not all(isinstance(w, str) for w in value):
            raise KindsError(f"{path}: {key} must be a list of strings")
    return Rules(**{key: tuple(value) for key, value in table.items()})


def by_archive_name(folder: str, name: str) -> str:
    """The kind of a tarball nobody could list (pax, gzip), from its name and folder."""
    match = BB_HF_TAR.search(name)
    if match:
        return "wf_bin_" + match.group(2)
    if folder.endswith("_bb") or "_bb/" in folder + "/":
        return "wf_bin_bb"
    if re.search(r"(^|_)lf\.tar", name) or "outbin" in name or "_lf_" in name:
        return "lf_mix"
    if re.search(r"(^|_)im\.tar", name):
        return "im"
    if "source" in name or "srf" in name:
        return "src"
    if re.search(r"(^|_)vms?\.tar", name) or "_vm_" in name:
        return "vm"
    return "unknown_archive"


def shares(kind: str) -> dict[str, float]:
    """A kind as {kind: fraction}: an unlisted LF tarball is shared out by LF_SPLIT."""
    return dict(LF_SPLIT) if kind == "lf_mix" else {kind: 1.0}
