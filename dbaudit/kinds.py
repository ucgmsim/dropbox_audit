"""Kinds of data: what each byte of a listing is, judged by its folder and file name.

A kind is a fine sort of file (LF binaries, velocity-model files, xyts, rupture sources and
so on), and each kind belongs to one group, the coarse sorts a reader is shown. The name
rules follow the outputs of EMOD3D-based ground-motion simulations. They run in order, so a
file named like an xyts or a velocity model wins over the folder it sits in. Every byte gets
a kind, but some kinds rest on naming conventions only: it is a first cut.

Folder names that only mean something on one Dropbox are not in the code. They come from a
TOML file of site rules (`load_rules`).

`walk` is one read-only pass over an audit listing, and over the archive index when there is
one: every byte comes out with its kind, whether it is loose or inside a tarball, and its date.
A tarball the index lists is shared out over its members; any other tarball is judged by its
name. `tally` adds the bytes up by kind, by month and by age.
"""

from __future__ import annotations

import csv
import functools
import os
import re
import sqlite3
import time
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import report
from .archive.tarwalk import REGULAR_MEMBER_TYPES
from .store import Store

GROUPS = (
    "simulated waveforms",
    "recorded waveforms",
    "velocity models",
    "kept velocity models",
    "xyts",
    "rupture sources",
    "IM and hazard results",
    "other",
)
SHORT = {
    "simulated waveforms": "sim wf",
    "recorded waveforms": "rec wf",
    "velocity models": "vm",
    "kept velocity models": "vm kept",
    "xyts": "xyts",
    "rupture sources": "src",
    "IM and hazard results": "im",
    "other": "other",
}
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
VM_FOLDERS = (
    "vm",
    "vms",
    "velocity_model",
    "velocity_models",
    "velocitymodel",
    "velocitymodels",
)
# What the LF tarballs the index could list held, by bytes, in October 2026. An LF tarball it
# could not list (`lf_mix`) is shared out the same way.
LF_SPLIT = {"wf_seis_tar": 0.303, "xyts": 0.684, "other": 0.013}
# Tarballs, and split parts of one, by name: every name classify judges as an archive
# (.tar, or .tar. anywhere, as in .tar.aa, .tar.gz or .tar.zst), and parts named .partaa
# or _part_aa.
TARLIKE = re.compile(r"\.tar$|\.tar\.|\.part[a-z]{2}$|_part_[a-z]{2}$")
# How each byte was placed: a loose file, shared out over the members the archive index
# lists, or a tarball judged by its name.
LOOSE, MEMBERS, BY_NAME = "loose", "members", "name"
AGES = (0.5, 1, 2, 3, 5, 10)  # years back from the listing that ages are counted at
YEAR = 365.25 * 86400
# A walked archive whose listing holds all it has: the walk reached its end, or the tar
# itself ends early.
FINISHED = ("complete", "truncated")
FOLDER_CACHE = 1 << 16  # folders whose tests classify remembers


class KindsError(ValueError):
    """A rules file or archive index that kinds cannot use. The message says why."""


@dataclass(frozen=True)
class Rules:
    """The classifier, given the folder words that only mean something on one Dropbox.

    Parameters
    ----------
    recorded : tuple of str, optional
        Words in a folder path that mark recorded waveforms.
    im_folders : tuple of str, optional
        Folders that hold IM and hazard results.
    lf_converted : tuple of str, optional
        Words that mark LF output converted to netCDF.
    vm_folders : tuple of str, optional
        More folder names that hold velocity models, besides the built-in ones.
    keep : tuple of str, optional
        Path prefixes whose velocity models are kept.

    Notes
    -----
    Every word is lowercased on construction, so the matching is without case.
    """

    recorded: tuple[
        str, ...
    ] = ()  # words in a folder path that mark recorded waveforms
    im_folders: tuple[str, ...] = ()  # folders that hold IM and hazard results
    lf_converted: tuple[str, ...] = ()  # words that mark LF converted to netCDF
    vm_folders: tuple[str, ...] = ()  # more folder names that hold velocity models
    keep: tuple[str, ...] = ()  # path prefixes whose velocity models are kept
    _im_dirs: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _vm_dir: re.Pattern = field(init=False, repr=False, compare=False)
    _folder: Callable[[str], tuple[bool, bool, bool, bool, bool]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Lowercase the words and set up the folder tests and their cache."""
        for name in RULE_KEYS:
            object.__setattr__(
                self, name, tuple(w.lower() for w in getattr(self, name))
            )
        object.__setattr__(
            self, "_im_dirs", tuple("/" + w.strip("/") for w in self.im_folders)
        )
        words = VM_FOLDERS + self.vm_folders
        object.__setattr__(
            self,
            "_vm_dir",
            re.compile(r"/(" + "|".join(map(re.escape, words)) + r")(/|$)|_vm(/|$)"),
        )
        # Files come folder by folder, so each folder's tests are worked out once.
        object.__setattr__(
            self,
            "_folder",
            functools.lru_cache(maxsize=FOLDER_CACHE)(self._folder_tests),
        )

    def _folder_tests(self, folder: str) -> tuple[bool, bool, bool, bool, bool]:
        """Hold what classify asks of a folder.

        That is: recorded, LF converted, velocity models, sources and IM results.

        Parameters
        ----------
        folder : str
            A lowercased folder path.

        Returns
        -------
        tuple of bool
            Whether the folder is a recorded-waveform folder, an LF-converted folder, a
            velocity-model folder, a rupture-source folder and an IM-results folder, in that
            order.
        """
        return (
            any(w in folder for w in self.recorded),
            any(w in folder for w in self.lf_converted),
            self._vm_dir.search(folder) is not None,
            "/srf" in folder or "/sources/" in folder + "/",
            any(d in folder for d in self._im_dirs),
        )

    def classify(self, folder: str, name: str) -> str:
        """Return the kind of a file, both arguments lowercased.

        `folder` is a folder path, or an archive's path, "::" and a member's folder.

        Parameters
        ----------
        folder : str
            The lowercased folder path, or an archive's path, "::" and a member's folder.
        name : str
            The lowercased file name, or a member's base name.

        Returns
        -------
        str
            A key of KINDS, or ``"lf_mix"`` for an LF tarball nobody could list (see
            `shares`).
        """
        recorded, lf_converted, vm_folder, src_folder, im_folder = self._folder(folder)
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if "xyts" in name or "xyzts" in name:
            return "xyts"
        if ext in ("mseed", "miniseed", "sac", "v1a", "v2a"):
            return "wf_recorded"
        if ext in ("zip", "000", "090", "ver") and recorded:
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
        if ext == "nc" and lf_converted:
            return "wf_lf_converted"
        match = WF_BIN.search(name)
        if match:
            return "wf_bin_" + match.group(2)
        if ext in ("000", "090", "ver"):
            return "wf_ascii"
        if ext == "tar" or ".tar." in name:
            return by_archive_name(folder, name)
        if vm_folder:
            return "vm"
        if ext in ("srf", "stoch", "srfinfo") or "_source" in name or src_folder:
            return "src"
        if (
            ext in ("db", "csv", "parquet", "feather")
            or "intensity_measures" in name
            or im_folder
        ):
            return "im"
        return "other"

    def kept(self, folder: str) -> bool:
        """Return whether this folder (lowercased) is under a `keep` prefix.

        Parameters
        ----------
        folder : str
            A lowercased folder path.

        Returns
        -------
        bool
            True if it starts with one of the `keep` prefixes.
        """
        return any(folder.startswith(prefix) for prefix in self.keep)


RULE_KEYS = tuple(f.name for f in fields(Rules) if f.init)


def load_rules(path: str | None) -> Rules:
    """Return the rules in a TOML file of site words, or the built-in rules alone without one.

    Parameters
    ----------
    path : str or None
        The TOML file; None gives the built-in rules.

    Returns
    -------
    Rules
        The rules with the file's words.

    Raises
    ------
    KindsError
        The file cannot be read, is not TOML, has an unknown key, or holds anything
        but lists of strings.
    """
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
            raise KindsError(
                f"{path}: unknown key {key!r} (the keys are {', '.join(RULE_KEYS)})"
            )
        if not isinstance(value, list) or not all(isinstance(w, str) for w in value):
            raise KindsError(f"{path}: {key} must be a list of strings")
    return Rules(**{key: tuple(value) for key, value in table.items()})


def by_archive_name(folder: str, name: str) -> str:
    """Return the kind of a tarball nobody could list (pax, gzip), from its name and folder.

    Parameters
    ----------
    folder : str
        The lowercased folder path.
    name : str
        The lowercased tarball name.

    Returns
    -------
    str
        A key of KINDS, or ``"lf_mix"`` for an LF tarball.
    """
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
    """Return a kind as {kind: fraction}: an unlisted LF tarball is shared out by LF_SPLIT.

    Parameters
    ----------
    kind : str
        A kind, or ``"lf_mix"``.

    Returns
    -------
    dict of str to float
        Each kind that gets a share of the bytes, with its fraction; the fractions
        add up to 1.
    """
    return dict(LF_SPLIT) if kind == "lf_mix" else {kind: 1.0}


# ---- the walk -------------------------------------------------------------------

# Each duplicated content's oldest date: a loose copy is dated by its content's oldest copy.
OLDEST = (
    "SELECT content_hash, MIN(client_modified) AS oldest FROM files "
    "WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING COUNT(*) > 1"
)
# One byte count from the walk: (dir_id, kind, size, how, date).
WalkRow = tuple[int, str, float, str, float | None]
# Each archive as (trusted, name, folder, listed bytes), by archive id.
Archives = dict[int, tuple[bool, str, str, int]]
_TYPES = sorted(REGULAR_MEMBER_TYPES)
_REGULAR = f"size > 0 AND type IN ({', '.join('?' * len(_TYPES))})"


def archive_index(path: str) -> sqlite3.Connection:
    """Open the archive index at `path`, read-only.

    One still being written is read through its write-ahead log, so it is never opened as
    immutable.

    Parameters
    ----------
    path : str
        The archive index database.

    Returns
    -------
    sqlite3.Connection
        A read-only connection in autocommit mode. The caller closes it.

    Raises
    ------
    KindsError
        The file cannot be opened or holds no archive index.
    """
    try:
        conn = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None
        )
    except sqlite3.Error:
        raise KindsError(f"{path} holds no archive index") from None
    try:
        conn.execute("SELECT 1 FROM archives LIMIT 1").fetchall()
    except sqlite3.Error:
        conn.close()
        raise KindsError(f"{path} holds no archive index") from None
    return conn


def walk(
    store: Store,
    rules: Rules,
    index: sqlite3.Connection | None = None,
    under: str | None = None,
) -> Iterator[WalkRow]:
    """Yield every byte of the listing, or of the folder `under`, as rows.

    The rows are (dir_id, kind, size, how, date), `how` being LOOSE, MEMBERS or BY_NAME.
    Bytes inside a tarball count toward the folder that holds it. Reads both databases,
    each in one read transaction, and writes neither.

    Parameters
    ----------
    store : Store
        The audit database of the listing.
    rules : Rules
        The classifier.
    index : sqlite3.Connection, optional
        The archive index (see `archive_index`); None judges every tarball by its name.
    under : str, optional
        Walk only the files at or below this folder; None walks the whole listing.

    Yields
    ------
    tuple
        ``(dir_id, kind, size, how, date)``: the folder's id, a key of KINDS, the
        bytes (a share of a file when it is split), `how` and the date in seconds
        since the Unix epoch (None if the file has none).
    """
    if index is not None:
        index.execute("BEGIN")
    try:
        archives, owner = _archives(index) if index is not None else ({}, {})
        aside: dict[int, list[tuple[int, int]]] = defaultdict(list)
        yield from _files(store, rules, archives, owner, aside, under)
        if index is not None:  # nothing is set aside without an index
            for archive_id in sorted(aside):
                yield from _members(
                    index, rules, archive_id, archives[archive_id], aside[archive_id]
                )
    finally:
        if index is not None and index.in_transaction:
            index.execute("COMMIT")


def _archives(index: sqlite3.Connection) -> tuple[Archives, dict[bytes, int]]:
    """Return the archives and the archive each part's content hash belongs to.

    Each archive is (trusted, name, folder, listed bytes).

    A listing is trusted to show what its archive holds when the walk reached the end, or
    when it holds at least half the archive's bytes. Below half it might hold only an
    archive's small first files (its parameters and logs) and not the binary whose header
    stopped the walk. A part can belong to more than one archive row: a trusted row wins
    over one that is not, a finished walk over any other (a stale row lists the archive as
    it was before a part went), and between equals the latest row wins.

    Parameters
    ----------
    index : sqlite3.Connection
        The archive index, inside a read transaction.

    Returns
    -------
    tuple
        The archives by archive id, each as ``(trusted, name, folder, listed bytes)``,
        and the archive id that owns each part's content hash (as bytes).
    """
    listed = dict(
        index.execute(
            f"SELECT archive_id, SUM(size) FROM members WHERE {_REGULAR} GROUP BY archive_id",
            _TYPES,
        )
    )
    archives: Archives = {}
    rank: dict[int, tuple[bool, bool, int]] = {}
    for archive_id, state, total, name, folder in index.execute(
        "SELECT id, state, total_size, name, folder FROM archives"
    ):
        held = listed.get(archive_id, 0)
        trusted = held > 0 and (state in FINISHED or held >= total / 2)
        archives[archive_id] = (trusted, name, folder or "", held)
        rank[archive_id] = (trusted, state in FINISHED, archive_id)
    owner: dict[bytes, int] = {}
    for archive_id, digest in index.execute(
        "SELECT archive_id, content_hash FROM parts WHERE content_hash IS NOT NULL "
        "AND content_hash != ''"
    ):
        try:
            key = bytes.fromhex(digest)
        except ValueError:  # no file's hash can match it
            continue
        held = owner.get(key)
        if held is None or rank[archive_id] > rank[held]:
            owner[key] = archive_id
    return archives, owner


def _files(
    store: Store,
    rules: Rules,
    archives: Archives,
    owner: dict[bytes, int],
    aside: dict[int, list[tuple[int, int]]],
    under: str | None,
) -> Iterator[WalkRow]:
    """Yield the listing's files, each with its kind.

    A file that is part of a trusted archive is set aside under that archive instead.

    Parameters
    ----------
    store : Store
        The audit database.
    rules : Rules
        The classifier.
    archives : dict
        The archives, from `_archives`.
    owner : dict of bytes to int
        The archive each part's content hash belongs to, from `_archives`.
    aside : dict of int to list of tuple
        Filled in here: for each trusted archive, the ``(dir_id, size)`` of each of its
        parts and copies, which `_members` then shares out.
    under : str or None
        Only files at or below this folder; None for all.

    Yields
    ------
    tuple
        ``(dir_id, kind, size, how, date)`` as `walk` gives them, for the files not set
        aside.
    """
    conn = store.connect()
    opened = not conn.in_transaction
    if opened:
        conn.execute("BEGIN")
    try:
        sql = (
            "SELECT f.dir_id, f.name, f.size, f.content_hash, f.client_modified, "
            "d.path_lower, o.oldest FROM files f JOIN dirs d ON d.id = f.dir_id "
            f"LEFT JOIN ({OLDEST}) o ON o.content_hash = f.content_hash"
        )
        values: list[str] = []
        if under is not None:
            condition, values = report._subtree(under)
            sql += f" WHERE {condition}"
        for dir_id, name, size, digest, modified, folder, oldest in conn.execute(
            sql, values
        ):
            archive_id = owner.get(digest) if digest else None
            if archive_id is not None:
                trusted, archive_name, archive_folder, _ = archives[archive_id]
                if trusted:
                    aside[archive_id].append((dir_id, size))
                    continue
                kind = by_archive_name(archive_folder.lower(), archive_name.lower())
                for part, w in shares(kind).items():
                    yield dir_id, part, size * w, BY_NAME, modified
                continue
            name = name.lower()
            kind = rules.classify(folder, name)
            if kind == "vm" and rules.kept(folder):
                kind = "vm_kept"
            if kind == "lf_mix" or TARLIKE.search(name):
                for part, w in shares(kind).items():
                    yield dir_id, part, size * w, BY_NAME, modified
            else:
                yield dir_id, kind, size, LOOSE, modified if oldest is None else oldest
    finally:
        if opened and conn.in_transaction:
            conn.execute("COMMIT")


def _members(
    index: sqlite3.Connection,
    rules: Rules,
    archive_id: int,
    archive: tuple[bool, str, str, int],
    files: list[tuple[int, int]],
) -> Iterator[WalkRow]:
    """Yield a trusted archive's bytes, in each of its copies and parts (`files`).

    They are shared out over its listed members by size and dated by each member's own
    time.

    Parameters
    ----------
    index : sqlite3.Connection
        The archive index.
    rules : Rules
        The classifier.
    archive_id : int
        The archive's id in the index.
    archive : tuple
        ``(trusted, name, folder, listed bytes)``, from `_archives`.
    files : list of tuple
        ``(dir_id, size)`` of each copy and part of the archive in the listing.

    Yields
    ------
    tuple
        ``(dir_id, kind, size, how, date)`` as `walk` gives them, with each member's
        share of each copy's bytes and the member's own time.
    """
    _, name, folder, listed = archive
    for where, member, size, mtime in index.execute(
        f"SELECT dir, name, size, mtime FROM members WHERE archive_id = ? AND {_REGULAR}",
        [archive_id, *_TYPES],
    ):
        kind = rules.classify(
            (folder + "/" + name + "::" + where).lower(),
            member.rsplit("/", 1)[-1].lower(),
        )
        for part, w in shares(kind).items():
            for dir_id, file_size in files:
                yield dir_id, part, file_size * size * w / listed, MEMBERS, mtime


# ---- the tally ------------------------------------------------------------------


@dataclass
class Tally:
    """Bytes by kind, each split [loose, in tarballs], as the walk gave them.

    Parameters
    ----------
    listed : float
        When the listing was taken, in seconds since the Unix epoch.
    kinds : dict, optional
        Each kind to ``[loose, tarred]`` bytes.
    months : dict, optional
        Each ``(kind, "YYYY-MM")`` to ``[loose, tarred]`` bytes.
    older : dict, optional
        Each ``(kind, years)`` to the ``[loose, tarred]`` bytes older than that many years.
    members : float, optional
        Tarred bytes shared over listed members.
    by_name : float, optional
        Tarred bytes judged by their names.
    undated : float, optional
        Bytes with no usable date.
    """

    listed: float  # when the listing was taken
    kinds: dict[str, list[float]] = field(
        default_factory=dict
    )  # kind -> [loose, tarred]
    months: dict[tuple[str, str], list[float]] = field(
        default_factory=dict
    )  # (kind, "YYYY-MM") -> [loose, tarred]
    older: dict[tuple[str, float], list[float]] = field(
        default_factory=dict
    )  # (kind, years) -> [loose, tarred]
    members: float = 0.0  # tarred bytes shared over listed members
    by_name: float = 0.0  # tarred bytes judged by their names
    undated: float = 0.0

    def total(self) -> float:
        """Every byte counted, loose and in tarballs.

        Returns
        -------
        float
            The total, in bytes.
        """
        return sum(loose + tarred for loose, tarred in self.kinds.values())

    def _by_group(
        self, figures: Iterable[tuple[str, Sequence[float]]]
    ) -> dict[str, list[float]]:
        """Add kinds' figures up by group.

        Parameters
        ----------
        figures : iterable of tuple
            ``(kind, [loose, tarred])`` pairs.

        Returns
        -------
        dict of str to list of float
            Each group that appears, to ``[loose, tarred]`` bytes.
        """
        out: dict[str, list[float]] = {}
        for kind, (loose, tarred) in figures:
            group = out.setdefault(KINDS[kind][1], [0, 0])
            group[0] += loose
            group[1] += tarred
        return out

    def groups(self) -> dict[str, list[float]]:
        """Group -> [loose, tarred], in GROUPS order, for the groups that hold bytes.

        Returns
        -------
        dict of str to list of float
            The groups in GROUPS order, each with ``[loose, tarred]`` bytes.
        """
        out = self._by_group(self.kinds.items())
        return {
            group: out[group] for group in GROUPS if group in out and sum(out[group])
        }

    def group_older(self, years: float) -> dict[str, list[float]]:
        """Group -> [loose, tarred] older than `years`, for each group that holds bytes.

        Parameters
        ----------
        years : float
            One of AGES.

        Returns
        -------
        dict of str to list of float
            Each group that holds bytes, with ``[loose, tarred]`` bytes older than `years`.
        """
        out = self._by_group((kind, self.older[kind, years]) for kind in self.kinds)
        return {group: out[group] for group in self.groups()}

    def group_years(self) -> dict[str, dict[int, float]]:
        """Group -> {year: bytes}.

        Returns
        -------
        dict of str to dict of int to float
            Each group that appears, with its bytes by the year they were last modified.
        """
        out: dict[str, dict[int, float]] = {}
        for (kind, month), (loose, tarred) in self.months.items():
            years = out.setdefault(KINDS[kind][1], {})
            year = int(month[:4])
            years[year] = years.get(year, 0) + loose + tarred
        return out


def tarball_note(
    counted: Tally, looked_inside: bool, size: Callable[[float], str]
) -> str:
    """One sentence on how the tarballs' bytes were placed; `size` formats a byte count.

    Parameters
    ----------
    counted : Tally
        The tally.
    looked_inside : bool
        Whether an archive index was given.
    size : callable
        Formats a byte count as text.

    Returns
    -------
    str
        The sentence.
    """
    if looked_inside:
        return (
            f"In tarballs: {size(counted.members)} shared out over the members the archive "
            f"index lists, {size(counted.by_name)} judged by their names."
        )
    return (
        f"In tarballs: {size(counted.members + counted.by_name)}, all judged by their names "
        "(give --archive-db to look inside them)."
    )


def tally(rows: Iterable[WalkRow], listed: float) -> Tally:
    """Add up the walk's rows.

    Ages count back from `listed`, months are UTC, and a date after the listing (a wrong
    clock) is in its own month but older than nothing.

    Parameters
    ----------
    rows : iterable of tuple
        The rows `walk` gives: ``(dir_id, kind, size, how, date)``.
    listed : float
        When the listing was taken, in seconds since the Unix epoch.

    Returns
    -------
    Tally
        The bytes by kind, month and age. Rows of zero size are skipped.
    """
    out = Tally(listed)
    month_of = _months()
    for _, kind, size, how, date in rows:
        if not size:  # an empty file: nothing to count, and no month to show
            continue
        side = 0 if how == LOOSE else 1
        figures: list[float] | None = out.kinds.get(kind)
        if figures is None:
            figures = out.kinds[kind] = [0, 0]
            for years in AGES:
                out.older[kind, years] = [0, 0]
        figures[side] += size
        if how == MEMBERS:
            out.members += size
        elif how == BY_NAME:
            out.by_name += size
        month = None if date is None else month_of(date)
        if date is None or month is None:
            out.undated += size
            continue
        figures = out.months.get((kind, month))
        if figures is None:
            figures = out.months[kind, month] = [0, 0]
        figures[side] += size
        age = listed - date
        for years in AGES:
            if age <= years * YEAR:
                break
            out.older[kind, years][side] += size
    return out


def _months() -> Callable[[float], str | None]:
    """Return a function giving a Unix time's UTC month as "YYYY-MM".

    The months are remembered day by day, and the function gives None for a time no
    calendar holds.

    Returns
    -------
    callable
        Takes a time in seconds since the Unix epoch and returns its month as
        ``"YYYY-MM"``, or None.
    """

    @functools.cache
    def of_day(day: int) -> str | None:
        """Return the UTC month of a day, remembered.

        Parameters
        ----------
        day : int
            Days since the Unix epoch.

        Returns
        -------
        str or None
            ``"YYYY-MM"``, or None for a day no calendar holds.
        """
        try:
            return time.strftime("%Y-%m", time.gmtime(day * 86400))
        except (OverflowError, OSError, ValueError):
            return None

    return lambda date: of_day(int(date // 86400))


def write_csv(counted: Tally, outdir: str, about: dict[str, str]) -> list[str]:
    """Write the tally into `outdir` as four CSV files, and return the paths.

    The files are in whole bytes, one row per kind that holds bytes; `about` says what
    listing it came from.

    Parameters
    ----------
    counted : Tally
        The tally.
    outdir : str
        The folder to write into; created if it does not exist.
    about : dict of str to str
        Column names and values for the one-row listing file.

    Returns
    -------
    list of str
        The paths written.
    """
    os.makedirs(outdir, exist_ok=True)
    held = [kind for kind in KINDS if sum(counted.kinds.get(kind, (0, 0)))]
    tables = {
        "kinds": (
            ["kind", "label", "group", "bytes", "loose", "tarred"],
            [
                (
                    kind,
                    *KINDS[kind],
                    round(sum(counted.kinds[kind])),
                    *map(round, counted.kinds[kind]),
                )
                for kind in held
            ],
        ),
        "kinds_by_month": (
            ["kind", "group", "month", "loose", "tarred"],
            [
                (kind, KINDS[kind][1], month, *map(round, figures))
                for kind in held
                for (of, month), figures in sorted(counted.months.items())
                if of == kind
            ],
        ),
        "kinds_older_than": (
            ["kind", "group", "years", "loose", "tarred"],
            [
                (kind, KINDS[kind][1], years, *map(round, counted.older[kind, years]))
                for kind in held
                for years in AGES
            ],
        ),
        "kinds_listing": (list(about), [list(about.values())]),
    }
    written = []
    for name, (header, rows) in tables.items():
        path = os.path.join(outdir, f"{name}.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(path)
    return written
