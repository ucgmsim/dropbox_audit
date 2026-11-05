"""An archive's parts, addressed as a single coordinate space.

A split archive is not a special case of a whole one: both are an ordered list of
parts, and a global offset maps to (part index, offset within that part). A read that
straddles a boundary is split into one read per part, so nothing above this module
needs to know where the parts divide -- which is what lets the tar walker treat 18
files of 300 GiB as one 5 TiB file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


class ArchiveSetError(ValueError):
    """The parts do not form a complete, ordered archive."""


@dataclass(frozen=True)
class Part:
    """One file of an archive set.

    Parameters
    ----------
    idx : int
        Position in the set, from 0.
    name : str
        File name.
    size : int
        Size in bytes.
    offset : int
        Where this part starts in the whole archive, in bytes.
    path_display : str
        The part's Dropbox path.
    dbx_id : str
        Dropbox's file id.
    rev : str
        Dropbox's revision.
    content_hash : str
        Dropbox's content hash, in hex.
    """

    idx: int
    name: str
    size: int
    offset: int  # where this part starts in the whole archive
    path_display: str = ""
    dbx_id: str = ""
    rev: str = ""
    content_hash: str = ""


def _suffix_value(suffix: str) -> int:
    """Read a part suffix as a number.

    Parameters
    ----------
    suffix : str
        Digits, or lowercase letters counted from ``a`` as ``split`` names them.

    Returns
    -------
    int
        The suffix's value, from 0.
    """
    if suffix.isdigit():
        return int(suffix)
    value = 0
    for char in suffix:
        value = value * 26 + (ord(char) - ord("a"))
    return value


def _suffix_text(value: int, width: int, numeric: bool) -> str:
    """Write a number as a part suffix; the inverse of ``_suffix_value``.

    Parameters
    ----------
    value : int
        The number.
    width : int
        Number of characters in the suffix.
    numeric : bool
        Digits if true, letters from ``a`` if false.

    Returns
    -------
    str
        The suffix.
    """
    if numeric:
        return str(value).zfill(width)
    out = []
    for _ in range(width):
        out.append(chr(ord("a") + value % 26))
        value //= 26
    return "".join(reversed(out))


class ArchiveSet:
    """An archive as an ordered, gapless run of parts, addressed by byte offset.

    Parameters
    ----------
    parts : list of Part
        The parts in order, each starting where the one before ends.
    """

    def __init__(self, parts: list[Part]):  # noqa: D107
        self.parts = parts

    @property
    def total_size(self) -> int:
        """The sum of the parts' sizes, in bytes.

        Returns
        -------
        int
            Bytes.
        """
        return sum(p.size for p in self.parts)

    @classmethod
    def from_entries(cls, entries: list[dict], base_name: str) -> ArchiveSet:
        """Order Dropbox (or local) entries into a gapless part sequence.

        ``base_name`` is the archive's own name, e.g. ``v01p0_incomplete.tar``. Parts are
        matched against it rather than guessed, because a plain ``FaultSZ03_Source.tar``
        would otherwise parse as a part with the suffix ``tar``.

        Parameters
        ----------
        entries : list of dict
            One per file, with ``name``, ``size`` and optionally ``path_display``,
            ``id``, ``rev`` and ``content_hash``.
        base_name : str
            The archive's name.

        Returns
        -------
        ArchiveSet
            The parts in order.

        Raises
        ------
        ArchiveSetError
            The entries are not one whole file or a complete, uniformly named set.
        """
        if not entries:
            raise ArchiveSetError(f"no files given for {base_name!r}")
        pattern = re.compile(
            rf"^{re.escape(base_name)}\.(?:part[-_]?)?(?P<suffix>[a-z]{{2,}}|\d{{2,}})$"
        )
        whole = [e for e in entries if e["name"] == base_name]
        if whole:
            if len(entries) > 1:
                raise ArchiveSetError(
                    f"{base_name!r} exists alongside {len(entries) - 1} other files; "
                    "an archive is either one whole file or a set of parts"
                )
            return cls([cls._part(0, whole[0], 0)])

        keyed = []
        for entry in entries:
            match = pattern.match(entry["name"])
            if match is None:
                raise ArchiveSetError(
                    f"{entry['name']!r} is not a part of {base_name!r}"
                )
            keyed.append((match.group("suffix"), entry))

        widths = {len(s) for s, _ in keyed}
        if len(widths) != 1:
            raise ArchiveSetError(
                f"mixed suffix widths for {base_name!r}: {sorted(widths)}"
            )
        width = widths.pop()
        numeric = keyed[0][0].isdigit()
        if any(s.isdigit() != numeric for s, _ in keyed):
            raise ArchiveSetError(f"mixed suffix styles for {base_name!r}")

        keyed.sort(key=lambda item: _suffix_value(item[0]))
        first = _suffix_value(keyed[0][0])
        # A gap *before* the lowest part is still a gap. Without this the second part
        # becomes part 0, every offset shifts by a whole part, and `index` calls the
        # archive corrupt at 0 -- blaming the data for a missing file. `split` counts
        # letters from `aa`; numbers from `00`, or `01` in tools that count from one.
        starts = (0, 1) if numeric else (0,)
        if first not in starts:
            named = " or ".join(repr(_suffix_text(v, width, numeric)) for v in starts)
            raise ArchiveSetError(
                f"{base_name!r} is missing its first part ({named}); the lowest part "
                f"present is {keyed[0][0]!r}"
            )
        for position, (suffix, _) in enumerate(keyed):
            expected = first + position
            if _suffix_value(suffix) != expected:
                raise ArchiveSetError(
                    f"{base_name!r} is missing part "
                    f"{_suffix_text(expected, width, numeric)!r}"
                )

        parts, offset = [], 0
        for idx, (_, entry) in enumerate(keyed):
            parts.append(cls._part(idx, entry, offset))
            offset += int(entry["size"])
        return cls(parts)

    @staticmethod
    def _part(idx: int, entry: dict, offset: int) -> Part:
        """Make a Part from one entry.

        Parameters
        ----------
        idx : int
            The part's position in the set.
        entry : dict
            The entry, as given to ``from_entries``.
        offset : int
            Where the part starts in the whole archive, in bytes.

        Returns
        -------
        Part
            The part.
        """
        return Part(
            idx=idx,
            name=entry["name"],
            size=int(entry["size"]),
            offset=offset,
            path_display=entry.get("path_display", ""),
            dbx_id=entry.get("id", ""),
            rev=entry.get("rev", ""),
            content_hash=entry.get("content_hash", ""),
        )

    def locate(self, offset: int) -> tuple[int, int]:
        """Find the part holding an offset.

        Parameters
        ----------
        offset : int
            A byte offset in the whole archive.

        Returns
        -------
        tuple of int
            ``(part index, offset within that part)``.

        Raises
        ------
        ArchiveSetError
            The offset is outside the archive.
        """
        if offset < 0 or offset >= self.total_size:
            raise ArchiveSetError(
                f"offset {offset} outside archive of {self.total_size}"
            )
        for part in self.parts:
            if offset < part.offset + part.size:
                return part.idx, offset - part.offset
        raise ArchiveSetError(f"offset {offset} outside archive of {self.total_size}")

    def slices(self, offset: int, length: int) -> list[tuple[int, int, int]]:
        """Split a read into (part index, offset within part, length) pieces.

        Parameters
        ----------
        offset : int
            Where the read starts in the whole archive, in bytes.
        length : int
            How many bytes; cut short at the end of the archive.

        Returns
        -------
        list of tuple of int
            One ``(part index, offset within part, length)`` per part touched.
        """
        length = min(length, max(self.total_size - offset, 0))
        out = []
        while length > 0:
            idx, within = self.locate(offset)
            take = min(length, self.parts[idx].size - within)
            out.append((idx, within, take))
            offset += take
            length -= take
        return out

    def set_hash(self) -> str:
        """Identity: the ordered part content hashes. Moving the parts does not change it.

        Returns
        -------
        str
            A SHA-256 hex digest over each part's content hash and size.
        """
        digest = hashlib.sha256()
        for part in self.parts:
            digest.update(f"{part.content_hash}:{part.size}\n".encode())
        return digest.hexdigest()
