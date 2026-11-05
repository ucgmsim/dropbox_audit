"""Sizes and durations as people read them, in every command's output and log lines."""

import math


def human_bytes(n: float) -> str:
    """Format a byte count with a binary unit.

    Parameters
    ----------
    n : float
        The size in bytes; may be negative.

    Returns
    -------
    str
        The size in B, KiB, MiB, GiB, TiB or PiB, to one decimal place (whole bytes for B).
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n} B"


def human_duration(seconds: float) -> str:
    """Format a duration in its two largest units.

    Parameters
    ----------
    seconds : float
        The duration in seconds. NaN, infinite and negative values are unknown.

    Returns
    -------
    str
        For example ``"3d 4h"``, ``"2h 5m"``, ``"7m 9s"`` or ``"12s"``; ``"unknown"`` for a duration that is
        not a finite, non-negative number.
    """
    if math.isnan(seconds) or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
