"""Timestamp formatting and parsing shared by chunking, citations and the UI."""

from __future__ import annotations

import re

_TS_RE = re.compile(r"^(?:(\d+):)?([0-5]?\d):([0-5]\d)(?:\.(\d+))?$")


def format_timestamp(seconds: float) -> str:
    """Render seconds as ``M:SS`` (or ``H:MM:SS`` past an hour).

    Negative inputs clamp to zero; fractional seconds truncate, so a citation
    never points past the moment it describes.
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_range(start: float, end: float) -> str:
    """Render an interval the way citations display it, e.g. ``12:30–13:10``."""
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


def parse_timestamp(value: str) -> float:
    """Parse ``M:SS``/``H:MM:SS`` back into seconds. Raises ``ValueError``."""
    match = _TS_RE.match(value.strip())
    if not match:
        raise ValueError(f"not a timestamp: {value!r}")
    hours, minutes, secs, frac = match.groups()
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
    if frac:
        total += float(f"0.{frac}")
    return float(total)


def format_duration(seconds: float) -> str:
    """Human duration for stats output, e.g. ``42m 18s`` or ``1h 04m``."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
