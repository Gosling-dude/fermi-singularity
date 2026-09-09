"""Resident-set reporting, so memory is observed rather than guessed at.

A container that exceeds its limit is killed without a Python traceback: the
process simply disappears and the platform reports a restart. The only way to
attribute that to a component is to record RSS as each one loads, which is
what this module exists for.
"""

from __future__ import annotations

import os

from companion.utils.logging import get_logger

log = get_logger("mem")


def rss_mb() -> float:
    """Resident set size of this process, in MB.

    Reads /proc on Linux (what a container is actually judged on) and falls
    back to getrusage elsewhere. ``ru_maxrss`` is bytes on macOS and
    kilobytes on Linux, hence the split.
    """
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
        return raw / divisor


def log_rss(stage: str, **extra: object) -> float:
    """Record RSS at a named point in start-up or a request."""
    value = rss_mb()
    log.info("rss", stage=stage, mb=round(value, 1), **extra)
    return value
