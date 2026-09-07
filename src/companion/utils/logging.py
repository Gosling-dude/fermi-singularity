"""Structured, stage-tagged logging.

Log lines are prefixed with the pipeline stage (``[INGEST]``, ``[RETRIEVAL]``,
``[CHAT]``, ``[EVAL]``) so a run can be read top to bottom as a trace.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_SECRET_HINTS = ("api_key", "apikey", "authorization", "token", "secret")


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Never let a credential reach a log sink."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if any(hint in key.lower() for hint in _SECRET_HINTS):
            clean[key] = "<redacted>"
        else:
            clean[key] = value
    return clean


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )
    # Third-party libraries are chatty at INFO; the pipeline's own stage logs
    # are the signal we want.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers",
                  "faster_whisper", "chromadb", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


class StageLogger:
    """A logger bound to one pipeline stage."""

    def __init__(self, stage: str) -> None:
        configure_logging()
        self.stage = stage.upper()
        self._log = logging.getLogger(f"companion.{stage.lower()}")

    def _emit(self, level: int, message: str, **fields: Any) -> None:
        parts = [f"[{self.stage}] {message}"]
        if fields:
            rendered = " ".join(
                f"{k}={v}" for k, v in _redact(fields).items() if v is not None
            )
            if rendered:
                parts.append(rendered)
        self._log.log(level, " | ".join(parts))

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, **fields)


def get_logger(stage: str) -> StageLogger:
    return StageLogger(stage)
