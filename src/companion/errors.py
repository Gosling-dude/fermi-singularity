"""Typed errors carrying operator-actionable remediation text.

Every error surfaced to a human goes through ``CompanionError`` so the CLI, the
web API and the eval runner can present a fix instead of a traceback.
"""

from __future__ import annotations


class CompanionError(Exception):
    """Base error. ``remedy`` is shown to the user verbatim beneath the message."""

    def __init__(self, message: str, remedy: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def render(self) -> str:
        if self.remedy:
            return f"{self.message}\n\n  → {self.remedy}"
        return self.message


class ConfigError(CompanionError):
    """Configuration or credentials are missing/invalid."""


class AudioError(CompanionError):
    """Audio discovery, validation or normalisation failed."""


class TranscriptionError(CompanionError):
    """ASR failed for a specific episode."""


class IndexError_(CompanionError):
    """The search index is missing or unusable."""


class RetrievalError(CompanionError):
    """Retrieval could not be completed."""


class ProviderError(CompanionError):
    """The LLM provider rejected, timed out on, or failed a request."""
