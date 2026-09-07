"""Citation extraction and validation.

An answer is only verifiable if its citations point at speech that actually
exists. Every citation the model emits is parsed into a structured object and
checked against the passages that were retrieved for that turn:

* the episode must be one of the supplied episodes;
* the interval must be well formed and inside the episode's duration;
* the interval must overlap a passage that was actually in the context.

The last check is the one that catches a fabricated citation that happens to
name a real episode and a plausible time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from companion.models import Chunk, Episode, RetrievedChunk
from companion.utils.timefmt import format_range, parse_timestamp

# (Ep. "Title" 12:30-13:10) — the dash may be a hyphen or an en dash, and the
# model sometimes drops the `Ep.` prefix, so both are tolerated on parse while
# the prompt asks for the canonical form.
CITATION_RE = re.compile(
    r"""\(\s*(?:Ep\.?|Episode)?\s*"(?P<title>[^"]+)"\s*
        (?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*
        (?P<end>\d{1,2}:\d{2}(?::\d{2})?)\s*\)""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class Citation:
    """One parsed citation plus the outcome of validating it."""

    episode_title: str
    start: float
    end: float
    raw: str
    episode_id: str | None = None
    valid: bool = False
    reason: str | None = None

    @property
    def formatted(self) -> str:
        return format_range(self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode_title,
            "episode_id": self.episode_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "formatted": self.formatted,
            "valid": self.valid,
            "reason": self.reason,
        }


def extract_citations(text: str) -> list[Citation]:
    """Parse every inline citation in an answer, in order of appearance."""
    citations: list[Citation] = []
    for match in CITATION_RE.finditer(text):
        try:
            start = parse_timestamp(match.group("start"))
            end = parse_timestamp(match.group("end"))
        except ValueError:
            continue
        citations.append(
            Citation(
                episode_title=match.group("title").strip(),
                start=start,
                end=end,
                raw=match.group(0),
            )
        )
    return citations


def _overlaps(citation: Citation, chunk: Chunk, tolerance: float = 5.0) -> bool:
    """True when the cited interval overlaps the chunk's real interval.

    A small tolerance absorbs the model rounding a timestamp label to whole
    seconds; it is not wide enough to let an invented time slip through.
    """
    return (
        citation.start <= chunk.end + tolerance
        and citation.end >= chunk.start - tolerance
    )


def validate_citations(
    citations: Iterable[Citation],
    retrieved: list[RetrievedChunk],
    episodes: list[Episode] | None = None,
) -> list[Citation]:
    """Mark each citation valid or invalid, with a reason when it fails."""
    by_title: dict[str, list[Chunk]] = {}
    for item in retrieved:
        by_title.setdefault(item.chunk.episode_title.lower(), []).append(item.chunk)
    durations = {
        episode.title.lower(): episode.duration for episode in (episodes or [])
    }
    ids = {episode.title.lower(): episode.episode_id for episode in (episodes or [])}

    validated: list[Citation] = []
    for citation in citations:
        key = citation.episode_title.lower()
        citation.episode_id = ids.get(key)

        if citation.start < 0 or citation.end <= citation.start:
            citation.reason = "malformed interval (end must follow start)"
        elif key not in by_title:
            citation.reason = "cites an episode that was not in the retrieved context"
        elif key in durations and citation.start > durations[key]:
            citation.reason = "starts after the end of the episode"
        elif not any(_overlaps(citation, chunk) for chunk in by_title[key]):
            citation.reason = "timestamp does not overlap any retrieved passage"
        else:
            citation.valid = True
        validated.append(citation)
    return validated


def citation_validity_rate(citations: list[Citation]) -> float:
    """Fraction of citations that survived validation (1.0 when there are none)."""
    if not citations:
        return 1.0
    return sum(1 for citation in citations if citation.valid) / len(citations)


def strip_invalid_citations(text: str, citations: list[Citation]) -> str:
    """Remove citations that failed validation from the rendered answer.

    Showing a citation we know to be wrong is worse than showing none: the
    learner would click through to the wrong moment and lose trust in the
    ones that are right.
    """
    cleaned = text
    for citation in citations:
        if not citation.valid:
            cleaned = cleaned.replace(citation.raw, "")
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def format_sources(retrieved: list[RetrievedChunk], citations: list[Citation]) -> list[dict[str, Any]]:
    """Build the source list shown under an answer.

    Sources are derived from the passages the model actually cited, so the
    list reflects what supported the answer rather than everything retrieved.
    """
    cited_titles = {c.episode_title.lower() for c in citations if c.valid}
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in retrieved:
        chunk = item.chunk
        if cited_titles and chunk.episode_title.lower() not in cited_titles:
            continue
        if chunk.chunk_id in seen:
            continue
        matches_citation = not cited_titles or any(
            c.valid
            and c.episode_title.lower() == chunk.episode_title.lower()
            and _overlaps(c, chunk)
            for c in citations
        )
        if not matches_citation:
            continue
        seen.add(chunk.chunk_id)
        sources.append(
            {
                "chunk_id": chunk.chunk_id,
                "episode_id": chunk.episode_id,
                "episode_title": chunk.episode_title,
                "start": chunk.start,
                "end": chunk.end,
                "timestamp": chunk.timestamp_label,
                "source_file": chunk.source_file,
                "score": round(item.score, 6),
            }
        )
    return sources
