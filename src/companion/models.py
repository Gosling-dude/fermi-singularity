"""Core domain types shared across ingestion, retrieval, chat and evaluation.

These are deliberately plain dataclasses with explicit (de)serialisation so the
on-disk artefacts under ``data/`` stay readable and diffable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from companion.utils.timefmt import format_range


@dataclass
class Segment:
    """One ASR segment with its timestamps in seconds from episode start."""

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Segment":
        return cls(
            id=int(payload["id"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=str(payload["text"]),
        )


@dataclass
class Transcript:
    """A full episode transcript produced by our own ASR pass."""

    episode_id: str
    episode_title: str
    source_file: str
    source_sha256: str
    duration: float
    asr_model: str
    asr_provider: str
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Transcript":
        return cls(
            episode_id=payload["episode_id"],
            episode_title=payload["episode_title"],
            source_file=payload["source_file"],
            source_sha256=payload["source_sha256"],
            duration=float(payload["duration"]),
            asr_model=payload.get("asr_model", "unknown"),
            asr_provider=payload.get("asr_provider", "unknown"),
            language=payload.get("language", "en"),
            segments=[Segment.from_dict(s) for s in payload.get("segments", [])],
        )


@dataclass
class Chunk:
    """A timestamp-bounded passage — the unit that is embedded and retrieved.

    ``segment_ids`` records which ASR segments the chunk was built from, which
    is what lets citation validation confirm a cited interval corresponds to
    real transcribed speech rather than an invented span.
    """

    chunk_id: str
    episode_id: str
    episode_title: str
    start: float
    end: float
    text: str
    source_file: str
    segment_ids: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def timestamp_label(self) -> str:
        return format_range(self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=payload["chunk_id"],
            episode_id=payload["episode_id"],
            episode_title=payload["episode_title"],
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=payload["text"],
            source_file=payload["source_file"],
            segment_ids=list(payload.get("segment_ids", [])),
        )


@dataclass
class Episode:
    """Catalogue-level metadata, surfaced to the learner in the UI."""

    episode_id: str
    title: str
    source_file: str
    duration: float
    sha256: str
    num_segments: int
    num_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Episode":
        return cls(**payload)


@dataclass
class RetrievedChunk:
    """A chunk plus the scores that put it in the context window."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "episode_id": self.chunk.episode_id,
            "episode_title": self.chunk.episode_title,
            "start": self.chunk.start,
            "end": self.chunk.end,
            "timestamp": self.chunk.timestamp_label,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
        }
