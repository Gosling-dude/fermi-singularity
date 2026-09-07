"""Timestamp-aware chunking.

Chunks are built by accumulating whole ASR segments, never by slicing on
character counts, so every chunk boundary coincides with a real boundary in
the audio and the ``start``/``end`` of a chunk is always a true interval of
speech. A chunk is closed when it reaches the target duration *and* the last
segment ends on sentence-final punctuation; it is force-closed at the hard
maximum so a speaker who never pauses cannot produce an unbounded chunk.

Consecutive chunks overlap by a configurable number of seconds so an idea
explained across a boundary is still retrievable as a unit.
"""

from __future__ import annotations

from companion.config import Settings, get_settings
from companion.models import Chunk, Segment, Transcript
from companion.utils.logging import get_logger

log = get_logger("ingest")

_SENTENCE_END = (".", "!", "?", '."', '?"', '!"', ".)", "?)", "!)")


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_END)


def _overlap_start_index(
    segments: list[Segment], closed_end: float, overlap_seconds: float
) -> int:
    """Index of the first segment to replay at the head of the next chunk.

    Walks back from the end of the closed chunk until ``overlap_seconds`` of
    audio has been covered, so the overlap is measured in time rather than in
    a fixed number of segments.
    """
    if overlap_seconds <= 0:
        return len(segments)
    cutoff = closed_end - overlap_seconds
    for index in range(len(segments) - 1, -1, -1):
        if segments[index].start < cutoff:
            return min(index + 1, len(segments))
    return 0


def chunk_transcript(
    transcript: Transcript, settings: Settings | None = None
) -> list[Chunk]:
    """Split one transcript into overlapping, timestamp-bounded chunks."""
    settings = settings or get_settings()
    target = settings.chunk_target_seconds
    hard_max = max(settings.chunk_max_seconds, target)
    overlap = min(settings.chunk_overlap_seconds, target / 2)

    chunks: list[Chunk] = []
    buffer: list[Segment] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(segment.text.strip() for segment in buffer).strip()
        if not text:
            buffer.clear()
            return
        chunks.append(
            Chunk(
                chunk_id=f"{transcript.episode_id}_{len(chunks):04d}",
                episode_id=transcript.episode_id,
                episode_title=transcript.episode_title,
                start=round(buffer[0].start, 3),
                end=round(buffer[-1].end, 3),
                text=text,
                source_file=transcript.source_file,
                segment_ids=[segment.id for segment in buffer],
            )
        )

    for segment in transcript.segments:
        buffer.append(segment)
        span = buffer[-1].end - buffer[0].start
        ready = span >= target and _ends_sentence(segment.text)
        if ready or span >= hard_max:
            flush()
            closed_end = buffer[-1].end
            keep_from = _overlap_start_index(buffer, closed_end, overlap)
            buffer = buffer[keep_from:]

    # The tail is emitted only if it carries content that is not already
    # fully contained in the previous chunk's overlap.
    if buffer:
        tail_end = buffer[-1].end
        if not chunks or tail_end > chunks[-1].end + 1.0:
            flush()

    log.info(
        "chunked",
        episode=transcript.episode_id,
        chunks=len(chunks),
        mean_seconds=(
            round(sum(c.duration for c in chunks) / len(chunks), 1) if chunks else 0
        ),
    )
    return chunks


def chunk_all(
    transcripts: list[Transcript], settings: Settings | None = None
) -> list[Chunk]:
    settings = settings or get_settings()
    chunks: list[Chunk] = []
    for transcript in transcripts:
        chunks.extend(chunk_transcript(transcript, settings))
    return chunks
