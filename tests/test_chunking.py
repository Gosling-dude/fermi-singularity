"""Chunking must preserve real audio intervals and cover the whole episode."""

from __future__ import annotations

from companion.config import Settings
from companion.ingest.chunk import chunk_transcript
from companion.models import Segment, Transcript


def _settings(**overrides) -> Settings:
    base = {
        "CHUNK_TARGET_SECONDS": 60.0,
        "CHUNK_MAX_SECONDS": 90.0,
        "CHUNK_OVERLAP_SECONDS": 15.0,
    }
    base.update({k: v for k, v in overrides.items()})
    return Settings(**base)


def _transcript(texts, step=10.0) -> Transcript:
    segments = [
        Segment(id=i, start=i * step, end=i * step + step, text=text)
        for i, text in enumerate(texts)
    ]
    return Transcript(
        episode_id="ep", episode_title="Ep", source_file="ep.mp3",
        source_sha256="c" * 64, duration=len(texts) * step,
        asr_model="tiny", asr_provider="local", language="en", segments=segments,
    )


def test_chunks_carry_real_intervals(transcript):
    chunks = chunk_transcript(transcript, _settings())
    assert chunks
    starts = {segment.start for segment in transcript.segments}
    ends = {segment.end for segment in transcript.segments}
    for chunk in chunks:
        assert chunk.start in starts, "chunk start must be a real segment start"
        assert chunk.end in ends, "chunk end must be a real segment end"
        assert chunk.end > chunk.start
        assert chunk.segment_ids


def test_chunk_ids_are_unique_and_ordered(transcript):
    chunks = chunk_transcript(transcript, _settings())
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_full_episode_is_covered(transcript):
    chunks = chunk_transcript(transcript, _settings())
    assert chunks[0].start == transcript.segments[0].start
    assert chunks[-1].end == transcript.segments[-1].end


def test_consecutive_chunks_overlap():
    texts = [f"Sentence number {i} about physics." for i in range(40)]
    chunks = chunk_transcript(_transcript(texts), _settings())
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end, "chunks must overlap in time"


def test_hard_maximum_is_respected_without_sentence_breaks():
    # No sentence-final punctuation anywhere: the only thing that can close a
    # chunk is the hard duration cap.
    texts = [f"clause {i} continuing onward" for i in range(60)]
    chunks = chunk_transcript(_transcript(texts), _settings())
    assert chunks
    for chunk in chunks:
        assert chunk.duration <= 90.0 + 1e-6


def test_prefers_closing_on_sentence_boundaries():
    texts = []
    for i in range(30):
        texts.append(f"clause {i} continuing" if i % 4 else f"Sentence {i} ends here.")
    chunks = chunk_transcript(_transcript(texts), _settings())
    closing_on_sentence = sum(1 for c in chunks if c.text.rstrip().endswith("."))
    assert closing_on_sentence >= len(chunks) - 1


def test_single_short_segment_still_produces_a_chunk():
    chunks = chunk_transcript(_transcript(["Just one line."]), _settings())
    assert len(chunks) == 1
    assert chunks[0].text == "Just one line."
