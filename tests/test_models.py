"""Domain types must round-trip through the on-disk JSON artefacts."""

from __future__ import annotations

from companion.models import Chunk, Transcript


def test_transcript_roundtrip(transcript):
    restored = Transcript.from_dict(transcript.to_dict())
    assert restored.episode_id == transcript.episode_id
    assert len(restored.segments) == len(transcript.segments)
    assert restored.segments[0].start == transcript.segments[0].start


def test_transcript_text_joins_segments(transcript):
    assert "quantum of action" in transcript.text
    assert "Tukey" in transcript.text


def test_chunk_roundtrip_and_labels(chunks):
    restored = Chunk.from_dict(chunks[0].to_dict())
    assert restored.chunk_id == chunks[0].chunk_id
    assert restored.segment_ids == [0, 1]
    assert restored.timestamp_label == "0:00–1:30"
    assert restored.duration == 90.0
