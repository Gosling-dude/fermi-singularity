"""Shared fixtures.

Tests that need a corpus build one from synthetic transcripts in a tmp
directory, so the suite runs without audio, without an API key and without
touching the real ``data/`` artefacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from companion.chat.providers import LLMProvider, LLMResponse  # noqa: E402
from companion.models import Chunk, Episode, RetrievedChunk, Segment, Transcript  # noqa: E402


@pytest.fixture
def segments() -> list[Segment]:
    texts = [
        "Planck introduced the quantum of action in December nineteen hundred.",
        "He assumed the oscillators could only have discrete energies.",
        "The constant is roughly six point six two six times ten to the minus thirty four.",
        "This resolved what later became known as the ultraviolet catastrophe.",
        "He described the assumption as an act of desperation.",
        "Shannon defined entropy as a measure of information in nineteen forty eight.",
        "The unit of information is the bit, a name credited to Tukey.",
        "Channel capacity sets a speed limit rather than an accuracy limit.",
    ]
    return [
        Segment(id=i, start=i * 30.0, end=i * 30.0 + 29.0, text=text)
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def transcript(segments) -> Transcript:
    return Transcript(
        episode_id="quantum",
        episode_title="The Birth of the Quantum",
        source_file="quantum.mp3",
        source_sha256="a" * 64,
        duration=240.0,
        asr_model="medium",
        asr_provider="local:faster-whisper",
        language="en",
        segments=segments,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="quantum_0000",
            episode_id="quantum",
            episode_title="The Birth of the Quantum",
            start=0.0, end=90.0,
            text="Planck introduced the quantum of action. He assumed oscillators "
                 "could only have discrete energies proportional to frequency.",
            source_file="quantum.mp3", segment_ids=[0, 1],
        ),
        Chunk(
            chunk_id="quantum_0001",
            episode_id="quantum",
            episode_title="The Birth of the Quantum",
            start=90.0, end=180.0,
            text="The ultraviolet catastrophe was the prediction of infinite "
                 "radiated energy. Planck called his fix an act of desperation.",
            source_file="quantum.mp3", segment_ids=[2, 3],
        ),
        Chunk(
            chunk_id="shannon_0000",
            episode_id="shannon",
            episode_title="Shannon and the Birth of Information",
            start=0.0, end=95.0,
            text="Shannon defined entropy as a measure of information. The unit "
                 "is the bit and channel capacity sets a speed limit.",
            source_file="shannon.mp3", segment_ids=[0, 1],
        ),
    ]


@pytest.fixture
def retrieved(chunks) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunk, score=1.0 / (index + 1), dense_rank=index + 1)
        for index, chunk in enumerate(chunks)
    ]


@pytest.fixture
def episodes() -> list[Episode]:
    return [
        Episode("quantum", "The Birth of the Quantum", "quantum.mp3", 240.0,
                "a" * 64, 8, 2),
        Episode("shannon", "Shannon and the Birth of Information", "shannon.mp3",
                200.0, "b" * 64, 6, 1),
    ]


class ScriptedProvider(LLMProvider):
    """A provider that returns queued replies — no network, no API key."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.model = "scripted-model"

    def complete(self, system, messages, *, max_tokens=4000, effort="medium",
                 temperature=None) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages})
        text = self.replies.pop(0) if self.replies else ""
        return LLMResponse(text=text, model=self.model, provider=self.name,
                           input_tokens=100, output_tokens=50, latency_ms=1.0)


@pytest.fixture
def scripted_provider():
    return ScriptedProvider
