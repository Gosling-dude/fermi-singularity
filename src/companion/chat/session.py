"""Conversation state.

A session holds the turn history and the last turn's retrieval, which is what
makes "take me to that part" and "explain the second point" work. History is
kept bounded: only a short window is ever sent to the model, because the full
transcript of a long conversation would both cost more and drown the current
question.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from companion.models import RetrievedChunk

MAX_HISTORY_TURNS = 8


@dataclass
class Turn:
    """One exchange plus the evidence that produced it."""

    question: str
    answer: str
    retrieval_query: str
    mode: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    not_covered: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieval_query": self.retrieval_query,
            "mode": self.mode,
            "not_covered": self.not_covered,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class Session:
    """A single learner conversation."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    episode_filter: str | None = None
    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def clear(self) -> None:
        self.turns.clear()

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def last_retrieved(self) -> list[RetrievedChunk]:
        """Passages behind the most recent answer — used by 'locate'."""
        return self.last_turn.retrieved if self.last_turn else []

    def history(self, limit: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
        """Recent turns as provider-shaped messages."""
        messages: list[dict[str, str]] = []
        for turn in self.turns[-limit:]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def is_followup(self) -> bool:
        return bool(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "episode_filter": self.episode_filter,
            "turns": [turn.to_dict() for turn in self.turns],
        }
