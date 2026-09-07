"""Loading and validation of the evaluation case set."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CASES = Path(__file__).resolve().parent / "cases.yaml"


@dataclass
class EvalCase:
    """One evaluation case, single- or multi-turn."""

    id: str
    category: str
    turns: list[str]
    expected: dict[str, Any] = field(default_factory=dict)
    must_cite: bool = False
    forbidden_claims: list[str] = field(default_factory=list)

    @property
    def scored_input(self) -> str:
        """The turn whose answer is scored — always the last one."""
        return self.turns[-1]

    @property
    def is_refusal(self) -> bool:
        return bool(self.expected.get("refusal"))

    @property
    def clarify_ok(self) -> bool:
        return bool(self.expected.get("clarify_ok"))

    @property
    def expected_episodes(self) -> list[str]:
        if "episodes" in self.expected:
            return list(self.expected["episodes"])
        if "episode" in self.expected:
            return [self.expected["episode"]]
        return []

    @property
    def must_retrieve(self) -> list[str]:
        return list(self.expected.get("must_retrieve", []))

    @property
    def key_points(self) -> list[str]:
        return list(self.expected.get("key_points", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "turns": self.turns,
            "expected": self.expected,
            "must_cite": self.must_cite,
            "forbidden_claims": self.forbidden_claims,
        }


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Read and validate cases.yaml, failing loudly on a malformed set."""
    path = path or DEFAULT_CASES
    if not path.exists():
        raise FileNotFoundError(f"evaluation cases not found at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_cases = payload.get("cases") or []
    if not raw_cases:
        raise ValueError(f"{path} contains no cases")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        case_id = raw.get("id") or f"case_{index}"
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        turns = raw.get("turns") or ([raw["input"]] if raw.get("input") else [])
        if not turns:
            raise ValueError(f"case '{case_id}' has neither 'input' nor 'turns'")
        cases.append(
            EvalCase(
                id=case_id,
                category=raw.get("category", "uncategorised"),
                turns=[str(turn) for turn in turns],
                expected=raw.get("expected") or {},
                must_cite=bool(raw.get("must_cite", False)),
                forbidden_claims=[str(c) for c in (raw.get("forbidden_claims") or [])],
            )
        )
    return cases
