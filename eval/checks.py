"""Deterministic checks.

These run without an LLM and without a network call. They cover the failures
that are objectively decidable — a fabricated citation, an out-of-scope
question that got answered anyway, a claim the episodes never make — so the
judge is only ever asked about things that genuinely require judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from companion.chat.agent import ChatResponse
from eval.cases import EvalCase


@dataclass
class CheckResult:
    """Outcome of the deterministic layer for one case."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "failures": self.failures,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


def _contains_term(haystack: str, term: str) -> bool:
    """Word-boundary-aware substring test, case-insensitive."""
    pattern = r"\b" + re.escape(term.lower()).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, haystack.lower()) is not None


def retrieval_metrics(case: EvalCase, response: ChatResponse) -> dict[str, float]:
    """Score retrieval quality against the case's ground truth.

    ``episode_hit`` is the fraction of expected episodes that appear in the
    retrieved context; ``term_coverage`` is the fraction of the ground-truth
    terms that appear in the retrieved text. Both are computed from the
    passages alone, so they measure the retriever rather than the model.
    """
    retrieved = response.retrieval.chunks if response.retrieval else []
    context = " ".join(item.chunk.text for item in retrieved)
    episodes = {item.chunk.episode_id for item in retrieved}

    expected_episodes = case.expected_episodes
    episode_hit = (
        sum(1 for episode in expected_episodes if episode in episodes)
        / len(expected_episodes)
        if expected_episodes
        else 1.0
    )
    terms = case.must_retrieve
    term_coverage = (
        sum(1 for term in terms if _contains_term(context, term)) / len(terms)
        if terms
        else 1.0
    )
    return {
        "episode_hit": episode_hit,
        "term_coverage": term_coverage,
        "num_retrieved": float(len(retrieved)),
        "num_episodes_retrieved": float(len(episodes)),
        "top_score": response.retrieval.top_score if response.retrieval else 0.0,
        "max_dense_similarity": float(
            (response.retrieval.diagnostics.get("max_dense_similarity") or 0.0)
            if response.retrieval
            else 0.0
        ),
    }


def run_checks(case: EvalCase, response: ChatResponse) -> CheckResult:
    """Apply every deterministic check to one answered case."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    answer = response.answer or ""

    checks["non_empty_answer"] = bool(answer.strip())
    if not checks["non_empty_answer"]:
        failures.append("the answer was empty")

    # Refusal behaviour is the single most important deterministic property.
    if case.is_refusal:
        checks["refused_correctly"] = response.not_covered
        if not response.not_covered:
            failures.append("out-of-scope question was answered instead of refused")
    else:
        checks["answered_when_covered"] = not response.not_covered
        if response.not_covered:
            failures.append("in-scope question was wrongly refused as not covered")

    # A true fact the episodes never state is direct evidence the model fell
    # back on its own knowledge.
    leaked = [
        claim for claim in case.forbidden_claims if _contains_term(answer, claim)
    ]
    checks["no_forbidden_claims"] = not leaked
    if leaked:
        failures.append(f"used facts not in the episodes: {', '.join(leaked)}")

    # Citations
    if case.must_cite and not response.not_covered:
        checks["has_citation"] = bool(response.citations)
        if not response.citations:
            failures.append("no citation was given for a claim-bearing answer")
    invalid = [c for c in response.citations if not c.valid]
    checks["citations_valid"] = not invalid
    if invalid:
        failures.append(
            f"{len(invalid)} citation(s) failed validation: "
            + "; ".join(sorted({c.reason or "unknown" for c in invalid}))
        )

    metrics = retrieval_metrics(case, response)
    metrics["citation_validity"] = response.citation_validity
    metrics["num_citations"] = float(len(response.citations))
    metrics["latency_ms"] = response.latency_ms

    # Retrieval ground truth is only meaningful for cases that should find
    # something; for refusal cases, retrieving nothing relevant is correct.
    if not case.is_refusal:
        checks["retrieved_expected_episode"] = metrics["episode_hit"] >= 0.999
        if metrics["episode_hit"] < 0.999:
            failures.append(
                f"retrieval missed expected episode(s) "
                f"(hit {metrics['episode_hit']:.0%})"
            )

    return CheckResult(
        passed=all(checks.values()), checks=checks, failures=failures, metrics=metrics
    )


def retrieval_only_result(case: EvalCase, response: ChatResponse) -> CheckResult:
    """Score retrieval alone — no LLM, no API key, no cost.

    For answerable cases the retriever must surface the right episode and a
    majority of the ground-truth terms. For refusal cases the correct
    behaviour is for the evidence gate to fire, which is exactly what
    ``not_covered`` records when no LLM is called.
    """
    metrics = retrieval_metrics(case, response)
    checks: dict[str, bool] = {}
    failures: list[str] = []

    if case.is_refusal:
        checks["evidence_gate_fired"] = response.not_covered
        if not response.not_covered:
            failures.append(
                f"evidence gate did not fire for an out-of-scope question "
                f"(max similarity {metrics['max_dense_similarity']:.3f})"
            )
    else:
        checks["retrieved_expected_episode"] = metrics["episode_hit"] >= 0.999
        if metrics["episode_hit"] < 0.999:
            failures.append(
                f"missed expected episode(s) (hit {metrics['episode_hit']:.0%})"
            )
        checks["term_coverage"] = metrics["term_coverage"] >= 0.5
        if metrics["term_coverage"] < 0.5:
            failures.append(
                f"retrieved passages covered only "
                f"{metrics['term_coverage']:.0%} of the expected terms"
            )
        checks["evidence_gate_open"] = not response.not_covered
        if response.not_covered:
            failures.append("evidence gate wrongly refused an answerable question")

    return CheckResult(
        passed=all(checks.values()), checks=checks, failures=failures, metrics=metrics
    )
