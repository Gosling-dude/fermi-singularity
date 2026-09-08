"""The evaluation system itself must be trustworthy."""

from __future__ import annotations

import pytest

from companion.chat.agent import ChatResponse
from companion.chat.citations import validate_citations, extract_citations
from companion.retrieve.retriever import RetrievalResult
from eval.cases import EvalCase, load_cases
from eval.checks import run_checks, retrieval_only_result
from eval.judge import JudgeResult


def _response(answer, retrieved, episodes, *, not_covered=False):
    citations = validate_citations(extract_citations(answer), retrieved, episodes)
    return ChatResponse(
        answer=answer, question="q", retrieval_query="q", mode="default",
        not_covered=not_covered, citations=citations,
        retrieval=RetrievalResult(
            query="q", chunks=retrieved,
            diagnostics={"max_dense_similarity": 0.8, "max_rerank_score": 4.0,
                         "relevance_gate": -9.98},
        ),
    )


# --- the shipped case set ------------------------------------------------
def test_shipped_cases_load_and_are_varied():
    cases = load_cases()
    assert len(cases) >= 12, "the brief requires at least 10 cases"
    categories = {case.category for case in cases}
    assert {"factual", "cross_episode", "refusal", "followup", "locate"} <= categories
    assert sum(1 for c in cases if c.category == "refusal") >= 2
    assert any(len(c.turns) > 1 for c in cases), "need a multi-turn case"


def test_every_refusal_case_declares_forbidden_claims():
    for case in load_cases():
        if case.is_refusal:
            assert case.forbidden_claims, f"{case.id} needs forbidden_claims"


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n  - id: a\n    input: x\n  - id: a\n    input: y\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_case_without_input_is_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n  - id: a\n    category: factual\n")
    with pytest.raises(ValueError, match="neither"):
        load_cases(path)


# --- deterministic checks ------------------------------------------------
def test_grounded_answer_passes(retrieved, episodes):
    case = EvalCase(id="t", category="factual", turns=["q"],
                    expected={"episode": "quantum"}, must_cite=True)
    answer = 'Planck assumed discrete energies (Ep. "The Birth of the Quantum" 0:10-1:00).'
    assert run_checks(case, _response(answer, retrieved, episodes)).passed


def test_missing_citation_fails_when_required(retrieved, episodes):
    case = EvalCase(id="t", category="factual", turns=["q"],
                    expected={"episode": "quantum"}, must_cite=True)
    result = run_checks(case, _response("Planck assumed discrete energies.",
                                        retrieved, episodes))
    assert not result.passed
    assert any("no citation" in f for f in result.failures)


def test_forbidden_claim_is_caught(retrieved, episodes):
    case = EvalCase(id="t", category="refusal", turns=["q"],
                    expected={"refusal": True},
                    forbidden_claims=["Canberra"])
    result = run_checks(
        case, _response("The capital is Canberra.", retrieved, episodes,
                        not_covered=False)
    )
    assert not result.passed
    assert any("Canberra" in f for f in result.failures)


def test_answering_an_out_of_scope_question_fails(retrieved, episodes):
    case = EvalCase(id="t", category="refusal", turns=["q"],
                    expected={"refusal": True}, forbidden_claims=[])
    result = run_checks(case, _response("Here is an answer.", retrieved, episodes))
    assert not result.passed
    assert any("instead of refused" in f for f in result.failures)


def test_correct_refusal_passes(retrieved, episodes):
    case = EvalCase(id="t", category="refusal", turns=["q"],
                    expected={"refusal": True}, forbidden_claims=["Canberra"])
    result = run_checks(
        case, _response("Not covered.", retrieved, episodes, not_covered=True)
    )
    assert result.passed


def test_wrongly_refusing_a_covered_question_fails(retrieved, episodes):
    case = EvalCase(id="t", category="factual", turns=["q"],
                    expected={"episode": "quantum"})
    result = run_checks(
        case, _response("Not covered.", retrieved, episodes, not_covered=True)
    )
    assert not result.passed
    assert any("wrongly refused" in f for f in result.failures)


def test_invalid_citation_fails_the_run(retrieved, episodes):
    case = EvalCase(id="t", category="factual", turns=["q"],
                    expected={"episode": "quantum"}, must_cite=True)
    answer = 'X (Ep. "The Birth of the Quantum" 0:10-1:00) Y (Ep. "Fake" 1:00-2:00).'
    result = run_checks(case, _response(answer, retrieved, episodes))
    assert not result.passed


def test_retrieval_metrics_measure_the_retriever(retrieved, episodes):
    case = EvalCase(id="t", category="factual", turns=["q"],
                    expected={"episode": "quantum",
                              "must_retrieve": ["ultraviolet catastrophe", "nonsense"]})
    result = run_checks(case, _response("x", retrieved, episodes))
    assert result.metrics["episode_hit"] == 1.0
    assert result.metrics["term_coverage"] == 0.5


def test_retrieval_only_mode_scores_the_gate(retrieved, episodes):
    case = EvalCase(id="t", category="refusal", turns=["q"],
                    expected={"refusal": True})
    fired = retrieval_only_result(
        case, _response("", retrieved, episodes, not_covered=True)
    )
    missed = retrieval_only_result(case, _response("", retrieved, episodes))
    assert fired.passed and not missed.passed


# --- judge ---------------------------------------------------------------
def test_judge_pass_requires_grounding_not_just_fluency():
    fluent_but_ungrounded = JudgeResult(
        scores={"faithfulness": 2, "completeness": 5, "citation_correctness": 5,
                "refusal_correctness": 5, "clarity": 5}
    )
    assert not fluent_but_ungrounded.passed

    grounded = JudgeResult(
        scores={"faithfulness": 5, "completeness": 3, "citation_correctness": 4,
                "refusal_correctness": 5, "clarity": 3}
    )
    assert grounded.passed


def test_judge_failure_is_recorded_not_raised():
    assert not JudgeResult(error="rate limited").ok
    assert not JudgeResult(error="rate limited").passed


def test_echoing_a_forbidden_term_from_the_question_is_not_a_leak(
    retrieved, episodes
):
    """A refusal must be able to name what it is declining.

    "The episodes don't mention Hawking radiation" repeats a term from the
    learner's own question; that is not evidence of ungrounded generation.
    """
    case = EvalCase(
        id="t", category="refusal",
        turns=["What is Hawking radiation?"],
        expected={"refusal": True},
        forbidden_claims=["Hawking", "event horizon"],
    )
    result = run_checks(
        case,
        _response("The episodes don't mention Hawking radiation.",
                  retrieved, episodes, not_covered=True),
    )
    assert result.passed, result.failures


def test_a_term_absent_from_the_question_is_still_caught(retrieved, episodes):
    case = EvalCase(
        id="t", category="refusal",
        turns=["What is Hawking radiation?"],
        expected={"refusal": True},
        forbidden_claims=["Hawking", "event horizon"],
    )
    result = run_checks(
        case,
        _response("It is emitted at the event horizon.", retrieved, episodes,
                  not_covered=True),
    )
    assert not result.passed
    assert any("event horizon" in f for f in result.failures)


def test_refusal_message_makes_no_unverifiable_process_claim():
    from companion.chat.agent import NOT_COVERED_MESSAGE

    lowered = NOT_COVERED_MESSAGE.lower()
    assert "i searched" not in lowered
    assert "every episode" not in lowered
    assert "don't cover" in lowered
