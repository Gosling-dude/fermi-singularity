"""LLM-as-judge.

The judge grades five dimensions on a 1-5 scale. It is given the question,
the passages that were actually retrieved, the answer, and the rubric — and
it is told explicitly that its own knowledge of physics is not admissible
evidence. That constraint matters: a judge that grades on world knowledge
would penalise a correct refusal and reward a fluent hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from companion.chat.agent import ChatResponse
from companion.chat.prompt import render_context
from companion.chat.providers import LLMProvider, get_provider
from companion.config import Settings, get_settings
from companion.errors import ProviderError
from eval.cases import EvalCase

DIMENSIONS = (
    "faithfulness",
    "completeness",
    "citation_correctness",
    "refusal_correctness",
    "clarity",
)

JUDGE_SYSTEM = """\
You grade answers produced by a podcast question-answering system.

CRITICAL RULE: judge ONLY against the transcript passages supplied to you. \
Your own knowledge of physics, history or mathematics is NOT evidence. If the \
answer states something true that the passages do not contain, that is a \
FAITHFULNESS FAILURE, not a bonus. If the answer declines to answer and the \
passages genuinely do not contain the answer, that is CORRECT behaviour and \
must score highly.

Score each dimension from 1 to 5.

faithfulness — is every factual claim supported by the passages?
  5 = every claim traceable to a passage
  3 = mostly supported, some unsupported elaboration
  1 = substantial claims that the passages do not support

completeness — does it address what was asked, given the rubric's key points?
  5 = covers the key points that the passages support
  3 = partial
  1 = does not address the question
  For a correct refusal, score completeness on whether the refusal is clear \
and explains what IS covered where relevant.

citation_correctness — do citations name the right episode and a plausible \
timestamp for the claims they attach to?
  5 = citations present, correctly attached, drawn from the passages
  3 = present but loosely attached
  1 = missing where required, or pointing at the wrong material
  If the answer is a correct refusal and needs no citation, score 5.

refusal_correctness — did it refuse when it should and answer when it should?
  5 = exactly right
  1 = answered an unsupported question, or refused a supported one

clarity — is it clear and useful to a motivated undergraduate?
  5 = clear, well organised, no padding
  1 = confusing or evasive

Return ONLY a JSON object, no prose around it:
{"faithfulness": n, "completeness": n, "citation_correctness": n,
 "refusal_correctness": n, "clarity": n,
 "unsupported_claims": ["..."], "rationale": "one or two sentences"}

"unsupported_claims" lists any statement in the answer you could not find \
support for in the passages. Use [] when there are none."""


@dataclass
class JudgeResult:
    """Scores plus the judge's own accounting, or an error if it failed."""

    scores: dict[str, int] = field(default_factory=dict)
    unsupported_claims: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.scores)

    @property
    def passed(self) -> bool:
        """A judge pass requires groundedness, not just a good average.

        Faithfulness, citation correctness and refusal correctness are the
        trust dimensions; a fluent but ungrounded answer must not pass on the
        strength of its clarity score.
        """
        if not self.ok:
            return False
        return (
            self.scores.get("faithfulness", 0) >= 4
            and self.scores.get("citation_correctness", 0) >= 4
            and self.scores.get("refusal_correctness", 0) >= 4
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "unsupported_claims": self.unsupported_claims,
            "rationale": self.rationale,
            "passed": self.passed,
            "error": self.error,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def _clamp(value: Any) -> int:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return 1


def build_judge_prompt(case: EvalCase, response: ChatResponse) -> str:
    retrieved = response.retrieval.chunks if response.retrieval else []
    rubric = "\n".join(f"- {point}" for point in case.key_points) or "- (none given)"
    expectation = (
        "The correct behaviour is to REFUSE — the passages do not support an answer."
        if case.is_refusal
        else "The correct behaviour is to ANSWER from the passages."
    )
    if case.clarify_ok:
        expectation += (
            " Asking a clarifying question is also acceptable for this case."
        )
    conversation = ""
    if len(case.turns) > 1:
        earlier = "\n".join(f"  turn {i + 1}: {t}" for i, t in enumerate(case.turns[:-1]))
        conversation = f"\nEARLIER TURNS IN THIS CONVERSATION:\n{earlier}\n"

    return (
        f"TRANSCRIPT PASSAGES SUPPLIED TO THE SYSTEM:\n{render_context(retrieved)}\n"
        f"{conversation}\n"
        f"QUESTION BEING SCORED: {case.scored_input}\n\n"
        f"EXPECTED BEHAVIOUR: {expectation}\n\n"
        f"RUBRIC — key points a good answer covers:\n{rubric}\n\n"
        f"SYSTEM'S ANSWER:\n{response.answer}\n\n"
        f"Grade it now."
    )


def judge_response(
    case: EvalCase,
    response: ChatResponse,
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
) -> JudgeResult:
    """Grade one answered case. Never raises — failures are recorded."""
    settings = settings or get_settings()
    try:
        provider = provider or get_provider("judge", settings)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return JudgeResult(error=f"judge unavailable: {exc}")

    try:
        raw = provider.complete(
            JUDGE_SYSTEM,
            [{"role": "user", "content": build_judge_prompt(case, response)}],
            max_tokens=settings.judge_max_tokens,
            effort=settings.judge_effort,
        )
    except ProviderError as exc:
        return JudgeResult(error=str(exc), model=provider.model)

    try:
        from companion.chat.providers import _extract_json

        payload = _extract_json(raw.text)
    except ProviderError as exc:
        return JudgeResult(error=f"unparseable judge output: {exc}",
                           model=provider.model)

    return JudgeResult(
        scores={dimension: _clamp(payload.get(dimension)) for dimension in DIMENSIONS},
        unsupported_claims=[str(c) for c in (payload.get("unsupported_claims") or [])],
        rationale=str(payload.get("rationale", ""))[:600],
        model=raw.model,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        estimated_cost_usd=raw.estimated_cost_usd,
    )
