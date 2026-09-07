"""Evaluation runner.

Every case is executed against the real production pipeline — the same
``CompanionAgent``, the same retriever, the same prompts that ``make chat``
uses. There is no evaluation-only shortcut anywhere in this file.

Two modes:

* ``--retrieval-only`` scores retrieval and the evidence gate deterministically.
  It needs no API key and costs nothing, and it is the mode used to measure the
  retrieval improvement in EVAL.md.
* the default mode runs the full pipeline and adds LLM-judge scores.

Raw inputs and outputs for every case are written to ``raw.jsonl`` so results
can be inspected rather than taken on trust.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from companion.chat.agent import ChatResponse, CompanionAgent
from companion.chat.providers import provider_available
from companion.chat.session import Session
from companion.config import get_settings
from companion.errors import CompanionError
from companion.utils.logging import configure_logging, get_logger
from eval.cases import EvalCase, load_cases
from eval.checks import CheckResult, run_checks, retrieval_only_result
from eval.judge import DIMENSIONS, JudgeResult, judge_response

console = Console()
log = get_logger("eval")

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


@dataclass
class CaseOutcome:
    """Everything one case produced — the raw record written to disk."""

    case: EvalCase
    response: ChatResponse | None
    checks: CheckResult | None
    judge: JudgeResult | None = None
    error: str | None = None
    conversation: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.error or not self.checks:
            return False
        if not self.checks.passed:
            return False
        if self.judge is not None and self.judge.ok:
            return self.judge.passed
        return True

    @property
    def estimated_cost_usd(self) -> float:
        total = self.response.estimated_cost_usd if self.response else 0.0
        if self.judge:
            total += self.judge.estimated_cost_usd
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "conversation": self.conversation,
            "response": self.response.to_dict() if self.response else None,
            "checks": self.checks.to_dict() if self.checks else None,
            "judge": self.judge.to_dict() if self.judge else None,
            "error": self.error,
            "passed": self.passed,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def run_case(
    case: EvalCase, agent: CompanionAgent, *, retrieval_only: bool, judge: bool
) -> CaseOutcome:
    """Execute one case, replaying every turn through a single session."""
    session = Session()
    conversation: list[dict[str, str]] = []
    response: ChatResponse | None = None
    try:
        for turn in case.turns:
            response = agent.ask(turn, session, retrieval_only=retrieval_only)
            conversation.append({"role": "user", "content": turn})
            conversation.append({"role": "assistant", "content": response.answer})
    except CompanionError as exc:
        return CaseOutcome(case=case, response=response, checks=None,
                           error=exc.render(), conversation=conversation)
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
        return CaseOutcome(case=case, response=response, checks=None,
                           error=f"{type(exc).__name__}: {exc}",
                           conversation=conversation)

    assert response is not None
    checks = (
        retrieval_only_result(case, response)
        if retrieval_only
        else run_checks(case, response)
    )
    judged = (
        judge_response(case, response) if (judge and not retrieval_only) else None
    )
    return CaseOutcome(case=case, response=response, checks=checks,
                       judge=judged, conversation=conversation)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(
    outcomes: list[CaseOutcome], *, retrieval_only: bool, elapsed: float
) -> dict[str, Any]:
    """Aggregate metrics, overall and per category."""
    settings = get_settings()
    scored = [outcome for outcome in outcomes if outcome.checks]
    judged = [o for o in outcomes if o.judge and o.judge.ok]

    per_category: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        bucket = per_category.setdefault(
            outcome.case.category, {"total": 0, "passed": 0, "cases": []}
        )
        bucket["total"] += 1
        bucket["passed"] += int(outcome.passed)
        bucket["cases"].append(
            {"id": outcome.case.id, "passed": outcome.passed,
             "failures": outcome.checks.failures if outcome.checks else [outcome.error]}
        )
    for bucket in per_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4)

    refusal_cases = [o for o in outcomes if o.case.is_refusal and o.checks]
    answerable = [o for o in outcomes if not o.case.is_refusal and o.checks]

    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "retrieval_only" if retrieval_only else "full",
        "num_cases": len(outcomes),
        "passed": sum(1 for outcome in outcomes if outcome.passed),
        "pass_rate": round(
            sum(1 for outcome in outcomes if outcome.passed) / len(outcomes), 4
        )
        if outcomes
        else 0.0,
        "errors": sum(1 for outcome in outcomes if outcome.error),
        "config": {
            "retrieval_mode": settings.retrieval_mode,
            "query_rewrite": settings.enable_query_rewrite,
            "compare_mode": settings.enable_compare_mode,
            "top_k": settings.top_k,
            "dense_k": settings.dense_k,
            "bm25_k": settings.bm25_k,
            "min_relevance": settings.min_relevance,
            "chat_model": None if retrieval_only else settings.chat_model,
            "judge_model": settings.judge_model if judged else None,
            "embed_model": settings.embed_model,
            "chunk_target_seconds": settings.chunk_target_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "retrieval": {
            "mean_episode_hit": round(
                _mean([o.checks.metrics.get("episode_hit", 0.0) for o in answerable]), 4
            ),
            "mean_term_coverage": round(
                _mean([o.checks.metrics.get("term_coverage", 0.0) for o in answerable]),
                4,
            ),
            "mean_top_score": round(
                _mean([o.checks.metrics.get("top_score", 0.0) for o in scored]), 6
            ),
        },
        "refusal": {
            "num_refusal_cases": len(refusal_cases),
            "refusal_accuracy": round(
                _mean([float(o.passed) for o in refusal_cases]), 4
            ),
            "false_refusals": sum(
                1
                for o in answerable
                if o.response and o.response.not_covered
            ),
        },
        "latency": {
            "mean_ms": round(
                _mean([o.response.latency_ms for o in outcomes if o.response]), 1
            ),
            "max_ms": round(
                max((o.response.latency_ms for o in outcomes if o.response),
                    default=0.0),
                1,
            ),
        },
        "cost": {
            "estimated_usd": round(
                sum(outcome.estimated_cost_usd for outcome in outcomes), 6
            ),
        },
        "per_category": per_category,
        "elapsed_seconds": round(elapsed, 1),
    }

    if not retrieval_only:
        summary["citations"] = {
            "mean_validity": round(
                _mean(
                    [
                        o.checks.metrics.get("citation_validity", 1.0)
                        for o in scored
                    ]
                ),
                4,
            ),
            "total": int(
                sum(o.checks.metrics.get("num_citations", 0.0) for o in scored)
            ),
            "invalid": sum(
                1
                for o in outcomes
                if o.response
                for c in o.response.citations
                if not c.valid
            ),
        }
        summary["hallucination"] = {
            "forbidden_claim_violations": sum(
                1
                for o in scored
                if not o.checks.checks.get("no_forbidden_claims", True)
            ),
            "judge_unsupported_claims": sum(
                len(o.judge.unsupported_claims) for o in judged
            ),
        }
    if judged:
        summary["judge"] = {
            "num_judged": len(judged),
            "judge_pass_rate": round(
                _mean([float(o.judge.passed) for o in judged]), 4
            ),
            **{
                f"mean_{dimension}": round(
                    _mean([float(o.judge.scores.get(dimension, 0)) for o in judged]), 3
                )
                for dimension in DIMENSIONS
            },
        }
    return summary


def _print_summary(summary: dict[str, Any], outcomes: list[CaseOutcome]) -> None:
    from rich.table import Table

    table = Table(title=f"Evaluation — {summary['mode']} mode",
                  header_style="bold cyan")
    table.add_column("Case")
    table.add_column("Category")
    table.add_column("Result")
    table.add_column("Notes", style="dim", max_width=60)
    for outcome in outcomes:
        notes = (
            outcome.error
            or "; ".join(outcome.checks.failures if outcome.checks else [])
            or ""
        )
        if outcome.judge and outcome.judge.ok and not notes:
            notes = " ".join(
                f"{d[:4]}={outcome.judge.scores.get(d)}" for d in DIMENSIONS
            )
        table.add_row(
            outcome.case.id,
            outcome.case.category,
            "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]",
            notes[:160],
        )
    console.print(table)

    console.print(
        f"\n[bold]Pass rate:[/bold] {summary['passed']}/{summary['num_cases']} "
        f"({summary['pass_rate']:.0%})"
    )
    retrieval = summary["retrieval"]
    console.print(
        f"[bold]Retrieval:[/bold] episode hit "
        f"{retrieval['mean_episode_hit']:.0%} · term coverage "
        f"{retrieval['mean_term_coverage']:.0%}"
    )
    refusal = summary["refusal"]
    console.print(
        f"[bold]Refusal:[/bold] accuracy {refusal['refusal_accuracy']:.0%} "
        f"over {refusal['num_refusal_cases']} cases · "
        f"false refusals {refusal['false_refusals']}"
    )
    if "citations" in summary:
        citations = summary["citations"]
        console.print(
            f"[bold]Citations:[/bold] validity "
            f"{citations['mean_validity']:.0%} · {citations['invalid']} invalid "
            f"of {citations['total']}"
        )
    if "judge" in summary:
        judge = summary["judge"]
        console.print(
            "[bold]Judge:[/bold] "
            + " · ".join(
                f"{d} {judge[f'mean_{d}']:.2f}" for d in DIMENSIONS
            )
        )
    console.print(
        f"[bold]Latency:[/bold] mean {summary['latency']['mean_ms'] / 1000:.2f}s · "
        f"[bold]Cost:[/bold] ${summary['cost']['estimated_usd']:.4f}"
    )


def run_evaluation(
    *,
    label: str | None = None,
    retrieval_only: bool = False,
    judge: bool = True,
    only: list[str] | None = None,
    cases_path: Path | None = None,
) -> Path:
    """Run the full case set and write a timestamped result directory."""
    configure_logging()
    settings = get_settings()
    cases = load_cases(cases_path)
    if only:
        wanted = set(only)
        cases = [
            case for case in cases
            if case.id in wanted or case.category in wanted
        ]
        if not cases:
            raise SystemExit(f"no cases matched: {', '.join(sorted(wanted))}")

    if not retrieval_only and not provider_available("chat", settings):
        raise SystemExit(
            "No API key found for CHAT_PROVIDER="
            f"{settings.chat_provider}.\n"
            "  → Add it to .env, or run the free offline evaluation:\n"
            "      make eval-retrieval"
        )
    if judge and not retrieval_only and not provider_available("judge", settings):
        console.print(
            "[yellow]No judge API key — running deterministic checks only.[/yellow]"
        )
        judge = False

    agent = CompanionAgent(settings)
    console.print(
        f"Running {len(cases)} cases "
        f"({'retrieval only' if retrieval_only else 'full pipeline'}"
        f"{', with LLM judge' if judge and not retrieval_only else ''})\n"
    )

    started = time.perf_counter()
    outcomes: list[CaseOutcome] = []
    for index, case in enumerate(cases, start=1):
        console.print(
            f"[dim]{index}/{len(cases)}[/dim] {case.id} "
            f"[dim]({case.category})[/dim]"
        )
        outcome = run_case(case, agent, retrieval_only=retrieval_only, judge=judge)
        outcomes.append(outcome)
        log.info("case complete", case=case.id, passed=outcome.passed)
    elapsed = time.perf_counter() - started

    summary = summarise(outcomes, retrieval_only=retrieval_only, elapsed=elapsed)
    run_dir = RESULTS_ROOT / (
        label or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print()
    _print_summary(summary, outcomes)
    console.print(f"\n[dim]Raw results: {run_dir}/raw.jsonl[/dim]")
    console.print(f"[dim]Summary:     {run_dir}/summary.json[/dim]")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--label", help="name the results directory")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval deterministically; no API key needed")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM judge")
    parser.add_argument("--only", nargs="+",
                        help="run only these case ids or categories")
    parser.add_argument("--cases", type=Path, help="path to a cases file")
    args = parser.parse_args(argv)

    run_evaluation(
        label=args.label,
        retrieval_only=args.retrieval_only,
        judge=not args.no_judge,
        only=args.only,
        cases_path=args.cases,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
