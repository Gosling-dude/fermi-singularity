"""Baseline vs improved reporting.

``make eval-compare`` renders two runs side by side. Regressions are shown in
red rather than hidden: a comparison that only ever moves up is not evidence,
it is marketing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
console = Console()

# (label, dotted path into summary.json, formatter, higher_is_better)
METRICS: list[tuple[str, str, str, bool]] = [
    ("Pass rate", "pass_rate", "pct", True),
    ("Retrieval: episode hit", "retrieval.mean_episode_hit", "pct", True),
    ("Retrieval: term coverage", "retrieval.mean_term_coverage", "pct", True),
    ("Refusal accuracy", "refusal.refusal_accuracy", "pct", True),
    ("False refusals", "refusal.false_refusals", "int", False),
    ("Citation validity", "citations.mean_validity", "pct", True),
    ("Invalid citations", "citations.invalid", "int", False),
    ("Forbidden-claim leaks", "hallucination.forbidden_claim_violations", "int", False),
    ("Judge: faithfulness", "judge.mean_faithfulness", "score", True),
    ("Judge: completeness", "judge.mean_completeness", "score", True),
    ("Judge: citation correctness", "judge.mean_citation_correctness", "score", True),
    ("Judge: refusal correctness", "judge.mean_refusal_correctness", "score", True),
    ("Judge: clarity", "judge.mean_clarity", "score", True),
    ("Mean latency", "latency.mean_ms", "ms", False),
    ("Estimated cost", "cost.estimated_usd", "usd", False),
]


def _dig(payload: dict[str, Any], path: str) -> Any:
    node: Any = payload
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "score":
        return f"{value:.2f}"
    if kind == "ms":
        return f"{value / 1000:.2f}s"
    if kind == "usd":
        return f"${value:.4f}"
    return f"{int(value)}"


def _delta(before: Any, after: Any, kind: str, higher_is_better: bool) -> str:
    if before is None or after is None:
        return "[dim]—[/dim]"
    change = after - before
    if abs(change) < 1e-9:
        return "[dim]0[/dim]"
    if kind == "pct":
        rendered = f"{change:+.1%}"
    elif kind == "score":
        rendered = f"{change:+.2f}"
    elif kind == "ms":
        rendered = f"{change / 1000:+.2f}s"
    elif kind == "usd":
        rendered = f"{change:+.4f}"
    else:
        rendered = f"{int(change):+d}"
    improved = change > 0 if higher_is_better else change < 0
    return f"[green]{rendered}[/green]" if improved else f"[red]{rendered}[/red]"


def load_summary(name_or_path: str) -> tuple[str, dict[str, Any]]:
    """Resolve a run by directory name or explicit path."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        path = candidate / "summary.json"
    elif candidate.suffix == ".json" and candidate.exists():
        path = candidate
    else:
        path = RESULTS_ROOT / name_or_path / "summary.json"
    if not path.exists():
        raise SystemExit(
            f"No summary found for '{name_or_path}' (looked at {path}).\n"
            f"  → Available runs: {', '.join(sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())) or 'none'}"
        )
    return path.parent.name, json.loads(path.read_text(encoding="utf-8"))


def _case_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    raw = run_dir / "raw.jsonl"
    if not raw.exists():
        return {}
    records = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["case"]["id"]] = record
    return records


def compare(baseline_name: str, improved_name: str) -> int:
    baseline_label, baseline = load_summary(baseline_name)
    improved_label, improved = load_summary(improved_name)

    if baseline.get("mode") != improved.get("mode"):
        console.print(
            f"[yellow]Warning: comparing a '{baseline.get('mode')}' run with a "
            f"'{improved.get('mode')}' run — the metrics are not equivalent."
            f"[/yellow]\n"
        )

    table = Table(
        title=f"{baseline_label}  →  {improved_label}", header_style="bold cyan"
    )
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Improved", justify="right")
    table.add_column("Delta", justify="right")

    for label, path, kind, higher_is_better in METRICS:
        before, after = _dig(baseline, path), _dig(improved, path)
        if before is None and after is None:
            continue
        table.add_row(
            label, _fmt(before, kind), _fmt(after, kind),
            _delta(before, after, kind, higher_is_better),
        )
    console.print(table)

    # Per-category movement
    categories = sorted(
        set(baseline.get("per_category", {})) | set(improved.get("per_category", {}))
    )
    if categories:
        cat_table = Table(title="Per category", header_style="bold cyan")
        cat_table.add_column("Category")
        cat_table.add_column("Baseline", justify="right")
        cat_table.add_column("Improved", justify="right")
        for category in categories:
            before = baseline.get("per_category", {}).get(category)
            after = improved.get("per_category", {}).get(category)
            cat_table.add_row(
                category,
                f"{before['passed']}/{before['total']}" if before else "—",
                f"{after['passed']}/{after['total']}" if after else "—",
            )
        console.print(cat_table)

    # Case-level movement, so regressions are named rather than averaged away.
    before_cases = _case_map(RESULTS_ROOT / baseline_label)
    after_cases = _case_map(RESULTS_ROOT / improved_label)
    fixed, broken = [], []
    for case_id in sorted(set(before_cases) & set(after_cases)):
        was, now = before_cases[case_id]["passed"], after_cases[case_id]["passed"]
        if not was and now:
            fixed.append(case_id)
        elif was and not now:
            broken.append(case_id)
    if fixed:
        console.print(f"\n[green]Fixed ({len(fixed)}):[/green] {', '.join(fixed)}")
    if broken:
        console.print(f"[red]Regressed ({len(broken)}):[/red] {', '.join(broken)}")
        for case_id in broken:
            record = after_cases[case_id]
            reasons = (record.get("checks") or {}).get("failures") or [record.get("error")]
            console.print(f"  [dim]{case_id}: {'; '.join(str(r) for r in reasons)}[/dim]")
    if not fixed and not broken:
        console.print("\n[dim]No case-level changes.[/dim]")
    return 0


def show(run_name: str) -> int:
    label, summary = load_summary(run_name)
    console.print_json(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("baseline", nargs="?", default="run_baseline")
    parser.add_argument("improved", nargs="?", default="run_improved")
    parser.add_argument("--show", help="print one run's summary instead")
    args = parser.parse_args(argv)
    if args.show:
        return show(args.show)
    return compare(args.baseline, args.improved)


if __name__ == "__main__":
    sys.exit(main())
