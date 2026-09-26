"""Compares an eval report against the committed CI baseline and fails if
retrieval got worse.

The eval harness prints numbers; on its own it cannot tell a regression
from a Tuesday. This is the part that decides. It reads a report written
by `python -m app.evals` and the expected numbers in evals/ci_baseline.json,
and exits non-zero when dense precision@5 or recall@10 has dropped by more
than the baseline's stated tolerance. Improvements never fail.

Both paths are explicit arguments with no env-var default, same rule as
`python -m app.evals` and scripts/label_golden_set.py: which report and
which baseline are per-call inputs, not deployment config.

When GITHUB_STEP_SUMMARY is set, the comparison table is appended to it so
a pull request shows dense against BM25 and the verdict without anyone
opening the job log.

Usage:
  .venv/bin/python -m scripts.check_eval_baseline \
      --report evals/results/eval-account9001-<stamp>.json \
      --baseline evals/ci_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_METRICS = (("precision_at_5", "precision@5"), ("recall_at_10", "recall@10"))


def _rows(report: dict, baseline: dict) -> list[tuple[str, float, float, float, bool]]:
    """(label, actual, expected, drop, ok) per gated metric."""
    tolerance = float(baseline["tolerance"])
    rows = []
    for key, label in _METRICS:
        actual = float(report["dense"][key])
        expected = float(baseline["dense"][key])
        drop = expected - actual
        rows.append((label, actual, expected, drop, drop <= tolerance))
    return rows


def _summary_table(report: dict, baseline: dict, rows: list) -> str:
    lines = [
        "### Eval gate (synthetic CI fixture)",
        "",
        f"Account {report['account_id']}, {report['pairs_scored']} golden pairs scored.",
        "",
        "| metric | dense | bm25 | baseline | change | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (key, label), (_, actual, expected, drop, ok) in zip(_METRICS, rows, strict=True):
        lines.append(
            f"| {label} | {actual:.3f} | {float(report['bm25'][key]):.3f} | "
            f"{expected:.3f} | {-drop:+.3f} | {'pass' if ok else 'FAIL'} |"
        )
    lines += [
        "",
        f"Tolerance: {float(baseline['tolerance']):.3f} absolute drop per metric.",
        "",
        "These numbers come from invented data and are a canary for retrieval "
        "breaking, not a measure of retrieval quality.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if retrieval regressed against the baseline."
    )
    parser.add_argument("--report", type=Path, required=True, help="eval report JSON to check")
    parser.add_argument("--baseline", type=Path, required=True, help="committed baseline JSON")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    baseline = json.loads(args.baseline.read_text())

    if report["account_id"] != baseline["account_id"]:
        raise SystemExit(
            f"report is for account {report['account_id']}, baseline expects "
            f"{baseline['account_id']}"
        )
    if report["pairs_scored"] != baseline["pairs_scored"]:
        # A pair that silently stopped being scored (an unknown collection,
        # an empty label list) would otherwise raise both averages and read
        # as an improvement.
        raise SystemExit(
            f"{report['pairs_scored']} pairs scored, baseline expects "
            f"{baseline['pairs_scored']}; the golden set or the fixture changed"
        )

    rows = _rows(report, baseline)
    for label, actual, expected, drop, ok in rows:
        verdict = "ok" if ok else "REGRESSED"
        print(f"{label:14}{actual:8.3f}  baseline {expected:.3f}  change {-drop:+.3f}  {verdict}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as handle:
            handle.write(_summary_table(report, baseline, rows) + "\n")

    if not all(row[4] for row in rows):
        raise SystemExit(
            "retrieval regressed beyond tolerance; see evals/ci_baseline.json for how "
            "to update the baseline when the change is intended"
        )
    print("eval gate passed")


if __name__ == "__main__":
    main()
