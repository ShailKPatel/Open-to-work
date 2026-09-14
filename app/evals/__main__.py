"""`python -m app.evals ACCOUNT_ID [--no-groundedness]`, same "explicit
argument, no env-var default" convention as `python -m app.ingest.github`
(an account id is a per-call input, not deployment config). Runs the eval
suite, prints a metrics table, and writes the full report to evals/results/
(`make eval`).
"""

from __future__ import annotations

import argparse

from app.core.db import init_db
from app.evals.run import run_eval, write_report


def _print_report(report) -> None:  # noqa: ANN001 - MetricsReport, kept loosely typed for the CLI print
    print(f"Eval report for account {report.account_id} ({report.generated_at})")
    print(
        f"Golden set: {report.golden_set_size} pairs for this account, "
        f"{report.pairs_scored} scored"
    )
    print()
    print(f"{'':20}{'precision@5':>14}{'recall@10':>14}{'pairs':>10}")
    print(
        f"{'dense':20}{report.dense['precision_at_5']:>14.3f}"
        f"{report.dense['recall_at_10']:>14.3f}{report.dense['pairs_scored']:>10}"
    )
    print(
        f"{'bm25 baseline':20}{report.bm25['precision_at_5']:>14.3f}"
        f"{report.bm25['recall_at_10']:>14.3f}{report.bm25['pairs_scored']:>10}"
    )
    print()
    if report.hybrid_beats_baseline is None:
        print("dense vs baseline: not measured (no scored pairs)")
    else:
        verdict = "beats" if report.hybrid_beats_baseline else "does NOT beat"
        print(f"dense {verdict} the BM25 baseline")
    if report.context_precision is not None:
        print(f"context precision (top-10, vs ground truth): {report.context_precision:.3f}")
    if report.groundedness is not None:
        print(
            f"groundedness: {report.groundedness:.3f} "
            f"({report.groundedness_checked} bullets checked)"
        )
    else:
        print(f"groundedness: not measured ({report.groundedness_checked} bullets checked)")
    print(
        f"cost this run: ${report.cost_usd:.4f} · avg latency: "
        f"{report.latency_ms_avg:.0f}ms · {report.llm_calls} LLM calls"
    )
    if report.notes:
        print()
        print("Notes:")
        for note in report.notes:
            print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval/groundedness eval suite.")
    parser.add_argument("account_id", type=int, help="account to run the eval for")
    parser.add_argument(
        "--no-groundedness",
        action="store_true",
        help="skip the LLM-judge groundedness pass (no LLM calls, no cost)",
    )
    args = parser.parse_args()

    init_db()
    report = run_eval(args.account_id, include_groundedness=not args.no_groundedness)
    _print_report(report)
    out_path = write_report(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
