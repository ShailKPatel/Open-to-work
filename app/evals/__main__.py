"""`python -m app.evals ACCOUNT_ID [--no-groundedness]`, same "explicit
argument, no env-var default" convention as `python -m app.ingest.github`
(an account id is a per-call input, not deployment config). Runs the eval
suite, prints a metrics table, and writes the full report to evals/results/
(`make eval`).
"""

from __future__ import annotations

import argparse

from app.core.db import init_db
from app.evals.run import MetricsReport, run_eval, write_report


def _score_row(label: str, score: dict) -> str:
    values = (
        f"{score['precision_at_5']:>12.3f}{score['recall_at_10']:>12.3f}"
        f"{score.get('ndcg_at_10', 0.0):>12.3f}{score.get('mrr', 0.0):>12.3f}"
        f"{score['pairs_scored']:>12}{score.get('retrieved_avg', 0.0):>12.1f}"
    )
    return f"{label:20}{values}"


def _print_report(report: MetricsReport) -> None:
    print(f"Eval report for account {report.account_id} ({report.generated_at})")
    print(
        f"Golden set: {report.golden_set_size} pairs for this account, "
        f"{report.pairs_scored} scored"
    )
    print()
    header = ("precision@5", "recall@10", "nDCG@10", "MRR", "pairs", "returned")
    print(f"{'':20}" + "".join(f"{h:>12}" for h in header))
    systems = (
        ("retrieval (app)", report.retrieval),
        ("dense (reference)", report.dense),
        ("bm25 baseline", report.bm25),
    )
    for label, score in systems:
        print(_score_row(label, score))
    for name, score in report.candidates.items():
        print(_score_row("candidate " + name, score))
    print()
    for comparison, metrics in report.differences.items():
        for metric, diff in metrics.items():
            interval = diff["ci95"]
            span = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
            print(f"{comparison} {metric}: {diff['mean']:+.3f}, 95% CI {span}")
    if report.differences:
        print("(an interval spanning zero means these pairs do not separate the two)")
        print()
    if report.retrieval_beats_bm25 is None:
        print("retrieval vs baseline: not measured (no scored pairs)")
    else:
        verdict = "beats" if report.retrieval_beats_bm25 else "does NOT beat"
        print(f"retrieval {verdict} the BM25 baseline")
    if report.precision_at_10 is not None:
        print(f"precision@10 (vs ground truth): {report.precision_at_10:.3f}")
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
