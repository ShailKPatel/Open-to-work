"""Scores retrieval on the synthetic dataset (evals/synthetic/), or with
--real on the labeled real-text set (evals/real/): seeds the profiles into
a throwaway database and in-process Qdrant, runs the eval harness per
profile against the golden pairs, prints the results and deletes
everything it created. Your own database and Qdrant server are never
opened (see app/evals/synthetic.py's isolated_environment).

--real needs the downloaded text: run
`python -m scripts.fetch_real_eval_data` first.

Loads the real embedding model; makes no LLM calls.

With --write, the summary table is also saved as markdown under
evals/results/ so a run can be committed as evidence. It holds only
aggregate numbers, never document text.

Usage: .venv/bin/python -m scripts.run_synthetic_eval [--real] [--write]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from app.evals.metrics import bootstrap_ci, paired_difference
from app.evals.real import build_jobs, build_portfolio
from app.evals.synthetic import load_jobs, load_personas, run_retrieval_eval

_RESULTS_DIR = Path("evals/results")

_SYNTHETIC_NOTE = (
    "Golden pairs are rule-labeled over invented data (a document is relevant",
    "when it names a skill the posting asks for), which favours keyword",
    "matching. Read these as a regression signal across varied profiles, not",
    "as real-world retrieval quality.",
)
_REAL_NOTE = (
    "Public job postings scored against a composite portfolio of public",
    "open-source repositories, with relevance labeled by judgment rather than",
    "string match. Small sample, one labeler; see evals/real/DATA_SOURCES.md.",
)


_SYSTEMS = (
    ("retrieval", "app retrieval (hybrid)"),
    ("dense", "dense, single query (previous)"),
    ("bm25", "BM25 keyword baseline"),
)
_METRICS = (
    ("precision_at_5", "p@5"),
    ("recall_at_10", "r@10"),
    ("ndcg_at_10", "nDCG@10"),
    ("reciprocal_rank", "MRR"),
)


def _cell(values: list[float]) -> str:
    mean = sum(values) / len(values) if values else 0.0
    interval = bootstrap_ci(values)
    if interval is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{interval[0]:.2f}, {interval[1]:.2f}]"


def _summary(reports: list) -> list[str]:
    """One row per system, every metric pooled over all pairs of all
    profiles with a bootstrap 95 percent interval, then the paired
    differences that say whether the gaps are real."""
    pooled = {key: [p for r in reports for p in r.per_pair[key]] for key, _ in _SYSTEMS}
    lines = [
        "| system | " + " | ".join(label for _, label in _METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in _METRICS) + " |",
    ]
    for key, label in _SYSTEMS:
        cells = [_cell([p[metric] for p in pooled[key]]) for metric, _ in _METRICS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Paired differences over the same pairs (95 percent bootstrap interval):", ""]
    for other in ("bm25", "dense"):
        for metric, label in _METRICS[:3]:
            mean, interval = paired_difference(
                [p[metric] for p in pooled["retrieval"]], [p[metric] for p in pooled[other]]
            )
            span = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
            lines.append(f"- retrieval minus {other}, {label}: {mean:+.3f} {span}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval on the synthetic or real set.")
    parser.add_argument(
        "--real", action="store_true", help="score the labeled real-text set instead"
    )
    parser.add_argument(
        "--write", action="store_true", help="also save the summary under evals/results/"
    )
    args = parser.parse_args()

    if args.real:
        name = "real-text"
        personas = [build_portfolio()]
        jobs = build_jobs()
    else:
        name = "synthetic"
        personas = load_personas()
        jobs = load_jobs()
    reports = run_retrieval_eval(personas, jobs)

    pairs = sum(r.pairs_scored for r in reports)
    stamp = dt.datetime.now(dt.UTC)
    lines = [
        f"# {name.capitalize()} retrieval eval "
        f"({len(personas)} profiles, {len(jobs)} postings)",
        "",
        f"Generated {stamp.isoformat()}. {pairs} scored golden pairs.",
        "",
        *_summary(reports),
        "",
        *(_REAL_NOTE if args.real else _SYNTHETIC_NOTE),
    ]
    print("\n".join(lines))

    if args.write:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = _RESULTS_DIR / f"{name}-{stamp.strftime('%Y%m%dT%H%M%SZ')}.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
