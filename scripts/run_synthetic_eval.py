"""Scores retrieval on the synthetic dataset (evals/synthetic/), or with
--real on the labeled real-text set (evals/real/): seeds the profiles into
a throwaway database and in-process Qdrant, runs the eval harness per
profile against the golden pairs, prints the results and deletes
everything it created. Your own database and Qdrant server are never
opened (see app/evals/synthetic.py's isolated_environment).

--real needs the downloaded text: run
`python -m scripts.fetch_real_eval_data` first. --public dev|test scores
the LinkedIn software postings (evals/public/) against the same portfolio;
it needs `python -m scripts.fetch_public_eval_data` and the extractions
saved by `python -m scripts.run_llm_evals --only public public-dev`.

Loads the real embedding model; makes no LLM calls.

With --write, the summary table is also saved as markdown under
evals/results/ so a run can be committed as evidence. It holds only
aggregate numbers, never document text.

Usage: .venv/bin/python -m scripts.run_synthetic_eval [--real | --public SPLIT] [--write]
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
_PUBLIC_NOTE = (
    "LinkedIn software postings (public dataset, CC BY-SA 4.0) scored against",
    "the composite open-source portfolio. Queries are the app's own LLM",
    "extraction of each posting. Relevance was labeled before the search was",
    "first run on these postings; the test split was never tuned against.",
    "One labeler; see evals/public/DATA_SOURCES.md.",
)
_FRESH2_NOTE = (
    "150 LinkedIn software postings drawn after every earlier set, scored",
    "once. Relevance labels are by an LLM annotator validated against the 80",
    "hand-labeled original postings before use (kappa 0.849, micro F1 0.859),",
    "written before the search ran on these postings: evals/public/",
    "retrieval_labels_llm.yaml, docs/RETRIEVAL_IMPROVEMENTS.md entry 4.",
)
_SKILLSPAN_NOTE = (
    "SkillSpan tech postings (Zhang et al., NAACL 2022, CC BY 4.0). A portfolio",
    "skill is relevant when a span its annotators marked as knowledge names it",
    "exactly. Covers only named skills and favours keyword matching; a",
    "regression check on independent labels, not the headline.",
)
_REAL_NOTE = (
    "Public job postings scored against a composite portfolio of public",
    "open-source repositories, with relevance labeled by judgment rather than",
    "string match. Small sample, one labeler; see evals/real/DATA_SOURCES.md.",
)


_SYSTEMS = (
    ("retrieval", "app retrieval (hybrid)"),
    ("dense", "dense alone, max similarity per skill"),
    ("dense_single", "dense alone, single query (previous)"),
    ("bm25", "BM25 keyword baseline"),
)
_DIFFERENCES = (
    ("retrieval", "bm25"),
    ("retrieval", "dense"),
    ("dense", "bm25"),
    ("dense", "dense_single"),
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
    scored = [
        r.candidates["skills"]
        for r in reports
        if r.candidates.get("skills", {}).get("pairs_scored")
    ]
    if scored:
        total = sum(s["pairs_scored"] for s in scored)
        recall = sum(s["recall_returned"] * s["pairs_scored"] for s in scored) / total
        lines += [
            "",
            f"Candidate skills handed to the model (up to 25, deduplicated by name): "
            f"{recall:.3f} of relevant skills present, over {total} pairs.",
        ]
    lines += ["", "Paired differences over the same pairs (95 percent bootstrap interval):", ""]
    for system, other in _DIFFERENCES:
        for metric, label in _METRICS[:3]:
            mean, interval = paired_difference(
                [p[metric] for p in pooled[system]], [p[metric] for p in pooled[other]]
            )
            span = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
            lines.append(f"- {system} minus {other}, {label}: {mean:+.3f} {span}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval on the synthetic or real set.")
    parser.add_argument(
        "--real", action="store_true", help="score the labeled real-text set instead"
    )
    parser.add_argument(
        "--public", choices=("dev", "test", "fresh", "fresh2"),
        help="score a split of the LinkedIn set instead (fresh, fresh2: later test sets)",
    )
    parser.add_argument(
        "--skillspan", action="store_true", help="score the SkillSpan tech postings instead"
    )
    parser.add_argument(
        "--write", action="store_true", help="also save the summary under evals/results/"
    )
    args = parser.parse_args()

    note: tuple[str, ...] = _SYNTHETIC_NOTE
    if args.public:
        from app.evals.public import build_retrieval_jobs

        name = f"public-{args.public}"
        personas = [build_portfolio()]
        if args.public == "fresh":
            jobs = build_retrieval_jobs("test", groups=("software-fresh",))
        elif args.public == "fresh2":
            from app.evals.public import FRESH2_GROUP, PUBLIC_DIR

            jobs = build_retrieval_jobs(
                "test",
                groups=(FRESH2_GROUP,),
                labels_path=PUBLIC_DIR / "retrieval_labels_llm.yaml",
            )
            note = _FRESH2_NOTE
        else:
            jobs = build_retrieval_jobs(args.public)
        if args.public != "fresh2":
            note = _PUBLIC_NOTE
    elif args.skillspan:
        from app.evals.skillspan import build_retrieval_jobs as skillspan_jobs

        name = "skillspan"
        personas = [build_portfolio()]
        jobs = skillspan_jobs()
        note = _SKILLSPAN_NOTE
    elif args.real:
        name = "real-text"
        personas = [build_portfolio()]
        jobs = build_jobs()
        note = _REAL_NOTE
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
        *note,
    ]
    print("\n".join(lines))

    if args.write:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = _RESULTS_DIR / f"{name}-{stamp.strftime('%Y%m%dT%H%M%SZ')}.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
