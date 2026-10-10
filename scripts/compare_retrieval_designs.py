"""Scores successive retrieval designs on the same pairs, with paired intervals:
the designs of entries 2, 3 and 8 of docs/RETRIEVAL_IMPROVEMENTS.md.

Earlier designs are recreated by setting the constants later entries
changed, so every run shares all other code.

Usage: .venv/bin/python -m scripts.compare_retrieval_designs
       (--public SPLIT | --skillspan) [--write]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import app.resume_build.orchestrator as orchestrator
import app.retrieval.search as search
from app.evals.metrics import bootstrap_ci, paired_difference
from app.evals.real import build_portfolio
from app.evals.synthetic import run_retrieval_eval

_DESIGNS = {
    "entry 2": {"cap": 12, "mentioned": 0.0, "per_slot": 1, "tech": False},
    "entry 3": {"cap": 40, "mentioned": 0.5, "per_slot": 3, "tech": False},
    "entry 8 (current)": {
        "cap": search._MAX_SKILL_QUERIES,
        "mentioned": search._MENTIONED_WEIGHT,
        "per_slot": orchestrator._SKILL_HITS_PER_SLOT,
        "tech": search._TECH_TOKENS,
    },
}
_METRICS = (
    ("retrieval", "precision_at_5", "hits p@5"),
    ("retrieval", "ndcg_at_10", "hits nDCG@10"),
    ("candidate_skills", "recall_returned", "candidate pool recall"),
    ("candidate_skills", "precision_at_5", "candidate skills p@5"),
)


def _jobs(args: argparse.Namespace) -> tuple[str, list]:
    if args.skillspan:
        from app.evals.skillspan import build_retrieval_jobs

        return "skillspan", build_retrieval_jobs()
    from app.evals.public import FRESH2_GROUP, PUBLIC_DIR, build_retrieval_jobs

    if args.public == "fresh2":
        return "public-fresh2", build_retrieval_jobs(
            "test", groups=(FRESH2_GROUP,), labels_path=PUBLIC_DIR / "retrieval_labels_llm.yaml"
        )
    return f"public-{args.public}", build_retrieval_jobs(args.public)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two retrieval designs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--public", choices=("dev", "fresh2"))
    group.add_argument("--skillspan", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    name, jobs = _jobs(args)
    per_design: dict[str, dict[str, list[float]]] = {}
    for design, values in _DESIGNS.items():
        search._MAX_SKILL_QUERIES = values["cap"]
        search._MENTIONED_WEIGHT = values["mentioned"]
        orchestrator._SKILL_HITS_PER_SLOT = values["per_slot"]
        search._TECH_TOKENS = values["tech"]
        (report,) = run_retrieval_eval([build_portfolio()], jobs)
        per_design[design] = {
            label: [p[metric] for p in report.per_pair[system]]
            for system, metric, label in _METRICS
        }
        per_design[design]["bm25 p@5"] = [p["precision_at_5"] for p in report.per_pair["bm25"]]

    names = list(_DESIGNS)
    current = per_design[names[-1]]
    pairs = len(current["hits p@5"])
    lines = [
        f"# Retrieval designs compared: {name} ({pairs} pairs)",
        "",
        f"Generated {dt.datetime.now(dt.UTC).isoformat()}.",
        "",
        "| metric | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for _, _, label in _METRICS:
        cells = []
        for design in names:
            values = per_design[design][label]
            mean = sum(values) / len(values)
            ci = bootstrap_ci(values)
            cells.append(f"{mean:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{mean:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Paired differences, each design minus the one before it (95% bootstrap):", ""]
    for before, after in zip(names, names[1:], strict=False):
        for _, _, label in _METRICS:
            diff, interval = paired_difference(
                per_design[after][label], per_design[before][label]
            )
            span = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
            lines.append(f"- {after} minus {before}, {label}: {diff:+.3f} {span}")
    diff, interval = paired_difference(current["hits p@5"], current["bm25 p@5"])
    span = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "n/a"
    bm25 = sum(current["bm25 p@5"]) / pairs
    lines += ["", f"BM25 p@5 {bm25:.3f}; current hybrid minus BM25: {diff:+.3f} {span}."]
    print("\n".join(lines))
    if args.write:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = Path("evals/results") / f"designs-{name}-{stamp}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
