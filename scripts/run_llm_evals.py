"""Runs the LLM-backed evals (app/evals/llm_evals.py) and the prompt
injection detector eval (app/evals/injection.py), and prints one markdown
report.

Makes real, billed calls, so the key is passed in explicitly and nothing
else is used: LIVE_LLM_API_KEY (required; several keys for the same
provider may be given comma-separated), LIVE_LLM_PROVIDER (default
"gemini"), and optionally LIVE_LLM_BULK_MODEL / LIVE_LLM_QUALITY_MODEL as
"<provider>/<model>". The same variables tests/live reads. On a free tier
set LIVE_LLM_RPM to the requests per minute allowed: calls are spaced to
match, and a rate-limited call waits a minute and retries. Everything runs
in a throwaway environment; the app's own database, keys and spend records
are never opened.

About 160 calls in all on the default sets: one per synthetic and
real-text posting, one per red team posting, one per persona resume, one
per labeled judge bullet. Responses are cached by content hash inside the
throwaway database only, so every run pays again.

A free Gemini tier allows 20 requests a day per model and project, so the
whole suite does not fit in one day on one free key. When every key is out
until much later the run stops, keeps what it measured, and says so; run
the remaining parts with --only once the quota resets, or --limit N to
sample each part.

Usage:
  LIVE_LLM_API_KEY=... .venv/bin/python -m scripts.run_llm_evals \\
      [--only jobs resumes judge injection] [--limit N] [--write]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path

from app.evals.injection import DetectionScore, run_detection_eval
from app.evals.llm_evals import (
    PACING,
    ExtractionScore,
    JudgeScore,
    RobustnessScore,
    llm_environment,
    run_job_extraction,
    run_judge_validation,
    run_resume_extraction,
    run_robustness,
)
from app.evals.synthetic import (
    SYNTHETIC_DIR,
    load_jobs,
    load_judge_bullets,
    load_personas,
    load_redteam,
)

_RESULTS_DIR = Path("evals/results")
_PARTS = ("jobs", "resumes", "judge", "injection")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _extraction_lines(score: ExtractionScore) -> list[str]:
    lines = [
        f"### {score.name}",
        "",
        f"{score.items} items, {len(score.failures)} failed calls.",
        "",
        "| field | accuracy | scored |",
        "| --- | --- | --- |",
    ]
    for name, tally in sorted(score.fields.items()):
        lines.append(f"| {name} | {_pct(tally.accuracy)} | {tally.scored} |")
    recall = _pct(score.mean_skill_recall)
    lines.append(f"| skills (recall) | {recall} | {len(score.skill_recalls)} |")
    if score.invented_salary:
        lines += ["", f"Salary invented where none was stated: {', '.join(score.invented_salary)}"]
    if score.mismatches:
        lines += ["", "<details><summary>Mismatches</summary>", ""]
        lines += [f"- {m}" for m in score.mismatches]
        lines += ["", "</details>"]
    if score.failures:
        lines += ["", "Failed calls:", *[f"- {f}" for f in score.failures]]
    return lines + [""]


def _judge_lines(score: JudgeScore) -> list[str]:
    lines = [
        "### Groundedness judge against human labels",
        "",
        f"{score.bullets} labeled bullets, {len(score.unusable)} unusable replies.",
        "",
        f"- Accuracy: {_pct(score.accuracy)}",
        f"- Cohen's kappa: {_pct(score.kappa)}",
        f"- Ungrounded bullets caught: {_pct(score.ungrounded_recall)}",
        "",
        "| how the bullet was made | judge agrees | bullets |",
        "| --- | --- | --- |",
    ]
    for kind, tally in sorted(score.by_kind.items()):
        lines.append(f"| {kind} | {_pct(tally.accuracy)} | {tally.scored} |")
    if score.disagreements:
        lines += ["", "Disagreements:", *[f"- {d}" for d in score.disagreements]]
    return lines + [""]


def _robustness_lines(score: RobustnessScore) -> list[str]:
    rate = score.resisted / score.attacks if score.attacks else None
    lines = [
        f"### Extraction under attack: {score.name}",
        "",
        f"- Attacks resisted: {score.resisted}/{score.attacks} ({_pct(rate)})",
        f"- Benign controls read correctly: {score.benign_correct}/{score.benign}",
    ]
    if score.obeyed:
        lines += ["", "Not resisted:", *[f"- {o}" for o in score.obeyed]]
    if score.failures:
        lines += ["", "Failed calls:", *[f"- {f}" for f in score.failures]]
    return lines + [""]


def _detection_lines(scores: list[DetectionScore]) -> list[str]:
    lines = [
        "### Prompt injection detector (no LLM)",
        "",
        "| set | detected | false positives |",
        "| --- | --- | --- |",
    ]
    for s in scores:
        detected = f"{s.detected}/{s.attacks}" if s.attacks else "-"
        lines.append(f"| {s.name} | {detected} | {s.false_positives}/{s.benign} |")
    lines += [
        "",
        "The development set is what the rules were written against; the held-out",
        "set was written afterwards and never tuned against, so its rate is the one",
        "to quote.",
    ]
    return lines + [""]


def _stopped_note(score: object) -> list[str]:
    stopped = getattr(score, "stopped", None)
    return [f"Stopped early: {stopped}. Counts above are partial.", ""] if stopped else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM-backed evals.")
    parser.add_argument("--only", nargs="*", choices=_PARTS, default=list(_PARTS))
    parser.add_argument(
        "--limit", type=int, default=None, help="score at most this many items per part"
    )
    parser.add_argument("--write", action="store_true", help="save under evals/results/")
    args = parser.parse_args()

    keys = [k.strip() for k in os.environ.get("LIVE_LLM_API_KEY", "").split(",") if k.strip()]
    if not keys:
        raise SystemExit("set LIVE_LLM_API_KEY to the key these evals should use")
    rpm = float(os.environ.get("LIVE_LLM_RPM") or 0)
    PACING.min_interval = 60.0 / rpm if rpm > 0 else 0.0
    PACING.retries = 5

    stamp = dt.datetime.now(dt.UTC)
    out = _RESULTS_DIR / f"llm-{stamp.strftime('%Y%m%dT%H%M%SZ')}.md"
    lines = [f"# LLM evals ({stamp.date().isoformat()})", ""]
    if args.limit:
        lines += [f"Sampled: at most {args.limit} items per part.", ""]

    def emit(section: list[str]) -> None:
        """Adds a finished section, printing it and saving the report so far,
        so a run stopped later keeps every part it completed."""
        lines.extend(section)
        print("\n".join(section), flush=True)
        if args.write:
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines) + "\n")

    def take(items: list) -> list:
        return items[: args.limit] if args.limit else items

    if "injection" in args.only:
        emit(_detection_lines(run_detection_eval()))

    parts: list[tuple[str, Callable[[], tuple[list[str], object]]]] = []
    if "jobs" in args.only:
        parts.append(("jobs", lambda: _scored(
            _extraction_lines,
            run_job_extraction(take(load_jobs()), "job extraction (synthetic postings)"),
        )))
        parts.append(("jobs", _real_jobs(take)))
    if "resumes" in args.only:
        parts.append(("resumes", lambda: _scored(
            _extraction_lines, run_resume_extraction(take(load_personas()))
        )))
    if "judge" in args.only:
        parts.append(("judge", lambda: _scored(
            _judge_lines, run_judge_validation(load_personas(), take(load_judge_bullets()))
        )))
    if "injection" in args.only:
        holdout = load_redteam(SYNTHETIC_DIR / "redteam_holdout.yaml")
        parts.append(("injection", lambda: _scored(
            _robustness_lines, run_robustness(take(load_redteam()), "development set")
        )))
        parts.append(("injection", lambda: _scored(
            _robustness_lines, run_robustness(take(holdout), "held-out set")
        )))

    with llm_environment(
        keys,
        provider=os.environ.get("LIVE_LLM_PROVIDER", "gemini"),
        bulk_model=os.environ.get("LIVE_LLM_BULK_MODEL") or None,
        quality_model=os.environ.get("LIVE_LLM_QUALITY_MODEL") or None,
    ):
        for _, run in parts:
            section, score = run()
            emit(section + _stopped_note(score))
            if getattr(score, "stopped", None):
                emit(["Remaining parts skipped: the provider's quota is used up.", ""])
                break

    if args.write:
        print(f"\nWritten to {out}")


def _scored(render: Callable, score: object) -> tuple[list[str], object]:
    return render(score), score


def _real_jobs(take: Callable[[list], list]) -> Callable[[], tuple[list[str], object]]:
    def run() -> tuple[list[str], object]:
        from app.evals.real import CacheMissingError, build_jobs

        try:
            jobs = build_jobs()
        except CacheMissingError as e:
            return [f"Real postings skipped: {e}", ""], None
        return _scored(
            _extraction_lines, run_job_extraction(take(jobs), "job extraction (real postings)")
        )

    return run


if __name__ == "__main__":
    main()
