"""Runs the LLM-backed evals (app/evals/llm_evals.py) and the prompt
injection detector eval (app/evals/injection.py), and prints one markdown
report.

Makes real, billed calls, so the key is passed in explicitly and nothing
else is used: LIVE_LLM_API_KEY (several keys for the same provider may be
given comma-separated) or, when that is unset, evals/.llm-eval-keys (one
key per line, gitignored), LIVE_LLM_PROVIDER (default
"gemini"), and optionally LIVE_LLM_BULK_MODEL / LIVE_LLM_QUALITY_MODEL as
"<provider>/<model>". The same variables tests/live reads. On a free tier
calls are spaced to LIVE_LLM_RPM requests per minute (default 6), the run
rests LIVE_LLM_PAUSE_SECONDS (default 90) every LIVE_LLM_PAUSE_EVERY calls
(default 20), and a rate-limited call waits a minute and retries. Everything runs
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
      [--only PART ...] [--limit N] [--shard K/N] [--write]

--shard splits every part's items across N processes run side by side;
with --only public, the saved extractions from all shards are combined by
a final run without --shard, which scores everything from the saved files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path

from app.evals.generated import (
    FilterScore,
    load_generated,
    load_perturbed,
    run_judge_generated,
    score_filter,
)
from app.evals.injection import DetectionScore, run_detection_eval
from app.evals.llm_evals import (
    ExtractionScore,
    JudgeScore,
    RobustnessScore,
    llm_environment,
    models_used,
    pacing_from_env,
    run_job_extraction,
    run_judge_validation,
    run_resume_extraction,
    run_robustness,
)
from app.evals.metrics import wilson_interval
from app.evals.synthetic import (
    SYNTHETIC_DIR,
    load_jobs,
    load_judge_bullets,
    load_personas,
    load_redteam,
)

_RESULTS_DIR = Path("evals/results")
# Keys for these evals only, one per line, "#" for comments. Gitignored and
# kept out of the Docker build context; the app never reads it.
KEYS_FILE = Path("evals/.llm-eval-keys")


def eval_keys() -> list[str]:
    """LIVE_LLM_API_KEY (comma-separated) when set, else KEYS_FILE."""
    from_env = [k.strip() for k in os.environ.get("LIVE_LLM_API_KEY", "").split(",")]
    keys = [k for k in from_env if k]
    if keys or not KEYS_FILE.exists():
        return keys
    lines = (line.strip() for line in KEYS_FILE.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


_PARTS = (
    "jobs", "public", "public-dev", "public-fresh", "public-fresh-2", "skillspan", "resumes",
    "judge", "judge-generated", "injection",
)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _interval(correct: int, total: int) -> str:
    bounds = wilson_interval(correct, total)
    return "n/a" if bounds is None else f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"


def _extraction_lines(score: ExtractionScore) -> list[str]:
    lines = [
        f"### {score.name}",
        "",
        f"{score.items} items, {len(score.failures)} failed calls{_cached_note(score)}.",
        "",
        "| field | accuracy | 95% CI (Wilson) | scored |",
        "| --- | --- | --- | --- |",
    ]
    for name, tally in sorted(score.fields.items()):
        ci = _interval(tally.correct, tally.scored)
        lines.append(f"| {name} | {_pct(tally.accuracy)} | {ci} | {tally.scored} |")
    if score.skill_recalls:
        recall = _pct(score.mean_skill_recall)
        lines.append(f"| skills (recall) | {recall} | | {len(score.skill_recalls)} |")
    if score.invented_salary:
        lines += ["", f"Salary invented where none was stated: {', '.join(score.invented_salary)}"]
    if score.mismatches:
        lines += ["", "<details><summary>Mismatches</summary>", ""]
        lines += [f"- {m}" for m in score.mismatches]
        lines += ["", "</details>"]
    if score.failures:
        lines += ["", "Failed calls:", *[f"- {f}" for f in score.failures]]
    return lines + [""]


def _judge_lines(score: JudgeScore, title: str = "human labels") -> list[str]:
    lines = [
        f"### Groundedness judge against {title}",
        "",
        f"{score.bullets} labeled bullets judged, {len(score.unusable)} unusable replies, "
        f"{len(score.errors)} failed calls.",
        "",
        f"- Accuracy: {_pct(score.accuracy)} {_interval(_agreements(score), len(score.labels))}",
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


def _cached_note(score: ExtractionScore) -> str:
    if not score.from_cache:
        return ""
    return f", {score.from_cache} scored from an earlier run's saved output"


def _agreements(score: JudgeScore) -> int:
    return sum(1 for lab, v in zip(score.labels, score.verdicts, strict=True) if lab == v)


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
    parser.add_argument(
        "--split", choices=("dev", "test"), default="test",
        help="split of the generated bullets to judge (judge-generated)",
    )
    parser.add_argument(
        "--shard", default="0/1",
        help="K/N: take every Nth item from the Kth, to run N processes side by side",
    )
    args = parser.parse_args()
    shard, shards = (int(n) for n in args.shard.split("/"))

    keys = eval_keys()
    if not keys:
        raise SystemExit(
            f"set LIVE_LLM_API_KEY, or list keys one per line in {KEYS_FILE} (gitignored)"
        )
    pacing_from_env()

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
        items = items[shard::shards]
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
    if "public" in args.only:
        parts.append(("public", _public_jobs(take, "test")))
    if "public-dev" in args.only:
        parts.append(("public-dev", _public_jobs(take, "dev")))
    if "public-fresh" in args.only:
        parts.append(("public-fresh", _fresh_extractions(take)))
    if "skillspan" in args.only:
        from app.evals.skillspan import CACHE_DIR as SKILLSPAN_CACHE
        from app.evals.skillspan import build_extraction_jobs

        parts.append(("skillspan", lambda: _scored(_extraction_lines, run_job_extraction(
            take(build_extraction_jobs()), "job extraction (SkillSpan tech postings)",
            save_dir=SKILLSPAN_CACHE / "extractions",
        ))))
    if "public-fresh-2" in args.only:
        parts.append(("public-fresh-2", _fresh_extractions(take, "software-fresh-2")))
    if "resumes" in args.only:
        parts.append(("resumes", lambda: _scored(
            _extraction_lines, run_resume_extraction(take(load_personas()))
        )))
    if "judge" in args.only:
        parts.append(("judge", lambda: _scored(
            _judge_lines, run_judge_validation(load_personas(), take(load_judge_bullets()))
        )))
        hard = load_judge_bullets(SYNTHETIC_DIR / "judge_bullets_hard.yaml")
        parts.append(("judge", lambda: _scored(
            lambda score: _judge_lines(score, "held-out subtle cases"),
            run_judge_validation(load_personas(), take(hard)),
        )))
        scope = load_judge_bullets(SYNTHETIC_DIR / "judge_bullets_scope.yaml")
        parts.append(("judge", lambda: _scored(
            lambda score: _judge_lines(score, "held-out scope claims"),
            run_judge_validation(load_personas(), take(scope)),
        )))
    if "judge-generated" in args.only:
        parts.append(("judge-generated", _generated_judge(take, args.split)))
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
        used = models_used()
        if used:
            counts = ", ".join(f"{model} {n}" for model, n in sorted(used.items()))
            emit([f"Models that answered (calls): {counts}.", ""])

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


def _filter_lines(score: FilterScore) -> list[str]:
    caught = score.dropped_ungrounded
    false_drops = score.dropped_grounded
    grounded = score.kept_grounded + score.dropped_grounded
    return [
        "### Deterministic grounding filter on the same bullets (no LLM)",
        "",
        f"- Ungrounded bullets dropped: {caught}/{score.ungrounded} "
        f"{_interval(caught, score.ungrounded)}",
        f"- Grounded bullets wrongly dropped: {false_drops}/{grounded} "
        f"{_interval(false_drops, grounded)}",
        "",
    ]


def _generated_judge(
    take: Callable[[list], list], split: str
) -> Callable[[], tuple[list[str], object]]:
    def run() -> tuple[list[str], object]:
        natural = load_generated(split)
        if not natural:
            return [f"No labeled generated bullets in the {split} split.", ""], None
        perturbed = load_perturbed() if split == "test" else []
        items = take(natural + perturbed)
        ungrounded = sum(1 for i in natural if not i.grounded)
        header = [
            f"### Bullets the app wrote ({split} split)",
            "",
            f"{len(natural)} labeled bullets as written, {ungrounded} labeled ungrounded "
            f"(95% CI {_interval(ungrounded, len(natural))} of what the writer produces).",
            f"Plus {len(perturbed)} of them changed to carry one unsupported claim each.",
            "",
        ]
        score = run_judge_generated(items)
        title = f"bullets the app wrote ({split} split)"
        filtered = _filter_lines(score_filter(items))
        return header + filtered + _judge_lines(score, title), score

    return run


def _fresh_extractions(
    take: Callable[[list], list], group: str = "software-fresh"
) -> Callable[[], tuple[list[str], object]]:
    """Extracts the fresh software postings so the retrieval eval can
    search with them. They carry no extraction labels of their own beyond
    the form fields, which are scored the same way."""

    def run() -> tuple[list[str], object]:
        from app.evals.public import (
            CACHE_DIR,
            expected_fields,
            load_sources,
            posting_key,
            posting_text,
            read_cached,
        )
        from app.evals.synthetic import Job

        jobs = []
        for item in load_sources()["postings"]:
            if item.get("group") != group:
                continue
            row = read_cached(item["id"])
            if row is None:
                return [f"{posting_key(item['id'])} is not cached.", ""], None
            jobs.append(Job(
                key=posting_key(item["id"]), for_personas=[], covers=[group],
                text=posting_text(row), expected=expected_fields(row),
            ))
        name = f"job extraction (LinkedIn {group} postings)"
        return _scored(
            _extraction_lines,
            run_job_extraction(take(jobs), name, save_dir=CACHE_DIR / "extractions"),
        )

    return run


def _public_jobs(
    take: Callable[[list], list], split: str
) -> Callable[[], tuple[list[str], object]]:
    def run() -> tuple[list[str], object]:
        from app.evals.public import CACHE_DIR, build_jobs
        from app.evals.real import CacheMissingError

        try:
            jobs = build_jobs(split)
        except CacheMissingError as e:
            return [f"Public postings skipped: {e}", ""], None
        name = f"job extraction (LinkedIn postings, {split} split)"
        return _scored(
            _extraction_lines,
            run_job_extraction(take(jobs), name, save_dir=CACHE_DIR / "extractions"),
        )

    return run


if __name__ == "__main__":
    main()
