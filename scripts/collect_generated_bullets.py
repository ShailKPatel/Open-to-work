"""Collects project bullets the app's resume writer actually produces, for
labeling and for validating the groundedness judge on real output rather
than on hand-written bullets.

Each persona is paired with the postings written for it (the synthetic
set), and the composite open-source portfolio with the real postings it
has relevant evidence for (the real-text set). For every pair the app
builds a one-page resume the same way the API does, and every bullet the
model wrote is recorded before the deterministic grounding filter runs,
with whether that filter kept it. One labeled file then scores both the
filter and the LLM judge against the same bullets.

The evidence each bullet was written from is stored with it, so the file
is a frozen benchmark: later changes to personas or sources do not change
what a label refers to. Labels are kept apart, in
evals/judge/generated_labels.yaml; each bullet carries its dev or test
split (app/evals/splits.py).

Makes one quality-tier call per pair, paced as scripts/run_llm_evals.py
is, and keys come from the same place. LIVE_LLM_QUALITY_MODEL pins the
writer's model; each bullet records the model that actually answered.
Re-running skips pairs already in the output file, so a run stopped by
quota continues where it left off.

Usage:
  .venv/bin/python -m scripts.collect_generated_bullets [--real N] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from app.evals.llm_evals import QuotaExhaustedError, _paced, llm_environment, pacing_from_env
from app.evals.real import build_jobs, build_portfolio
from app.evals.splits import split_for
from app.evals.synthetic import load_jobs, load_personas, repo_id, seed
from scripts.run_llm_evals import KEYS_FILE, eval_keys

DEFAULT_OUT = Path("evals/judge/generated_bullets.yaml")

_HEADER = """\
# Project bullets written by the app's resume writer, recorded before its
# deterministic grounding filter, with the evidence the writer was shown.
# Built by scripts/collect_generated_bullets.py; see that script and
# evals/judge/LABELING.md for how `grounded` is assigned.
#
# passed_filter  whether app/resume_build/grounding.py kept the bullet
# split          dev or test, by posting (app/evals/splits.py)
# model          the model that wrote it
#
# Labels are in generated_labels.yaml, keyed by id.
"""


def _pairs(real_limit: int) -> list[tuple[Any, Any]]:
    personas = {p.key: p for p in load_personas()}
    pairs = [(personas[name], job) for job in load_jobs() for name in job.for_personas]
    portfolio = build_portfolio()
    real = [job for job in build_jobs() if job.for_personas][:real_limit]
    return pairs + [(portfolio, job) for job in real]


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(yaml.safe_load(path.read_text(encoding="utf-8")) or [])


def _save(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(records, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(_HEADER + "\n" + body, encoding="utf-8")


def _build(persona: Any, job: Any, recorded: list[dict[str, Any]]) -> None:
    """One resume build for the pair; the bullets land in `recorded`."""
    from app.core.db import Account, get_db
    from app.resume_build import orchestrator
    from app.retrieval.search import Query

    original = orchestrator._ground_projects

    def recording(projects, candidates_by_id, vocabulary):  # type: ignore[no-untyped-def]
        kept, dropped = original(projects, candidates_by_id, vocabulary)
        kept_points = {(p["repo_id"], b) for p in kept for b in p["points"]}
        for project in projects:
            candidate = candidates_by_id[project["repo_id"]]
            evidence = orchestrator.format_project_evidence(
                candidate["name"],
                candidate["description"],
                candidate["skills"],
                project.get("user_note"),
            )
            for bullet in project["points"]:
                recorded.append(
                    {
                        "repo_id": project["repo_id"],
                        "evidence": evidence,
                        "bullet": bullet,
                        "passed_filter": (project["repo_id"], bullet) in kept_points,
                    }
                )
        return kept, dropped

    orchestrator._ground_projects = recording
    db = get_db()
    try:
        account = db.get(Account, persona.account_id)
        assert account is not None, f"{persona.key} is not seeded"
        query = Query(text=job.query_text(), skills=tuple(job.skills))
        orchestrator._build_resume_data_for_text(
            db, account, persona.account_id, job.text, "onepage", retrieval_query=query
        )
    finally:
        db.close()
        orchestrator._ground_projects = original


def _last_model() -> str:
    """The model that answered the latest call. The app falls back to the
    bulk model when the quality model is unavailable, so the configured
    model is not always the one that wrote the bullets."""
    from sqlalchemy import select

    from app.core.db import LLMCall, get_db

    db = get_db()
    try:
        call = db.execute(select(LLMCall).order_by(LLMCall.id.desc()).limit(1)).scalar()
        return call.model if call is not None else ""
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", type=int, default=15, help="real-text postings to use")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    keys = eval_keys()
    if not keys:
        raise SystemExit(f"list keys one per line in {KEYS_FILE} (gitignored)")
    records = _load(args.out)
    done = {(r["posting"], r["persona"]) for r in records}
    pairs = [(p, j) for p, j in _pairs(args.real) if (j.key, p.key) not in done]
    print(f"{len(pairs)} pairs to build, {len(records)} bullets already recorded")

    pacing_from_env()
    with llm_environment(
        keys,
        bulk_model=os.environ.get("LIVE_LLM_BULK_MODEL") or None,
        quality_model=os.environ.get("LIVE_LLM_QUALITY_MODEL") or None,
    ):
        personas = {p.key: p for p, _ in _pairs(args.real)}
        seed(list(personas.values()))
        repo_keys = {repo_id(p, i): r.key for p in personas.values() for i, r in enumerate(p.repos)}
        for persona, job in pairs:
            recorded: list[dict[str, Any]] = []
            try:
                _paced(partial(_build, persona, job, recorded))
            except QuotaExhaustedError as e:
                print(f"stopped: {e}")
                break
            except Exception as e:  # noqa: BLE001 - one failed build should not end the run
                print(f"{job.key} / {persona.key}: build failed: {e}")
                continue
            model = _last_model()
            for index, item in enumerate(recorded):
                records.append(
                    {
                        "id": f"{job.key}/{persona.key}#{index}",
                        "posting": job.key,
                        "persona": persona.key,
                        "repo": repo_keys[item["repo_id"]],
                        "split": split_for(job.key),
                        "model": model,
                        "evidence": item["evidence"],
                        "bullet": item["bullet"],
                        "passed_filter": item["passed_filter"],
                    }
                )
            _save(args.out, records)
            print(f"{job.key} / {persona.key}: {len(recorded)} bullets")
    print(f"{len(records)} bullets in {args.out}")


if __name__ == "__main__":
    main()
