"""Labeled bullets the app's resume writer actually produced
(evals/judge/generated_bullets.yaml, collected by
scripts/collect_generated_bullets.py), and the two checks scored on them:

- the deterministic grounding filter (app/resume_build/grounding.py),
  whose keep or drop decision was recorded when each bullet was written,
  so scoring it needs no call at all. It is a pure function of the
  bullet, its evidence and the account's skill names, so replay_filter()
  reruns it offline for bullets it never saw (the perturbed set); on the
  collected bullets the replay reproduces every recorded decision;
- the LLM groundedness judge (app/evals/groundedness.py), which is called
  again on each bullet with the evidence stored next to it.

Labels live in their own file (evals/judge/generated_labels.yaml), keyed by
bullet id, so collecting more bullets never touches a label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml

JUDGE_DIR = Path(__file__).resolve().parents[2] / "evals" / "judge"


@dataclass
class GeneratedBullet:
    id: str
    split: str
    evidence: str
    bullet: str
    passed_filter: bool
    grounded: bool
    borderline: bool = False
    perturbation: str | None = None

    @property
    def kind(self) -> str:
        if self.perturbation:
            return f"perturbed: {self.perturbation}"
        return "borderline" if self.borderline else "clear"


def load_generated(
    split: str | None, bullets_path: Path | None = None, labels_path: Path | None = None
) -> list[GeneratedBullet]:
    """Labeled bullets in one split (None for both). Bullets not labeled
    yet are left out."""
    records = yaml.safe_load(
        (bullets_path or JUDGE_DIR / "generated_bullets.yaml").read_text(encoding="utf-8")
    ) or []
    labels = yaml.safe_load(
        (labels_path or JUDGE_DIR / "generated_labels.yaml").read_text(encoding="utf-8")
    ) or {}
    out = []
    for record in records:
        label = labels.get(record["id"])
        if label is None or (split is not None and record["split"] != split):
            continue
        out.append(
            GeneratedBullet(
                id=record["id"],
                split=record["split"],
                evidence=record["evidence"],
                bullet=record["bullet"],
                passed_filter=bool(record["passed_filter"]),
                grounded=bool(label["grounded"]),
                borderline=bool(label.get("borderline", False)),
            )
        )
    return out


def load_perturbed(path: Path | None = None) -> list[GeneratedBullet]:
    """evals/judge/perturbed_test.yaml: test-split bullets the app wrote,
    each changed to carry one unsupported claim. The filter's verdict is
    replayed, since it never ran on the changed text."""
    records = yaml.safe_load(
        (path or JUDGE_DIR / "perturbed_test.yaml").read_text(encoding="utf-8")
    ) or []
    out = []
    for record in records:
        persona = record["source"].split("/", 1)[1].split("#", 1)[0]
        item = GeneratedBullet(
            id=record["id"],
            split="test",
            evidence=record["evidence"],
            bullet=record["bullet"],
            passed_filter=replay_filter(record["bullet"], record["evidence"], persona),
            grounded=bool(record["grounded"]),
            perturbation=record["kind"],
        )
        out.append(item)
    return out


_SKILLS = re.compile(r"skills=(.*?)(?: \[User Note:|$)")


def replay_filter(bullet: str, evidence: str, persona_key: str) -> bool:
    """app/resume_build/grounding.py's verdict on one bullet, with the
    persona's skill names as the technology vocabulary, as the app builds
    it from the account's evidence."""
    from app.resume_build.grounding import is_grounded_bullet

    match = _SKILLS.search(evidence)
    skills = [s.strip() for s in match.group(1).split(",")] if match else []
    return is_grounded_bullet(bullet, evidence, skills, _vocabulary(persona_key))


_VOCABULARIES: dict[str, Any] = {}


def _vocabulary(persona_key: str) -> Any:
    from app.evals.real import build_portfolio
    from app.evals.synthetic import evidence_rows, load_personas
    from app.resume_build.grounding import TechVocabulary

    if not _VOCABULARIES:
        personas = {p.key: p for p in load_personas()}
        personas["real-portfolio"] = build_portfolio()
        for key, persona in personas.items():
            _VOCABULARIES[key] = TechVocabulary({r.skill for r in evidence_rows(persona)})
    return _VOCABULARIES[persona_key]


@dataclass
class FilterScore:
    """The deterministic filter as a classifier of ungrounded bullets:
    a dropped bullet is a positive."""

    kept_grounded: int = 0
    dropped_grounded: int = 0
    kept_ungrounded: int = 0
    dropped_ungrounded: int = 0

    @property
    def total(self) -> int:
        return (
            self.kept_grounded + self.dropped_grounded
            + self.kept_ungrounded + self.dropped_ungrounded
        )

    @property
    def ungrounded(self) -> int:
        return self.kept_ungrounded + self.dropped_ungrounded


def score_filter(items: list[GeneratedBullet]) -> FilterScore:
    score = FilterScore()
    for item in items:
        if item.grounded:
            if item.passed_filter:
                score.kept_grounded += 1
            else:
                score.dropped_grounded += 1
        elif item.passed_filter:
            score.kept_ungrounded += 1
        else:
            score.dropped_ungrounded += 1
    return score


def run_judge_generated(items: list[GeneratedBullet]) -> Any:
    """The judge on each bullet, with the evidence the writer was shown."""
    from app.evals.groundedness import judge_bullet
    from app.evals.llm_evals import JudgeScore, QuotaExhaustedError, _paced

    score = JudgeScore()
    for item in items:
        try:
            verdict = _paced(partial(judge_bullet, item.evidence, item.bullet, None))
        except QuotaExhaustedError as e:
            score.stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001
            score.errors.append(f"{item.id}: {e}")
            continue
        score.add(item.id, item.kind, item.grounded, verdict)
    return score
