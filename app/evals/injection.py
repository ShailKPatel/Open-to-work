"""Scores the prompt injection detector (app/profile/injection.py).

Three numbers, kept apart because they answer different questions:
  - development set (evals/synthetic/redteam.yaml): the attacks the rules
    were written alongside. A high rate here is expected and says little.
  - held-out set (evals/synthetic/redteam_holdout.yaml): attacks written
    after the rules were fixed and never tuned against. This is the
    detection rate to quote.
  - false positives on ordinary postings: every posting in the synthetic
    and real-text eval sets, none of which contains an attack, plus the
    benign controls in both red team files.

No LLM calls. Whether the model actually obeys an attack is a separate
question, scored by app/evals/llm_evals.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.evals.synthetic import SYNTHETIC_DIR, RedTeamPosting, load_jobs, load_redteam
from app.profile.injection import detect_injection


@dataclass
class DetectionScore:
    name: str
    attacks: int
    detected: int
    benign: int
    false_positives: int
    missed: list[str] = field(default_factory=list)
    flagged_benign: list[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float | None:
        return self.detected / self.attacks if self.attacks else None

    @property
    def false_positive_rate(self) -> float | None:
        return self.false_positives / self.benign if self.benign else None


def score_postings(name: str, postings: list[RedTeamPosting]) -> DetectionScore:
    score = DetectionScore(name=name, attacks=0, detected=0, benign=0, false_positives=0)
    for posting in postings:
        flagged = bool(detect_injection(posting.text))
        if posting.injection:
            score.attacks += 1
            if flagged:
                score.detected += 1
            else:
                score.missed.append(posting.key)
        else:
            score.benign += 1
            if flagged:
                score.false_positives += 1
                score.flagged_benign.append(posting.key)
    return score


def score_ordinary(texts: dict[str, str]) -> DetectionScore:
    """Postings known to hold no attack: any detection is a false positive."""
    score = DetectionScore(name="ordinary postings", attacks=0, detected=0, benign=0,
                           false_positives=0)
    for key, text in sorted(texts.items()):
        score.benign += 1
        if detect_injection(text):
            score.false_positives += 1
            score.flagged_benign.append(key)
    return score


def ordinary_postings(include_real: bool = True) -> dict[str, str]:
    """The synthetic postings, and the real-text ones when their cache has
    been downloaded (app/evals/real.py)."""
    texts = {f"synthetic/{job.key}": job.text for job in load_jobs()}
    if include_real:
        from app.evals.real import CacheMissingError, build_jobs

        try:
            texts.update({f"real/{job.key}": job.text for job in build_jobs()})
        except CacheMissingError:
            pass
    return texts


def run_detection_eval(
    development: Path | None = None, holdout: Path | None = None, include_real: bool = True
) -> list[DetectionScore]:
    return [
        score_postings("development set", load_redteam(development)),
        score_postings(
            "held-out set", load_redteam(holdout or SYNTHETIC_DIR / "redteam_holdout.yaml")
        ),
        score_ordinary(ordinary_postings(include_real)),
    ]
