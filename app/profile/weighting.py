"""Evidence weighting: pure, deterministic, no LLM involved. Weighted
across authored-vs-forked, commit recency, commit volume, and
declared-vs-described. The LLM only decides *what* a skill is; how much it
counts is decided here, in code, so it's auditable and doesn't drift call
to call.
"""

from __future__ import annotations

import datetime as dt
import math

from app.profile.claims import EvidenceType

_BASE_WEIGHT: dict[EvidenceType, float] = {
    # Certain to exist (parsed from a real manifest), but a declared
    # dependency alone is weak evidence of skill; it could be transitive,
    # boilerplate, or barely touched.
    "declared_dependency": 0.5,
    # Uncertain to exist (LLM-inferred from prose), but the author chose to
    # call it out explicitly, a stronger signal of intentional skill use,
    # scaled by the LLM's own stated confidence.
    "readme_described": 0.6,
    # Same LLM-inferred uncertainty as readme_described, but from a much
    # thinner source (GitHub's one-line "About" text, not a full README),
    # so the base is lower even before confidence scaling.
    "description_described": 0.4,
}

_FORK_MULTIPLIER = 0.3
_RECENCY_FULL_WEIGHT_MONTHS = 6
_RECENCY_DECAY_MONTHS = 36
_RECENCY_FLOOR = 0.3
_RECENCY_UNKNOWN = 0.5  # last_commit_at is None: stats not yet available
_VOLUME_SATURATION_COMMITS = 50
_VOLUME_FLOOR = 0.3


def _recency_multiplier(last_commit_at: dt.datetime | None, now: dt.datetime) -> float:
    if last_commit_at is None:
        return _RECENCY_UNKNOWN
    if last_commit_at.tzinfo is None:
        last_commit_at = last_commit_at.replace(tzinfo=dt.UTC)
    months_ago = (now - last_commit_at).days / 30.0
    if months_ago <= _RECENCY_FULL_WEIGHT_MONTHS:
        return 1.0
    decay_progress = (months_ago - _RECENCY_FULL_WEIGHT_MONTHS) / _RECENCY_DECAY_MONTHS
    return max(_RECENCY_FLOOR, 1.0 - decay_progress)


def _volume_multiplier(commits_authored: int) -> float:
    if commits_authored <= 0:
        return _VOLUME_FLOOR
    progress = math.log1p(commits_authored) / math.log1p(_VOLUME_SATURATION_COMMITS)
    return min(1.0, _VOLUME_FLOOR + (1.0 - _VOLUME_FLOOR) * progress)


def compute_weight(
    evidence_type: EvidenceType,
    confidence: float,
    *,
    is_fork: bool,
    last_commit_at: dt.datetime | None,
    commits_authored: int,
    now: dt.datetime | None = None,
) -> float:
    now = now or dt.datetime.now(dt.UTC)
    base = _BASE_WEIGHT[evidence_type] * max(0.0, min(1.0, confidence))
    fork_mult = _FORK_MULTIPLIER if is_fork else 1.0
    recency_mult = _recency_multiplier(last_commit_at, now)
    volume_mult = _volume_multiplier(commits_authored)
    return max(0.0, min(1.0, base * fork_mult * recency_mult * volume_mult))
