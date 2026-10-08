"""Golden set: hand-labeled (query, relevant evidence ids) pairs the eval
harness (app/evals/run.py) measures retrieval quality against. The
target size is 50 hand-labeled job-to-profile pairs.

Labeling needs a human judgment ("is this piece of evidence relevant to
this job posting"). scripts/label_golden_set.py is the interactive tool that walks a person
through building this file from their own real data, driven by their own
real retrieval results; this module is only the file format + load/save.

File format: one YAML file, a plain list of pair dicts (see GoldenPair's
fields below), human-readable and diffable in a PR, since this file is
meant to be committed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# The two account-scoped, evidence-shaped collections a real match
# actually retrieves from (see app/retrieval/index.py). "resumes" and
# "role_families"/"job_postings" aren't retrieval targets in the same
# sense (a resume isn't evidence FOR a job, a role family isn't evidence
# at all), so aren't scored by this harness.
SCORABLE_COLLECTIONS = ("skill_evidence", "experience_points")


@dataclass
class GoldenPair:
    """One labeled query. `query_text` is typically a job posting's
    title+skills+summary (the same shape app/retrieval/index.py's
    job_posting_text builds, though any text works). `relevant_ids` are
    the ids in `collection` a human confirmed are actually relevant
    evidence for that query. For "skill_evidence", that's the SAME id
    space app/retrieval/search.py's hits already come back in (offset ids
    for experience-linked evidence, see index.py's
    experience_evidence_point_id), not a raw SkillEvidence/
    ExperienceSkillEvidence row id.
    """

    id: str
    account_id: int
    collection: str
    query_text: str
    relevant_ids: list[int]
    job_posting_id: int | None = None
    notes: str = ""
    # True when the labeler did not reach a decision on every candidate they
    # were shown. Its own field rather than a marker inside `notes`, because
    # app/evals/run.py branches on it. Files written before it existed carry
    # no such key and inherit False.
    partial: bool = False
    labeled_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())


def load_golden_set(path: Path) -> list[GoldenPair]:
    """Missing file -> empty list, not an error: a fresh checkout with no
    golden set yet is a valid (if uninformative) state for run_eval() to
    handle, not a crash."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    return [GoldenPair(**item) for item in raw]


def save_golden_set(pairs: list[GoldenPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(p) for p in pairs]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def upsert_pair(pairs: list[GoldenPair], pair: GoldenPair) -> list[GoldenPair]:
    """Replaces an existing pair with the same id (re-labeling), else
    appends. scripts/label_golden_set.py relies on this so re-running a
    label session on the same query updates it in place instead of
    duplicating it in the committed file."""
    return [p for p in pairs if p.id != pair.id] + [pair]
