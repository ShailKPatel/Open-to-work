"""Seeds the small synthetic account the CI eval gate scores against.

The eval harness (app/evals/) needs a populated account: repositories,
skill evidence, experience points, all indexed into Qdrant. Real profile
data lives under data/, which is gitignored and personal, so CI has none
and cannot be given any. This builds a stand-in instead: one account of
hand-written fake repositories and roles, indexed through the real
app/retrieval/index.py write path rather than written into Qdrant
directly, because the indexing step (what text gets embedded, which id a
point lands on) is part of what the eval measures.

Deterministic and idempotent. Every row carries an explicit primary key
rather than an autoincrement one, so ids are identical on a fresh
database and on the tenth re-run, which is what lets the committed
golden set (evals/golden/ci_fixture.yaml) reference them by number.
Re-running deletes the fixture's own rows and rebuilds them; the Qdrant
points upsert over themselves at the same ids. Embedding is a pure
function of the text and the model, so the vectors repeat too.

No network, no GitHub, no LLM calls. The embedding model runs locally.

The account id is a constant here rather than an argument, unlike
scripts/label_golden_set.py's --account: it is not a per-call input but
part of the fixture's definition, since the committed golden pairs name
it. The id sits far above anything a real local profile reaches so the
fixture cannot collide with real data if this is ever pointed at a real
database by mistake.

Usage: .venv/bin/python -m scripts.seed_eval_fixture
"""

from __future__ import annotations

import argparse
import datetime as dt

from sqlalchemy import delete

from app.core.db import (
    Account,
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    Repository,
    SkillEvidence,
    get_db,
    init_db,
)
from app.retrieval.index import (
    index_experience_points,
    index_experience_skill_evidence,
    index_skill_evidence,
)

FIXTURE_ACCOUNT_ID = 9001

# Fixed timestamp for every row this writes. Nothing here is embedded or
# scored, but a frozen clock keeps two runs of the script byte-identical
# in the database as well as in Qdrant.
_FIXED_TIME = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

# (repo_id, github_id, name, primary_language, stars, description)
_REPOSITORIES: tuple[tuple[int, int, str, str, int, str], ...] = (
    (9101, 990101, "stream-ingest", "Python", 41, "Kafka ingestion pipeline with replay."),
    (9102, 990102, "graphql-gateway", "TypeScript", 17, "Federated GraphQL gateway."),
    (9103, 990103, "tenant-operator", "Go", 63, "Kubernetes operator for tenant namespaces."),
    (9104, 990104, "feature-store", "Python", 28, "Offline feature store for ranking models."),
    (9105, 990105, "pg-tuning-kit", "Python", 9, "PostgreSQL index and vacuum analysis."),
    (9106, 990106, "infra-modules", "HCL", 22, "Reusable Terraform modules."),
)

# (evidence_id, repo_id, skill, evidence_type, weight, confidence). Only
# skill and evidence_type reach the embedding (see index.py's
# evidence_text), so these are what the golden queries actually match on.
_SKILL_EVIDENCE: tuple[tuple[int, int, str, str, float, float], ...] = (
    (9201, 9101, "Python", "readme", 1.0, 0.9),
    (9202, 9101, "Apache Kafka", "manifest", 1.0, 0.95),
    (9203, 9101, "Apache Airflow", "manifest", 0.8, 0.8),
    (9204, 9102, "TypeScript", "readme", 1.0, 0.9),
    (9205, 9102, "GraphQL", "manifest", 1.0, 0.95),
    (9206, 9102, "Node.js", "manifest", 0.9, 0.85),
    (9207, 9103, "Go", "readme", 1.0, 0.9),
    (9208, 9103, "Kubernetes", "manifest", 1.0, 0.95),
    (9209, 9103, "Docker", "manifest", 0.9, 0.9),
    (9210, 9104, "PyTorch", "manifest", 1.0, 0.9),
    (9211, 9104, "pandas", "manifest", 0.8, 0.85),
    (9212, 9105, "PostgreSQL", "readme", 1.0, 0.95),
    (9213, 9105, "SQL", "readme", 0.9, 0.9),
    (9214, 9106, "Terraform", "manifest", 1.0, 0.95),
    (9215, 9106, "AWS", "readme", 0.9, 0.85),
    # Distractors. No golden query names these, so they are the rows a
    # working ranker has to keep out of the top of the list.
    (9216, 9101, "Redis", "manifest", 0.7, 0.8),
    (9217, 9101, "RabbitMQ", "manifest", 0.6, 0.75),
    (9218, 9102, "React", "manifest", 0.8, 0.85),
    (9219, 9102, "gRPC", "manifest", 0.7, 0.8),
    (9220, 9103, "Rust", "readme", 0.6, 0.7),
    (9221, 9104, "Elasticsearch", "manifest", 0.7, 0.8),
    (9222, 9105, "MongoDB", "manifest", 0.6, 0.75),
    (9223, 9106, "Ansible", "manifest", 0.7, 0.8),
)

# (experience_id, title, company, start, end)
_EXPERIENCES: tuple[tuple[int, str, str, str, str | None], ...] = (
    (9301, "Backend Engineer", "Northwind Data", "Mar 2022", "Jun 2024"),
    (9302, "Platform Engineer", "Helix Systems", "Jul 2024", None),
    (9303, "Software Engineer", "Cobalt Labs", "Aug 2020", "Feb 2022"),
)

# (point_id, experience_id, order_index, text). Unlike skill evidence,
# a point's full text is what gets embedded.
_EXPERIENCE_POINTS: tuple[tuple[int, int, int, str], ...] = (
    (
        9401, 9301, 0,
        "Built a Kafka ingestion pipeline handling 40 million events per day "
        "with exactly-once delivery.",
    ),
    (
        9402, 9301, 1,
        "Cut p99 API latency from 850ms to 120ms with read replicas and a query cache.",
    ),
    (
        9403, 9301, 2,
        "Migrated a 400GB PostgreSQL database to partitioned tables with no downtime.",
    ),
    (
        9404, 9301, 3,
        "Designed the GraphQL schema and resolver layer serving twelve client applications.",
    ),
    (
        9405, 9301, 4,
        "Automated nightly feature extraction feeding a PyTorch recommendation model.",
    ),
    (
        9406, 9302, 0,
        "Wrote a Kubernetes operator in Go that reconciles tenant namespaces and quotas.",
    ),
    (
        9407, 9302, 1,
        "Replaced hand-rolled deploy scripts with Terraform modules covering thirty AWS accounts.",
    ),
    (
        9408, 9302, 2,
        "Reduced container image sizes by 70 percent by restructuring multi-stage Docker builds.",
    ),
    (
        9409, 9302, 3,
        "Introduced CI gates for linting, type checking and coverage across fourteen repositories.",
    ),
    (
        9410, 9302, 4,
        "Halved alerting noise by rewriting Prometheus alert rules and on-call runbooks.",
    ),
    # Distractors, as above: subject matter no golden query asks about, so
    # a top-10 window is a real cut rather than the entire collection.
    (
        9411, 9303, 0,
        "Rebuilt the mobile checkout flow and lifted completed purchases by 8 percent.",
    ),
    (
        9412, 9303, 1,
        "Localised the storefront into nine languages including right-to-left layouts.",
    ),
    (
        9413, 9303, 2,
        "Led an accessibility audit and brought every checkout screen to WCAG 2.1 AA.",
    ),
    (
        9414, 9303, 3,
        "Built the shared component library the web and marketing teams both ship from.",
    ),
    (
        9415, 9303, 4,
        "Integrated a second payment provider as a failover for card authorisation.",
    ),
    (
        9416, 9303, 5,
        "Wrote the customer support console that replaced three spreadsheets.",
    ),
    (
        9417, 9303, 6,
        "Implemented the GDPR data deletion flow and its audit trail.",
    ),
    (
        9418, 9303, 7,
        "Ran the A/B testing framework and its weekly experiment review.",
    ),
)

# (evidence_id, experience_id, skill). These share the skill_evidence
# collection with the repo-linked rows above, at offset point ids (see
# index.py's _EXPERIENCE_EVIDENCE_ID_OFFSET), which is why the golden set
# refers to them as 1_000_009_50x rather than 950x.
_EXPERIENCE_SKILL_EVIDENCE: tuple[tuple[int, int, str], ...] = (
    (9501, 9301, "Apache Kafka"),
    (9502, 9301, "PostgreSQL"),
    (9503, 9301, "GraphQL"),
    (9504, 9302, "Kubernetes"),
    (9505, 9302, "Terraform"),
    (9506, 9302, "Prometheus"),
    (9507, 9303, "React"),
    (9508, 9303, "Redis"),
)


def _clear_fixture() -> None:
    """Deletes only rows this script owns, by explicit id, so pointing it
    at a database that has real accounts in it cannot touch them."""
    db = get_db()
    try:
        db.execute(
            delete(ExperienceSkillEvidence).where(
                ExperienceSkillEvidence.id.in_([e[0] for e in _EXPERIENCE_SKILL_EVIDENCE])
            )
        )
        db.execute(
            delete(ExperiencePoint).where(
                ExperiencePoint.id.in_([p[0] for p in _EXPERIENCE_POINTS])
            )
        )
        db.execute(delete(Experience).where(Experience.id.in_([e[0] for e in _EXPERIENCES])))
        db.execute(
            delete(SkillEvidence).where(
                SkillEvidence.id.in_([e[0] for e in _SKILL_EVIDENCE])
            )
        )
        db.execute(delete(Repository).where(Repository.id.in_([r[0] for r in _REPOSITORIES])))
        db.execute(delete(Account).where(Account.id == FIXTURE_ACCOUNT_ID))
        db.commit()
    finally:
        db.close()


def _write_rows() -> None:
    db = get_db()
    try:
        db.add(
            Account(
                id=FIXTURE_ACCOUNT_ID,
                first_name="Fixture",
                last_name="Account",
                github_username="ci-fixture",
                created_at=_FIXED_TIME,
            )
        )
        for repo_id, github_id, name, language, stars, description in _REPOSITORIES:
            db.add(
                Repository(
                    id=repo_id,
                    account_id=FIXTURE_ACCOUNT_ID,
                    github_id=github_id,
                    name=name,
                    full_name=f"ci-fixture/{name}",
                    url=f"https://example.invalid/ci-fixture/{name}",
                    primary_language=language,
                    stars=stars,
                    description=description,
                    commits_authored=stars,
                    fetched_at=_FIXED_TIME,
                )
            )
        for ev_id, repo_id, skill, evidence_type, weight, confidence in _SKILL_EVIDENCE:
            db.add(
                SkillEvidence(
                    id=ev_id,
                    repo_id=repo_id,
                    skill=skill,
                    evidence_type=evidence_type,
                    weight=weight,
                    confidence=confidence,
                )
            )
        for exp_id, title, company, start, end in _EXPERIENCES:
            db.add(
                Experience(
                    id=exp_id,
                    account_id=FIXTURE_ACCOUNT_ID,
                    title=title,
                    company=company,
                    start_date=start,
                    end_date=end,
                    created_at=_FIXED_TIME,
                    updated_at=_FIXED_TIME,
                )
            )
        for point_id, exp_id, order_index, text in _EXPERIENCE_POINTS:
            db.add(
                ExperiencePoint(
                    id=point_id,
                    experience_id=exp_id,
                    text=text,
                    order_index=order_index,
                    created_at=_FIXED_TIME,
                    updated_at=_FIXED_TIME,
                )
            )
        for ev_id, exp_id, skill in _EXPERIENCE_SKILL_EVIDENCE:
            db.add(
                ExperienceSkillEvidence(
                    id=ev_id,
                    experience_id=exp_id,
                    skill=skill,
                    evidence_type="manual",
                )
            )
        db.commit()
    finally:
        db.close()


def _index() -> tuple[int, int, int]:
    """Indexes through the real write path, in the same order the app
    does, so a change to what index.py embeds moves the eval numbers
    instead of being invisible to it."""
    db = get_db()
    try:
        repo_evidence = [
            db.get(SkillEvidence, ev_id) for ev_id, *_ in _SKILL_EVIDENCE
        ]
        exp_evidence = [
            db.get(ExperienceSkillEvidence, ev_id) for ev_id, *_ in _EXPERIENCE_SKILL_EVIDENCE
        ]
        points = [db.get(ExperiencePoint, point_id) for point_id, *_ in _EXPERIENCE_POINTS]
        written_evidence = index_skill_evidence(
            [r for r in repo_evidence if r is not None], account_id=FIXTURE_ACCOUNT_ID
        )
        written_exp_evidence = index_experience_skill_evidence(
            [r for r in exp_evidence if r is not None], account_id=FIXTURE_ACCOUNT_ID
        )
        written_points = index_experience_points(
            [p for p in points if p is not None], account_id=FIXTURE_ACCOUNT_ID
        )
        return written_evidence, written_exp_evidence, written_points
    finally:
        db.close()


def main() -> None:
    argparse.ArgumentParser(
        description=(
            "Seed the synthetic account the CI eval gate scores against "
            f"(account id {FIXTURE_ACCOUNT_ID})."
        )
    ).parse_args()

    init_db()
    _clear_fixture()
    _write_rows()
    evidence, exp_evidence, points = _index()
    print(
        f"seeded account {FIXTURE_ACCOUNT_ID}: "
        f"{len(_REPOSITORIES)} repositories, {len(_EXPERIENCES)} experiences"
    )
    print(
        f"indexed: {evidence} repo skill evidence, {exp_evidence} experience skill evidence, "
        f"{points} experience points"
    )


if __name__ == "__main__":
    main()
