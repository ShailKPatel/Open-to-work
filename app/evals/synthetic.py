"""The synthetic eval dataset: invented people, job postings, labeled
resume bullets and prompt injection postings under evals/synthetic/, plus
the code that turns them into a populated, scored, then discarded profile.

Every name, company, contact detail and number in that directory is made
up. Nothing here reads data/ or the configured database: the only way to
seed is inside isolated_environment(), which points the app at a temporary
SQLite file and an in-process Qdrant, and deletes both on the way out. A
run therefore leaves no profile behind, and it cannot write into a real
one either, because seed() refuses to run outside that context.

What the data feeds:
  - retrieval: golden_pairs() turns each posting's `for` personas into
    GoldenPairs scored by app/evals/run.py. Relevance is rule-labeled, not
    hand-labeled: a document is relevant when it carries a skill the
    posting asks for. That is a coarser judgment than a person makes, and
    it favours keyword matching, so these numbers are a diverse
    regression signal, not a claim about real-world quality.
  - extraction: each posting's `expected` fields, and expected_resume()
    for each persona against render_resume_text().
  - the groundedness judge: judge_bullets.yaml, human labels per bullet.
  - prompt injection: redteam.yaml.

Ids are derived from the persona's account id (account_id * 1000 plus a
per-table offset), so they are stable across runs and never collide with
the CI fixture (scripts/seed_eval_fixture.py, account 9001).
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import signal
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.evals.golden import SCORABLE_COLLECTIONS, GoldenPair

SYNTHETIC_DIR = Path(__file__).resolve().parents[2] / "evals" / "synthetic"

# Synthetic accounts live in this range and nowhere else.
ACCOUNT_ID_MIN = 9101
ACCOUNT_ID_MAX = 9199

# Per-table offsets inside a persona's id block (account_id * 1000).
_REPO_OFFSET = 1
_EVIDENCE_OFFSET = 100
_EXPERIENCE_OFFSET = 300
_POINT_OFFSET = 400
_EXPERIENCE_EVIDENCE_OFFSET = 600
_EDUCATION_OFFSET = 900
_BLOCK_LIMITS = {
    "repos": _EVIDENCE_OFFSET - _REPO_OFFSET,
    "evidence": _EXPERIENCE_OFFSET - _EVIDENCE_OFFSET,
    "roles": _POINT_OFFSET - _EXPERIENCE_OFFSET,
    "points": _EXPERIENCE_EVIDENCE_OFFSET - _POINT_OFFSET,
    "role_evidence": _EDUCATION_OFFSET - _EXPERIENCE_EVIDENCE_OFFSET,
}

_FIXED_TIME = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
_LABELED_AT = "2026-10-08T00:00:00+00:00"

_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


@dataclass
class Point:
    text: str
    skills: list[str]


@dataclass
class Role:
    key: str
    company: str
    title: str
    location: str
    start: str | None
    end: str | None
    skills: list[str]
    points: list[Point]


@dataclass
class Repo:
    key: str
    name: str
    language: str | None
    stars: int
    description: str
    readme: str | None
    manifests: dict[str, list[str]]
    readme_skills: list[str]
    profile_readme: bool = False


@dataclass
class EducationEntry:
    institution: str
    degree: str
    location: str
    start: str | None
    end: str | None
    grade: str
    details: list[str]


@dataclass
class Link:
    platform: str
    url: str
    label: str | None = None


@dataclass
class Persona:
    key: str
    account_id: int
    first_name: str
    last_name: str
    github_username: str
    headline: str
    location: str
    email: str
    phone: str
    links: list[Link]
    resume_style: str
    date_style: str
    covers: list[str]
    repos: list[Repo]
    roles: list[Role]
    education: list[EducationEntry]

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class Job:
    key: str
    for_personas: list[str]
    covers: list[str]
    text: str
    expected: dict[str, Any]
    # Skills whose evidence counts as relevant to this posting, when that
    # is a judgment wider than the skills it names (the real-text set,
    # app/evals/real.py). None falls back to the named skills.
    relevant_skills: list[str] | None = None

    @property
    def skills(self) -> list[str]:
        return list(self.expected.get("skills") or [])

    @property
    def match_skills(self) -> list[str]:
        return self.skills if self.relevant_skills is None else list(self.relevant_skills)

    def query_text(self) -> str:
        """The posting as retrieval sees it, the same shape
        app/retrieval/index.py's job_posting_text builds from an extracted
        posting: title at company, summary, skills."""
        parts = [f"{self.expected['title']} at {self.expected['company']}".strip()]
        if self.expected.get("summary"):
            parts.append(self.expected["summary"])
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills))
        return "\n".join(parts)


@dataclass
class JudgeBullet:
    persona: str
    repo: str
    bullet: str
    grounded: bool
    kind: str


@dataclass
class RedTeamPosting:
    key: str
    injection: bool
    technique: str
    text: str
    expected: dict[str, Any]
    must_not_contain: list[str] = field(default_factory=list)


@dataclass
class EvidenceRow:
    """One skill claim as seeded: `row_id` is the table's primary key and
    `point_id` the id it is indexed under in the skill_evidence
    collection, which differs for role evidence (see
    app/retrieval/index.py's experience_evidence_point_id)."""

    row_id: int
    point_id: int
    skill: str
    evidence_type: str
    repo_key: str | None = None
    role_key: str | None = None


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _persona_from_dict(raw: dict[str, Any]) -> Persona:
    return Persona(
        key=raw["key"],
        account_id=int(raw["account_id"]),
        first_name=raw["first_name"],
        last_name=raw["last_name"],
        github_username=raw["github_username"],
        headline=raw["headline"],
        location=raw["location"],
        email=raw["email"],
        phone=str(raw["phone"]),
        links=[Link(**link) for link in raw.get("links") or []],
        resume_style=raw["resume_style"],
        date_style=raw["date_style"],
        covers=list(raw.get("covers") or []),
        repos=[
            Repo(
                key=r["key"],
                name=r["name"],
                language=r.get("language"),
                stars=int(r.get("stars") or 0),
                description=r["description"],
                readme=r.get("readme"),
                manifests={k: list(v) for k, v in (r.get("manifests") or {}).items()},
                readme_skills=list(r.get("readme_skills") or []),
                profile_readme=bool(r.get("profile_readme", False)),
            )
            for r in raw.get("repos") or []
        ],
        roles=[
            Role(
                key=r["key"],
                company=r["company"],
                title=r["title"],
                location=r.get("location") or "",
                start=_date_or_none(r.get("start")),
                end=_date_or_none(r.get("end")),
                skills=list(r.get("skills") or []),
                points=[
                    Point(text=p["text"], skills=list(p.get("skills") or []))
                    for p in r["points"]
                ],
            )
            for r in raw.get("roles") or []
        ],
        education=[
            EducationEntry(
                institution=e["institution"],
                degree=e["degree"],
                location=e.get("location") or "",
                start=_date_or_none(e.get("start")),
                end=_date_or_none(e.get("end")),
                grade=e.get("grade") or "",
                details=list(e.get("details") or []),
            )
            for e in raw.get("education") or []
        ],
    )


def _date_or_none(value: object) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value).strip()


def load_personas(directory: Path | None = None) -> list[Persona]:
    """Every persona file, ordered by account id."""
    directory = directory or SYNTHETIC_DIR / "personas"
    personas = [_persona_from_dict(_read_yaml(p)) for p in sorted(directory.glob("*.yaml"))]
    return sorted(personas, key=lambda p: p.account_id)


def load_jobs(path: Path | None = None) -> list[Job]:
    raw = _read_yaml(path or SYNTHETIC_DIR / "jobs.yaml") or []
    return [
        Job(
            key=item["key"],
            for_personas=list(item.get("for") or []),
            covers=list(item.get("covers") or []),
            text=item["text"],
            expected=dict(item["expected"]),
        )
        for item in raw
    ]


def load_judge_bullets(path: Path | None = None) -> list[JudgeBullet]:
    raw = _read_yaml(path or SYNTHETIC_DIR / "judge_bullets.yaml") or []
    return [JudgeBullet(**item) for item in raw]


def load_redteam(path: Path | None = None) -> list[RedTeamPosting]:
    raw = _read_yaml(path or SYNTHETIC_DIR / "redteam.yaml") or []
    return [RedTeamPosting(**item) for item in raw]


def _block(persona: Persona) -> int:
    return persona.account_id * 1000


def repo_id(persona: Persona, index: int) -> int:
    return _block(persona) + _REPO_OFFSET + index


def experience_id(persona: Persona, index: int) -> int:
    return _block(persona) + _EXPERIENCE_OFFSET + index


def point_ids(persona: Persona) -> list[tuple[int, Role, Point]]:
    """(ExperiencePoint id, its role, the point) in file order."""
    out = []
    n = 0
    for role in persona.roles:
        for point in role.points:
            out.append((_block(persona) + _POINT_OFFSET + n, role, point))
            n += 1
    return out


def _wrapped_manifests(repo: Repo) -> dict[str, dict[str, Any]]:
    """The manifests in the shape app/ingest/github/sync.py stores."""
    from app.ingest.github.manifests import MANIFEST_FILENAMES

    return {
        filename: {"ecosystem": MANIFEST_FILENAMES.get(filename, ""), "dependencies": deps}
        for filename, deps in repo.manifests.items()
    }


def repo_evidence(repo: Repo) -> list[tuple[str, str, list[str]]]:
    """(skill, evidence_type, source_files) for one repo: declared
    dependencies resolved through the real manifest map
    (app/profile/manifest_skills.py), then the README or description
    skills that map did not already produce."""
    from app.profile.manifest_skills import skills_from_manifests

    out: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for claim in skills_from_manifests(_wrapped_manifests(repo)):
        out.append((claim.skill, "declared_dependency", list(claim.source_files)))
        seen.add(claim.skill.casefold())
    prose_type = "readme_described" if repo.readme else "description_described"
    for skill in repo.readme_skills:
        if skill.casefold() not in seen:
            out.append((skill, prose_type, []))
            seen.add(skill.casefold())
    return out


def evidence_rows(persona: Persona) -> list[EvidenceRow]:
    """Every skill claim the persona is seeded with, repo evidence first,
    then role evidence, with the ids they are stored and indexed under."""
    from app.retrieval.index import experience_evidence_point_id

    rows: list[EvidenceRow] = []
    n = 0
    for repo in persona.repos:
        for skill, evidence_type, _ in repo_evidence(repo):
            row_id = _block(persona) + _EVIDENCE_OFFSET + n
            rows.append(EvidenceRow(row_id, row_id, skill, evidence_type, repo_key=repo.key))
            n += 1
    m = 0
    for role in persona.roles:
        for skill in role.skills:
            row_id = _block(persona) + _EXPERIENCE_EVIDENCE_OFFSET + m
            rows.append(
                EvidenceRow(
                    row_id, experience_evidence_point_id(row_id), skill, "resume",
                    role_key=role.key,
                )
            )
            m += 1
    return rows


def block_usage(persona: Persona) -> dict[str, int]:
    """How many ids each table uses in the persona's block, checked
    against _BLOCK_LIMITS by the tests so a growing persona cannot spill
    into the next table's range."""
    rows = evidence_rows(persona)
    return {
        "repos": len(persona.repos),
        "evidence": sum(1 for r in rows if r.repo_key is not None),
        "roles": len(persona.roles),
        "points": len(point_ids(persona)),
        "role_evidence": sum(1 for r in rows if r.role_key is not None),
    }


def block_limits() -> dict[str, int]:
    return dict(_BLOCK_LIMITS)


def golden_pairs(personas: list[Persona], jobs: list[Job]) -> list[GoldenPair]:
    """One pair per (posting, persona it is for, collection) that has at
    least one relevant document. Relevant: skill evidence whose skill is
    one of the posting's match_skills, or an experience point tagged with
    one of them, compared case-insensitively."""
    by_key = {p.key: p for p in personas}
    pairs: list[GoldenPair] = []
    for job in jobs:
        wanted = {s.casefold() for s in job.match_skills}
        for persona_key in job.for_personas:
            persona = by_key[persona_key]
            relevant_by_collection = {
                "skill_evidence": sorted(
                    r.point_id for r in evidence_rows(persona) if r.skill.casefold() in wanted
                ),
                "experience_points": sorted(
                    pid
                    for pid, _, point in point_ids(persona)
                    if any(s.casefold() in wanted for s in point.skills)
                ),
            }
            for collection in SCORABLE_COLLECTIONS:
                relevant = relevant_by_collection[collection]
                if not relevant:
                    continue
                pairs.append(
                    GoldenPair(
                        id=f"{job.key}@{persona.key}-{collection}",
                        account_id=persona.account_id,
                        collection=collection,
                        query_text=job.query_text(),
                        relevant_ids=relevant,
                        notes=(
                            "Synthetic, rule-labeled: documents naming a skill "
                            "the posting asks for."
                        ),
                        labeled_at=_LABELED_AT,
                    )
                )
    return pairs


def _format_date(value: str | None, style: str) -> str:
    """A stored "mar 2021" or "2021" as the persona's resume prints it."""
    if value is None:
        return "Present"
    parts = value.split()
    if len(parts) == 1:
        return parts[0]
    month, year = parts[0].lower(), parts[1]
    if style == "numeric":
        return f"{_MONTHS.index(month) + 1:02d}/{year}"
    if style == "year_only":
        return year
    return f"{month.title()} {year}"


def _date_range(start: str | None, end: str | None, style: str) -> str:
    return f"{_format_date(start, style)} - {_format_date(end, style)}"


def resume_skills(persona: Persona) -> list[str]:
    """Every skill the persona's resume shows, first spelling kept."""
    out: list[str] = []
    seen: set[str] = set()
    for skill in [s for r in persona.roles for s in r.skills] + [
        s for repo in persona.repos for s, _, _ in repo_evidence(repo)
    ]:
        if skill.casefold() not in seen:
            out.append(skill)
            seen.add(skill.casefold())
    return out


def render_resume_text(persona: Persona) -> str:
    """The persona as a plain text resume, laid out per its resume_style
    (classic, skills_first, compact) with dates in its date_style, so
    resume extraction is tested against more than one layout."""
    style = persona.date_style
    header = [
        persona.full_name,
        persona.headline,
        " | ".join([persona.location, persona.email, persona.phone]),
        " | ".join(link.url.removeprefix("https://") for link in persona.links),
    ]
    skills = ["SKILLS", ", ".join(resume_skills(persona))]

    experience = ["EXPERIENCE"]
    for role in persona.roles:
        if persona.resume_style == "compact":
            experience.append(
                f"{role.title}, {role.company}, {role.location} "
                f"({_date_range(role.start, role.end, style)})"
            )
        else:
            experience.append(f"{role.company} - {role.location}")
            experience.append(f"{role.title}    {_date_range(role.start, role.end, style)}")
        experience.extend(f"- {p.text}" for p in role.points)
        experience.append("")

    projects = ["PROJECTS"]
    for repo in persona.repos:
        if repo.profile_readme:
            continue
        projects.append(f"{repo.name}: {repo.description}")

    education = ["EDUCATION"]
    for entry in persona.education:
        line = f"{entry.institution}, {entry.degree}"
        if entry.location:
            line += f", {entry.location}"
        education.append(line)
        education.append(_date_range(entry.start, entry.end, style))
        if entry.grade:
            education.append(entry.grade)
        education.extend(entry.details)

    if persona.resume_style == "skills_first":
        sections = [header, skills, education, experience, projects]
    else:
        sections = [header, experience, projects, education, skills]
    return "\n\n".join("\n".join(lines).strip() for lines in sections) + "\n"


def expected_resume(persona: Persona) -> dict[str, Any]:
    """What app/profile/resume_extract.py should read from
    render_resume_text(persona), in that module's stored date format."""
    return {
        "contact": {
            "name": persona.full_name,
            "location": persona.location,
            "emails": [persona.email],
            "phones": [persona.phone],
            "links": [link.url for link in persona.links],
        },
        "experiences": [
            {"company": r.company, "title": r.title, "start_date": r.start, "end_date": r.end}
            for r in persona.roles
        ],
        "education": [
            {
                "institution": e.institution,
                "degree": e.degree,
                "start_date": e.start,
                "end_date": e.end,
                "grade": e.grade,
            }
            for e in persona.education
        ],
        "tags": resume_skills(persona),
    }


_isolated_root: Path | None = None

_ISOLATED_ENV_KEYS = (
    "DATABASE_URL",
    "QDRANT_URL",
    "EVALS_GOLDEN_DIR",
    "EVALS_RESULTS_DIR",
    "APP_SECRET_KEY",
)


def _reset_caches() -> None:
    import app.core.crypto as crypto_module
    import app.core.db as db_module
    from app.core.settings import get_settings
    from app.retrieval.vectorstore import get_client

    db_module.reset_engine()
    get_settings.cache_clear()
    get_client.cache_clear()
    crypto_module._fernet = None


@contextmanager
def isolated_environment() -> Iterator[Path]:
    """Points the app at a fresh temporary SQLite database, an in-process
    Qdrant and a throwaway encryption key for the duration of the block,
    then deletes them and restores the previous settings. The configured
    database and Qdrant server are never opened, so neither a real profile
    nor its vectors can be read or written from inside.

    A SIGTERM during the block (a long eval run being stopped) is turned
    into a normal exit, so the temporary directory is still removed."""
    global _isolated_root
    if _isolated_root is not None:
        raise RuntimeError("isolated_environment() is already active")
    root = Path(tempfile.mkdtemp(prefix="open-to-work-synthetic-"))
    saved = {key: os.environ.get(key) for key in _ISOLATED_ENV_KEYS}
    previous_handler = _exit_on_sigterm()
    try:
        (root / "golden").mkdir()
        (root / "results").mkdir()
        os.environ["DATABASE_URL"] = f"sqlite:///{root / 'synthetic.db'}"
        os.environ["QDRANT_URL"] = ":memory:"
        os.environ["EVALS_GOLDEN_DIR"] = str(root / "golden")
        os.environ["EVALS_RESULTS_DIR"] = str(root / "results")
        # A throwaway encryption key, so an API key stored for an LLM eval
        # (app/evals/llm_evals.py) never reads or creates data/.secret_key.
        from cryptography.fernet import Fernet

        os.environ["APP_SECRET_KEY"] = Fernet.generate_key().decode("ascii")
        _reset_caches()
        _isolated_root = root
        yield root
    finally:
        _isolated_root = None
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_caches()
        shutil.rmtree(root, ignore_errors=True)
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)


def _exit_on_sigterm() -> Any:
    """Installs a SIGTERM handler that raises SystemExit, returning the
    handler it replaced, or None off the main thread, where Python does not
    allow installing one."""

    def _raise(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    try:
        return signal.signal(signal.SIGTERM, _raise)
    except ValueError:
        return None


def _require_isolation() -> Path:
    """The guard every write goes through: the active isolated root, after
    checking the database the app would actually open lives inside it."""
    from app.core.settings import get_settings

    root = _isolated_root
    if root is None:
        raise RuntimeError(
            "synthetic data is only seeded inside isolated_environment(), "
            "never into the configured database"
        )
    if str(root) not in get_settings().database_url:
        raise RuntimeError("database_url points outside the isolated directory")
    return root


def seed(personas: list[Persona]) -> dict[str, int]:
    """Writes every persona's rows and indexes them through the real
    app/retrieval/index.py write path. Only inside isolated_environment()."""
    _require_isolation()
    from app.core.db import (
        Account,
        Education,
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

    init_db()
    counts = {"accounts": 0, "evidence": 0, "experience_evidence": 0, "points": 0}
    db = get_db()
    try:
        for persona in personas:
            db.add(
                Account(
                    id=persona.account_id,
                    first_name=persona.first_name,
                    last_name=persona.last_name,
                    github_username=persona.github_username,
                    contact_email=persona.email,
                    contact_phone=persona.phone,
                    contact_location=persona.location,
                    created_at=_FIXED_TIME,
                )
            )
            repo_ids = {}
            for i, repo in enumerate(persona.repos):
                repo_ids[repo.key] = repo_id(persona, i)
                db.add(
                    Repository(
                        id=repo_ids[repo.key],
                        account_id=persona.account_id,
                        github_id=repo_ids[repo.key],
                        name=repo.name,
                        full_name=f"{persona.github_username}/{repo.name}",
                        url=f"https://example.invalid/{persona.github_username}/{repo.name}",
                        primary_language=repo.language,
                        stars=repo.stars,
                        readme=repo.readme,
                        description=repo.description,
                        manifests_json=_wrapped_manifests(repo),
                        commits_authored=repo.stars,
                        is_profile_readme=repo.profile_readme,
                        skill_extraction_status="extracted",
                        skills_extracted_at=_FIXED_TIME,
                        fetched_at=_FIXED_TIME,
                    )
                )
            role_ids = {}
            for i, role in enumerate(persona.roles):
                role_ids[role.key] = experience_id(persona, i)
                db.add(
                    Experience(
                        id=role_ids[role.key],
                        account_id=persona.account_id,
                        title=role.title,
                        company=role.company,
                        location=role.location or None,
                        start_date=role.start,
                        end_date=role.end,
                        created_at=_FIXED_TIME,
                        updated_at=_FIXED_TIME,
                    )
                )
            for i, entry in enumerate(persona.education):
                db.add(
                    Education(
                        id=_block(persona) + _EDUCATION_OFFSET + i,
                        account_id=persona.account_id,
                        institution=entry.institution,
                        degree=entry.degree,
                        location=entry.location or None,
                        start_date=entry.start,
                        end_date=entry.end,
                        grade=entry.grade or None,
                        details=entry.details,
                        created_at=_FIXED_TIME,
                        updated_at=_FIXED_TIME,
                    )
                )
            db.flush()
            sources = {
                (repo.key, skill): files
                for repo in persona.repos
                for skill, _, files in repo_evidence(repo)
            }
            for row in evidence_rows(persona):
                if row.repo_key is not None:
                    db.add(
                        SkillEvidence(
                            id=row.row_id,
                            repo_id=repo_ids[row.repo_key],
                            skill=row.skill,
                            evidence_type=row.evidence_type,
                            weight=1.0 if row.evidence_type == "declared_dependency" else 0.8,
                            confidence=1.0 if row.evidence_type == "declared_dependency" else 0.85,
                            source_files_json=sources[(row.repo_key, row.skill)],
                        )
                    )
                elif row.role_key is not None:
                    db.add(
                        ExperienceSkillEvidence(
                            id=row.row_id,
                            experience_id=role_ids[row.role_key],
                            skill=row.skill,
                            evidence_type=row.evidence_type,
                        )
                    )
            for pid, role, point in point_ids(persona):
                db.add(
                    ExperiencePoint(
                        id=pid,
                        experience_id=role_ids[role.key],
                        text=point.text,
                        order_index=role.points.index(point),
                        created_at=_FIXED_TIME,
                        updated_at=_FIXED_TIME,
                    )
                )
        db.commit()

        for persona in personas:
            rows = evidence_rows(persona)
            repo_rows = [db.get(SkillEvidence, r.row_id) for r in rows if r.repo_key]
            role_rows = [db.get(ExperienceSkillEvidence, r.row_id) for r in rows if r.role_key]
            point_rows = [db.get(ExperiencePoint, pid) for pid, _, _ in point_ids(persona)]
            counts["accounts"] += 1
            counts["evidence"] += index_skill_evidence(
                [r for r in repo_rows if r is not None], account_id=persona.account_id
            )
            counts["experience_evidence"] += index_experience_skill_evidence(
                [r for r in role_rows if r is not None], account_id=persona.account_id
            )
            counts["points"] += index_experience_points(
                [p for p in point_rows if p is not None], account_id=persona.account_id
            )
    finally:
        db.close()
    return counts


def run_retrieval_eval(
    personas: list[Persona] | None = None, jobs: list[Job] | None = None
) -> list[Any]:
    """Seeds every persona into a throwaway environment, scores each one
    with app/evals/run.py against the derived golden pairs, and returns the
    MetricsReports. Nothing persists: the database, vectors and golden file
    are deleted before this returns. No LLM calls (groundedness is
    skipped; judge_bullets.yaml is that judge's own eval)."""
    from app.evals.golden import save_golden_set
    from app.evals.run import run_eval

    personas = personas if personas is not None else load_personas()
    jobs = jobs if jobs is not None else load_jobs()
    reports = []
    with isolated_environment() as root:
        seed(personas)
        golden_path = root / "golden" / "golden_set.yaml"
        save_golden_set(golden_pairs(personas, jobs), golden_path)
        for persona in personas:
            reports.append(
                run_eval(persona.account_id, golden_path=golden_path, include_groundedness=False)
            )
    return reports
