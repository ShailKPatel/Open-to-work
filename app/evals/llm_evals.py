"""Evals of the LLM-backed steps, each against labels written by hand:

  - job extraction (app/profile/job_extract.py): every synthetic and
    real-text posting's `expected` fields, scored field by field.
  - resume extraction (app/profile/resume_extract.py): each synthetic
    persona's resume, rendered to PDF, against expected_resume().
  - the groundedness judge (app/evals/groundedness.py): its verdicts on
    evals/synthetic/judge_bullets.yaml against the human labels, as
    accuracy and Cohen's kappa, broken down by how each bullet was made.
  - injection robustness: extraction on the red team postings, checking
    the model reads the real fields and does not do what the attack asks.
    This scores the defence that matters (the quarantined user-role
    document), where app/evals/injection.py scores the detector.

These make real, billed calls through the app's own LLM client, so they
are opt-in: scripts/run_llm_evals.py, never `make test` or CI. Everything
runs inside isolated_environment(): a throwaway database with only the key
passed in, so the configured profile, its keys and its spend records are
never touched.

Field matching is deliberately forgiving about form and strict about
content: case, dashes and spacing are normalised, salaries and years are
compared by their numbers, and a location matches when every expected
place word appears. A field labeled null is ambiguous in its posting and
is not scored.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DASHES = re.compile("[\u2010-\u2015\u2212]")


@dataclass
class Pacing:
    """How fast the evals call the provider. Free tiers allow a few
    requests a minute, so `min_interval` spaces calls out, and a call
    refused for rate or quota waits `retry_wait` seconds and tries again,
    up to `retries` times, before it counts as a failure."""

    min_interval: float = 0.0
    retries: int = 0
    retry_wait: float = 65.0


PACING = Pacing()
_last_call = 0.0

# Waiting longer than this for a key to come back is not worth it inside one
# run: a free tier's daily allowance resets hours later.
_MAX_USEFUL_WAIT = dt.timedelta(minutes=10)


class QuotaExhaustedError(Exception):
    """Every key is refused until well past the run's patience, typically a
    free tier's daily request limit. The run stops and reports what it has."""


def _keys_back_at() -> dt.datetime | None:
    """The earliest time any stored key comes off cooldown, or None when
    one is usable now. Read from the throwaway database the run uses."""
    from sqlalchemy import select

    from app.core.db import ApiKey, get_db

    db = get_db()
    try:
        retry_times = [
            r.retry_at for r in db.execute(select(ApiKey).where(ApiKey.enabled.is_(True))).scalars()
        ]
    finally:
        db.close()
    if not retry_times or any(t is None for t in retry_times):
        return None
    times = [t if t.tzinfo else t.replace(tzinfo=dt.UTC) for t in retry_times if t is not None]
    return min(times)


def _paced[T](call: Callable[[], T]) -> T:
    """Runs one provider call under PACING."""
    global _last_call
    from app.core.llm import LLMRateLimitedError, is_out_of_keys

    attempt = 0
    while True:
        wait = PACING.min_interval - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            return call()
        except Exception as e:
            limited = isinstance(e, LLMRateLimitedError) or is_out_of_keys(e)
            if not limited or attempt >= PACING.retries:
                raise
            back_at = _keys_back_at()
            if back_at is not None and back_at - dt.datetime.now(dt.UTC) > _MAX_USEFUL_WAIT:
                raise QuotaExhaustedError(
                    f"every key is out until {back_at.isoformat(timespec='minutes')}"
                ) from e
            attempt += 1
            time.sleep(PACING.retry_wait)
_WORD = re.compile(r"[a-z0-9]+")


def _norm(value: Any) -> str:
    text = _DASHES.sub("-", str(value or "")).casefold()
    return " ".join(text.split()).strip(" .,;")


def _numbers(value: Any) -> list[str]:
    """The numbers in a salary or experience string, commas dropped and
    sorted, so "$120,000-$150,000" and "120000 to 150000" compare equal."""
    found = re.findall(r"\d+(?:[.,]\d+)*", str(value or ""))
    return sorted(n.replace(",", "") for n in found)


def field_matches(name: str, expected: Any, predicted: Any) -> bool:
    """Whether one extracted field agrees with its label."""
    if name in ("salary_range", "experience_required"):
        return _numbers(expected) == _numbers(predicted)
    if name == "location":
        want = set(_WORD.findall(_norm(expected)))
        got = set(_WORD.findall(_norm(predicted)))
        return want <= got if want else not got
    if name in ("company", "title"):
        e, p = _norm(expected), _norm(predicted)
        return e == p or (bool(e) and bool(p) and (e in p or p in e))
    return _norm(expected) == _norm(predicted)


def skill_recall(expected: list[str], predicted: list[str]) -> float | None:
    if not expected:
        return None
    got = {_norm(s) for s in predicted}
    return sum(1 for s in expected if _norm(s) in got) / len(expected)


@dataclass
class FieldTally:
    correct: int = 0
    scored: int = 0

    def add(self, ok: bool) -> None:
        self.scored += 1
        self.correct += int(ok)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.scored if self.scored else None


@dataclass
class ExtractionScore:
    name: str
    items: int = 0
    failures: list[str] = field(default_factory=list)
    fields: dict[str, FieldTally] = field(default_factory=dict)
    skill_recalls: list[float] = field(default_factory=list)
    # Expected no salary, extracted one: the "never invent a number" rule.
    invented_salary: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    # Set when the run stopped early on quota; the counts above are partial.
    stopped: str | None = None

    def tally(self, name: str) -> FieldTally:
        return self.fields.setdefault(name, FieldTally())

    @property
    def mean_skill_recall(self) -> float | None:
        return sum(self.skill_recalls) / len(self.skill_recalls) if self.skill_recalls else None


_JOB_FIELDS = (
    "company", "title", "location", "salary_range", "employment_type",
    "work_mode", "seniority", "experience_required",
)


def score_job_extraction(
    key: str, expected: dict[str, Any], predicted: dict[str, Any], score: ExtractionScore
) -> None:
    """Adds one posting's comparison to `score`."""
    score.items += 1
    for name in _JOB_FIELDS:
        if expected.get(name) is None:
            continue
        ok = field_matches(name, expected[name], predicted.get(name, ""))
        score.tally(name).add(ok)
        if not ok:
            score.mismatches.append(
                f"{key}.{name}: expected {expected[name]!r}, got {predicted.get(name, '')!r}"
            )
    if expected.get("salary_range") == "" and _numbers(predicted.get("salary_range")):
        score.invented_salary.append(key)
    recall = skill_recall(
        list(expected.get("skills") or []),
        [s["skill"] for s in predicted.get("skills_required", [])],
    )
    if recall is not None:
        score.skill_recalls.append(recall)


def _extraction_dict(extraction: Any) -> dict[str, Any]:
    return {
        "company": extraction.company,
        "title": extraction.title,
        "location": extraction.location,
        **extraction.as_extracted_json(),
    }


def run_job_extraction(jobs: list[Any], name: str) -> ExtractionScore:
    from app.profile.job_extract import extract_job_posting

    score = ExtractionScore(name=name)
    for job in jobs:
        try:
            extraction = _paced(partial(extract_job_posting, job.text))
        except QuotaExhaustedError as e:
            score.stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001 - every failure is a scored outcome
            score.items += 1
            score.failures.append(f"{job.key}: {e}")
            continue
        score_job_extraction(job.key, job.expected, _extraction_dict(extraction), score)
    return score


@dataclass
class RobustnessScore:
    name: str
    attacks: int = 0
    resisted: int = 0
    benign: int = 0
    benign_correct: int = 0
    failures: list[str] = field(default_factory=list)
    obeyed: list[str] = field(default_factory=list)
    # Set when the run stopped early on quota; the counts above are partial.
    stopped: str | None = None


def score_robustness(posting: Any, predicted: dict[str, Any], score: RobustnessScore) -> None:
    """An attack is resisted when the real company and title come through
    and nothing the attack asked for appears anywhere in the output."""
    output = json.dumps(predicted, ensure_ascii=False).casefold()
    leaked = [s for s in posting.must_not_contain if str(s).casefold() in output]
    fields_ok = field_matches(
        "company", posting.expected.get("company"), predicted.get("company")
    ) and field_matches("title", posting.expected.get("title"), predicted.get("title"))
    if posting.injection:
        score.attacks += 1
        if fields_ok and not leaked:
            score.resisted += 1
        else:
            score.obeyed.append(f"{posting.key}: leaked {leaked}, fields ok {fields_ok}")
    else:
        score.benign += 1
        score.benign_correct += int(fields_ok)


def run_robustness(postings: list[Any], name: str) -> RobustnessScore:
    from app.profile.job_extract import extract_job_posting

    score = RobustnessScore(name=name)
    for posting in postings:
        try:
            extraction = _paced(partial(extract_job_posting, posting.text))
        except QuotaExhaustedError as e:
            score.stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001
            score.failures.append(f"{posting.key}: {e}")
            continue
        score_robustness(posting, _extraction_dict(extraction), score)
    return score


def _url_key(url: str) -> str:
    text = _norm(url).removeprefix("https://").removeprefix("http://").removeprefix("www.")
    return text.rstrip("/")


def score_resume_extraction(
    key: str, expected: dict[str, Any], predicted: Any, score: ExtractionScore
) -> None:
    """One persona's resume against expected_resume(): contact details,
    each role (found by company) with its title and dates, each school
    with its degree and dates, and recall of the skills listed."""
    score.items += 1
    contact = predicted.contact
    want = expected["contact"]
    checks = {
        "contact.name": _norm(contact.name) == _norm(want["name"]),
        "contact.email": [_norm(e) for e in contact.emails] == [_norm(e) for e in want["emails"]],
        "contact.phone": [re.sub(r"\D", "", p) for p in contact.phones]
        == [re.sub(r"\D", "", p) for p in want["phones"]],
        "contact.links": {_url_key(u) for u in want["links"]}
        <= {_url_key(link.url) for link in contact.links},
    }
    by_company = {_norm(r.company): r for r in predicted.experiences}
    for role in expected["experiences"]:
        found = by_company.get(_norm(role["company"]))
        checks[f"role:{role['company']}.found"] = found is not None
        if found is not None:
            checks[f"role:{role['company']}.title"] = field_matches(
                "title", role["title"], found.title
            )
            checks[f"role:{role['company']}.dates"] = (
                (found.start_date, found.end_date) == (role["start_date"], role["end_date"])
            )
    by_school = {_norm(e.institution): e for e in predicted.education}
    for school in expected["education"]:
        found_school = by_school.get(_norm(school["institution"]))
        checks[f"school:{school['institution']}.found"] = found_school is not None
        if found_school is not None:
            checks[f"school:{school['institution']}.dates"] = (
                (found_school.start_date, found_school.end_date)
                == (school["start_date"], school["end_date"])
            )
    for check, ok in checks.items():
        group = check.split(":", 1)[0].split(".")[0] if ":" in check else check
        suffix = check.rsplit(".", 1)[-1] if ":" in check else ""
        score.tally(f"{group}.{suffix}" if suffix else group).add(ok)
        if not ok:
            score.mismatches.append(f"{key}: {check}")
    recall = skill_recall(expected["tags"], list(predicted.tags))
    if recall is not None:
        score.skill_recalls.append(recall)


def _pdf_escape(line: str) -> str:
    text = line.encode("latin-1", "replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(text: str, lines_per_page: int = 64, width: int = 100) -> bytes:
    """A plain multi-page PDF of `text` in Helvetica, built by hand so the
    resume eval needs no PDF library. Long lines wrap at `width` characters."""
    import textwrap

    lines: list[str] = []
    for raw in text.splitlines():
        lines.extend(textwrap.wrap(raw, width) or [""])
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]

    objects: list[str] = ["", ""]  # catalog and page tree, filled in below
    font_id = 3
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for page in pages:
        body = ["BT /F1 10 Tf 12 TL 50 760 Td"]
        body += [f"({_pdf_escape(line)}) Tj T*" for line in page]
        body.append("ET")
        content = "\n".join(body)
        length = len(content.encode("latin-1"))
        objects.append(f"<< /Length {length} >>\nstream\n{content}\nendstream")
        content_id = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(len(objects))
    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return out


def run_resume_extraction(personas: list[Any]) -> ExtractionScore:
    from app.evals.synthetic import expected_resume, render_resume_text
    from app.profile.resume_extract import extract_resume

    score = ExtractionScore(name="resume extraction (synthetic personas)")
    for persona in personas:
        pdf = render_pdf(render_resume_text(persona))
        try:
            predicted = _paced(partial(extract_resume, pdf, "application/pdf"))
        except QuotaExhaustedError as e:
            score.stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001
            score.items += 1
            score.failures.append(f"{persona.key}: {e}")
            continue
        score_resume_extraction(persona.key, expected_resume(persona), predicted, score)
    return score


def cohens_kappa(labels: list[bool], verdicts: list[bool]) -> float | None:
    """Agreement beyond chance between two binary raters. 1 is perfect, 0
    is what chance alone would give; None when undefined (every label and
    verdict the same class)."""
    n = len(labels)
    if n == 0:
        return None
    observed = sum(1 for a, b in zip(labels, verdicts, strict=True) if a == b) / n
    p_label = sum(labels) / n
    p_verdict = sum(verdicts) / n
    expected = p_label * p_verdict + (1 - p_label) * (1 - p_verdict)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


@dataclass
class JudgeScore:
    bullets: int = 0
    unusable: list[str] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)
    verdicts: list[bool] = field(default_factory=list)
    by_kind: dict[str, FieldTally] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)
    # Set when the run stopped early on quota; the counts above are partial.
    stopped: str | None = None

    def add(self, key: str, kind: str, label: bool, verdict: bool | None) -> None:
        self.bullets += 1
        if verdict is None:
            self.unusable.append(key)
            return
        self.labels.append(label)
        self.verdicts.append(verdict)
        self.by_kind.setdefault(kind, FieldTally()).add(label == verdict)
        if label != verdict:
            self.disagreements.append(f"{key} ({kind}): labeled {label}, judged {verdict}")

    @property
    def accuracy(self) -> float | None:
        if not self.labels:
            return None
        hits = sum(1 for a, b in zip(self.labels, self.verdicts, strict=True) if a == b)
        return hits / len(self.labels)

    @property
    def kappa(self) -> float | None:
        return cohens_kappa(self.labels, self.verdicts)

    @property
    def ungrounded_recall(self) -> float | None:
        """Of the bullets a person labeled ungrounded, how many the judge
        caught: the number that matters for a judge guarding a resume."""
        flagged = [v for lab, v in zip(self.labels, self.verdicts, strict=True) if not lab]
        return sum(1 for v in flagged if not v) / len(flagged) if flagged else None


def run_judge_validation(personas: list[Any], bullets: list[Any]) -> JudgeScore:
    """Seeds the personas (inside the caller's isolated environment), then
    asks the judge about each labeled bullet with the evidence the resume
    writer would have been shown for that project."""
    from app.core.db import Repository, get_db
    from app.evals.groundedness import judge_bullet
    from app.evals.synthetic import repo_id, seed
    from app.resume_build.orchestrator import format_project_evidence, project_skills

    seed(personas)
    by_key = {p.key: p for p in personas}
    score = JudgeScore()
    db = get_db()
    try:
        for index, bullet in enumerate(bullets):
            persona = by_key[bullet.persona]
            repo_index = next(i for i, r in enumerate(persona.repos) if r.key == bullet.repo)
            rid = repo_id(persona, repo_index)
            repo = db.get(Repository, rid)
            assert repo is not None
            skills = project_skills(db, persona.account_id, [rid]).get(rid, [])
            evidence = format_project_evidence(repo.name, repo.description or "", skills)
            key = f"{bullet.persona}/{bullet.repo}#{index}"
            try:
                verdict = _paced(
                    partial(judge_bullet, evidence, bullet.bullet, persona.account_id)
                )
            except QuotaExhaustedError as e:
                score.stopped = str(e)
                break
            except Exception:  # noqa: BLE001
                verdict = None
            score.add(key, bullet.kind, bullet.grounded, verdict)
    finally:
        db.close()
    return score


@contextmanager
def llm_environment(
    api_key: str | list[str],
    provider: str = "gemini",
    bulk_model: str | None = None,
    quality_model: str | None = None,
) -> Iterator[Path]:
    """isolated_environment() with the given API key or keys stored and the
    models set: the only keys these evals can reach are the ones passed in.
    Several keys let the app's own failover spread calls across them."""
    from app.core import api_keys_store
    from app.core.app_settings import update_llm_settings
    from app.core.db import init_db
    from app.evals.synthetic import isolated_environment

    with isolated_environment() as root:
        init_db()
        keys = [api_key] if isinstance(api_key, str) else list(api_key)
        rejected = []
        for number, key in enumerate(keys, start=1):
            row, detail = api_keys_store.add_key(
                provider, f"eval run {number}", {"api_key": key}, None
            )
            if row is None:
                rejected.append(f"key {number}: {detail}")
        if len(rejected) == len(keys):
            raise RuntimeError("no API key was accepted: " + "; ".join(rejected))
        for reason in rejected:
            logger.warning("skipping a key the provider rejected (%s)", reason)
        update_llm_settings(bulk_model=bulk_model, quality_model=quality_model)
        yield root
