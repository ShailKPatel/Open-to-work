"""app/resume_build/orchestrator.py: generate_resume(). complete() and
search_skill_evidence() are both mocked (real LLM/Qdrant calls aren't
what this module is responsible for proving work, see test_search.py and
tests/live/ for those); what's under test here is the grounding logic
(never trust an LLM-invented repo_id or skill name), the job-text/system-
prompt separation, and correct assembly into the render_resume() call.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.db import (
    Account,
    Experience,
    ExperiencePoint,
    JobPosting,
    Repository,
    SkillEvidence,
    get_db,
    init_db,
)
from app.resume_build.orchestrator import (
    build_resume_data_from_seed,
    edit_resume_content,
    generate_resume,
)


def _reset_db(tmp_path: Path):
    import os

    import app.core.db as db_module
    import app.retrieval.vectorstore as vectorstore_module
    from app.core.settings import get_settings

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    # build_resume_data() now calls search_experience_points() for every
    # role unconditionally (per-role point selection); without this, that
    # call falls through to whatever QDRANT_URL .env has configured.
    # :memory: keeps every test here
    # isolated, same convention every other Qdrant-touching test file uses.
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _seed(tmp_path: Path) -> tuple[int, int]:
    _reset_db(tmp_path)
    db = get_db()
    account = Account(first_name="Jane", last_name="Doe", github_username="janedoe")
    db.add(account)
    db.commit()
    db.refresh(account)

    repo = Repository(
        account_id=account.id, github_id=1, name="cool-project", full_name="janedoe/cool-project",
        url="https://github.com/janedoe/cool-project", is_fork=False, description="A cool project",
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    db.add(
        SkillEvidence(
            skill="Python", repo_id=repo.id, evidence_type="declared_dependency",
            weight=0.7, confidence=1.0,
        )
    )

    role = Experience(account_id=account.id, title="Engineer", company="Acme")
    db.add(role)
    db.commit()
    db.refresh(role)
    db.add(ExperiencePoint(experience_id=role.id, text="Shipped a thing", order_index=1))

    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="hash1", company="Acme",
        title="Backend Engineer", raw_text_quarantined="We need a Python backend engineer.",
        content_hash="hash1",
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)

    account_id, posting_id = account.id, posting.id
    db.close()
    return account_id, posting_id


def _stub_search(monkeypatch, repo_id: int, skill: str = "Python"):
    from app.retrieval.search import Hit

    hits = [Hit(id=1, score=0.9, payload={"repo_id": repo_id, "skill": skill})]
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence",
        lambda query_text, account_id, top_k=10, source_type=None: hits,
    )


def test_unknown_account_raises(tmp_path):
    _, posting_id = _seed(tmp_path)
    with pytest.raises(ValueError, match="no account"):
        generate_resume(999999, posting_id)


def test_unknown_job_posting_raises(tmp_path):
    account_id, _ = _seed(tmp_path)
    with pytest.raises(ValueError, match="no job posting"):
        generate_resume(account_id, 999999)


def test_llm_response_must_be_valid_json(tmp_path, monkeypatch):
    account_id, posting_id = _seed(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence", lambda *a, **k: []
    )
    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        generate_resume(account_id, posting_id)


def test_grounds_out_invented_repo_id_and_skill(tmp_path, monkeypatch):
    """The LLM referencing a repo_id/skill it was never offered must be
    silently dropped, not trusted straight into the resume."""
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    repo_id = db.query(Repository).filter_by(account_id=account_id).first().id
    db.close()

    _stub_search(monkeypatch, repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "A concise, grounded summary.",
        "projects": [
            {"repo_id": repo_id, "tagline": "Nice tool", "points": ["Built X with Python"]},
            {"repo_id": 999999, "tagline": "Fake", "points": ["Should be dropped"]},
        ],
        "skills": ["Python", "InventedSkillNoOneHas"],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    tex = generate_resume(account_id, posting_id)

    assert "cool-project" in tex
    assert "Built X with Python" in tex
    assert "Fake" not in tex
    assert "Should be dropped" not in tex
    assert "Python" in tex
    assert "InventedSkillNoOneHas" not in tex


def test_job_text_is_its_own_message_not_the_system_prompt(tmp_path, monkeypatch):
    account_id, posting_id = _seed(tmp_path)
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])

    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    generate_resume(account_id, posting_id)

    messages = fake_complete.call_args.args[1]
    assert messages[0]["role"] == "system"
    assert "We need a Python backend engineer." not in messages[0]["content"]
    job_text_messages = [
        m
        for m in messages
        if m["role"] == "user" and "We need a Python backend engineer." in m["content"]
    ]
    assert len(job_text_messages) == 1
    assert "reference material" in job_text_messages[0]["content"]


def test_experience_role_always_included_points_fall_back_when_search_empty(tmp_path, monkeypatch):
    """No mocked search_experience_points here: with QDRANT_URL forced to
    :memory: (see _reset_db), the real search runs against an empty
    collection and comes back with nothing, exercising the fallback path,
    not a hallucinated result. The role itself is never optional either
    way.
    """
    account_id, posting_id = _seed(tmp_path)
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    tex = generate_resume(account_id, posting_id)

    assert "Shipped a thing" in tex
    assert "Acme" in tex


def test_experience_points_narrowed_by_semantic_search(tmp_path, monkeypatch):
    """The role stays, in full; which of its points render is decided per
    role by search_experience_points, not "always every point."""
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    role = db.query(Experience).filter_by(account_id=account_id).first()
    db.add(ExperiencePoint(experience_id=role.id, text="Organized the office party", order_index=2))
    db.commit()
    role_id = role.id
    db.close()

    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])

    from app.retrieval.search import Hit

    def _fake_points_search(query_text, account_id, top_k=10, experience_id=None):
        if experience_id == role_id:
            return [Hit(id=1, score=0.9, payload={"text": "Shipped a thing"})]
        return []

    monkeypatch.setattr("app.retrieval.search.search_experience_points", _fake_points_search)

    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    tex = generate_resume(account_id, posting_id)

    assert "Shipped a thing" in tex
    assert "Organized the office party" not in tex


def test_length_guidance_varies_by_template(tmp_path, monkeypatch):
    account_id, posting_id = _seed(tmp_path)
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    generate_resume(account_id, posting_id, template="onepage")
    onepage_messages = [
        m["content"] for m in fake_complete.call_args.args[1] if m["role"] == "user"
    ]
    assert any("one-page template" in m for m in onepage_messages)

    generate_resume(account_id, posting_id, template="twopage")
    twopage_messages = [
        m["content"] for m in fake_complete.call_args.args[1] if m["role"] == "user"
    ]
    assert any("two-page template" in m for m in twopage_messages)


def test_education_included_in_full(tmp_path, monkeypatch):
    from app.core.db import Education

    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    db.add(Education(account_id=account_id, institution="State University", degree="B.Sc"))
    db.commit()
    db.close()

    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    tex = generate_resume(account_id, posting_id)

    assert "State University" in tex
    assert "B.Sc" in tex


# --- build_resume_data_from_seed() / edit_resume_content() ---------------------------------------
# The "adopt an uploaded resume, then AI-edit it" path (app/api/resume.py's
# POST /{id}/edit): same grounding rules as generate_resume() above, just
# seeded from arbitrary text instead of a JobPosting row.


def test_build_resume_data_from_seed_grounds_like_a_job_posting(tmp_path, monkeypatch):
    account_id, _ = _seed(tmp_path)
    db = get_db()
    repo_id = db.query(Repository).filter_by(account_id=account_id).first().id
    db.close()
    _stub_search(monkeypatch, repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "Seeded summary.",
        "projects": [
            {"repo_id": repo_id, "tagline": "Nice tool", "points": ["Built X with Python"]}
        ],
        "skills": ["Python"],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = build_resume_data_from_seed(account_id, "Backend generalist, Python", "onepage")

    assert data["summary"] == "Seeded summary."
    assert data["projects"][0]["name"] == "cool-project"
    assert data["skills"] == ["Python"]
    assert len(data["experience"]) == 1  # deterministic context still built in full


def test_build_resume_data_from_seed_unknown_account_raises(tmp_path):
    _seed(tmp_path)
    with pytest.raises(ValueError, match="no account"):
        build_resume_data_from_seed(999999, "seed text", "onepage")


def test_edit_resume_content_never_sends_experience_to_the_llm(tmp_path, monkeypatch):
    """Grounding rule: experience/education/header are rebuilt fresh from
    the DB, not part of what the edit LLM call is given at all. This is
    what makes them AI-immutable structurally, not just by prompt
    wording."""
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    repo_id = db.query(Repository).filter_by(account_id=account_id).first().id
    posting = db.get(JobPosting, posting_id)
    job_text = posting.raw_text_quarantined
    db.close()
    _stub_search(monkeypatch, repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "Edited summary.",
        "projects": [
            {"repo_id": repo_id, "tagline": "Nice tool", "points": ["Built X with Python"]}
        ],
        "skills": ["Python"],
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    current = {"summary": "Old summary.", "projects": [], "skills": []}
    data = edit_resume_content(account_id, job_text, current, "Emphasize Python", "onepage")

    assert data["summary"] == "Edited summary."
    assert len(data["experience"]) == 1
    assert data["experience"][0]["company"] == "Acme"  # untouched, real DB data

    messages = fake_complete.call_args.args[1]
    joined = " ".join(str(m["content"]) for m in messages)
    assert "Acme" not in joined  # experience never sent to this LLM call
    assert "Emphasize Python" in joined  # the account holder's own instruction, not quarantined
    assert "Old summary." in joined  # current content given as context to edit from


def test_edit_resume_content_grounds_out_invented_repo_id(tmp_path, monkeypatch):
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    repo_id = db.query(Repository).filter_by(account_id=account_id).first().id
    posting = db.get(JobPosting, posting_id)
    job_text = posting.raw_text_quarantined
    db.close()
    _stub_search(monkeypatch, repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "S",
        "projects": [{"repo_id": 999999, "tagline": "Fake", "points": ["nope"]}],
        "skills": ["InventedSkillNoOneHas"],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = edit_resume_content(account_id, job_text, {}, "Change something", "onepage")

    assert data["projects"] == []
    assert data["skills"] == []


def _seed_with_leftovers(tmp_path: Path) -> tuple[int, int, int, int]:
    """Same shape as _seed(), plus a second repository and a second
    experience point, so there is real content left over for the reserve
    to hold once the model has picked. Returns account, posting and both
    repo ids.
    """
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    picked_repo = db.query(Repository).filter_by(account_id=account_id).first()
    spare = Repository(
        account_id=account_id, github_id=2, name="spare-project",
        full_name="janedoe/spare-project", url="https://github.com/janedoe/spare-project",
        is_fork=False, description="A spare project nobody picked",
    )
    db.add(spare)
    role = db.query(Experience).filter_by(account_id=account_id).first()
    db.add(ExperiencePoint(experience_id=role.id, text="Also did a spare thing", order_index=2))
    db.commit()
    db.refresh(spare)
    ids = (account_id, posting_id, picked_repo.id, spare.id)
    db.close()
    return ids


def _stub_reserve_searches(monkeypatch, picked_repo_id: int, spare_repo_id: int):
    from app.retrieval.search import Hit

    skill_hits = [
        Hit(id=1, score=0.9, payload={"repo_id": picked_repo_id, "skill": "Python"}),
        Hit(id=2, score=0.5, payload={"repo_id": spare_repo_id, "skill": "Spare Skill"}),
    ]
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence",
        lambda query_text, account_id, top_k=10, source_type=None: skill_hits,
    )
    # Only the first of the role's two points matches, so the other is
    # narrowed away and should land in the reserve rather than vanish.
    monkeypatch.setattr(
        "app.retrieval.search.search_experience_points",
        lambda query_text, account_id, top_k=5, experience_id=None: [
            Hit(id=3, score=0.9, payload={"text": "Shipped a thing"})
        ],
    )


def test_reserve_holds_what_the_resume_is_not_showing(tmp_path, monkeypatch):
    """Everything real that got left out has to be recoverable, because
    app/resume_build/pagefit.py adds it back when a resume falls short of
    its target page count."""
    from app.resume_build.orchestrator import build_resume_data

    account_id, posting_id, picked_repo_id, spare_repo_id = _seed_with_leftovers(tmp_path)
    _stub_reserve_searches(monkeypatch, picked_repo_id, spare_repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "A concise, grounded summary.",
        "projects": [
            {"repo_id": picked_repo_id, "tagline": "Nice tool", "points": ["Built X"]}
        ],
        "skills": ["Python"],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = build_resume_data(account_id, posting_id)
    reserve = data["reserve"]

    assert [p["name"] for p in reserve["projects"]] == ["spare-project"]
    # A held-back project's bullet is the repository's own description,
    # never anything a model wrote for it.
    assert reserve["projects"][0]["points"] == ["A spare project nobody picked"]
    assert reserve["experience_points"] == {"Acme": ["Also did a spare thing"]}
    assert reserve["skills"] == ["Spare Skill"]


def test_reserve_never_reaches_the_rendered_tex(tmp_path, monkeypatch):
    account_id, posting_id, picked_repo_id, spare_repo_id = _seed_with_leftovers(tmp_path)
    _stub_reserve_searches(monkeypatch, picked_repo_id, spare_repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "A concise, grounded summary.",
        "projects": [
            {"repo_id": picked_repo_id, "tagline": "Nice tool", "points": ["Built X"]}
        ],
        "skills": ["Python"],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    tex = generate_resume(account_id, posting_id)

    assert "spare-project" not in tex
    assert "Spare Skill" not in tex
    assert "Also did a spare thing" not in tex


def test_long_repo_description_is_truncated_before_it_is_sent(tmp_path, monkeypatch):
    """A repo that put a whole README in its GitHub "About" field would
    otherwise pay for it in every resume build for that account."""
    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    repo = db.query(Repository).filter(Repository.account_id == account_id).one()
    repo.description = "A cool project. " + ("padding " * 500)
    repo_id = repo.id
    db.commit()
    db.close()
    _stub_search(monkeypatch, repo_id)

    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    generate_resume(account_id, posting_id)

    candidates_message = next(
        m["content"]
        for m in fake_complete.call_args.args[1]
        if m["role"] == "user" and "CANDIDATE PROJECTS" in m["content"]
    )
    assert "A cool project." in candidates_message
    assert candidates_message.count("padding") < 50


def test_resume_build_call_is_tagged_with_its_purpose(tmp_path, monkeypatch):
    """So /monitor's by_purpose breakdown can tell resume spend apart from
    ingestion spend."""
    account_id, posting_id = _seed(tmp_path)
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    generate_resume(account_id, posting_id)

    assert fake_complete.call_args.kwargs["purpose"] == "resume_build"


def test_candidate_projects_skip_profile_readme(tmp_path, monkeypatch):
    """The profile README's evidence is indexed (it is a skill source), so
    retrieval can rank it first; it must still never be offered as a
    project, nor take a candidate slot from a real one."""
    from app.resume_build.orchestrator import _candidate_projects
    from app.retrieval.search import Hit

    account_id, _ = _seed(tmp_path)
    db = get_db()
    project_id = db.query(Repository).filter_by(account_id=account_id).first().id
    profile = Repository(
        account_id=account_id, github_id=2, name="janedoe", full_name="janedoe/janedoe",
        url="https://github.com/janedoe/janedoe", is_fork=False, is_profile_readme=True,
    )
    db.add(profile)
    db.commit()
    profile_id = profile.id

    hits = [
        Hit(id=1, score=0.95, payload={"repo_id": profile_id, "skill": "Python"}),
        Hit(id=2, score=0.5, payload={"repo_id": project_id, "skill": "Python"}),
    ]
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence",
        lambda query_text, account_id, top_k=10, source_type=None: hits,
    )

    candidates = _candidate_projects(db, account_id, "Python backend")
    db.close()

    assert [c["repo_id"] for c in candidates] == [project_id]


# --- picking experience points, leaving entries off --------------------------


def _add_points(account_id: int, texts: list[str]) -> int:
    db = get_db()
    role = db.query(Experience).filter_by(account_id=account_id).first()
    for i, text in enumerate(texts, start=2):
        db.add(ExperiencePoint(experience_id=role.id, text=text, order_index=i))
    db.commit()
    role_id = role.id
    db.close()
    return role_id


def test_model_picks_experience_points_by_number(tmp_path, monkeypatch):
    """The model picks by number into the role's pool, in its own order,
    and an invented role id or out-of-range number is ignored. A point
    returned without new wording keeps the account's own text."""
    from app.resume_build.orchestrator import build_resume_data

    account_id, posting_id = _seed(tmp_path)
    role_id = _add_points(account_id, ["Organized the office party", "Cut build time in half"])
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "S", "projects": [], "skills": [],
        "experience": [
            {
                "role_id": role_id,
                "points": [
                    {"id": 2, "text": "Cut build time in half"},
                    {"id": 99, "text": "Invented"},
                    {"id": 0, "text": ""},
                ],
            },
            {"role_id": 424242, "points": [{"id": 0, "text": "Invented"}]},
        ],
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.resume_build.orchestrator.complete", fake_complete)

    data = build_resume_data(account_id, posting_id, points_per_role=2)

    assert data["experience"][0]["points"] == ["Cut build time in half", "Shipped a thing"]
    prompt = "\n".join(m["content"] for m in fake_complete.call_args.args[1])
    assert "[1] Organized the office party" in prompt
    assert "at most 2 points" in prompt


def test_reworded_experience_point_is_kept_unless_it_adds_a_number(tmp_path, monkeypatch):
    """A rewording is used as long as it states no number the original
    does not; one that invents a metric falls back to the account's own
    text. source_points keeps the real point behind each shown line, so
    the reserve does not offer the same point again."""
    from app.resume_build.orchestrator import build_resume_data

    account_id, posting_id = _seed(tmp_path)
    role_id = _add_points(account_id, ["Organized the office party", "Cut build time in half"])
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {
        "summary": "S", "projects": [], "skills": [],
        "experience": [
            {
                "role_id": role_id,
                "points": [
                    {"id": 2, "text": "Halved CI build time"},
                    {"id": 0, "text": "Shipped a thing used by 10000 people"},
                ],
            },
        ],
    }
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = build_resume_data(account_id, posting_id, points_per_role=2)

    role = data["experience"][0]
    assert role["points"] == ["Halved CI build time", "Shipped a thing"]
    assert role["source_points"] == ["Cut build time in half", "Shipped a thing"]
    held = data["reserve"]["experience_points"].get(role["company"], [])
    assert "Cut build time in half" not in held
    assert "Organized the office party" in held


def test_role_the_model_skips_keeps_its_best_retrieval_points(tmp_path, monkeypatch):
    from app.resume_build.orchestrator import build_resume_data

    account_id, posting_id = _seed(tmp_path)
    _add_points(account_id, ["Second", "Third", "Fourth"])
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = build_resume_data(account_id, posting_id, points_per_role=2)

    assert data["experience"][0]["points"] == ["Shipped a thing", "Second"]


def test_unticked_education_is_left_off(tmp_path, monkeypatch):
    from app.core.db import Education
    from app.resume_build.orchestrator import build_resume_data

    account_id, posting_id = _seed(tmp_path)
    db = get_db()
    kept = Education(account_id=account_id, institution="State University", degree="B.Sc")
    dropped = Education(account_id=account_id, institution="Night School", degree="Cert")
    db.add_all([kept, dropped])
    db.commit()
    kept_id = kept.id
    db.close()

    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "S", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    data = build_resume_data(account_id, posting_id, selected_education_ids=[kept_id])

    assert [e["institution"] for e in data["education"]] == ["State University"]


def test_edit_keeps_entries_and_points_the_resume_already_showed(tmp_path, monkeypatch):
    """An AI edit rebuilds experience and education from the account's
    own data, but a role or degree left off the resume stays off, and the
    roles it does show keep the points they showed."""
    from app.core.db import Education

    account_id, _ = _seed(tmp_path)
    role_id = _add_points(account_id, ["Organized the office party"])
    db = get_db()
    other_role = Experience(account_id=account_id, title="Intern", company="Initech")
    school = Education(account_id=account_id, institution="Night School", degree="Cert")
    db.add_all([other_role, school])
    db.commit()
    db.close()

    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    fake_response = MagicMock()
    fake_response.parsed = {"summary": "New", "projects": [], "skills": []}
    monkeypatch.setattr(
        "app.resume_build.orchestrator.complete", MagicMock(return_value=fake_response)
    )

    current = {
        "summary": "Old",
        "experience": [{"id": role_id, "points": ["Organized the office party"]}],
        "education": [],
    }
    data = edit_resume_content(account_id, "job text", current, "Shorter summary", "onepage")

    assert [r["company"] for r in data["experience"]] == ["Acme"]
    assert data["experience"][0]["points"] == ["Organized the office party"]
    assert data["education"] == []


def test_resume_building_never_sees_entries_left_off_resumes(tmp_path, monkeypatch):
    """A project or role marked exclude_from_resume is not a candidate,
    not in the experience section, and its evidence suggests no skill."""
    from app.resume_build.context import build_experience_context
    from app.resume_build.orchestrator import _candidate_projects, _candidate_skills
    from app.retrieval.search import Hit

    account_id, _ = _seed(tmp_path)
    db = get_db()
    project_id = db.query(Repository).filter_by(account_id=account_id).first().id
    hidden = Repository(
        account_id=account_id, github_id=2, name="qr-tool", full_name="janedoe/qr-tool",
        url="https://github.com/janedoe/qr-tool", is_fork=False, exclude_from_resume=True,
    )
    hidden_role = Experience(
        account_id=account_id, title="Intern", company="Side Gig", exclude_from_resume=True
    )
    db.add_all([hidden, hidden_role])
    db.commit()

    hits = [
        Hit(id=1, score=0.95, payload={"repo_id": hidden.id, "skill": "QR codes"}),
        Hit(id=2, score=0.9, payload={"experience_id": hidden_role.id, "skill": "Excel"}),
        Hit(id=3, score=0.5, payload={"repo_id": project_id, "skill": "Python"}),
    ]
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence",
        lambda query_text, account_id, top_k=10, source_type=None: hits,
    )

    candidates = _candidate_projects(db, account_id, "Python", selected_project_ids=[hidden.id])
    skills = _candidate_skills(db, account_id, "Python")
    roles = build_experience_context(db, account_id)
    db.close()

    assert [c["repo_id"] for c in candidates] == [project_id]
    assert skills == ["Python"]
    assert [r["company"] for r in roles] == ["Acme"]


def test_resume_building_never_sees_archived_education_or_skills(tmp_path, monkeypatch):
    """An archived education entry is left out of the education section,
    and a skill archived by hand is neither a candidate skill nor listed
    under a candidate project, even when a live project backs it."""
    from app.core.db import Education, SkillArchive, SkillEvidence
    from app.resume_build.context import build_education_context
    from app.resume_build.orchestrator import _candidate_projects, _candidate_skills
    from app.retrieval.search import Hit

    account_id, _ = _seed(tmp_path)
    db = get_db()
    project_id = db.query(Repository).filter_by(account_id=account_id).first().id
    db.add_all([
        Education(account_id=account_id, institution="State University", degree="B.Sc"),
        Education(
            account_id=account_id, institution="Night School", degree="Cert",
            exclude_from_resume=True,
        ),
        SkillEvidence(
            repo_id=project_id, skill="QR Codes", evidence_type="readme_described",
            weight=1.0, confidence=1.0,
        ),
        SkillArchive(account_id=account_id, name_key="qr codes"),
    ])
    db.commit()

    hits = [
        Hit(id=1, score=0.95, payload={"repo_id": project_id, "skill": "QR Codes"}),
        Hit(id=2, score=0.5, payload={"repo_id": project_id, "skill": "Python"}),
    ]
    monkeypatch.setattr(
        "app.retrieval.search.search_skill_evidence",
        lambda query_text, account_id, top_k=10, source_type=None: hits,
    )

    schools = build_education_context(db, account_id)
    skills = _candidate_skills(db, account_id, "Python")
    candidates = _candidate_projects(db, account_id, "Python")
    db.close()

    assert [e["institution"] for e in schools] == ["State University"]
    assert skills == ["Python"]
    assert "QR Codes" not in candidates[0]["skills"]
