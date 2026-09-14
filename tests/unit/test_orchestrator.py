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

    db_module._engine = None
    db_module._SessionLocal = None
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
