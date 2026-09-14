import datetime as dt
from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.db import ProjectLink, Repository, get_db, init_db
from app.core.llm import BudgetExceededError, LLMRateLimitedError
from app.core.settings import get_settings
from app.profile.build import build_profile, reprocess_repo, skill_evidence_for_repos
from app.profile.claims import LinkClaim, SkillClaim
from app.profile.extract import NoSourceTextError

NOW = dt.datetime(2026, 8, 24, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _stub_link_extraction(monkeypatch):
    """None of the skill-extraction tests below care about link extraction.
    Without this, every build_profile() call in this file would fall
    through to the real extract_links_from_repo and attempt a real LLM
    dispatch (blocking on network / hanging with no API key configured
    is a much worse test failure mode than a wrong assertion). Tests that
    do care override this via monkeypatch same as they already do for
    extract_skills_from_repo.
    """
    monkeypatch.setattr("app.profile.build.extract_links_from_repo", lambda repo: [])


@pytest.fixture(autouse=True)
def _stub_indexing(monkeypatch):
    """build_profile()/reprocess_repo() now index each repo's SkillEvidence
    into Qdrant after every commit (see _index_repo_evidence): real,
    intentional wiring, not something these tests care about asserting.
    Without this, every test below would hit the real sentence-transformers
    model and a real (likely unreachable, from a unit test) Qdrant
    connection. Best-effort/try-except in production code means a real
    failure wouldn't break these tests either way, but this keeps the
    suite fast and deterministic. app/retrieval/test_search.py and
    app/retrieval/test_index.py cover the indexing behavior itself.
    """
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts]
    )


def _reset_db(tmp_path: Path):
    import os

    import app.retrieval.vectorstore as vectorstore_module

    db_module._engine = None
    db_module._SessionLocal = None
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _persist_repo(**overrides) -> Repository:
    db = get_db()
    defaults = dict(
        github_id=1,
        name="proj",
        full_name="octocat/proj",
        url="https://github.com/octocat/proj",
        is_fork=False,
        readme="uses React and gRPC",
        manifests_json={"requirements.txt": {"ecosystem": "pip", "dependencies": ["fastapi"]}},
        commits_authored=20,
        last_commit_at=dt.datetime(2026, 8, 1, tzinfo=dt.UTC),
    )
    defaults.update(overrides)
    repo = Repository(**defaults)
    db.add(repo)
    db.commit()
    db.refresh(repo)
    db.close()  # detached, mirrors how repos actually arrive at build_profile
    return repo


def _fresh(repo_id: int) -> Repository:
    """Re-query from a brand new session, so we're checking what's actually
    persisted, not the caller's possibly-stale in-memory object."""
    db = get_db()
    fetched = db.get(Repository, repo_id)
    db.expunge(fetched)
    db.close()
    return fetched


def test_build_profile_writes_skill_evidence_from_manifest_and_readme(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.profile.build.extract_skills_from_repo",
        lambda repo: [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)],
    )
    repo = _persist_repo()

    profile = build_profile([repo], now=NOW)

    assert "fastapi" in profile.skills_json
    assert "React" in profile.skills_json
    assert profile.skills_json["fastapi"]["repo_count"] == 1
    assert profile.skills_json["fastapi"]["evidence_types"] == ["declared_dependency"]

    evidence = skill_evidence_for_repos([repo.id])
    assert {e.skill for e in evidence} == {"fastapi", "React"}


def test_build_profile_persists_extracted_status(tmp_path, monkeypatch):
    """Mutating a detached repo object's
    status doesn't persist unless it's re-attached to the commit session."""
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    repo = _persist_repo()
    assert repo.skill_extraction_status == "pending"

    build_profile([repo], now=NOW)

    reloaded = _fresh(repo.id)
    assert reloaded.skill_extraction_status == "extracted"
    assert reloaded.skills_extracted_at is not None


def test_no_source_text_marks_no_signal_not_failed(tmp_path, monkeypatch):
    _reset_db(tmp_path)

    def raise_no_source(repo):
        raise NoSourceTextError("nothing to work with")

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", raise_no_source)
    repo = _persist_repo(readme=None, description=None)

    build_profile([repo], now=NOW)

    reloaded = _fresh(repo.id)
    assert reloaded.skill_extraction_status == "no_signal"
    assert reloaded.skill_extraction_error is None
    # manifest-based evidence still gets written even with no LLM signal
    evidence = skill_evidence_for_repos([repo.id])
    assert {e.skill for e in evidence} == {"fastapi"}


def test_llm_failure_on_one_repo_does_not_lose_other_repos(tmp_path, monkeypatch):
    """API key/budget runs out
    mid-batch. One repo fails; the rest still get processed and committed."""
    _reset_db(tmp_path)

    def flaky_extract(repo):
        if repo.full_name == "octocat/bad":
            raise RuntimeError("budget exceeded")
        return [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)]

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", flaky_extract)
    good = _persist_repo(github_id=1, full_name="octocat/good")
    bad = _persist_repo(github_id=2, full_name="octocat/bad")

    profile = build_profile([good, bad], now=NOW)

    good_reloaded = _fresh(good.id)
    bad_reloaded = _fresh(bad.id)
    assert good_reloaded.skill_extraction_status == "extracted"
    assert bad_reloaded.skill_extraction_status == "failed"
    assert "budget exceeded" in bad_reloaded.skill_extraction_error

    # good repo's evidence survived despite bad repo's failure
    good_evidence = {e.skill for e in skill_evidence_for_repos([good.id])}
    assert good_evidence == {"fastapi", "React"}
    # bad repo still got its manifest-based (free, no-LLM) evidence
    bad_evidence = {e.skill for e in skill_evidence_for_repos([bad.id])}
    assert bad_evidence == {"fastapi"}
    # profile aggregation reflects both repos' surviving evidence
    assert profile.skills_json["fastapi"]["repo_count"] == 2


def test_rate_limit_mid_batch_stops_and_leaves_rest_untouched(tmp_path, monkeypatch):
    """A 429 on repo 2 of 3 gives repo 2 a clean "rate_limited" status and
    stops the batch: repo 3 is never attempted and stays "pending"."""
    _reset_db(tmp_path)
    calls = []

    def hits_limit_on_second(repo):
        calls.append(repo.full_name)
        if repo.full_name == "octocat/b":
            raise LLMRateLimitedError("quota exceeded, retry in 16s")
        return [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)]

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", hits_limit_on_second)
    a = _persist_repo(github_id=1, full_name="octocat/a")
    b = _persist_repo(github_id=2, full_name="octocat/b")
    c = _persist_repo(github_id=3, full_name="octocat/c")

    build_profile([a, b, c], now=NOW)

    assert calls == ["octocat/a", "octocat/b"]  # c never attempted at all

    a_reloaded = _fresh(a.id)
    b_reloaded = _fresh(b.id)
    c_reloaded = _fresh(c.id)
    assert a_reloaded.skill_extraction_status == "extracted"
    assert b_reloaded.skill_extraction_status == "rate_limited"
    assert "quota exceeded" not in (b_reloaded.skill_extraction_error or "")  # cleaned up
    assert "later" in b_reloaded.skill_extraction_error.lower()
    assert c_reloaded.skill_extraction_status == "pending"  # untouched, not "failed"


def test_budget_exceeded_mid_batch_also_stops_the_batch(tmp_path, monkeypatch):
    _reset_db(tmp_path)

    def hits_budget_cap(repo):
        raise BudgetExceededError("monthly budget $20.00 reached")

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", hits_budget_cap)
    a = _persist_repo(github_id=1, full_name="octocat/a")
    b = _persist_repo(github_id=2, full_name="octocat/b")

    build_profile([a, b], now=NOW)

    assert _fresh(a.id).skill_extraction_status == "rate_limited"
    assert _fresh(b.id).skill_extraction_status == "pending"  # never attempted


def test_generic_error_mid_batch_does_not_stop_the_batch(tmp_path, monkeypatch):
    """Only rate-limit/budget stops the whole batch; a per-repo
    failure (existing behavior) still lets the rest proceed normally."""
    _reset_db(tmp_path)

    def fails_only_on_b(repo):
        if repo.full_name == "octocat/b":
            raise RuntimeError("malformed response")
        return []

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", fails_only_on_b)
    a = _persist_repo(github_id=1, full_name="octocat/a")
    b = _persist_repo(github_id=2, full_name="octocat/b")
    c = _persist_repo(github_id=3, full_name="octocat/c")

    build_profile([a, b, c], now=NOW)

    assert _fresh(a.id).skill_extraction_status == "extracted"
    assert _fresh(b.id).skill_extraction_status == "failed"
    assert _fresh(c.id).skill_extraction_status == "extracted"  # still attempted


def test_rate_limited_repo_is_retried_automatically_without_force(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    call_count = {"n": 0}

    def rate_limited_once(repo):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise LLMRateLimitedError("quota exceeded")
        return []

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", rate_limited_once)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    assert _fresh(repo.id).skill_extraction_status == "rate_limited"

    build_profile([repo], now=NOW)  # "rate_limited" isn't a skip status, so it's retried
    assert _fresh(repo.id).skill_extraction_status == "extracted"
    assert call_count["n"] == 2


def test_already_extracted_repo_is_skipped_without_force(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    calls = []

    def counting_extract(repo):
        calls.append(repo.full_name)
        return []

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", counting_extract)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    assert len(calls) == 1

    build_profile([repo], now=NOW)  # same repo, already "extracted"
    assert len(calls) == 1  # not called again


def test_failed_repo_is_retried_automatically_without_force(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    call_count = {"n": 0}

    def fail_once(repo):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient error")
        return []

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", fail_once)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    assert _fresh(repo.id).skill_extraction_status == "failed"

    build_profile([repo], now=NOW)  # "failed" isn't a skip status, so it's retried
    assert _fresh(repo.id).skill_extraction_status == "extracted"
    assert call_count["n"] == 2


def test_force_reprocesses_already_extracted_repo(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    calls = []

    def counting_extract(repo):
        calls.append(repo.full_name)
        return []

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", counting_extract)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    build_profile([repo], now=NOW, force=True)
    assert len(calls) == 2


def test_reprocess_repo_clears_old_evidence_before_rewriting(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    responses = [
        [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)],
        [SkillClaim(skill="Vue", evidence_type="readme_described", confidence=0.9)],
    ]
    call_index = {"i": 0}

    def sequenced_extract(repo):
        result = responses[call_index["i"]]
        call_index["i"] += 1
        return result

    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", sequenced_extract)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    assert {e.skill for e in skill_evidence_for_repos([repo.id])} == {"fastapi", "React"}

    reprocess_repo(repo.id, now=NOW)
    # React (from the first run) is gone: reprocess clears before rewriting,
    # doesn't just append
    assert {e.skill for e in skill_evidence_for_repos([repo.id])} == {"fastapi", "Vue"}


def test_reprocess_repo_unknown_id_raises(tmp_path):
    _reset_db(tmp_path)
    try:
        reprocess_repo(999999)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_same_skill_from_two_repos_aggregates_via_noisy_or(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    repo_a = _persist_repo(github_id=1, full_name="octocat/a")
    repo_b = _persist_repo(github_id=2, full_name="octocat/b")

    profile = build_profile([repo_a, repo_b], now=NOW)

    fastapi = profile.skills_json["fastapi"]
    assert fastapi["repo_count"] == 2
    # noisy-OR of two positive weights must exceed either weight alone
    single_repo_profile = build_profile([repo_a], now=NOW)
    assert fastapi["weight"] > single_repo_profile.skills_json["fastapi"]["weight"]


def test_profile_skills_property_matches_skills_json(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    repo = _persist_repo()

    profile = build_profile([repo], now=NOW)

    skills_by_name = {s["skill"]: s for s in profile.skills}
    assert skills_by_name["fastapi"]["weight"] == profile.skills_json["fastapi"]["weight"]


def test_skill_evidence_for_repos_empty_input(tmp_path):
    _reset_db(tmp_path)
    assert skill_evidence_for_repos([]) == []


def _links_for(repo_id: int) -> list[ProjectLink]:
    db = get_db()
    try:
        return list(db.query(ProjectLink).filter(ProjectLink.repo_id == repo_id).all())
    finally:
        db.close()


def test_build_profile_writes_links_found_in_readme(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    monkeypatch.setattr(
        "app.profile.build.extract_links_from_repo",
        lambda repo: [LinkClaim(label="YouTube Video", url="https://youtu.be/abc123")],
    )
    repo = _persist_repo()

    build_profile([repo], now=NOW)

    links = _links_for(repo.id)
    assert len(links) == 1
    assert links[0].label == "YouTube Video"
    assert links[0].url == "https://youtu.be/abc123"
    assert links[0].source == "readme_extracted"


def test_link_matching_repo_url_is_not_duplicated(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    monkeypatch.setattr(
        "app.profile.build.extract_links_from_repo",
        lambda repo: [LinkClaim(label="GitHub", url="https://github.com/octocat/proj")],
    )
    repo = _persist_repo(url="https://github.com/octocat/proj")

    build_profile([repo], now=NOW)

    assert _links_for(repo.id) == []


def test_reprocess_clears_extracted_links_but_keeps_manual_ones(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.profile.build.extract_skills_from_repo", lambda repo: [])
    monkeypatch.setattr("app.profile.build.extract_links_from_repo", lambda repo: [])
    repo = _persist_repo()
    build_profile([repo], now=NOW)  # gets it out of "pending" so reprocess is meaningful

    db = get_db()
    db.add(
        ProjectLink(repo_id=repo.id, label="Live Demo", url="https://example.com", source="manual")
    )
    db.commit()
    db.close()

    monkeypatch.setattr(
        "app.profile.build.extract_links_from_repo",
        lambda repo: [LinkClaim(label="Docs", url="https://docs.example.com")],
    )
    reprocess_repo(repo.id, now=NOW)

    links = {(link.label, link.source) for link in _links_for(repo.id)}
    assert links == {("Live Demo", "manual"), ("Docs", "readme_extracted")}
