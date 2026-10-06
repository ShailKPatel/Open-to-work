import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.core.db as db_module
from app.core.db import ProjectLink, Repository, SkillEvidence, get_db, init_db
from app.core.llm import BudgetExceededError, LLMProviderError, LLMRateLimitedError
from app.core.settings import get_settings
from app.profile.build import (
    build_profile,
    rebuild_manifest_evidence,
    reprocess_repo,
    skill_evidence_for_repos,
)
from app.profile.claims import LinkClaim, SkillClaim
from app.profile.extract import NoSourceTextError, RepoFacts

NOW = dt.datetime(2026, 8, 24, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _stub_extraction(monkeypatch):
    """Extraction answers nothing unless a test says otherwise. Without
    this, every build_profile() call in this file would fall through to the
    real extraction and attempt a real LLM dispatch (blocking on network /
    hanging with no API key configured is a much worse test failure mode
    than a wrong assertion).

    Two functions, because build.py takes two routes to the same answer:
    prefetch_repo_facts covers a whole batch in one call and
    extract_repo_facts covers whatever that missed. Stubbed empty here so
    tests using _stub_skills/_stub_links below exercise the per-repo route;
    test_batched_prefetch_* cover the batched one.
    """
    monkeypatch.setattr("app.profile.build.prefetch_repo_facts", lambda repos: {})
    monkeypatch.setattr("app.profile.build.extract_repo_facts", lambda repo, **_: RepoFacts())


def _stub_skills(monkeypatch, fn):
    """Point build.py's extraction call at `fn`, a repo -> skill claims
    callable (or one that raises, for the failure-path tests). Skills and
    links come back from one call now (see app/profile/extract.py), so a
    test that only cares about skills gets the links half empty rather than
    stubbing a second function."""
    monkeypatch.setattr(
        "app.profile.build.extract_repo_facts", lambda repo, **_: RepoFacts(skills=fn(repo))
    )


def _stub_links(monkeypatch, fn):
    """_stub_skills' twin for the tests that care about the links half."""
    monkeypatch.setattr(
        "app.profile.build.extract_repo_facts", lambda repo, **_: RepoFacts(links=fn(repo))
    )


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

    db_module.reset_engine()
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
    _stub_skills(
        monkeypatch,
        lambda repo: [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)],
    )
    repo = _persist_repo()

    build_profile([repo], now=NOW)

    evidence = {e.skill: e.evidence_type for e in skill_evidence_for_repos([repo.id])}
    assert evidence == {"FastAPI": "declared_dependency", "React": "readme_described"}


def test_build_profile_persists_extracted_status(tmp_path, monkeypatch):
    """Mutating a detached repo object's
    status doesn't persist unless it's re-attached to the commit session."""
    _reset_db(tmp_path)
    _stub_skills(monkeypatch, lambda repo: [])
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

    _stub_skills(monkeypatch, raise_no_source)
    repo = _persist_repo(readme=None, description=None)

    build_profile([repo], now=NOW)

    reloaded = _fresh(repo.id)
    assert reloaded.skill_extraction_status == "no_signal"
    assert reloaded.skill_extraction_error is None
    # manifest-based evidence still gets written even with no LLM signal
    evidence = skill_evidence_for_repos([repo.id])
    assert {e.skill for e in evidence} == {"FastAPI"}


def test_llm_failure_on_one_repo_does_not_lose_other_repos(tmp_path, monkeypatch):
    """API key/budget runs out
    mid-batch. One repo fails; the rest still get processed and committed."""
    _reset_db(tmp_path)

    def flaky_extract(repo):
        if repo.full_name == "octocat/bad":
            raise RuntimeError("budget exceeded")
        return [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)]

    _stub_skills(monkeypatch, flaky_extract)
    good = _persist_repo(github_id=1, full_name="octocat/good")
    bad = _persist_repo(github_id=2, full_name="octocat/bad")

    build_profile([good, bad], now=NOW)

    good_reloaded = _fresh(good.id)
    bad_reloaded = _fresh(bad.id)
    assert good_reloaded.skill_extraction_status == "extracted"
    assert bad_reloaded.skill_extraction_status == "failed"
    assert "budget exceeded" in bad_reloaded.skill_extraction_error

    # good repo's evidence survived despite bad repo's failure
    good_evidence = {e.skill for e in skill_evidence_for_repos([good.id])}
    assert good_evidence == {"FastAPI", "React"}
    # bad repo still got its manifest-based (free, no-LLM) evidence
    bad_evidence = {e.skill for e in skill_evidence_for_repos([bad.id])}
    assert bad_evidence == {"FastAPI"}


def test_rate_limit_mid_batch_stops_and_leaves_rest_untouched(tmp_path, monkeypatch):
    """A 429 on repo 2 of 3 gives repo 2 a clean "rate_limited" status and
    stops the batch: repo 3 is never attempted and stays "pending"."""
    _reset_db(tmp_path)
    calls = []

    def hits_limit_on_second(repo):
        calls.append(repo.full_name)
        if repo.full_name == "octocat/b":
            raise LLMRateLimitedError("Gemini is limiting requests from this key right now.")
        return [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)]

    _stub_skills(monkeypatch, hits_limit_on_second)
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
    # app/core/llm.py's own readable message, plus why the batch stopped
    assert b_reloaded.skill_extraction_error.startswith("Gemini is limiting requests")
    assert "later" in b_reloaded.skill_extraction_error.lower()
    assert c_reloaded.skill_extraction_status == "pending"  # untouched, not "failed"


def test_reprocess_rate_limit_leaves_out_the_batch_note(tmp_path, monkeypatch):
    """A single Reprocess has no remaining projects, so its error is the
    provider message alone, without the batch's "processing stopped" note."""
    _reset_db(tmp_path)

    def busy(repo):
        raise LLMRateLimitedError("The provider is too busy to answer right now.")

    _stub_skills(monkeypatch, busy)
    repo = _persist_repo(github_id=1, full_name="octocat/a")

    reprocessed = reprocess_repo(repo.id, now=NOW)

    assert reprocessed.skill_extraction_status == "rate_limited"
    assert reprocessed.skill_extraction_error == "The provider is too busy to answer right now."


def test_budget_exceeded_mid_batch_also_stops_the_batch(tmp_path, monkeypatch):
    _reset_db(tmp_path)

    def hits_budget_cap(repo):
        raise BudgetExceededError("monthly budget $20.00 reached")

    _stub_skills(monkeypatch, hits_budget_cap)
    a = _persist_repo(github_id=1, full_name="octocat/a")
    b = _persist_repo(github_id=2, full_name="octocat/b")

    build_profile([a, b], now=NOW)

    assert _fresh(a.id).skill_extraction_status == "rate_limited"
    assert _fresh(b.id).skill_extraction_status == "pending"  # never attempted


def test_running_out_of_keys_mid_batch_stops_it_the_same_way(tmp_path, monkeypatch):
    """Every stored key rejected is not "this repo is broken": the repos
    after it would fail identically. It reaches here as an
    LLMProviderError rather than a 429, so the stop has to be decided by
    app/core/llm.py's is_out_of_keys(), not by the exception type. Repos
    never attempted stay pending, so the next run continues from here."""
    _reset_db(tmp_path)

    def every_key_rejected(repo):
        error = LLMProviderError(
            "All 2 OpenAI keys failed on this request:\n"
            "- Personal: OpenAI rejected the API key.\n"
            "- Work: OpenAI is limiting requests from this key right now."
        )
        error.blames_key = True
        raise error

    _stub_skills(monkeypatch, every_key_rejected)
    a = _persist_repo(github_id=1, full_name="octocat/a")
    b = _persist_repo(github_id=2, full_name="octocat/b")

    build_profile([a, b], now=NOW)

    a_reloaded = _fresh(a.id)
    assert a_reloaded.skill_extraction_status == "rate_limited"
    assert "All 2 OpenAI keys failed" in a_reloaded.skill_extraction_error
    assert "run it again once a key is available" in a_reloaded.skill_extraction_error
    assert _fresh(b.id).skill_extraction_status == "pending"  # never attempted


def test_generic_error_mid_batch_does_not_stop_the_batch(tmp_path, monkeypatch):
    """Only rate-limit/budget stops the whole batch; a per-repo
    failure (existing behavior) still lets the rest proceed normally."""
    _reset_db(tmp_path)

    def fails_only_on_b(repo):
        if repo.full_name == "octocat/b":
            raise RuntimeError("malformed response")
        return []

    _stub_skills(monkeypatch, fails_only_on_b)
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

    _stub_skills(monkeypatch, rate_limited_once)
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

    _stub_skills(monkeypatch, counting_extract)
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

    _stub_skills(monkeypatch, fail_once)
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

    _stub_skills(monkeypatch, counting_extract)
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

    _stub_skills(monkeypatch, sequenced_extract)
    repo = _persist_repo()

    build_profile([repo], now=NOW)
    assert {e.skill for e in skill_evidence_for_repos([repo.id])} == {"FastAPI", "React"}

    reprocess_repo(repo.id, now=NOW)
    # React (from the first run) is gone: reprocess clears before rewriting,
    # doesn't just append
    assert {e.skill for e in skill_evidence_for_repos([repo.id])} == {"FastAPI", "Vue"}


def test_reprocess_removes_the_index_points_of_evidence_it_replaces(tmp_path):
    """A skill dropped from a repo must leave Qdrant too, or resume
    building would still find it there."""
    from app.retrieval.index import COLLECTION
    from app.retrieval.vectorstore import get_client

    _reset_db(tmp_path)
    repo = _persist_repo(
        readme=None,
        manifests_json={
            "requirements.txt": {"ecosystem": "pip", "dependencies": ["pandas", "numpy"]}
        },
    )
    build_profile([repo], now=NOW)

    db = get_db()
    db.get(Repository, repo.id).manifests_json = {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["flask"]}
    }
    db.commit()
    db.close()
    reprocess_repo(repo.id, now=NOW)

    points, _ = get_client().scroll(collection_name=COLLECTION, limit=100)
    sql = {(e.id, e.skill) for e in skill_evidence_for_repos([repo.id])}
    assert {(p.id, p.payload["skill"]) for p in points} == sql
    assert {skill for _, skill in sql} == {"Flask"}


def test_reprocess_repo_unknown_id_raises(tmp_path):
    _reset_db(tmp_path)
    try:
        reprocess_repo(999999)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_rebuild_manifest_evidence_replaces_only_manifest_rows(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    calls = []
    _stub_skills(
        monkeypatch,
        lambda repo: calls.append(1)
        or [SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)],
    )
    repo = _persist_repo(
        manifests_json={
            "requirements.txt": {"ecosystem": "pip", "dependencies": ["blinker", "fastapi"]}
        }
    )
    build_profile([repo], now=NOW)

    db = get_db()
    db.add(
        SkillEvidence(
            skill="blinker", repo_id=repo.id, evidence_type="declared_dependency",
            weight=0.5, confidence=1.0,
        )
    )
    db.add(
        SkillEvidence(
            skill="Docker", repo_id=repo.id, evidence_type="manual", weight=0.5, confidence=1.0
        )
    )
    db.commit()
    db.close()

    removed, written = rebuild_manifest_evidence(now=NOW)

    assert (removed, written) == (2, 1)
    assert len(calls) == 1  # no LLM call from the rebuild
    evidence = {(e.skill, e.evidence_type) for e in skill_evidence_for_repos([repo.id])}
    assert evidence == {
        ("FastAPI", "declared_dependency"),
        ("React", "readme_described"),
        ("Docker", "manual"),
    }
    assert _fresh(repo.id).skill_extraction_status == "extracted"


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
    _stub_skills(monkeypatch, lambda repo: [])
    _stub_links(
        monkeypatch,
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
    _stub_skills(monkeypatch, lambda repo: [])
    _stub_links(
        monkeypatch,
        lambda repo: [LinkClaim(label="GitHub", url="https://github.com/octocat/proj")],
    )
    repo = _persist_repo(url="https://github.com/octocat/proj")

    build_profile([repo], now=NOW)

    assert _links_for(repo.id) == []


def test_reprocess_clears_extracted_links_but_keeps_manual_ones(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _stub_skills(monkeypatch, lambda repo: [])
    _stub_links(monkeypatch, lambda repo: [])
    repo = _persist_repo()
    build_profile([repo], now=NOW)  # gets it out of "pending" so reprocess is meaningful

    db = get_db()
    db.add(
        ProjectLink(repo_id=repo.id, label="Live Demo", url="https://example.com", source="manual")
    )
    db.commit()
    db.close()

    _stub_links(
        monkeypatch,
        lambda repo: [LinkClaim(label="Docs", url="https://docs.example.com")],
    )
    reprocess_repo(repo.id, now=NOW)

    links = {(link.label, link.source) for link in _links_for(repo.id)}
    assert links == {("Live Demo", "manual"), ("Docs", "readme_extracted")}


def test_batched_prefetch_answer_is_used_without_a_second_call(tmp_path, monkeypatch):
    """A repo the batched pass already answered costs no call of its own."""
    _reset_db(tmp_path)
    repo_a = _persist_repo(github_id=1, name="a", full_name="octocat/a")
    repo_b = _persist_repo(github_id=2, name="b", full_name="octocat/b")
    monkeypatch.setattr(
        "app.profile.build.prefetch_repo_facts",
        lambda repos: {
            repo_a.id: RepoFacts(
                skills=[
                    SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)
                ],
                links=[LinkClaim(label="Docs", url="https://docs.example.com")],
            ),
            repo_b.id: RepoFacts(
                skills=[SkillClaim(skill="Rust", evidence_type="readme_described", confidence=0.8)]
            ),
        },
    )
    per_repo = MagicMock()
    monkeypatch.setattr("app.profile.build.extract_repo_facts", per_repo)

    build_profile([repo_a, repo_b], now=NOW)

    per_repo.assert_not_called()
    skills = {e.skill for e in skill_evidence_for_repos([repo_a.id, repo_b.id])}
    assert {"React", "Rust"} <= skills
    assert [link.label for link in _links_for(repo_a.id)] == ["Docs"]
    assert _fresh(repo_a.id).skill_extraction_status == "extracted"
    assert _fresh(repo_b.id).skill_extraction_status == "extracted"


def test_repo_the_batched_pass_missed_falls_back_to_its_own_call(tmp_path, monkeypatch):
    """Nothing depends on the batched pass succeeding: a repo it skipped (or
    a group whose call failed) is processed exactly as before."""
    _reset_db(tmp_path)
    repo_a = _persist_repo(github_id=1, name="a", full_name="octocat/a")
    repo_b = _persist_repo(github_id=2, name="b", full_name="octocat/b")
    monkeypatch.setattr(
        "app.profile.build.prefetch_repo_facts",
        lambda repos: {
            repo_a.id: RepoFacts(
                skills=[SkillClaim(skill="React", evidence_type="readme_described", confidence=0.9)]
            )
        },
    )
    asked: list[int] = []

    def _per_repo(repo, **_):
        asked.append(repo.id)
        return RepoFacts(
            skills=[SkillClaim(skill="Rust", evidence_type="readme_described", confidence=0.8)]
        )

    monkeypatch.setattr("app.profile.build.extract_repo_facts", _per_repo)

    build_profile([repo_a, repo_b], now=NOW)

    assert asked == [repo_b.id]
    skills = {e.skill for e in skill_evidence_for_repos([repo_a.id, repo_b.id])}
    assert {"React", "Rust"} <= skills


def test_prefetch_is_skipped_for_a_single_repo(tmp_path, monkeypatch):
    """One repo has nothing to amortize, so the batched prompt's extra
    instructions would cost more than the call they save."""
    _reset_db(tmp_path)
    prefetch = MagicMock(return_value={})
    monkeypatch.setattr("app.profile.build.prefetch_repo_facts", prefetch)
    _stub_skills(monkeypatch, lambda repo: [])
    repo = _persist_repo()

    build_profile([repo], now=NOW)

    prefetch.assert_not_called()


def test_prefetch_only_covers_repos_this_run_will_process(tmp_path, monkeypatch):
    """Already-extracted repos are skipped by the loop, so paying to
    prefetch them would be spend for nothing."""
    _reset_db(tmp_path)
    done = _persist_repo(
        github_id=1, name="done", full_name="octocat/done", skill_extraction_status="extracted"
    )
    pending_a = _persist_repo(github_id=2, name="a", full_name="octocat/a")
    pending_b = _persist_repo(github_id=3, name="b", full_name="octocat/b")
    seen: list[list[int]] = []

    def _prefetch(repos):
        seen.append([r.id for r in repos])
        return {}

    monkeypatch.setattr("app.profile.build.prefetch_repo_facts", _prefetch)
    _stub_skills(monkeypatch, lambda repo: [])

    build_profile([done, pending_a, pending_b], now=NOW)

    assert seen == [[pending_a.id, pending_b.id]]


def test_reprocess_moves_profile_readme_saved_as_project(tmp_path):
    """A "username/username" repo saved before the flag existed is a
    project until something touches it; the Reprocess button is enough."""
    _reset_db(tmp_path)
    repo = _persist_repo(name="octocat", full_name="octocat/octocat", is_profile_readme=False)

    reprocess_repo(repo.id, now=NOW)

    assert _fresh(repo.id).is_profile_readme is True


def test_extraction_leaves_ordinary_repo_a_project(tmp_path):
    _reset_db(tmp_path)
    repo = _persist_repo()

    build_profile([repo], now=NOW)

    assert _fresh(repo.id).is_profile_readme is False
