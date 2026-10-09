"""app/evals/synthetic.py and the data under evals/synthetic/: the files
load and agree with each other, the derived golden set is sane, and a full
seed-and-score run happens in a throwaway environment that leaves nothing
behind and never opens the configured database."""

import os
import re
from collections import Counter
from pathlib import Path

import pytest

import app.evals.synthetic as synthetic
from app.core.settings import get_settings
from app.evals.golden import SCORABLE_COLLECTIONS
from app.ingest.github.manifests import MANIFEST_FILENAMES


@pytest.fixture(scope="module")
def personas():
    return synthetic.load_personas()


@pytest.fixture(scope="module")
def jobs():
    return synthetic.load_jobs()


def _fake_embed(monkeypatch):
    """Two-dimensional vectors keyed on one word, the same stand-in
    test_evals_run.py uses, so no embedding model is loaded."""

    def _encode(texts):
        return [[1.0, 0.0] if "python" in t.lower() else [0.0, 1.0] for t in texts]

    import app.retrieval.index as index_module
    import app.retrieval.search as search_module

    monkeypatch.setattr(index_module, "embed", _encode)
    monkeypatch.setattr(search_module, "embed", _encode)


def test_personas_have_unique_keys_and_ids_in_the_synthetic_range(personas):
    assert len(personas) >= 10
    assert len({p.key for p in personas}) == len(personas)
    assert len({p.account_id for p in personas}) == len(personas)
    for persona in personas:
        assert synthetic.ACCOUNT_ID_MIN <= persona.account_id <= synthetic.ACCOUNT_ID_MAX


def test_persona_ids_fit_their_blocks(personas):
    limits = synthetic.block_limits()
    for persona in personas:
        usage = synthetic.block_usage(persona)
        assert all(usage[k] <= limits[k] for k in usage), (persona.key, usage)


def test_contact_details_are_reserved_or_placeholder(personas):
    """Every address is on an example domain and every link handle is
    marked as a demo, so none of it can reach a real inbox or profile."""
    for persona in personas:
        assert re.search(r"@example\.[a-z.]+$", persona.email), persona.email
        for link in persona.links:
            assert "demo-" in link.url or "example" in link.url, link.url


def test_manifest_files_are_ones_ingestion_recognises(personas):
    for persona in personas:
        for repo in persona.repos:
            assert set(repo.manifests) <= set(MANIFEST_FILENAMES), (persona.key, repo.key)


def test_repo_evidence_uses_real_manifest_mapping_and_prose_fallback(personas):
    by_key = {p.key: p for p in personas}
    ledger = next(r for r in by_key["backend-senior-us"].repos if r.key == "ledger-sync")
    evidence = {skill: kind for skill, kind, _ in synthetic.repo_evidence(ledger)}
    assert evidence["FastAPI"] == "declared_dependency"
    assert evidence["PostgreSQL"] == "declared_dependency"
    assert evidence["Docker"] == "readme_described"

    no_readme = next(r for r in by_key["devops-sre-uk"].repos if r.key == "tf-aws-baseline")
    kinds = {kind for _, kind, _ in synthetic.repo_evidence(no_readme)}
    assert kinds == {"description_described"}


def test_dataset_covers_a_spread_of_profiles(personas):
    styles = {p.resume_style for p in personas}
    date_styles = {p.date_style for p in personas}
    assert styles == {"classic", "skills_first", "compact"}
    assert date_styles == {"mon_year", "numeric", "year_only"}
    covers = {c for p in personas for c in p.covers}
    for needed in ("entry-level", "senior", "manager", "career-changer", "phd", "india", "europe"):
        assert needed in covers


def test_jobs_cover_the_extraction_edge_cases(jobs):
    assert len(jobs) >= 25
    assert len({j.key for j in jobs}) == len(jobs)
    modes = {j.expected["work_mode"] for j in jobs}
    types = {j.expected["employment_type"] for j in jobs}
    assert {"Remote", "Hybrid", "On-site", ""} <= modes
    assert {"Full-time", "Part-time", "Contract", "Internship", "Freelance", ""} <= types
    assert any(not j.for_personas for j in jobs), "needs extraction-only postings"
    assert any(j.expected["salary_range"] == "" for j in jobs)


def test_every_job_names_existing_personas_and_yields_pairs(personas, jobs):
    """A typo in a skill name or persona key would silently drop the pair;
    this makes it loud instead."""
    keys = {p.key for p in personas}
    pair_ids = {p.id for p in synthetic.golden_pairs(personas, jobs)}
    for job in jobs:
        for persona_key in job.for_personas:
            assert persona_key in keys, (job.key, persona_key)
            assert any(pid.startswith(f"{job.key}@{persona_key}-") for pid in pair_ids), (
                job.key,
                persona_key,
            )


def test_golden_pairs_reach_target_size_and_point_at_real_ids(personas, jobs):
    pairs = synthetic.golden_pairs(personas, jobs)
    assert len(pairs) >= 50
    assert {p.collection for p in pairs} == set(SCORABLE_COLLECTIONS)
    assert len({p.id for p in pairs}) == len(pairs)
    by_account = {p.account_id: p for p in personas}
    for pair in pairs:
        persona = by_account[pair.account_id]
        if pair.collection == "skill_evidence":
            valid = {r.point_id for r in synthetic.evidence_rows(persona)}
        else:
            valid = {pid for pid, _, _ in synthetic.point_ids(persona)}
        assert set(pair.relevant_ids) <= valid
        assert pair.relevant_ids


def test_golden_pairs_every_persona_is_queried(personas, jobs):
    counts = Counter(p.account_id for p in synthetic.golden_pairs(personas, jobs))
    assert set(counts) == {p.account_id for p in personas}


def test_query_text_matches_job_posting_text_shape(jobs):
    job = next(j for j in jobs if j.key == "payments-senior-backend")
    lines = job.query_text().split("\n")
    assert lines[0] == "Senior Backend Engineer - Payments Core at Brightline Pay"
    assert lines[-1].startswith("Skills: Python, Go")


def test_judge_bullets_name_real_repos_and_include_both_labels(personas):
    bullets = synthetic.load_judge_bullets()
    repos = {(p.key, r.key) for p in personas for r in p.repos}
    for bullet in bullets:
        assert (bullet.persona, bullet.repo) in repos, (bullet.persona, bullet.repo)
    kinds = Counter(b.kind for b in bullets)
    assert {"faithful", "paraphrase", "wrong_tech", "invented_metric",
            "invented_feature", "inflated_scope"} <= set(kinds)
    grounded = [b.grounded for b in bullets]
    assert grounded.count(True) >= 15 and grounded.count(False) >= 15
    for bullet in bullets:
        assert bullet.grounded == (bullet.kind in {"faithful", "paraphrase"})


def test_redteam_has_attacks_and_benign_controls():
    postings = synthetic.load_redteam()
    attacks = [p for p in postings if p.injection]
    benign = [p for p in postings if not p.injection]
    assert len(attacks) >= 10 and len(benign) >= 3
    assert len({p.technique for p in attacks}) >= 8
    assert any("​" in p.text for p in attacks)
    assert any("‮" in p.text for p in attacks)


@pytest.mark.parametrize("style", ["mon_year", "numeric", "year_only"])
def test_format_date_styles(style):
    expected = {"mon_year": "Mar 2021", "numeric": "03/2021", "year_only": "2021"}[style]
    assert synthetic._format_date("mar 2021", style) == expected
    assert synthetic._format_date("2019", style) == "2019"
    assert synthetic._format_date(None, style) == "Present"


def test_rendered_resume_contains_everything_expected(personas):
    for persona in personas:
        text = synthetic.render_resume_text(persona)
        expected = synthetic.expected_resume(persona)
        assert persona.full_name in text
        assert persona.email in text and persona.phone in text
        for role in expected["experiences"]:
            assert role["company"] in text and role["title"] in text
        for school in expected["education"]:
            assert school["institution"] in text
        if persona.resume_style == "skills_first":
            assert text.index("SKILLS") < text.index("EXPERIENCE")
        else:
            assert text.index("EXPERIENCE") < text.index("SKILLS")


def test_profile_readme_is_not_listed_as_a_project(personas):
    student = next(p for p in personas if p.key == "new-grad-student")
    text = synthetic.render_resume_text(student)
    assert "Profile README" not in text


def test_seed_refuses_outside_the_isolated_environment(personas):
    with pytest.raises(RuntimeError, match="isolated_environment"):
        synthetic.seed(personas)


def test_isolated_environment_cannot_nest():
    with synthetic.isolated_environment():
        with pytest.raises(RuntimeError, match="already active"):
            with synthetic.isolated_environment():
                pass


def test_isolated_environment_guards_against_a_redirected_database(monkeypatch, personas):
    with synthetic.isolated_environment():
        monkeypatch.setenv("DATABASE_URL", "sqlite:///somewhere/else.db")
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="outside the isolated directory"):
            synthetic.seed(personas)


def test_full_run_scores_every_persona_and_leaves_nothing_behind(
    tmp_path, monkeypatch, personas, jobs
):
    _fake_embed(monkeypatch)
    real_db = tmp_path / "real.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{real_db}")
    monkeypatch.setenv("QDRANT_URL", "http://real-qdrant.invalid:6333")
    get_settings.cache_clear()

    created: list[Path] = []
    original = synthetic.isolated_environment

    from contextlib import contextmanager

    @contextmanager
    def _spy():
        with original() as root:
            created.append(root)
            assert (root / "golden").is_dir()
            yield root

    monkeypatch.setattr(synthetic, "isolated_environment", _spy)

    reports = synthetic.run_retrieval_eval(personas, jobs)

    assert [r.account_id for r in reports] == [p.account_id for p in personas]
    assert sum(r.pairs_scored for r in reports) == len(synthetic.golden_pairs(personas, jobs))
    assert all(r.llm_calls == 0 for r in reports)
    assert all(r.groundedness is None for r in reports)
    # The throwaway directory is gone, the configured database was never
    # created, and the settings point back where they were.
    assert created and not created[0].exists()
    assert not real_db.exists()
    assert os.environ["DATABASE_URL"] == f"sqlite:///{real_db}"
    assert os.environ["QDRANT_URL"] == "http://real-qdrant.invalid:6333"


def test_isolated_environment_restores_unset_variables(monkeypatch):
    monkeypatch.delenv("EVALS_GOLDEN_DIR", raising=False)
    with synthetic.isolated_environment() as root:
        assert os.environ["EVALS_GOLDEN_DIR"] == str(root / "golden")
    assert "EVALS_GOLDEN_DIR" not in os.environ


def test_isolated_environment_uses_a_throwaway_secret_key(monkeypatch):
    import app.core.crypto as crypto

    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    with synthetic.isolated_environment():
        token = crypto.encrypt("secret")
        assert crypto.decrypt(token) == "secret"
        inside = os.environ["APP_SECRET_KEY"]
    assert "APP_SECRET_KEY" not in os.environ
    assert inside
    assert crypto._fernet is None


def test_sigterm_inside_isolated_environment_still_cleans_up():
    import signal

    before = signal.getsignal(signal.SIGTERM)
    created = []
    with pytest.raises(SystemExit):
        with synthetic.isolated_environment() as root:
            created.append(root)
            os.kill(os.getpid(), signal.SIGTERM)
    assert not created[0].exists()
    assert signal.getsignal(signal.SIGTERM) == before


def test_isolated_environment_off_the_main_thread_skips_the_handler():
    import threading

    results = []

    def _run():
        with synthetic.isolated_environment() as root:
            results.append(root.exists())

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()
    assert results == [True]


def test_hard_judge_bullets_are_consistent(personas):
    bullets = synthetic.load_judge_bullets(synthetic.SYNTHETIC_DIR / "judge_bullets_hard.yaml")
    repos = {(p.key, r.key) for p in personas for r in p.repos}
    grounded_kinds = {"grounded_inference", "generalization"}
    ungrounded_kinds = {"adjacent_tech", "unstated_number", "role_inflation", "scope_shift"}
    assert len(bullets) >= 25
    for bullet in bullets:
        assert (bullet.persona, bullet.repo) in repos
        assert bullet.kind in grounded_kinds | ungrounded_kinds
        assert bullet.grounded == (bullet.kind in grounded_kinds)
    assert {b.kind for b in bullets} == grounded_kinds | ungrounded_kinds


def test_seeding_twice_in_one_environment_skips_existing_personas(monkeypatch, personas):
    _fake_embed(monkeypatch)
    with synthetic.isolated_environment():
        first = synthetic.seed(personas[:2])
        second = synthetic.seed(personas[:3])
    assert first["accounts"] == 2
    assert second["accounts"] == 1


def test_scope_judge_bullets_are_balanced_and_consistent(personas):
    bullets = synthetic.load_judge_bullets(synthetic.SYNTHETIC_DIR / "judge_bullets_scope.yaml")
    repos = {(p.key, r.key) for p in personas for r in p.repos}
    for bullet in bullets:
        assert (bullet.persona, bullet.repo) in repos
        assert bullet.grounded == (bullet.kind == "grounded_control")
    kinds = [b.kind for b in bullets]
    assert kinds.count("scope_inflation") == kinds.count("grounded_control") >= 8
