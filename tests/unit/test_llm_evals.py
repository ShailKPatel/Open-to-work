"""app/evals/llm_evals.py: the scoring behind the LLM evals. Every model
call is faked, so these cover matching, tallying and the throwaway
environment, not model quality (scripts/run_llm_evals.py measures that)."""

import io

import pytest

import app.evals.llm_evals as llm_evals
from app.evals.synthetic import (
    JudgeBullet,
    RedTeamPosting,
    expected_resume,
    load_jobs,
    load_personas,
    render_resume_text,
)
from app.profile.job_extract import JobExtraction, RequiredSkill
from app.profile.resume_extract import (
    ContactClaim,
    EducationClaim,
    ExperienceClaim,
    LinkClaim,
    ResumeExtraction,
)


@pytest.mark.parametrize(
    "name, expected, predicted, ok",
    [
        ("salary_range", "$120,000 - $150,000 per year", "120000-150000 USD", True),
        ("salary_range", "", "", True),
        ("salary_range", "$120k", "$130k", False),
        ("experience_required", "5+ years", "at least 5 years", True),
        ("location", "Berlin, Germany", "Berlin, Germany (Hybrid)", True),
        ("location", "Berlin, Germany", "Munich, Germany", False),
        ("location", "Berlin, Germany", "Berlin", True),
        ("location", "United Kingdom", "UK", True),
        ("location", "London, UK", "London, United Kingdom", True),
        ("location", "United States", "US", True),
        ("location", "New York, NY", "NYC or remote US", False),
        ("location", "Toronto, ON", "Toronto, Canada", True),
        ("location", "New York, NY", "New York, New York, United States", True),
        ("location", "Toronto, ON", "Ottawa, ON", False),
        ("location", "", "", True),
        ("location", "", "Remote", False),
        ("title", "Senior Software Frontend Engineer - Dashboards",
         "Senior Software Frontend Engineer \u2013 Dashboards", True),
        ("company", "Acme", "Acme Inc.", True),
        ("company", "Acme", "", False),
        ("work_mode", "Hybrid", "hybrid", True),
        ("work_mode", "", "Remote", False),
    ],
)
def test_field_matches(name, expected, predicted, ok):
    assert llm_evals.field_matches(name, expected, predicted) is ok


def test_skill_recall_is_case_insensitive_and_none_when_nothing_expected():
    assert llm_evals.skill_recall(["Go", "SQL"], ["go", "Kafka"]) == 0.5
    assert llm_evals.skill_recall([], ["go"]) is None


def _extraction(**overrides) -> JobExtraction:
    fields = dict(
        company="Acme", title="Engineer", location="Berlin, Germany", salary_range="",
        employment_type="", seniority="", experience_required="", work_mode="Hybrid",
        skills_required=[RequiredSkill("Go", "")], other_requirements=[], role_summary="",
    )
    fields.update(overrides)
    return JobExtraction(**fields)


def test_score_job_extraction_skips_null_fields_and_flags_invented_salary():
    score = llm_evals.ExtractionScore(name="t")
    expected = {
        "company": "Acme", "title": "Engineer", "location": "Berlin, Germany",
        "salary_range": "", "employment_type": None, "work_mode": "Remote",
        "seniority": None, "experience_required": None, "skills": ["Go", "SQL"],
    }
    predicted = llm_evals._extraction_dict(_extraction(salary_range="$100k"))

    llm_evals.score_job_extraction("p1", expected, predicted, score)

    assert score.items == 1
    assert "employment_type" not in score.fields
    assert score.fields["company"].accuracy == 1.0
    assert score.fields["work_mode"].accuracy == 0.0
    assert score.invented_salary == ["p1"]
    assert score.mean_skill_recall == 0.5
    assert any("p1.work_mode" in m for m in score.mismatches)


def test_run_job_extraction_counts_failures(monkeypatch):
    jobs = load_jobs()[:2]
    calls = iter([_extraction(), RuntimeError("provider down")])

    def fake(text):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("app.profile.job_extract.extract_job_posting", fake)
    score = llm_evals.run_job_extraction(jobs, "t")

    assert score.items == 2
    assert len(score.failures) == 1 and "provider down" in score.failures[0]


def _red(key, injection, must_not=()):
    return RedTeamPosting(
        key=key, injection=injection, technique="t", text="x",
        expected={"company": "Acme", "title": "Engineer"}, must_not_contain=list(must_not),
    )


def test_score_robustness_resisted_obeyed_and_benign():
    score = llm_evals.RobustnessScore(name="t")
    clean = llm_evals._extraction_dict(_extraction())
    leaked = llm_evals._extraction_dict(_extraction(role_summary="HACKED"))
    wrong = llm_evals._extraction_dict(_extraction(company="Attacker"))

    llm_evals.score_robustness(_red("a", True, ["HACKED"]), clean, score)
    llm_evals.score_robustness(_red("b", True, ["HACKED"]), leaked, score)
    llm_evals.score_robustness(_red("c", True), wrong, score)
    llm_evals.score_robustness(_red("d", False), clean, score)

    assert (score.attacks, score.resisted) == (3, 1)
    assert [o.split(":")[0] for o in score.obeyed] == ["b", "c"]
    assert (score.benign, score.benign_correct) == (1, 1)


def test_run_robustness_counts_failures(monkeypatch):
    def boom(text):
        raise RuntimeError("no key")

    monkeypatch.setattr("app.profile.job_extract.extract_job_posting", boom)
    score = llm_evals.run_robustness([_red("a", True)], "t")
    assert score.failures == ["a: no key"]


def test_render_pdf_is_readable_and_paginates():
    from pypdf import PdfReader

    text = "\n".join(f"Line {i} (with parens) and a backslash \\" for i in range(150))
    reader = PdfReader(io.BytesIO(llm_evals.render_pdf(text, lines_per_page=64)))
    assert len(reader.pages) == 3
    first = reader.pages[0].extract_text()
    assert "Line 0 (with parens)" in first
    assert len(llm_evals.render_pdf("")) > 0


def _resume_from(persona) -> ResumeExtraction:
    want = expected_resume(persona)
    return ResumeExtraction(
        tags=list(want["tags"]),
        target_roles=[],
        summary="",
        experiences=[
            ExperienceClaim(r["company"], r["title"], r["start_date"], r["end_date"])
            for r in want["experiences"]
        ],
        education=[
            EducationClaim(e["institution"], e["degree"], None, e["start_date"], e["end_date"])
            for e in want["education"]
        ],
        contact=ContactClaim(
            name=want["contact"]["name"],
            emails=list(want["contact"]["emails"]),
            phones=list(want["contact"]["phones"]),
            links=[LinkClaim("other", u) for u in want["contact"]["links"]],
        ),
    )


def test_score_resume_extraction_perfect_and_degraded():
    persona = load_personas()[0]
    score = llm_evals.ExtractionScore(name="t")
    llm_evals.score_resume_extraction(
        persona.key, expected_resume(persona), _resume_from(persona), score
    )
    assert all(t.accuracy == 1.0 for t in score.fields.values())
    assert score.mean_skill_recall == 1.0

    degraded = _resume_from(persona)
    degraded.experiences = degraded.experiences[1:]
    degraded.contact.phones = []
    worse = llm_evals.ExtractionScore(name="t")
    llm_evals.score_resume_extraction(persona.key, expected_resume(persona), degraded, worse)
    assert worse.fields["role.found"].accuracy < 1.0
    assert worse.fields["contact.phone"].accuracy == 0.0


def test_run_resume_extraction_sends_a_pdf(monkeypatch):
    personas = load_personas()[:2]
    seen = []

    def fake(pdf_bytes, mime_type):
        seen.append((pdf_bytes[:5], mime_type))
        persona = personas[len(seen) - 1]
        if len(seen) == 2:
            raise RuntimeError("bad read")
        return _resume_from(persona)

    monkeypatch.setattr("app.profile.resume_extract.extract_resume", fake)
    score = llm_evals.run_resume_extraction(personas)

    assert seen[0] == (b"%PDF-", "application/pdf")
    assert score.items == 2 and len(score.failures) == 1
    assert render_resume_text(personas[0])


def test_cohens_kappa():
    assert llm_evals.cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert llm_evals.cohens_kappa([True, False], [False, True]) == -1.0
    assert llm_evals.cohens_kappa([True, True], [True, True]) is None
    assert llm_evals.cohens_kappa([], []) is None


def test_judge_score_metrics():
    score = llm_evals.JudgeScore()
    score.add("a", "faithful", True, True)
    score.add("b", "wrong_tech", False, False)
    score.add("c", "invented_metric", False, True)
    score.add("d", "faithful", True, None)

    assert score.bullets == 4 and score.unusable == ["d"]
    assert score.accuracy == pytest.approx(2 / 3)
    assert score.ungrounded_recall == 0.5
    assert score.by_kind["invented_metric"].accuracy == 0.0
    assert score.disagreements == ["c (invented_metric): labeled False, judged True"]
    assert llm_evals.JudgeScore().accuracy is None
    assert llm_evals.JudgeScore().ungrounded_recall is None


def test_run_judge_validation_uses_the_projects_evidence(monkeypatch):
    from app.evals.synthetic import isolated_environment

    def _encode(texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("app.retrieval.index.embed", _encode)
    personas = load_personas()[:1]
    bullets = [
        JudgeBullet("backend-senior-us", "ledger-sync", "Built a ledger.", True, "faithful"),
        JudgeBullet("backend-senior-us", "ledger-sync", "Built it in Rust.", False, "wrong_tech"),
    ]
    evidence_seen = []

    def fake_judge(evidence, bullet, account_id):
        evidence_seen.append(evidence)
        if "Rust" in bullet:
            raise RuntimeError("rate limited")
        return True

    monkeypatch.setattr("app.evals.groundedness.judge_bullet", fake_judge)
    with isolated_environment():
        score = llm_evals.run_judge_validation(personas, bullets)

    assert "ledger-sync" in evidence_seen[0] and "FastAPI" in evidence_seen[0]
    assert score.accuracy == 1.0
    assert score.unusable == []
    assert len(score.errors) == 1 and "rate limited" in score.errors[0]


def test_llm_environment_stores_only_the_given_key(monkeypatch):
    added = []

    def fake_add(provider, label, credentials, cap):
        added.append((provider, credentials))
        return {"id": 1}, ""

    monkeypatch.setattr("app.core.api_keys_store.add_key", fake_add)
    monkeypatch.setattr("app.core.app_settings.update_llm_settings", lambda **k: None)
    with llm_evals.llm_environment("k-123", provider="gemini") as root:
        assert root.exists()
    assert added == [("gemini", {"api_key": "k-123"})]


def test_llm_environment_rejects_a_bad_key(monkeypatch):
    monkeypatch.setattr(
        "app.core.api_keys_store.add_key", lambda *a, **k: (None, "invalid key")
    )
    with pytest.raises(RuntimeError, match="no API key was accepted.*invalid key"):
        with llm_evals.llm_environment("bad"):
            pass


def test_paced_retries_rate_limits_then_gives_up(monkeypatch):
    from app.core.llm import LLMRateLimitedError

    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing(retries=2, retry_wait=0))
    monkeypatch.setattr(llm_evals, "_keys_back_at", lambda: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise LLMRateLimitedError("slow down")
        return "ok"

    assert llm_evals._paced(flaky) == "ok"
    assert len(attempts) == 3

    def always():
        raise LLMRateLimitedError("slow down")

    with pytest.raises(LLMRateLimitedError):
        llm_evals._paced(always)


def test_paced_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing(retries=5, retry_wait=0))
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("bad json")

    with pytest.raises(ValueError):
        llm_evals._paced(broken)
    assert len(calls) == 1


def test_paced_spaces_calls(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing(min_interval=10.0))
    monkeypatch.setattr(llm_evals.time, "sleep", slept.append)
    llm_evals._paced(lambda: None)
    llm_evals._paced(lambda: None)
    assert slept and slept[-1] > 9


def test_llm_environment_stores_every_key(monkeypatch):
    added = []
    monkeypatch.setattr(
        "app.core.api_keys_store.add_key",
        lambda provider, label, credentials, cap: (added.append(label), ({"id": 1}, ""))[1],
    )
    monkeypatch.setattr("app.core.app_settings.update_llm_settings", lambda **k: None)
    with llm_evals.llm_environment(["a", "b"]):
        pass
    assert added == ["eval run 1", "eval run 2"]


def test_llm_environment_skips_a_rejected_key_when_another_works(monkeypatch):
    def fake_add(provider, label, credentials, cap):
        if credentials["api_key"] == "bad":
            return None, "invalid"
        return {"id": 1}, ""

    monkeypatch.setattr("app.core.api_keys_store.add_key", fake_add)
    monkeypatch.setattr("app.core.app_settings.update_llm_settings", lambda **k: None)
    with llm_evals.llm_environment(["good", "bad"]) as root:
        assert root.exists()


def test_quota_exhausted_when_every_key_cools_down_for_long(monkeypatch):
    import datetime as dt

    from app.core.llm import LLMRateLimitedError

    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing(retries=5, retry_wait=0))
    later = dt.datetime.now(dt.UTC) + dt.timedelta(hours=14)
    monkeypatch.setattr(llm_evals, "_keys_back_at", lambda: later)

    def limited():
        raise LLMRateLimitedError("daily quota")

    with pytest.raises(llm_evals.QuotaExhaustedError, match="every key is out until"):
        llm_evals._paced(limited)


def test_short_cooldown_is_waited_out(monkeypatch):
    import datetime as dt

    from app.core.llm import LLMRateLimitedError

    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing(retries=2, retry_wait=0))
    soon = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
    monkeypatch.setattr(llm_evals, "_keys_back_at", lambda: soon)
    calls = []

    def once_limited():
        calls.append(1)
        if len(calls) == 1:
            raise LLMRateLimitedError("per minute")
        return "ok"

    assert llm_evals._paced(once_limited) == "ok"


def test_keys_back_at_reads_the_isolated_key_rows():
    import datetime as dt

    from app.core.db import ApiKey, get_db, init_db
    from app.evals.synthetic import isolated_environment

    with isolated_environment():
        init_db()
        assert llm_evals._keys_back_at() is None
        db = get_db()
        later = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
        db.add(ApiKey(provider="gemini", label="a", encrypted_credentials="x",
                      masked_preview={}, retry_at=later))
        db.commit()
        db.close()
        assert llm_evals._keys_back_at() == later


def test_a_quota_stop_keeps_partial_results(monkeypatch):
    jobs = load_jobs()[:3]
    calls = []

    def fake(text):
        calls.append(1)
        if len(calls) == 2:
            raise llm_evals.QuotaExhaustedError("every key is out until tomorrow")
        return _extraction()

    monkeypatch.setattr("app.evals.llm_evals._paced", lambda call: call())
    monkeypatch.setattr("app.profile.job_extract.extract_job_posting", fake)
    score = llm_evals.run_job_extraction(jobs, "t")

    assert score.items == 1
    assert score.stopped == "every key is out until tomorrow"
    robust = llm_evals.RobustnessScore(name="t")
    assert robust.stopped is None


def test_eval_keys_prefers_the_environment_then_the_gitignored_file(tmp_path, monkeypatch):
    import scripts.run_llm_evals as runner

    keys_file = tmp_path / ".llm-eval-keys"
    keys_file.write_text("# comment\nk1\n\n k2 \n")
    monkeypatch.setattr(runner, "KEYS_FILE", keys_file)
    monkeypatch.setenv("LIVE_LLM_API_KEY", "e1, e2")
    assert runner.eval_keys() == ["e1", "e2"]
    monkeypatch.delenv("LIVE_LLM_API_KEY")
    assert runner.eval_keys() == ["k1", "k2"]
    monkeypatch.setattr(runner, "KEYS_FILE", tmp_path / "missing")
    assert runner.eval_keys() == []


def test_keys_file_is_gitignored():
    from pathlib import Path

    assert "evals/.llm-eval-keys" in Path(".gitignore").read_text()
    assert "evals/.llm-eval-keys" in Path(".dockerignore").read_text()


def test_paced_rests_every_n_calls(monkeypatch):
    slept = []
    monkeypatch.setattr(
        llm_evals, "PACING", llm_evals.Pacing(pause_every=2, pause_seconds=90.0)
    )
    monkeypatch.setattr(llm_evals, "_calls_made", 0)
    monkeypatch.setattr(llm_evals.time, "sleep", slept.append)
    for _ in range(5):
        llm_evals._paced(lambda: None)
    assert slept.count(90.0) == 2


def test_pacing_from_env(monkeypatch):
    monkeypatch.setattr(llm_evals, "PACING", llm_evals.Pacing())
    monkeypatch.setenv("LIVE_LLM_RPM", "4")
    monkeypatch.setenv("LIVE_LLM_PAUSE_EVERY", "10")
    monkeypatch.delenv("LIVE_LLM_PAUSE_SECONDS", raising=False)
    llm_evals.pacing_from_env()
    assert llm_evals.PACING.min_interval == 15.0
    assert llm_evals.PACING.pause_every == 10
    assert llm_evals.PACING.pause_seconds == 90.0
