import json

import pytest

from app.evals import public
from app.evals.splits import DEV, TEST, split_for


def _row(**overrides):
    row = {
        "job_id": 1,
        "company_name": "Acme",
        "title": "Backend Engineer",
        "location": "Austin, TX",
        "description": "Build APIs in Python.",
        "min_salary": None,
        "max_salary": None,
        "med_salary": None,
        "formatted_work_type": "Full-time",
        "remote_allowed": None,
    }
    return {**row, **overrides}


def test_split_is_stable_and_roughly_seventy_thirty():
    keys = [f"linkedin-{n}" for n in range(2000)]
    assert [split_for(k) for k in keys] == [split_for(k) for k in keys]
    share = sum(split_for(k) == TEST for k in keys) / len(keys)
    assert 0.66 < share < 0.74
    assert {split_for(k) for k in keys} == {DEV, TEST}


def test_header_fields_come_from_the_form():
    expected = public.expected_fields(_row())
    assert expected["company"] == "Acme"
    assert expected["title"] == "Backend Engineer"
    assert expected["location"] == "Austin, TX"


def test_salary_scored_when_the_description_states_both_ends():
    row = _row(min_salary="120000.0", max_salary="150000.0",
               description="Pay: $120,000 - $150,000 per year.")
    assert public.expected_fields(row)["salary_range"] == "120000-150000"


def test_salary_matches_k_and_hourly_forms():
    assert public.expected_fields(_row(
        min_salary="120000", max_salary="150000", description="Range $120k to $150k."
    ))["salary_range"] == "120000-150000"
    assert public.expected_fields(_row(
        min_salary="25.5", max_salary="30", description="$25.50 - $30.00 an hour"
    ))["salary_range"] == "25.5-30"


def test_salary_on_form_but_not_in_text_is_not_scored():
    row = _row(min_salary="120000", max_salary="150000", description="Great benefits.")
    assert public.expected_fields(row)["salary_range"] is None


def test_no_salary_anywhere_checks_for_invention():
    assert public.expected_fields(_row())["salary_range"] == ""


def test_dollar_amount_without_form_salary_is_not_scored():
    row = _row(description="We offer a $2,000 signing bonus.")
    assert public.expected_fields(row)["salary_range"] is None


def test_employment_type_scored_only_when_the_text_says_it():
    assert public.expected_fields(_row())["employment_type"] is None
    stated = _row(description="This is a full-time role building APIs.")
    assert public.expected_fields(stated)["employment_type"] == "Full-time"
    other = _row(formatted_work_type="Temporary", description="Temporary full-time role.")
    assert public.expected_fields(other)["employment_type"] is None


def test_remote_scored_only_when_form_and_text_agree():
    assert public.expected_fields(_row(remote_allowed="1.0"))["work_mode"] is None
    both = _row(remote_allowed="1.0", description="Fully remote team.")
    assert public.expected_fields(both)["work_mode"] == "Remote"
    assert public.expected_fields(_row(description="Fully remote."))["work_mode"] is None


def test_posting_text_leaves_salary_and_type_out_of_the_header():
    text = public.posting_text(_row(min_salary="1", max_salary="2"))
    assert text.splitlines()[:3] == ["Backend Engineer", "Acme", "Austin, TX"]
    assert "Full-time" not in text


@pytest.fixture
def public_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "PUBLIC_DIR", tmp_path)
    monkeypatch.setattr(public, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache" / "extractions").mkdir(parents=True)
    return tmp_path


def _write_posting(directory, job_id, **overrides):
    path = directory / "cache" / f"linkedin-{job_id}.json"
    path.write_text(json.dumps(_row(job_id=job_id, **overrides)))


def test_build_jobs_keeps_one_split(public_dir):
    ids = range(1, 41)
    for job_id in ids:
        _write_posting(public_dir, job_id)
    sources = {"postings": [{"id": n, "group": "general"} for n in ids]}
    test = public.build_jobs(TEST, sources)
    dev = public.build_jobs(DEV, sources)
    assert len(test) + len(dev) == 40
    assert all(split_for(j.key) == TEST for j in test)
    assert len(public.build_jobs(None, sources)) == 40


def test_build_jobs_reports_a_missing_cache(public_dir):
    from app.evals.real import CacheMissingError

    with pytest.raises(CacheMissingError):
        public.build_jobs(None, {"postings": [{"id": 5, "group": "general"}]})


def test_retrieval_jobs_search_with_the_saved_extraction(public_dir):
    key = "linkedin-7"
    (public_dir / "retrieval_labels.yaml").write_text(
        f"{key}: [Python, Docker]\nlinkedin-8: []\n"
    )
    for job_id in (7, 8):
        extraction = {
            "title": "Backend Engineer", "company": "Acme",
            "role_summary": "Builds APIs.",
            "skills_required": [{"skill": "Python", "level": ""}],
        }
        (public_dir / "cache" / "extractions" / f"linkedin-{job_id}.json").write_text(
            json.dumps(extraction)
        )
    sources = {"postings": [{"id": 7, "group": "software"}, {"id": 8, "group": "software"},
                            {"id": 9, "group": "general"}]}
    jobs = {j.key: j for j in public.build_retrieval_jobs(None, sources)}
    assert set(jobs) == {"linkedin-7", "linkedin-8"}
    job = jobs[key]
    assert job.for_personas == ["real-portfolio"]
    assert job.match_skills == ["Python", "Docker"]
    assert job.query_text() == "Backend Engineer at Acme\nBuilds APIs.\nSkills: Python"
    assert jobs["linkedin-8"].for_personas == []


def test_retrieval_jobs_need_the_extraction(public_dir):
    from app.evals.real import CacheMissingError

    (public_dir / "retrieval_labels.yaml").write_text("linkedin-7: [Python]\n")
    with pytest.raises(CacheMissingError):
        public.build_retrieval_jobs(None, {"postings": [{"id": 7, "group": "software"}]})


@pytest.mark.parametrize(
    "description",
    ["Direct hire - 160k a year.", "Rate- 50HR ON 1099", "Pay rate 13.07 / Hour", "$20 hourly"],
)
def test_pay_stated_without_a_form_salary_is_not_scored(description):
    assert public.expected_fields(_row(description=description))["salary_range"] is None


def test_a_single_rate_is_one_number():
    row = _row(min_salary="34.49", max_salary="34.49", description="Pay: $34.49 per hour.")
    assert public.expected_fields(row)["salary_range"] == "34.49"
