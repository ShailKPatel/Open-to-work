"""app/evals/real.py: the PII scrub applied to downloaded postings, the
committed sources and labels agreeing with each other, and building the
portfolio and postings from a cache. CI has no cache (it is gitignored),
so these tests write a small one of their own; the one check that needs the
real downloaded text skips when it is absent."""

import json

import pytest

import app.evals.real as real
from app.evals.synthetic import evidence_rows, golden_pairs
from scripts.fetch_real_eval_data import _html_to_text

_FIELDS = {
    "company", "title", "location", "salary_range", "employment_type", "work_mode",
    "seniority", "experience_required", "skills",
}


@pytest.fixture(scope="module")
def sources():
    return real.load_sources()


@pytest.fixture(scope="module")
def labels():
    return real.load_labels()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Email jane.doe@example.com for help", "Email [email removed] for help"),
        ("Call +1 (415) 555-0134 today", "Call [phone removed] today"),
        ("Call 415.555.0134", "Call [phone removed]"),
        ("Recruiter: Jane Doe", "[contact line removed]"),
        ("Point of contact - Ana Maria Lopez", "[contact line removed]"),
    ],
)
def test_scrub_pii_removes_contact_details(text, expected):
    assert real.scrub_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "$164,200-$229,900 USD",
        "The US base salary range is $220,500 to $245,000.",
        "Experience from 2018 - 2024 counts.",
        "If you need an accommodation, please contact your Recruiting Partner.",
        "The offered salary is dependent upon several factors. Please contact Human Resources.",
        "5+ years of experience",
    ],
)
def test_scrub_pii_leaves_pay_years_and_boilerplate(text):
    assert real.scrub_pii(text) == text


def test_html_to_text_keeps_one_list_item_per_line():
    raw = (
        "&lt;p&gt;About&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Go&lt;/li&gt;"
        "&lt;li&gt;SQL&amp;nbsp;&lt;/li&gt;&lt;/ul&gt;"
    )
    assert _html_to_text(raw) == "About\n- Go\n- SQL"


def test_every_source_has_labels_and_nothing_extra(sources, labels):
    repos = {r["repo"] for r in sources["repos"]}
    postings = {real.posting_key(p["board"], p["id"]) for p in sources["postings"]}
    assert set(labels["repos"]) == repos
    assert set(labels["postings"]) == postings


def test_sources_are_pinned_and_permissively_licensed(sources):
    allowed = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "PostgreSQL"}
    for repo in sources["repos"]:
        assert len(repo["sha"]) == 40, repo["repo"]
        assert repo["license"] in allowed, repo["repo"]
        assert repo["description"], repo["repo"]


def test_posting_labels_have_every_field(labels):
    for key, label in labels["postings"].items():
        assert set(label["expected"]) == _FIELDS, key
        assert isinstance(label["relevant_skills"], list), key
        assert label["summary"].strip(), key


def test_labels_include_off_domain_postings_with_nothing_relevant(labels):
    empty = [k for k, v in labels["postings"].items() if not v["relevant_skills"]]
    assert len(empty) >= 2


def test_committed_files_hold_no_document_text(sources, labels):
    """Only ids and labels are committed; posting and README text stays
    in the gitignored cache."""
    for path in (real.REAL_DIR / "sources.yaml", real.REAL_DIR / "labels.yaml"):
        assert len(path.read_text(encoding="utf-8")) < 20_000
    assert "evals/real/cache/" in (real.REAL_DIR.parents[1] / ".gitignore").read_text()


def _write_cache(cache, sources):
    (cache / "repos").mkdir(parents=True)
    (cache / "postings").mkdir(parents=True)
    for item in sources["repos"]:
        name = item["repo"].replace("/", "__")
        (cache / "repos" / f"{name}.json").write_text(
            json.dumps(
                {
                    "readme": f"# {item['repo']}",
                    "manifests": {"requirements.txt": {"ecosystem": "pip", "dependencies": []}},
                }
            )
        )
    for item in sources["postings"]:
        (cache / "postings" / f"{item['board']}-{item['id']}.json").write_text(
            json.dumps({"text": "posting text"})
        )


def test_build_from_cache_produces_pairs_only_for_relevant_postings(
    tmp_path, monkeypatch, sources, labels
):
    monkeypatch.setattr(real, "CACHE_DIR", tmp_path)
    _write_cache(tmp_path, sources)

    portfolio = real.build_portfolio(sources, labels)
    jobs = real.build_jobs(sources, labels)

    assert portfolio.account_id == real.REAL_ACCOUNT_ID
    assert len(portfolio.repos) == len(sources["repos"])
    assert portfolio.roles == []
    assert len(jobs) == len(sources["postings"])
    off_domain = [j for j in jobs if not j.relevant_skills]
    assert off_domain and all(not j.for_personas for j in off_domain)

    pairs = golden_pairs([portfolio], jobs)
    assert {p.collection for p in pairs} == {"skill_evidence"}
    assert len(pairs) == len(jobs) - len(off_domain)


def test_missing_cache_says_how_to_build_it(tmp_path, monkeypatch, sources, labels):
    monkeypatch.setattr(real, "CACHE_DIR", tmp_path)
    with pytest.raises(real.CacheMissingError, match="fetch_real_eval_data"):
        real.build_portfolio(sources, labels)
    with pytest.raises(real.CacheMissingError, match="posting"):
        real.build_jobs(sources, labels)


def test_read_cached_missing_file_is_none(tmp_path):
    assert real.read_cached(tmp_path / "nope.json") is None


@pytest.mark.skipif(
    not (real.CACHE_DIR / "repos").exists(), reason="real-text cache not downloaded"
)
def test_every_relevant_skill_names_real_portfolio_evidence(sources, labels):
    """A typo in a relevant skill would silently shrink its posting's
    relevant set; with the downloaded cache, check every one resolves."""
    portfolio = real.build_portfolio(sources, labels)
    skills = {r.skill.casefold() for r in evidence_rows(portfolio)}
    for key, label in labels["postings"].items():
        missing = [s for s in label["relevant_skills"] if s.casefold() not in skills]
        assert not missing, (key, missing)
