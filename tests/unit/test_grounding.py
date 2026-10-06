"""app/resume_build/grounding.py: the number and technology checks on
project bullets and the summary. Pure functions, no DB or LLM."""

import datetime as dt

from app.resume_build.grounding import (
    TechVocabulary,
    ground_summary,
    introduces_technology,
    is_grounded_bullet,
    numbers,
    years_of_experience,
)

_VOCAB = TechVocabulary(
    ["Python", "PostgreSQL", "Redis", "Docker", "Docker Compose", "Go", "REST", "Node.js"]
)
_EVIDENCE = (
    "name='api-server' description='REST API serving 40 requests per second' "
    "skills=Python, PostgreSQL"
)
_SKILLS = ["Python", "PostgreSQL"]


def test_numbers_ignore_thousands_separators():
    assert numbers("Handled 10,000 rows in 1.5 s") == {"10000", "1.5"}


def test_bullet_with_evidence_numbers_and_skills_is_kept():
    assert is_grounded_bullet(
        "Served 40 requests per second from Python", _EVIDENCE, _SKILLS, _VOCAB
    )


def test_bullet_with_invented_metric_fails():
    assert not is_grounded_bullet("Cut latency by 70% in Python", _EVIDENCE, _SKILLS, _VOCAB)


def test_bullet_naming_another_projects_tool_fails():
    assert not is_grounded_bullet("Cached responses in Redis", _EVIDENCE, _SKILLS, _VOCAB)


def test_alias_counts_as_the_skill_it_stands_for():
    assert is_grounded_bullet("Stored jobs in Postgres", _EVIDENCE, _SKILLS, _VOCAB)


def test_tool_named_in_the_project_description_is_allowed():
    assert is_grounded_bullet("Designed the REST API", _EVIDENCE, _SKILLS, _VOCAB)


def test_short_and_all_caps_names_match_case_sensitively():
    assert _VOCAB.mentions("Ready to go live, the rest followed") == set()
    assert _VOCAB.mentions("Rewrote it in Go behind a REST layer") == {"go", "rest"}


def test_longest_name_wins_an_overlap():
    assert _VOCAB.mentions("Shipped with Docker Compose") == {"docker compose"}
    assert _VOCAB.mentions("Shipped a Docker-based service on Node.js.") == {"docker", "node.js"}


def test_introduces_technology_compares_against_the_original():
    assert introduces_technology("Built the service in Python", "Built it in Go", _VOCAB)
    assert not introduces_technology("Built the service in Python", "Built it in Python", _VOCAB)


def test_summary_sentence_with_invented_number_is_removed():
    summary, dropped = ground_summary(
        "Backend engineer with 4 years of experience. Cut costs by 30%. Ships Python APIs.",
        [_EVIDENCE],
        years=4.25,
    )
    assert summary == "Backend engineer with 4 years of experience. Ships Python APIs."
    assert dropped == 1


def test_summary_with_nothing_left_is_none():
    assert ground_summary("Grew revenue 300%.", [_EVIDENCE]) == (None, 1)
    assert ground_summary("", [_EVIDENCE]) == (None, 0)


def test_years_span_earliest_start_to_latest_end():
    today = dt.date(2026, 10, 5)
    assert years_of_experience([("jan 2020", "dec 2021"), ("jan 2022", "dec 2023")], today) == 4
    assert years_of_experience([("2024", None)], today) == 2 + 10 / 12
    assert years_of_experience([(None, None)], today) is None
