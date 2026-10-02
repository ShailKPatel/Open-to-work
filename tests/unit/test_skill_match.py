"""app/resume_build/skill_match.py: exact and related skill matching, the
match percentage built on them, and the grounding of the model's
keep/drop review. embed() and complete() are stubbed: what is under test
is the tiering and the arithmetic, not the embedding model's judgment.
"""

from unittest.mock import MagicMock

from app.resume_build.skill_match import (
    coverage_pct,
    match_requirements,
    review_related,
    suggest_related,
)

# Hand-made unit vectors: "Golang" sits on top of "Go", "MySQL" close to
# "PostgreSQL" but not as close, everything else far from both.
_VECTORS = {
    "Go": [1.0, 0.0, 0.0],
    "Golang": [1.0, 0.0, 0.0],
    "PostgreSQL": [0.0, 1.0, 0.0],
    "MySQL": [0.0, 0.75, 0.66],
    "Cooking": [0.0, 0.0, 1.0],
    "Python": [0.0, 0.0, 1.0],
    "C++": [0.0, 1.0, 0.0],
    "C": [1.0, 0.0, 0.0],
}


def _stub_embed(monkeypatch):
    monkeypatch.setattr(
        "app.resume_build.skill_match.embed",
        lambda texts: [_VECTORS.get(t, [0.0, 0.0, 1.0]) for t in texts],
    )


def test_exact_match_folds_spelling(monkeypatch):
    _stub_embed(monkeypatch)
    matches = match_requirements(["Node.js", "C++"], ["nodejs", "C"])
    assert [(m.kind, m.have) for m in matches] == [("exact", "nodejs"), ("missing", None)]


def test_related_match_counts_half(monkeypatch):
    _stub_embed(monkeypatch)
    matches = match_requirements(["Python", "Go", "PostgreSQL"], ["Python", "Golang", "Cooking"])
    assert [m.kind for m in matches] == ["exact", "related", "missing"]
    assert matches[1].have == "Golang"
    # 1 exact + 0.5 related out of 3
    assert coverage_pct(matches) == 50


def test_no_required_skills_gives_no_percentage():
    assert coverage_pct([]) is None


def test_embedding_failure_leaves_exact_matching_standing(monkeypatch):
    def _boom(texts):
        raise RuntimeError("model not downloaded")

    monkeypatch.setattr("app.resume_build.skill_match.embed", _boom)
    matches = match_requirements(["Python", "Go"], ["Python", "Golang"])
    assert [m.kind for m in matches] == ["exact", "missing"]


def test_suggestions_skip_exact_matches_and_lower_the_bar(monkeypatch):
    _stub_embed(monkeypatch)
    # MySQL is 0.75 from PostgreSQL: under the automatic bar, over the
    # suggestion bar, so it is offered for review rather than counted.
    suggestions = suggest_related(
        ["PostgreSQL", "Python"], ["MySQL", "Python", "Golang"], exclude={"Python"}
    )
    assert suggestions == {"MySQL": ["PostgreSQL"]}
    assert match_requirements(["PostgreSQL"], ["MySQL"])[0].kind == "missing"


def test_review_drops_verdicts_for_skills_it_was_not_given(monkeypatch):
    response = MagicMock()
    response.parsed = {
        "verdicts": [
            {"skill": "mysql", "keep": True, "reason": "Same kind of database."},
            {"skill": "Kubernetes", "keep": True, "reason": "Invented."},
            {"skill": "Cooking", "keep": False, "reason": "Unrelated field."},
        ]
    }
    fake = MagicMock(return_value=response)
    monkeypatch.setattr("app.resume_build.skill_match.complete", fake)

    verdicts = review_related(
        1, "Backend Engineer", ["PostgreSQL"], {"MySQL": ["PostgreSQL"], "Cooking": []}
    )

    assert set(verdicts) == {"MySQL", "Cooking"}
    assert verdicts["MySQL"].keep is True
    assert verdicts["Cooking"].keep is False
    assert fake.call_args.kwargs["purpose"] == "resume_skill_review"


def test_review_with_nothing_to_judge_makes_no_call(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr("app.resume_build.skill_match.complete", fake)
    assert review_related(1, "Engineer", ["Go"], {}) == {}
    fake.assert_not_called()
