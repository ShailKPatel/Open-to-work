import pytest

from app.profile.experience_years import ExperienceYears, parse_experience


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3+ years", ExperienceYears(3, None)),
        ("2-4 yrs", ExperienceYears(2, 4)),
        ("3 to 5 years of experience", ExperienceYears(3, 5)),
        ("Minimum 5 years", ExperienceYears(5, None)),
        ("up to 2 years", ExperienceYears(0, 2)),
        ("6 months", ExperienceYears(0.5, None)),
        ("0 years", ExperienceYears(0, 0)),
        ("No experience required", ExperienceYears(0, 0)),
        ("Freshers welcome", ExperienceYears(0, 0)),
    ],
)
def test_parses_stated_years(text, expected):
    assert parse_experience(text) == expected


def test_seniority_stands_in_when_no_years():
    assert parse_experience("", "Senior") == ExperienceYears(5, None, estimated=True)
    assert parse_experience("Strong communication", "Intern") == ExperienceYears(
        0, None, estimated=True
    )
    # Stated years win over the label.
    assert parse_experience("2+ years", "Senior") == ExperienceYears(2, None)


def test_nothing_usable():
    assert parse_experience("", "") is None
    assert parse_experience(None) is None
