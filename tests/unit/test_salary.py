import pytest

from app.profile.salary import parse_salary


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.5 lakhs to 1.6 lakh per month", (1_800_000, 1_920_000, "INR")),
        ("₹12–18 LPA", (1_200_000, 1_800_000, "INR")),
        ("8L-10L", (800_000, 1_000_000, "INR")),
        ("18,00,000 - 24,00,000 per annum", (1_800_000, 2_400_000, "INR")),
        ("Rs. 30000 p.m.", (360_000, 360_000, "INR")),
        ("₹25,000/month stipend", (300_000, 300_000, "INR")),
        ("1.2 Cr", (12_000_000, 12_000_000, "INR")),
        ("$120k–$150k", (120_000, 150_000, "USD")),
        ("$60/hr", (124_800, 124_800, "USD")),
        ("€55,000 - €65,000 a year", (55_000, 65_000, "EUR")),
    ],
)
def test_parses_to_annual_whole_numbers(text, expected):
    salary = parse_salary(text)
    assert (salary.min, salary.max, salary.currency) == expected


def test_one_sided_bounds():
    upto = parse_salary("up to 20 LPA")
    assert (upto.min, upto.max) == (None, 2_000_000)
    plus = parse_salary("$150k+")
    assert (plus.min, plus.max) == (150_000, None)


@pytest.mark.parametrize("text", ["", "   ", None, "Competitive", "As per industry standards"])
def test_no_number_means_no_salary(text):
    salary = parse_salary(text)
    assert (salary.min, salary.max, salary.currency) == (None, None, None)
