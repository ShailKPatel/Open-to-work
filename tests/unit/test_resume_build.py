"""app/resume_build/: LaTeX escaping/rendering (latex.py) and the
deterministic header/experience context builders (context.py).

test_template_contains_no_personal_data is the load-bearing test here:
the template file was built from a real resume with every
personal value stripped out and replaced with a Jinja placeholder, and
this pins that down as a permanent regression check, not a one-time
manual review.
"""

import datetime as dt
import re
from pathlib import Path

import pytest

from app.core.db import (
    Account,
    Education,
    Experience,
    ExperiencePoint,
    SocialLink,
    get_db,
    init_db,
)
from app.resume_build.context import (
    build_education_context,
    build_experience_context,
    build_header_context,
)
from app.resume_build.latex import escape_latex, escape_latex_url, render_resume

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "app/resume_build/templates"
TEMPLATE_NAMES = ["onepage.tex.j2", "twopage.tex.j2"]

# Real values from the personal resume the template was built from, must
# never appear in the template file itself. Specific, not
# generic ("no email-shaped string"), so this test actually breaks if any
# of these ever gets pasted back in.
_FORBIDDEN_STRINGS = [
    "Shail",
    "Patel",
    "shailkpatel",
    "shailpatel.connect",
    "9023020937",
    "Ahmedabad",
    "RestaurantPilot",
    "GET MY SPACE",
    "PredictGrad",
    "predictgrad",
    "Beyond The Marks",
    "beyondthemarks",
    "LJ University",
    "zenodo.19686376",
]


def _reset_db(tmp_path: Path):
    import os

    import app.core.db as db_module
    from app.core.settings import get_settings

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_template_contains_no_personal_data(template_name):
    text = (TEMPLATES_DIR / template_name).read_text()
    for forbidden in _FORBIDDEN_STRINGS:
        assert forbidden not in text, f"personal data leaked into {template_name}: {forbidden!r}"


def test_escape_latex_handles_every_special_char():
    assert escape_latex("100% & $5 #1 _under_ {brace} ~tilde ^caret \\back") == (
        r"100\% \& \$5 \#1 \_under\_ \{brace\} \textasciitilde{}tilde "
        r"\textasciicircum{}caret \textbackslash{}back"
    )


def test_escape_latex_url_only_touches_breaking_chars():
    assert escape_latex_url("https://example.com/a_b-c~d?x=1&y=2%3") == (
        r"https://example.com/a_b-c~d?x=1\&y=2\%3"
    )


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_render_resume_escapes_injected_data_and_leaves_no_unrendered_tags(template_name):
    data = {
        "full_name": "A & B",
        "contact_items": [],
        "social_items": [],
        "summary": "Built systems handling 50% more load, C# included.",
        "experience": [],
        "projects": [],
        "education": [],
        "technologies": [],
        "skills": [],
    }
    rendered = render_resume(template_name, data)

    assert r"A \& B" in rendered
    assert r"50\% more" in rendered
    assert r"C\# included" in rendered
    assert "\\BLOCK{" not in rendered
    assert "\\VAR{" not in rendered


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_render_resume_omits_empty_sections(template_name):
    data = {
        "full_name": "Jane Doe",
        "contact_items": [],
        "social_items": [],
        "summary": None,
        "experience": [],
        "projects": [],
        "education": [],
        "technologies": [],
        "skills": [],
    }
    rendered = render_resume(template_name, data)

    assert r"\section{Summary}" not in rendered
    assert r"\section{Experience}" not in rendered
    assert r"\section{Projects}" not in rendered
    assert r"\section{Education}" not in rendered


def _make_account(**overrides) -> Account:
    defaults = dict(first_name="Jane", last_name="Doe", github_username="janedoe")
    defaults.update(overrides)
    db = get_db()
    account = Account(**defaults)
    db.add(account)
    db.commit()
    db.refresh(account)
    db.close()
    return account


def test_build_header_context_orders_contact_items_and_includes_github(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(
        contact_location="Remote", contact_email="jane@example.com", contact_phone="555-0100"
    )

    ctx = build_header_context(account, social_links=[])

    assert ctx["full_name"] == "Jane Doe"
    assert [i["text"] for i in ctx["contact_items"]] == ["Remote", "jane@example.com", "555-0100"]
    assert ctx["contact_items"][1]["href"] == "mailto:jane@example.com"
    assert ctx["social_items"][0] == {
        "icon": r"\faGithub", "text": "janedoe", "href": "https://github.com/janedoe",
    }


def test_build_header_context_skips_missing_contact_fields(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(github_username="")

    ctx = build_header_context(account, social_links=[])

    assert ctx["contact_items"] == []
    assert ctx["social_items"] == []


def test_build_header_context_maps_platform_to_icon(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(github_username="")
    link = SocialLink(account_id=account.id, platform="linkedin", url="https://linkedin.com/in/jane")

    ctx = build_header_context(account, social_links=[link])

    assert ctx["social_items"] == [
        {"icon": r"\faLinkedin", "text": "https://linkedin.com/in/jane", "href": "https://linkedin.com/in/jane"}
    ]


def test_build_header_context_other_platform_uses_label(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(github_username="")
    link = SocialLink(
        account_id=account.id, platform="other", url="https://example.com/x", label="Portfolio"
    )

    ctx = build_header_context(account, social_links=[link])

    assert ctx["social_items"][0]["text"] == "Portfolio"
    assert ctx["social_items"][0]["icon"] == r"\faLink"


def test_build_experience_context_orders_roles_and_points(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    db = get_db()
    older = Experience(
        account_id=account.id, title="Engineer I", company="Old Co",
        start_date=dt.date(2020, 1, 1), end_date=dt.date(2021, 1, 1),
    )
    current = Experience(
        account_id=account.id, title="Engineer II", company="New Co",
        start_date=dt.date(2022, 6, 1), end_date=None,
    )
    db.add_all([older, current])
    db.commit()
    db.refresh(older)
    db.refresh(current)
    db.add_all(
        [
            ExperiencePoint(experience_id=current.id, text="Did the first thing", order_index=1),
            ExperiencePoint(experience_id=current.id, text="Did the second thing", order_index=2),
        ]
    )
    db.commit()
    db.close()

    result = build_experience_context(get_db(), account.id)

    assert [r["company"] for r in result] == ["New Co", "Old Co"]
    assert result[0]["date_range"] == "Jun. 2022 -- present"
    assert result[0]["points"] == ["Did the first thing", "Did the second thing"]
    assert result[1]["points"] == []


def test_build_experience_context_empty_when_no_roles(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()

    assert build_experience_context(get_db(), account.id) == []


def test_build_experience_context_includes_role_id(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    db = get_db()
    role = Experience(account_id=account.id, title="Engineer", company="Acme")
    db.add(role)
    db.commit()
    db.refresh(role)
    role_id = role.id
    db.close()

    result = build_experience_context(get_db(), account.id)

    assert result[0]["id"] == role_id


def test_build_education_context_orders_newest_first(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    db = get_db()
    db.add_all(
        [
            Education(
                account_id=account.id, institution="Old College", degree="AA",
                start_date=dt.date(2015, 1, 1), end_date=dt.date(2017, 1, 1),
            ),
            Education(
                account_id=account.id, institution="New University", degree="B.Sc",
                start_date=dt.date(2018, 1, 1), end_date=dt.date(2022, 1, 1),
            ),
        ]
    )
    db.commit()
    db.close()

    result = build_education_context(get_db(), account.id)

    assert [r["institution"] for r in result] == ["New University", "Old College"]
    assert result[0]["degree"] == "B.Sc"
    assert result[0]["date_range"] == "Jan. 2018 -- Jan. 2022"


def test_build_education_context_empty_when_none(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()

    assert build_education_context(get_db(), account.id) == []


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_render_resume_escapes_href_targets(template_name):
    """Every href in a resume comes from account data (a social link's
    URL, a repo URL, the contact email/phone), so it can carry the three
    characters LaTeX still tokenizes inside \\href's target: `%`, `#` and
    `&`. An unescaped `%` comments out the rest of the line and silently
    drops the link; `#` and `&` fail the compile outright. The templates
    must route every href through the latex_url filter, not just the
    visible link text through latex.
    """
    data = {
        "full_name": "Ada Lovelace",
        "contact_items": [
            {"icon": r"\faEnvelope", "text": "a%b@example.com", "href": "mailto:a%b@example.com"}
        ],
        "social_items": [
            {"icon": r"\faGithub", "text": "ada", "href": "https://example.com/u?a=1&b=2#frag"}
        ],
        "summary": "",
        "experience": [],
        "projects": [
            {
                "name": "Proj",
                "tagline": "",
                "date_range": "2026",
                "href": "https://example.com/r%20d?x=1&y=2#top",
                "url_display": "example.com/r d",
                "points": [],
            }
        ],
        "education": [],
        "technologies": [],
        "skills": [],
    }
    rendered = render_resume(template_name, data)

    assert r"mailto:a\%b@example.com" in rendered
    assert r"https://example.com/u?a=1\&b=2\#frag" in rendered
    assert r"https://example.com/r\%20d?x=1\&y=2\#top" in rendered
    # No bare breaking char survives inside an href target.
    for target in re.findall(r"\\href(?:WithoutArrow)?\{(.*?)\}", rendered):
        assert not re.search(r"(?<!\\)[%#&]", target), f"unescaped href target: {target!r}"
