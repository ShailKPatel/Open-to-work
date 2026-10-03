"""app/resume_build/: LaTeX escaping/rendering (latex.py) and the
deterministic header/experience context builders (context.py).

test_template_contains_no_personal_data is the load-bearing test here:
the template file was built from a real resume with every
personal value stripped out and replaced with a Jinja placeholder, and
this pins that down as a permanent regression check, not a one-time
manual review.
"""

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

# Every word a template may contain outside its Jinja placeholders and
# LaTeX comments: section headings, package names, option keys. Anything
# else is text someone typed into the template instead of a placeholder,
# which is how a name, a city or a project title would get in. An
# allowlist rather than a list of known personal values, because a list of
# the values would itself publish them.
_TEMPLATE_WORDS = {
    "Education", "Experience", "LaTeX", "LastPage", "Projects", "RGB",
    "Resume", "Skills", "Summary", "Technologies", "adjustwidth", "amsmath",
    "array", "black", "bookmark", "bottom", "calc", "changepage", "cm",
    "colorlinks", "customFooterStyle", "document", "dvipsnames", "empty",
    "enumitem", "eso", "fontawesome", "footskip", "geometry",
    "glyphtounicode", "graphicx", "header", "highlights", "hyperref",
    "iftex", "ifthen", "ignoreheadfoot", "inputenc", "itemize", "itemsep",
    "lastpage", "left", "leftmargin", "letterpaper", "linkcolor", "lmodern",
    "needspace", "of", "onecolentry", "paracol", "parsep", "partopsep",
    "pdfauthor", "pdfcreator", "pdftitle", "pic", "primaryColor", "pscoord",
    "pt", "right", "secnumdepth", "tabularx", "titlesec", "top", "topsep",
    "true", "twocolentry", "urlcolor", "utf", "xcolor",
}

# The one URL the templates may carry, the credit for the styling they
# are based on.
_ALLOWED_URLS = {"github.com/rendercv/rendercv"}

# Optional extra check against values only the account holder knows: one
# string per line in a file that never leaves the machine (data/ is
# gitignored). Skipped when the file is not there, as in CI.
_PRIVATE_STRINGS_FILE = Path(__file__).parent.parent.parent / "data/private_strings.txt"


def _template_body(text: str) -> str:
    """The template with its LaTeX comments, Jinja placeholders and LaTeX
    command names removed: what is left is literal text."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(?:VAR|BLOCK)\{[^}]*\}", "", text)
    return re.sub(r"\\[A-Za-z@]+", "", text)


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
    body = _template_body(text)

    unknown = set(re.findall(r"[A-Za-z]{2,}", body)) - _TEMPLATE_WORDS
    assert not unknown, f"literal text in {template_name}, use a placeholder: {sorted(unknown)}"

    # Comments are free prose, so they are checked for the shapes personal
    # data takes rather than word by word.
    assert not re.search(r"\d{7,}", text), f"phone-like number in {template_name}"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), f"email in {template_name}"
    urls = set(re.findall(r"(?:https?://)?((?:[\w-]+\.)+[a-z]{2,}/[\w./-]*[\w/])", text))
    assert urls <= _ALLOWED_URLS, f"unexpected URL in {template_name}: {urls - _ALLOWED_URLS}"


@pytest.mark.skipif(not _PRIVATE_STRINGS_FILE.exists(), reason="no local private strings file")
@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_template_contains_no_private_strings(template_name):
    text = (TEMPLATES_DIR / template_name).read_text().casefold()
    for line in _PRIVATE_STRINGS_FILE.read_text().splitlines():
        value = line.strip()
        if value:
            assert value.casefold() not in text, f"private value leaked into {template_name}"


def test_escape_latex_handles_every_special_char():
    assert escape_latex("100% & $5 #1 _under_ {brace} ~tilde ^caret \\back") == (
        r"100\% \& \$5 \#1 \_under\_ \{brace\} \textasciitilde{}tilde "
        r"\textasciicircum{}caret \textbackslash{}back"
    )


def test_escape_latex_replaces_em_dashes_with_commas():
    assert escape_latex("Built a tool \u2014 fast") == "Built a tool, fast"
    assert escape_latex("A\u2014B, \u2014 C") == "A, B, C"
    assert escape_latex("2019 -- 2020") == "2019 -- 2020"


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
        {"icon": r"\faLinkedin", "text": "jane", "href": "https://linkedin.com/in/jane"}
    ]


def test_build_header_context_shows_links_without_scheme(tmp_path):
    """The href stays the full URL; the printed text drops the scheme,
    "www." and trailing slash, and a link saved without a scheme still
    gets one so it is clickable."""
    _reset_db(tmp_path)
    account = _make_account(github_username="")
    links = [
        SocialLink(account_id=account.id, platform="website", url="https://jane.github.io/"),
        SocialLink(account_id=account.id, platform="website", url="www.jane.dev/blog"),
        SocialLink(account_id=account.id, platform="linkedin", url="linkedin.com/in/jane-d/"),
    ]

    ctx = build_header_context(account, social_links=links)

    assert [(i["text"], i["href"]) for i in ctx["social_items"]] == [
        ("jane.github.io", "https://jane.github.io/"),
        ("jane.dev/blog", "https://www.jane.dev/blog"),
        ("jane-d", "https://linkedin.com/in/jane-d/"),
    ]


def test_build_header_context_other_platform_uses_label(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(github_username="")
    link = SocialLink(
        account_id=account.id, platform="other", url="https://example.com/x", label="Portfolio"
    )

    ctx = build_header_context(account, social_links=[link])

    assert ctx["social_items"][0]["text"] == "Portfolio"
    assert ctx["social_items"][0]["icon"] == r"\faBriefcase"


def test_build_header_context_custom_label_icons(tmp_path):
    _reset_db(tmp_path)
    account = _make_account(github_username="")
    links = [
        SocialLink(account_id=account.id, platform="other", url="https://a.example", label=label)
        for label in ("Certifications", "Blog", "LeetCode")
    ]

    ctx = build_header_context(account, social_links=links)

    assert [i["icon"] for i in ctx["social_items"]] == [
        r"\faCertificate", r"\faBlog", r"\faLink",
    ]


def test_build_experience_context_orders_roles_and_points(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    db = get_db()
    older = Experience(
        account_id=account.id, title="Engineer I", company="Old Co",
        start_date="jan 2020", end_date="jan 2021",
    )
    current = Experience(
        account_id=account.id, title="Engineer II", company="New Co",
        start_date="jun 2022", end_date=None,
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
                start_date="jan 2015", end_date="jan 2017",
            ),
            Education(
                account_id=account.id, institution="New University", degree="B.Sc",
                start_date="jan 2018", end_date="jan 2022",
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


def test_build_education_context_carries_grade_and_details(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    db = get_db()
    db.add_all(
        [
            Education(
                account_id=account.id, institution="New University", degree="B.Sc",
                start_date="jan 2018", grade="CGPA 8.9/10",
                details=["Ranked 1st in university"],
            ),
            Education(
                account_id=account.id, institution="Old College", degree="AA",
                start_date="jan 2015",
            ),
        ]
    )
    db.commit()
    db.close()

    result = build_education_context(get_db(), account.id)

    assert result[0]["grade"] == "CGPA 8.9/10"
    assert result[0]["details"] == ["Ranked 1st in university"]
    assert result[1]["grade"] is None
    assert result[1]["details"] == []


def _education_render_data(education):
    return {
        "full_name": "Jane Doe",
        "contact_items": [],
        "social_items": [],
        "summary": None,
        "experience": [],
        "projects": [],
        "education": education,
        "technologies": [],
        "skills": [],
    }


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_render_resume_shows_education_grade_and_details(template_name):
    rendered = render_resume(
        template_name,
        _education_render_data(
            [
                {
                    "institution": "State University", "degree": "B.Sc",
                    "date_range": "2018 -- 2022", "grade": "GPA 3.9/4 & honors",
                    "details": ["Coursework: Algorithms, 100% attendance"],
                }
            ]
        ),
    )

    education = rendered.split(r"\section{Education}", 1)[1]
    assert r"\textbar\kern 0.20 cm GPA 3.9/4 \& honors" in education
    assert r"\item Coursework: Algorithms, 100\% attendance" in education


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_render_resume_education_without_extras_adds_nothing(template_name):
    rendered = render_resume(
        template_name,
        _education_render_data(
            [
                {
                    "institution": "State University", "degree": "B.Sc",
                    "date_range": "2018 -- 2022", "grade": None, "details": [],
                }
            ]
        ),
    )

    education = rendered.split(r"\section{Education}", 1)[1]
    assert r"\textbar" not in education
    assert r"\begin{highlights}" not in education


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


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
def test_rendered_resume_stamps_the_pdf_creator(template_name):
    """app/profile/resume_ingest.py's is_generated_pdf relies on this to
    refuse a generated resume uploaded back into the library."""
    from app.resume_build.latex import GENERATED_PDF_CREATOR
    from app.resume_build.warm_tectonic import _sample_data

    tex = render_resume(template_name, _sample_data())
    assert f"pdfcreator={{{GENERATED_PDF_CREATOR}}}" in tex


def test_is_generated_pdf_ignores_non_pdfs_and_unreadable_pdfs():
    from app.profile.resume_ingest import is_generated_pdf

    assert not is_generated_pdf(b"\x89PNG not a pdf")
    assert not is_generated_pdf(b"%PDF-1.4 truncated garbage")
