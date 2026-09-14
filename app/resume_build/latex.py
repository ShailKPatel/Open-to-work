"""LaTeX template rendering: Jinja2 reconfigured with delimiters that don't
collide with LaTeX's own syntax (LaTeX uses `{}` constantly and `%` for
comments, which are Jinja2's defaults), plus the escaping every piece of
DB-sourced text must go through before it lands in a .tex file. Standard
"Jinja2 for LaTeX" pattern (Jinja2's own docs show this exact delimiter
set for LaTeX output).

This module never decides *what* goes in a resume (that's
app/resume_build/context.py for the deterministic parts: header, contact
line, experience, and the not-yet-built orchestrator for the
semantically-picked parts, projects/skills/summary). It only turns a data
dict into a .tex string, safely. No LLM call happens here.

Regex-based, not a string method chain, so there's exactly one place that
defines what "safe for LaTeX" means, applied identically everywhere,
without spending any LLM tokens deciding it per-field.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Order matters: backslash isn't in this map because every other
# replacement below already introduces literal backslashes into the
# *output*, but re.sub's pattern matches against the *original* input
# string in one pass, so replacement text is never re-scanned. A raw
# backslash in the input (rare in resume data, but possible in a pasted
# path or shell snippet) still needs handling, listed separately so it's
# scanned for at the same time as everything else in one compiled regex.
_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _LATEX_SPECIAL_CHARS))

# Narrower than escape_latex(): a URL is going into \href's *target*
# argument, not typeset as regular text. `_`, `~`, `-`, `.`, `/`, `:` are
# all common and structurally meaningful in real URLs and don't need
# escaping there; `%`, `#`, `&` do, they're the ones LaTeX's tokenizer
# still treats specially even inside an href target. Not yet covered by a
# real compile test.
_URL_ESCAPE_CHARS = {"%": r"\%", "#": r"\#", "&": r"\&"}
_URL_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _URL_ESCAPE_CHARS))


def escape_latex(text: str) -> str:
    """Escapes a plain string for use as LaTeX body text (names, bullet
    points, company names, skill names, ...). Never apply this to a
    literal LaTeX command string the template itself constructs (e.g. an
    icon macro like \\faGithub): only to actual data.
    """
    return _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_SPECIAL_CHARS[m.group()], text)


def escape_latex_url(url: str) -> str:
    """Escapes a URL for use as \\href's target argument. See the module
    comment above for why this differs from escape_latex().
    """
    return _URL_ESCAPE_RE.sub(lambda m: _URL_ESCAPE_CHARS[m.group()], url)


def _get_env() -> Environment:
    env = Environment(
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        loader=FileSystemLoader(_TEMPLATES_DIR),
    )
    env.filters["latex"] = escape_latex
    env.filters["latex_url"] = escape_latex_url
    return env


def render_resume(template_name: str, data: dict[str, Any]) -> str:
    """template_name is a filename under app/resume_build/templates/, e.g.
    "onepage.tex.j2". Returns the rendered .tex source as a string, no
    compilation happens here, see the (not-yet-built) compile-step slice.
    """
    return _get_env().get_template(template_name).render(**data)
