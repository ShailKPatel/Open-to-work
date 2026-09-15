"""Text-based skill extraction, covering what manifest parsing can't get:
technologies/practices described in prose (architecture, protocols, patterns)
that never show up as a declared dependency. Bulk tier (runs across every
repo).

Source text, in order: README (best signal) → GitHub's short "About"
description (repo metadata, not repo content, free with the repo-list
call, no extra fetch) → nothing. A repo with neither raises
NoSourceTextError rather than making an LLM call on empty input: an LLM
call on nothing burns tokens for a response that can only be noise, and
build.py needs to tell "nothing to extract" apart from "extraction failed"
so the UI can show "no README" instead of a retry-suggesting error.
Does not walk the rest of the repo tree looking for something else to feed
the model, for the same reason at a higher cost.

Narrow claim: this reads README/description text only, not
source files. "declared_dependency" (manifest_skills.py),
"readme_described", and "description_described" (here) are the evidence
types this ingestion supports. A stronger evidence type ("imported and
used across N files") would need scanning source file imports, which isn't
ingested yet. Don't let a prompt
tempt the model into claiming that tier of evidence for data we don't have.
"""

from __future__ import annotations

from app.core.db import Repository
from app.core.llm import complete, system_message, user_message
from app.profile.claims import EvidenceType, LinkClaim, SkillClaim

_README_CHAR_LIMIT = 4000

_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["skill", "confidence"],
            },
        }
    },
    "required": ["skills"],
}

_SYSTEM_PROMPT = (
    "You extract technical skills from text describing a GitHub repository "
    "(either its README or its short one-line description). List only "
    "technologies, tools, protocols, or engineering practices the text "
    "explicitly describes the project as building, using, or implementing, "
    "not things merely mentioned in badges, license text, contributor "
    "lists, or links to unrelated projects. Do not infer skills beyond what "
    "the text actually states. Name each skill the way it would appear on a "
    "resume, using its common official name (e.g. 'React', 'PostgreSQL', "
    "'Scikit-learn'). Name the library or framework, not individual classes, "
    "functions, or models inside it (e.g. 'Scikit-learn', not 'ElasticNetCV' "
    "or 'RobustScaler'). Skip code editors and IDEs. Assign each a confidence in [0, 1] "
    "reflecting how explicitly the text states it. If nothing qualifies, "
    "return an empty list."
)


_LINKS_SCHEMA = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["label", "url"],
            },
        }
    },
    "required": ["links"],
}

_LINKS_SYSTEM_PROMPT = (
    "You extract outbound project links from text describing a GitHub "
    "repository (either its README or its short one-line description). "
    "Find links the text gives for the project itself: its own GitHub repo "
    "(if a URL is given), a demo video (e.g. YouTube), a live/deployed "
    "version of the project, hosted docs, or similar project destinations. "
    "For each, give a short human-readable label such as 'GitHub', "
    "'YouTube Video', 'Live Demo', or 'Documentation', and the URL exactly "
    "as it appears in the text. Do not include badges, license links, "
    "contributor/profile links, or links to unrelated tools or projects. "
    "If nothing qualifies, return an empty list."
)


class NoSourceTextError(Exception):
    """No README and no description, so nothing worth an LLM call. Not a
    failure; build.py catches this specifically to mark the repo
    "no_signal" rather than "failed" (no retry implied, nothing to retry)."""


def _source_text(repo: Repository) -> tuple[str, EvidenceType]:
    if repo.readme and repo.readme.strip():
        return repo.readme[:_README_CHAR_LIMIT], "readme_described"
    if repo.description and repo.description.strip():
        return repo.description.strip(), "description_described"
    raise NoSourceTextError(f"{repo.full_name}: no README, no description")


def extract_skills_from_repo(repo: Repository) -> list[SkillClaim]:
    text, evidence_type = _source_text(repo)

    label = "README" if evidence_type == "readme_described" else "description"
    messages = [
        system_message(_SYSTEM_PROMPT),
        user_message(f"Repository: {repo.full_name}\n\n{label}:\n{text}"),
    ]

    response = complete("bulk", messages, schema=_SCHEMA, account_id=repo.account_id)
    if response.parsed is None:
        return []

    claims = []
    for item in response.parsed.get("skills", []):
        skill = str(item.get("skill", "")).strip()
        if not skill:
            continue
        confidence = float(item.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        claims.append(
            SkillClaim(
                skill=skill,
                evidence_type=evidence_type,
                confidence=confidence,
                source_files=[label],
            )
        )
    return claims


def extract_links_from_repo(repo: Repository) -> list[LinkClaim]:
    """Same source text and same fallback chain as extract_skills_from_repo
    (README → description → NoSourceTextError), separate LLM call: a
    different schema/prompt than skill extraction, so it stays its own
    function rather than a second return value bolted onto that one. Called
    from build.py's _process_repo, same first pass that runs skill
    extraction.
    """
    text, evidence_type = _source_text(repo)
    label = "README" if evidence_type == "readme_described" else "description"
    messages = [
        system_message(_LINKS_SYSTEM_PROMPT),
        user_message(f"Repository: {repo.full_name}\n\n{label}:\n{text}"),
    ]

    response = complete("bulk", messages, schema=_LINKS_SCHEMA, account_id=repo.account_id)
    if response.parsed is None:
        return []

    claims = []
    for item in response.parsed.get("links", []):
        link_label = str(item.get("label", "")).strip()
        url = str(item.get("url", "")).strip()
        if not link_label or not url:
            continue
        claims.append(LinkClaim(label=link_label, url=url))
    return claims
