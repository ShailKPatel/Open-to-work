"""Shared shape between the skill sources (manifest parsing, LLM text
extraction) so `build.py` doesn't care which one produced a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceType = Literal["declared_dependency", "readme_described", "description_described"]


@dataclass
class SkillClaim:
    skill: str
    evidence_type: EvidenceType
    confidence: float
    source_files: list[str] = field(default_factory=list)


@dataclass
class LinkClaim:
    """One outbound link found in a repo's README/description text; see
    app/profile/extract.py's extract_repo_facts. `label` is a short
    human-readable title ("GitHub", "YouTube Video", "Live Demo"), not an
    enum: same free-text reasoning as ProjectLink.label.
    """

    label: str
    url: str
