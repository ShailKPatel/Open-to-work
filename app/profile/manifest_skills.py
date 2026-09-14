"""Deterministic skill claims from parsed manifest dependencies: no LLM
call, no cost, no ambiguity: we already know for a fact these packages are
declared, `app/ingest/github/manifests.py` parsed them at ingestion time.
Confidence is always 1.0 (existence is certain); the LLM-derived signal
(`readme_described`, in `extract.py`) is what carries genuine uncertainty.
"""

from __future__ import annotations

from app.profile.claims import SkillClaim


def skills_from_manifests(manifests_json: dict) -> list[SkillClaim]:
    claims = []
    for filename, manifest in manifests_json.items():
        for dependency in manifest.get("dependencies", []):
            claims.append(
                SkillClaim(
                    skill=dependency,
                    evidence_type="declared_dependency",
                    confidence=1.0,
                    source_files=[filename],
                )
            )
    return claims
