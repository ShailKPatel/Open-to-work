"""Re-derives manifest-based skill evidence for every repo from the
manifests already stored in the database, using the current package-to-skill
mapping in app/profile/manifest_skills.py. Run after that mapping changes.

No LLM calls and no GitHub fetches. README-derived and manual skills are left
as they are; use Reprocess on a project to re-run README extraction.

Safe to run more than once.

Usage: .venv/bin/python -m scripts.rebuild_manifest_skills
"""

from __future__ import annotations

from app.core.db import init_db
from app.profile.build import rebuild_manifest_evidence


def main() -> None:
    init_db()
    removed, written = rebuild_manifest_evidence()
    print(f"replaced {removed} manifest skill row(s) with {written}")


if __name__ == "__main__":
    main()
