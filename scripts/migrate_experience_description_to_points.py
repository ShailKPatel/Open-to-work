"""One-time migration: folds each Experience row's old free-text
`description` into a single ExperiencePoint, then drops the now-unused
`description` column. Run once, on an existing database, after pulling the
change that removed `description` from app.core.db.Experience (see that
model's docstring for why: a paragraph is one unsplittable blob a
resume-building semantic search can only take or leave whole, points are
individually retrievable).

Safe to run more than once: it only touches rows where `description`
still has non-blank text, and the DROP COLUMN step is skipped once the
column is already gone. No rollback path is provided: nothing
downstream reads `description` after this.

Usage: .venv/bin/python -m scripts.migrate_experience_description_to_points
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select, text

from app.core.db import ExperiencePoint, get_db, init_db

logger = logging.getLogger(__name__)


def _has_description_column(db) -> bool:
    cols = db.execute(text("PRAGMA table_info(experiences)")).all()
    return any(col[1] == "description" for col in cols)


def migrate() -> int:
    """Returns the number of ExperiencePoint rows created."""
    init_db()
    db = get_db()
    try:
        if not _has_description_column(db):
            print("no `description` column on experiences; already migrated, nothing to do")
            return 0

        rows = db.execute(
            text(
                "SELECT id, description FROM experiences "
                "WHERE description IS NOT NULL AND trim(description) != ''"
            )
        ).all()

        created: list[ExperiencePoint] = []
        for exp_id, description in rows:
            next_order = (
                db.execute(
                    select(func.max(ExperiencePoint.order_index)).where(
                        ExperiencePoint.experience_id == exp_id
                    )
                ).scalar()
                or 0
            ) + 1
            point = ExperiencePoint(
                experience_id=exp_id, text=description.strip(), order_index=next_order
            )
            db.add(point)
            created.append(point)

        if created:
            db.commit()
            for point in created:
                db.refresh(point)

        # Best-effort: get the migrated points into Qdrant too, same
        # posture as every other indexing call in this codebase; a
        # Qdrant hiccup here shouldn't block the migration itself.
        if created:
            try:
                from app.retrieval.index import index_experience_points

                index_experience_points(created)
            except Exception:
                logger.exception(
                    "could not index migrated experience points; rerun "
                    "app.retrieval.index.index_experience_points against "
                    "them manually once Qdrant is reachable"
                )

        db.execute(text("ALTER TABLE experiences DROP COLUMN description"))
        db.commit()

        print(f"migrated {len(created)} description(s) into points, dropped description column")
        return len(created)
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
