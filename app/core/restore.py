"""Puts a backup from app/core/backup.py back in place, then rebuilds the
search index from it. Run with the app stopped, since it replaces the
database files under it:

    docker compose stop app
    docker compose run --rm app python -m app.core.restore --list
    docker compose run --rm app python -m app.core.restore --yes            # newest
    docker compose run --rm app python -m app.core.restore --yes --key otw-backup-...
    docker compose up -d

Needs the same BACKUP_* settings as the backup, and the same APP_SECRET_KEY
the backed-up instance used, or its stored API keys cannot be read. Files
already in place are moved aside with a `.before-restore-<time>` suffix,
not deleted. --file restores a local archive instead of downloading one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

from app.core.backup import _PREFIX, _client
from app.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class RestoreError(Exception):
    pass


def list_backups(settings: Settings) -> list[str]:
    """Backup names in the bucket, oldest first (the timestamp sorts)."""
    listed = _client(settings).list_objects_v2(Bucket=settings.backup_bucket, Prefix=_PREFIX)
    return sorted(item["Key"] for item in listed.get("Contents", []))


def _targets(settings: Settings) -> dict[str, Path]:
    """Where each archive member goes: db/<file name> and files/<folder name>."""
    main = Path(settings.database_url.removeprefix("sqlite:///"))
    graph = Path(settings.resume_graph_checkpoint_path)
    if main.name == graph.name:
        raise RestoreError("the database and the graph checkpoint share a file name")
    resumes = Path(settings.resume_storage_dir)
    screenshots = Path(settings.job_screenshot_storage_dir)
    return {
        f"db/{main.name}": main,
        f"db/{graph.name}": graph,
        f"files/{resumes.name}": resumes,
        f"files/{screenshots.name}": screenshots,
    }


def _move_aside(path: Path, stamp: str) -> None:
    for existing in (path, *(path.with_name(path.name + s) for s in ("-wal", "-shm"))):
        if existing.exists():
            existing.rename(existing.with_name(f"{existing.name}.before-restore-{stamp}"))


def restore_archive(settings: Settings, archive: Path) -> list[Path]:
    """Unpacks into a scratch folder first (tarfile's "data" filter refuses
    links, absolute paths and ..), checks the archive holds a main
    database, and only then moves things into place."""
    targets = _targets(settings)
    main_member = next(iter(targets))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as tmp:
        unpacked = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        if not (unpacked / main_member).is_file():
            raise RestoreError(f"{archive.name} has no {main_member}; not a backup of this app")
        placed = []
        for member, target in targets.items():
            source = unpacked / member
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            _move_aside(target, stamp)
            shutil.move(str(source), str(target))
            placed.append(target)
    return placed


def restore(key: str | None = None, file: Path | None = None) -> str:
    """Restores one backup (the newest when no key or file is given) and
    rebuilds the index. Returns what was restored."""
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        if file is not None:
            archive, name = file, file.name
        else:
            if not settings.backup_bucket:
                raise RestoreError("BACKUP_BUCKET is not set")
            names = list_backups(settings)
            if key is None:
                if not names:
                    raise RestoreError("the bucket has no backups")
                key = names[-1]
            elif key not in names:
                raise RestoreError(f"no backup named {key}")
            archive, name = Path(tmp) / key, key
            _client(settings).download_file(settings.backup_bucket, key, str(archive))
        placed = restore_archive(settings, archive)
    logger.info("restored %s: %s", name, ", ".join(str(p) for p in placed))

    # The restored database may predate columns this version adds.
    from app.core.db import init_db, reset_engine

    reset_engine()
    init_db()
    from app.retrieval.rebuild import rebuild_index

    counts = rebuild_index(fresh=True)
    logger.info("search index rebuilt: %s", counts)
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restore a backup made by app.core.backup.")
    parser.add_argument("--list", action="store_true", help="list backups and stop")
    parser.add_argument("--key", help="backup name; default is the newest")
    parser.add_argument("--file", type=Path, help="a local archive instead of the bucket")
    parser.add_argument("--yes", action="store_true", help="needed to replace the data")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.list:
        print("\n".join(list_backups(get_settings())) or "no backups")
        return
    if not args.yes:
        parser.error("this replaces the database: stop the app, then pass --yes")
    print(f"restored {restore(args.key, args.file)}")


if __name__ == "__main__":
    main()
