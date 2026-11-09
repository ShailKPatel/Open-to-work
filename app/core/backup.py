"""Nightly copy of a deployment's data to S3-compatible storage (Backblaze
B2, Cloudflare R2, S3). Run as `python -m app.core.backup`; cron on the VM
does that. Does nothing until BACKUP_BUCKET is set.

Databases are copied with SQLite's online backup API, because copying a
live WAL database file can capture it half written. Resume files and
screenshots go in the same archive. Keeps the newest BACKUP_KEEP archives,
so the bill or free allowance has a ceiling.

The archive holds resumes in the clear, so the bucket must be private. The
API keys inside are encrypted with APP_SECRET_KEY, which is deliberately
not included: keep that value somewhere else.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from app.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_PREFIX = "otw-backup-"


def _copy_sqlite(source: Path, dest: Path) -> None:
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(dest) as out:
        src.backup(out)


def _database_paths(settings: Settings) -> list[Path]:
    main = Path(settings.database_url.removeprefix("sqlite:///"))
    return [p for p in (main, Path(settings.resume_graph_checkpoint_path)) if p.is_file()]


def build_archive(settings: Settings, workdir: Path, stamp: str) -> Path:
    archive = workdir / f"{_PREFIX}{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for database in _database_paths(settings):
            copy = workdir / database.name
            _copy_sqlite(database, copy)
            tar.add(copy, arcname=f"db/{database.name}")
        for folder in (settings.resume_storage_dir, settings.job_screenshot_storage_dir):
            if Path(folder).is_dir():
                tar.add(folder, arcname=f"files/{Path(folder).name}")
    return archive


def _client(settings: Settings) -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=settings.backup_endpoint or None,
        region_name=settings.backup_region or None,
        aws_access_key_id=os.environ.get("BACKUP_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("BACKUP_SECRET_ACCESS_KEY"),
    )


def _prune(client: Any, bucket: str, keep: int) -> None:
    listed = client.list_objects_v2(Bucket=bucket, Prefix=_PREFIX).get("Contents", [])
    # The timestamp in the name sorts oldest first.
    keys = sorted(item["Key"] for item in listed)
    for key in keys[: max(0, len(keys) - keep)]:
        client.delete_object(Bucket=bucket, Key=key)


def run_backup() -> str | None:
    """The uploaded object's name, or None when backups are not configured."""
    settings = get_settings()
    if not settings.backup_bucket:
        logger.info("BACKUP_BUCKET is not set; nothing to do")
        return None
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    client = _client(settings)
    with tempfile.TemporaryDirectory() as tmp:
        archive = build_archive(settings, Path(tmp), stamp)
        client.upload_file(str(archive), settings.backup_bucket, archive.name)
    # Only after a successful upload, so a failing run cannot empty the bucket.
    _prune(client, settings.backup_bucket, settings.backup_keep)
    return archive.name


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_backup() or "backups not configured")
