"""app/core/backup.py against a fake bucket and real SQLite files."""

import sqlite3
import tarfile
from unittest.mock import MagicMock

import pytest

from app.core import backup
from app.core.settings import Settings, get_settings


@pytest.fixture
def data(tmp_path, monkeypatch):
    database = tmp_path / "app.db"
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("create table t (v text)")
        db.execute("insert into t values ('kept')")
    resumes = tmp_path / "resumes"
    (resumes / "1").mkdir(parents=True)
    (resumes / "1" / "a.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("RESUME_STORAGE_DIR", str(resumes))
    monkeypatch.setenv("RESUME_GRAPH_CHECKPOINT_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("JOB_SCREENSHOT_STORAGE_DIR", str(tmp_path / "nope"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_not_configured_does_nothing(data, monkeypatch):
    monkeypatch.setattr(backup, "_client", MagicMock(side_effect=AssertionError))
    assert backup.run_backup() is None


def test_archive_holds_a_readable_database_and_the_files(data):
    settings = Settings()
    work = data / "work"
    work.mkdir()

    archive = backup.build_archive(settings, work, "x")

    with tarfile.open(archive) as tar:
        names = tar.getnames()
        tar.extract("db/app.db", data / "out", filter="data")
    assert "files/resumes/1/a.pdf" in names
    with sqlite3.connect(data / "out" / "db" / "app.db") as db:
        assert db.execute("select v from t").fetchone() == ("kept",)


def _bucket(existing):
    client = MagicMock()
    client.list_objects_v2.return_value = {"Contents": [{"Key": k} for k in existing]}
    return client


def test_uploads_then_keeps_only_the_newest(data, monkeypatch):
    monkeypatch.setenv("BACKUP_BUCKET", "b")
    monkeypatch.setenv("BACKUP_KEEP", "2")
    get_settings.cache_clear()
    client = _bucket(["otw-backup-1", "otw-backup-3", "otw-backup-2"])
    monkeypatch.setattr(backup, "_client", lambda settings: client)

    name = backup.run_backup()

    assert client.upload_file.call_args.args[1:] == ("b", name)
    client.delete_object.assert_called_once_with(Bucket="b", Key="otw-backup-1")


def test_failed_upload_prunes_nothing(data, monkeypatch):
    monkeypatch.setenv("BACKUP_BUCKET", "b")
    get_settings.cache_clear()
    client = _bucket(["otw-backup-1"])
    client.upload_file.side_effect = RuntimeError("network")
    monkeypatch.setattr(backup, "_client", lambda settings: client)

    with pytest.raises(RuntimeError):
        backup.run_backup()

    client.delete_object.assert_not_called()
