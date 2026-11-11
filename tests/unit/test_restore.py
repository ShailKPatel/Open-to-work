"""app/core/restore.py and app/retrieval/rebuild.py: a backup goes back in
place and the search index is rebuilt from the database alone."""

import io
import sqlite3
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core import backup, restore
from app.core.settings import Settings, get_settings


@pytest.fixture
def data(tmp_path, monkeypatch):
    database = tmp_path / "live" / "app.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("create table t (v text)")
        db.execute("insert into t values ('backed up')")
    resumes = tmp_path / "live" / "resumes"
    (resumes / "1").mkdir(parents=True)
    (resumes / "1" / "a.pdf").write_bytes(b"%PDF old")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("RESUME_STORAGE_DIR", str(resumes))
    monkeypatch.setenv("RESUME_GRAPH_CHECKPOINT_PATH", str(tmp_path / "live" / "graph.db"))
    monkeypatch.setenv("JOB_SCREENSHOT_STORAGE_DIR", str(tmp_path / "live" / "shots"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _archive(data: Path) -> Path:
    work = data / "work"
    work.mkdir(exist_ok=True)
    return backup.build_archive(Settings(), work, "20261110T000000Z")


def test_round_trip_puts_the_backup_back_and_keeps_what_was_there(data):
    archive = _archive(data)
    database = data / "live" / "app.db"
    with sqlite3.connect(database) as db:
        db.execute("update t set v = 'changed later'")
    (data / "live" / "resumes" / "1" / "a.pdf").write_bytes(b"%PDF new")

    placed = restore.restore_archive(Settings(), archive)

    assert database in placed
    with sqlite3.connect(database) as db:
        assert db.execute("select v from t").fetchone() == ("backed up",)
    assert (data / "live" / "resumes" / "1" / "a.pdf").read_bytes() == b"%PDF old"
    aside = sorted(p.name for p in (data / "live").iterdir() if "before-restore" in p.name)
    assert any(name.startswith("app.db.before-restore-") for name in aside)
    assert any(name.startswith("resumes.before-restore-") for name in aside)


def _tar_with(members: dict[str, bytes], path: Path) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


def test_an_archive_without_the_database_changes_nothing(data):
    archive = _tar_with({"files/resumes/x.pdf": b"x"}, data / "bad.tar.gz")
    with pytest.raises(restore.RestoreError):
        restore.restore_archive(Settings(), archive)
    assert (data / "live" / "resumes" / "1" / "a.pdf").read_bytes() == b"%PDF old"


def test_an_archive_that_escapes_its_folder_is_refused(data):
    archive = _tar_with({"../../evil.txt": b"x", "db/app.db": b""}, data / "evil.tar.gz")
    with pytest.raises(tarfile.FilterError):
        restore.restore_archive(Settings(), archive)
    assert not (data.parent / "evil.txt").exists()


def test_restore_downloads_the_newest_then_rebuilds(data, monkeypatch):
    archive = _archive(data)
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": "otw-backup-20261101T000000Z.tar.gz"}, {"Key": archive.name}]
    }
    client.download_file.side_effect = lambda bucket, key, dest: Path(dest).write_bytes(
        archive.read_bytes()
    )
    monkeypatch.setenv("BACKUP_BUCKET", "b")
    get_settings.cache_clear()
    monkeypatch.setattr(restore, "_client", lambda settings: client)
    rebuild = MagicMock(return_value={})
    monkeypatch.setattr("app.retrieval.rebuild.rebuild_index", rebuild)
    monkeypatch.setattr("app.core.db.init_db", lambda: None)

    assert restore.restore() == archive.name
    assert client.download_file.call_args.args[1] == archive.name
    rebuild.assert_called_once_with(fresh=True)


def test_restore_refuses_an_unknown_key_and_an_empty_bucket(data, monkeypatch):
    client = MagicMock()
    client.list_objects_v2.return_value = {}
    monkeypatch.setenv("BACKUP_BUCKET", "b")
    get_settings.cache_clear()
    monkeypatch.setattr(restore, "_client", lambda settings: client)
    with pytest.raises(restore.RestoreError, match="no backups"):
        restore.restore()
    with pytest.raises(restore.RestoreError, match="no backup named"):
        restore.restore(key="otw-backup-nope.tar.gz")


def test_the_command_needs_yes(data):
    with pytest.raises(SystemExit):
        restore.main([])


def test_rebuild_refills_every_collection_from_the_database(tmp_path, monkeypatch):
    from app.core.db import (
        Account,
        Experience,
        ExperiencePoint,
        JobPosting,
        Repository,
        SkillEvidence,
        get_db,
        init_db,
    )
    from app.retrieval.index import COLLECTION, EXPERIENCE_POINTS_COLLECTION
    from app.retrieval.rebuild import COLLECTIONS, rebuild_index

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    init_db()
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    db = get_db()
    account = Account(first_name="A", last_name="B", github_username="a")
    db.add(account)
    db.commit()
    repo = Repository(
        github_id=1, name="p", full_name="a/p", url="https://x", manifests_json={},
        account_id=account.id,
    )
    exp = Experience(account_id=account.id, company="C", title="Dev")
    db.add_all([repo, exp])
    db.commit()
    db.add_all(
        [
            SkillEvidence(
                skill="Rust", repo_id=repo.id, evidence_type="declared_dependency",
                weight=1.0, confidence=1.0, source_files_json=[],
            ),
            ExperiencePoint(experience_id=exp.id, text="Built a thing", order_index=0),
            JobPosting(
                account_id=account.id, source="pasted", external_id="j", company="C",
                title="T", raw_text_quarantined="x", content_hash="h",
            ),
        ]
    )
    db.commit()
    account_id = account.id
    db.close()

    counts = rebuild_index(fresh=True)

    assert counts["skill_evidence"] == 1 and counts["experience_points"] == 1
    assert counts["job_postings"] == 0  # not extracted yet, so nothing to embed
    client = vectorstore_module.get_client()
    point = client.retrieve(COLLECTION, ids=[1], with_payload=True)[0]
    assert point.payload["account_id"] == account_id
    assert client.count(EXPERIENCE_POINTS_COLLECTION).count == 1

    # Fresh drops what the database no longer has.
    from qdrant_client.models import PointStruct

    client.upsert(
        COLLECTION,
        points=[PointStruct(id=999, vector=[0.0, 1.0], payload={"account_id": account_id})],
    )
    rebuild_index(fresh=True)
    assert client.count(COLLECTION).count == 1
    assert set(COLLECTIONS) >= {COLLECTION, EXPERIENCE_POINTS_COLLECTION}
    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    get_settings.cache_clear()
