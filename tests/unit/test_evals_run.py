"""app/evals/run.py's run_eval(): real DB, real :memory: Qdrant, injected
embedder, same convention as test_search.py. Test accounts have no
generated resumes, so score_groundedness() short-circuits with no LLM
call needed (see test_groundedness.py for that module's own coverage),
keeping these tests free of any LLM mocking noise.
"""

from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import (
    Account,
    Experience,
    ExperiencePoint,
    Repository,
    SkillEvidence,
    get_db,
    init_db,
)
from app.core.settings import get_settings
from app.evals.golden import GoldenPair, save_golden_set
from app.evals.run import run_eval, write_report
from app.retrieval.index import index_experience_points, index_skill_evidence


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _make_account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _fake_embed(monkeypatch):
    def _encode(texts):
        return [[1.0, 0.0] if "python" in t.lower() else [0.0, 1.0] for t in texts]

    import app.retrieval.index as index_module
    import app.retrieval.search as search_module

    monkeypatch.setattr(index_module, "embed", _encode)
    monkeypatch.setattr(search_module, "embed", _encode)


def test_no_golden_pairs_for_account_reports_zero_scored(tmp_path, monkeypatch, tmp_path_factory):
    _reset(tmp_path)
    account_id = _make_account()
    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set([], golden_path)

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)

    assert report.pairs_scored == 0
    assert report.dense_beats_bm25 is None
    assert any("no golden pairs" in n for n in report.notes)


def test_dense_and_bm25_scored_against_real_evidence(tmp_path, monkeypatch, tmp_path_factory):
    _reset(tmp_path)
    _fake_embed(monkeypatch)
    account_id = _make_account()

    db = get_db()
    repo = Repository(
        account_id=account_id, github_id=1, name="proj", full_name="octocat/proj",
        url="https://github.com/octocat/proj", is_fork=False,
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    repo_id = repo.id
    evidence = SkillEvidence(
        id=1, skill="Python", repo_id=repo_id, evidence_type="declared_dependency",
        weight=0.7, confidence=1.0,
    )
    other = SkillEvidence(
        id=2, skill="Woodworking", repo_id=repo_id, evidence_type="declared_dependency",
        weight=0.7, confidence=1.0,
    )
    db.add_all([evidence, other])
    db.commit()
    # index_skill_evidence only reads plain attribute values (skill,
    # evidence_type) that are already loaded; refresh before close so
    # they survive as in-memory values rather than expired-and-detached.
    db.refresh(evidence)
    db.refresh(other)
    db.close()

    index_skill_evidence([evidence, other], account_id=account_id)

    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set(
        [
            GoldenPair(
                id="p1", account_id=account_id, collection="skill_evidence",
                query_text="python backend engineer", relevant_ids=[1],
            )
        ],
        golden_path,
    )

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)

    assert report.pairs_scored == 1
    # only 2 rows exist total, both land in the top-5 window: precision is
    # 1 relevant of 2 retrieved, recall is the 1 relevant id fully found.
    assert report.dense["precision_at_5"] == 0.5
    assert report.dense["recall_at_10"] == 1.0
    assert report.precision_at_10 is not None
    assert report.groundedness is None  # skipped
    assert any("groundedness check skipped" in n for n in report.notes)


def test_experience_points_collection_also_scored(tmp_path, monkeypatch, tmp_path_factory):
    _reset(tmp_path)
    _fake_embed(monkeypatch)
    account_id = _make_account()

    db = get_db()
    exp = Experience(account_id=account_id, title="Eng", company="Acme")
    db.add(exp)
    db.commit()
    db.refresh(exp)
    point = ExperiencePoint(id=1, experience_id=exp.id, text="Built a Python backend service")
    db.add(point)
    db.commit()
    db.refresh(point)
    db.close()

    index_experience_points([point], account_id=account_id)

    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set(
        [
            GoldenPair(
                id="p1", account_id=account_id, collection="experience_points",
                query_text="python backend work", relevant_ids=[1],
            )
        ],
        golden_path,
    )

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)

    assert report.pairs_scored == 1
    assert report.dense["recall_at_10"] == 1.0


def test_pair_with_no_relevant_ids_is_skipped_not_scored(tmp_path, tmp_path_factory):
    _reset(tmp_path)
    account_id = _make_account()
    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set(
        [
            GoldenPair(
                id="p1", account_id=account_id, collection="skill_evidence",
                query_text="x", relevant_ids=[],
            )
        ],
        golden_path,
    )

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)

    assert report.pairs_scored == 0
    assert any("no labeled relevant ids" in n for n in report.notes)


def test_golden_pairs_scoped_to_the_requested_account(tmp_path, tmp_path_factory):
    _reset(tmp_path)
    account_id = _make_account()
    other_account_id = _make_account()
    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set(
        [
            GoldenPair(
                id="p1", account_id=other_account_id, collection="skill_evidence",
                query_text="x", relevant_ids=[1],
            )
        ],
        golden_path,
    )

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)

    assert report.golden_set_size == 0


def test_write_report_creates_a_json_file(tmp_path, tmp_path_factory):
    _reset(tmp_path)
    account_id = _make_account()
    golden_path = tmp_path_factory.mktemp("golden") / "golden_set.yaml"
    save_golden_set([], golden_path)

    report = run_eval(account_id, golden_path=golden_path, include_groundedness=False)
    results_dir = tmp_path / "results"
    out_path = write_report(report, results_dir=results_dir)

    assert out_path.exists()
    assert out_path.suffix == ".json"
    assert str(account_id) in out_path.name
