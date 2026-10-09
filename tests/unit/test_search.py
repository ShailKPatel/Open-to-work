"""app/retrieval/search.py: the read path over the collections
app/retrieval/index.py writes into. Uses real Qdrant (:memory: mode, no
mocked client, same escape hatch app/retrieval/vectorstore.py already
exposes for tests) so ranking and filtering are proven against the real
query API, not a stand-in.
"""

from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import ExperienceSkillEvidence, JobPosting, RoleFamily, SkillEvidence, init_db
from app.core.settings import get_settings
from app.retrieval.index import (
    index_experience_points,
    index_experience_skill_evidence,
    index_job_posting,
    index_role_family,
    index_skill_evidence,
)
from app.retrieval.search import (
    search_experience_points,
    search_job_postings,
    search_role_families,
    search_skill_evidence,
)


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _fake_embed(monkeypatch, target: str = "app.retrieval.index.embed"):
    """Deterministic stand-in vectors: text containing "python" points one
    direction, everything else points another, so a query for "python" can
    be asserted to actually rank the python-tagged point first rather than
    just returning something.
    """

    def _encode(texts):
        return [[1.0, 0.0] if "python" in t.lower() else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(target, _encode)
    monkeypatch.setattr("app.retrieval.search.embed", _encode)


def _evidence(**overrides) -> SkillEvidence:
    defaults = dict(
        id=1, skill="Python", repo_id=1, evidence_type="declared_dependency",
        weight=0.7, confidence=1.0,
    )
    defaults.update(overrides)
    return SkillEvidence(**defaults)


def test_search_returns_nothing_when_collection_never_created(tmp_path):
    _reset(tmp_path)
    assert search_skill_evidence("python backend", account_id=1) == []


def test_search_filters_to_the_right_account(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    index_skill_evidence([_evidence(id=1, skill="Python", repo_id=1)], account_id=1)
    index_skill_evidence([_evidence(id=2, skill="Python", repo_id=2)], account_id=2)

    hits = search_skill_evidence("python", account_id=1)

    assert len(hits) == 1
    assert hits[0].payload["repo_id"] == 1


def test_search_ranks_the_closer_match_first(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    index_skill_evidence(
        [
            _evidence(id=1, skill="Python", repo_id=1),
            _evidence(id=2, skill="Woodworking", repo_id=2),
        ],
        account_id=1,
    )

    hits = search_skill_evidence("python developer", account_id=1)

    assert hits[0].payload["skill"] == "Python"


def test_search_source_type_narrows_repo_vs_experience(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    index_skill_evidence([_evidence(id=1, skill="Python", repo_id=1)], account_id=1)
    index_experience_skill_evidence(
        [
            ExperienceSkillEvidence(
                id=1, skill="Python", experience_id=1, evidence_type="manual",
                weight=1.0, confidence=1.0,
            )
        ],
        account_id=1,
    )

    all_hits = search_skill_evidence("python", account_id=1)
    repo_only = search_skill_evidence("python", account_id=1, source_type="repo")
    experience_only = search_skill_evidence("python", account_id=1, source_type="experience")

    assert len(all_hits) == 2
    assert len(repo_only) == 1 and repo_only[0].payload["source_type"] == "repo"
    assert len(experience_only) == 1 and experience_only[0].payload["source_type"] == "experience"


def test_search_experience_points(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    from app.core.db import ExperiencePoint

    index_experience_points(
        [
            ExperiencePoint(
                id=1, experience_id=1, text="Built a Python backend service", order_index=1
            ),
            ExperiencePoint(
                id=2, experience_id=1, text="Organized a woodworking workshop", order_index=2
            ),
        ],
        account_id=1,
    )

    hits = search_experience_points("python backend work", account_id=1)

    assert hits[0].payload["text"] == "Built a Python backend service"


def test_search_experience_points_scoped_to_one_role(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    from app.core.db import ExperiencePoint

    index_experience_points(
        [
            ExperiencePoint(id=1, experience_id=1, text="Built a Python backend", order_index=1),
            ExperiencePoint(
                id=2, experience_id=2, text="Built another Python backend", order_index=1
            ),
        ],
        account_id=1,
    )

    hits = search_experience_points("python", account_id=1, experience_id=2)

    assert len(hits) == 1
    assert hits[0].payload["experience_id"] == 2


def test_search_top_k_limits_results(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    index_skill_evidence(
        [_evidence(id=i, skill="Python", repo_id=i) for i in range(1, 6)], account_id=1
    )

    hits = search_skill_evidence("python", account_id=1, top_k=2)

    assert len(hits) == 2


def test_search_role_families_is_not_account_scoped(tmp_path, monkeypatch):
    """The one global search in this module: no account_id
    filter at all, since role families are a shared taxonomy, not
    per-account data."""
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    index_role_family(RoleFamily(id=1, canonical_name="Python Backend Engineer"))
    index_role_family(RoleFamily(id=2, canonical_name="Woodworking Specialist"))

    hits = search_role_families("python role")

    assert hits[0].payload["canonical_name"] == "Python Backend Engineer"


def test_search_job_postings_scoped_by_account(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    def _posting(**overrides):
        defaults = dict(
            id=1, account_id=1, source="pasted", external_id="h", company="Acme",
            title="Python Backend Engineer", raw_text_quarantined="t", content_hash="h",
            extraction_status="extracted", extracted_json={"skills_required": []},
        )
        defaults.update(overrides)
        return JobPosting(**defaults)

    index_job_posting(_posting(id=1), account_id=1)
    index_job_posting(_posting(id=2, content_hash="h2"), account_id=2)

    hits = search_job_postings("python backend", account_id=1)

    assert len(hits) == 1
    assert hits[0].id == 1


def test_hybrid_finds_named_skills_the_vector_alone_misses(tmp_path, monkeypatch):
    """The stand-in embedding only knows "python", so a dense search for a
    posting about Go and Kafka ranks the rows arbitrarily. One keyword
    sub-query per named skill is what puts them first."""
    from app.retrieval.search import Query

    _reset(tmp_path)
    _fake_embed(monkeypatch)
    rows = [
        _evidence(id=1, skill="Python"),
        _evidence(id=2, skill="Go"),
        _evidence(id=3, skill="Apache Kafka"),
        _evidence(id=4, skill="Ruby"),
    ]
    index_skill_evidence(rows, account_id=1)

    query = Query(text="Backend engineer at Acme", skills=("Go", "Apache Kafka"))
    hits = search_skill_evidence(query, account_id=1, top_k=2)

    assert {h.payload["skill"] for h in hits} == {"Go", "Apache Kafka"}
    assert all(0 < h.score <= 1 for h in hits)


def test_a_skill_the_account_lacks_adds_no_ranking(tmp_path, monkeypatch):
    """Skills get keyword runs only. A dense run for "Angular", which this
    account has no evidence for, would still rank every row; a keyword
    run matches nothing and leaves the fusion alone."""
    from app.retrieval import search
    from app.retrieval.search import Query

    _reset(tmp_path)
    _fake_embed(monkeypatch)
    embedded: list[str] = []
    original = search.embed

    def counting(texts):
        embedded.extend(texts)
        return original(texts)

    monkeypatch.setattr("app.retrieval.search.embed", counting)
    index_skill_evidence(
        [_evidence(id=1, skill="Python"), _evidence(id=2, skill="Go")], account_id=1
    )

    with_missing = search_skill_evidence(
        Query(text="Backend engineer", skills=("Go", "Angular", "C#")), account_id=1, top_k=2
    )
    without = search_skill_evidence(
        Query(text="Backend engineer", skills=("Go",)), account_id=1, top_k=2
    )

    assert [h.id for h in with_missing] == [h.id for h in without]
    assert len(embedded) == 2  # one dense query per search, never one per skill


def test_dense_mode_is_the_single_vector_search(tmp_path, monkeypatch):
    from app.retrieval.search import Query

    _reset(tmp_path)
    _fake_embed(monkeypatch)
    index_skill_evidence(
        [_evidence(id=1, skill="Python"), _evidence(id=2, skill="Go")], account_id=1
    )

    hits = search_skill_evidence(Query(text="python", skills=("Go",)), account_id=1, mode="dense")

    # Skills are ignored and the score is cosine similarity.
    assert hits[0].payload["skill"] == "Python"
    assert hits[0].score == 1.0


def test_hybrid_experience_points_respect_the_role_filter(tmp_path, monkeypatch):
    from app.core.db import ExperiencePoint

    _reset(tmp_path)
    _fake_embed(monkeypatch)
    points = [
        ExperiencePoint(id=1, experience_id=10, text="Built python services", order_index=0),
        ExperiencePoint(id=2, experience_id=20, text="Built python pipelines", order_index=0),
    ]
    index_experience_points(points, account_id=1)

    hits = search_experience_points("python", account_id=1, experience_id=20)

    assert [h.id for h in hits] == [2]


def test_query_from_posting_text_reads_the_skills_line():
    from app.retrieval.search import Query

    query = Query.from_posting_text("Engineer at Acme\nBuilds APIs.\nSkills: Go, , Kafka")
    assert query.skills == ("Go", "Kafka")
    assert Query.from_posting_text("Engineer at Acme").skills == ()


def test_query_for_posting_uses_extracted_fields_when_present():
    from app.retrieval.search import query_for_posting

    extracted = JobPosting(
        company="Acme", title="Engineer", raw_text_quarantined="long raw text",
        extracted_json={
            "role_summary": "Builds APIs.",
            "skills_required": [{"skill": "Go", "level": ""}, {"skill": "SQL", "level": ""}],
        },
    )
    query = query_for_posting(extracted)
    assert query.text.startswith("Engineer at Acme")
    assert "long raw text" not in query.text
    assert query.skills == ("Go", "SQL")

    raw = JobPosting(
        company="Acme", title="Engineer", raw_text_quarantined="long raw text",
        extracted_json=None,
    )
    assert query_for_posting(raw).text == "long raw text"
    assert query_for_posting(raw).skills == ()


def test_sub_queries_dedupe_and_cap():
    from app.retrieval import search

    query = search.Query(text="Go", skills=("go", "SQL", "sql", *[f"s{i}" for i in range(40)]))
    subs = search._sub_queries(query)
    assert subs[:2] == ["Go", "SQL"]
    assert len(subs) == search._MAX_SKILL_QUERIES + 1


def test_fuse_scales_to_one_and_breaks_ties_by_id():
    from app.retrieval.search import _fuse

    fused = _fuse([[5, 9], [5, 9]])
    assert fused[0] == (5, 1.0)
    assert _fuse([[9], [4]]) == [(4, 0.5), (9, 0.5)]
