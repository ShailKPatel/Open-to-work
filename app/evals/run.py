"""The eval harness's entry point: `run_eval(account_id) -> MetricsReport`.
`make eval` calls this (see app/evals/__main__.py).

Depends on `app.retrieval` (the dense search/index this measures). The
groundedness pass (app/evals/groundedness.py) is scored against
app/resume_build/orchestrator.py's bullet generation.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sqlalchemy import select

from app.core.db import (
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    LLMCall,
    Repository,
    SkillEvidence,
    get_db,
)
from app.core.settings import get_settings
from app.evals.bm25 import Bm25Corpus
from app.evals.candidates import candidate_scores
from app.evals.golden import GoldenPair, load_golden_set
from app.evals.groundedness import score_groundedness
from app.evals.metrics import (
    PairScore,
    SystemScore,
    mean_system_score,
    paired_difference,
    precision_at_k,
    score_pair,
)
from app.retrieval.index import evidence_text, experience_evidence_point_id
from app.retrieval.search import (
    Query,
    SearchMode,
    search_experience_points,
    search_skill_evidence,
)

logger = logging.getLogger(__name__)

_RECALL_K = 10


@dataclass
class MetricsReport:
    generated_at: str
    account_id: int
    golden_set_size: int
    pairs_scored: int
    # The app's retrieval as it runs (hybrid, app/retrieval/search.py), its
    # dense run alone, and the keyword baseline. The single-vector dense
    # search the hybrid replaced is `dense_single`, below.
    retrieval: dict
    dense: dict
    bm25: dict
    retrieval_beats_bm25: bool | None
    precision_at_10: float | None
    groundedness: float | None
    groundedness_checked: int
    cost_usd: float
    latency_ms_avg: float
    llm_calls: int
    notes: list[str]
    # What the resume builder hands the model, scored against the same
    # pairs (app/evals/candidates.py): {"skills": ..., "projects": ...},
    # each shaped like `retrieval`.
    candidates: dict = field(default_factory=dict)
    # Paired differences per metric over the same pairs, keyed
    # "retrieval - bm25" and "retrieval - dense", each metric
    # {"mean": ..., "ci95": [low, high]}: an interval spanning zero means
    # these pairs do not separate the two systems.
    differences: dict = field(default_factory=dict)
    # Every pair's scores per system, {"retrieval": [PairScore as dict, ...]},
    # in the same pair order for each system, so results across several
    # profiles can be pooled and bootstrapped as one sample.
    per_pair: dict = field(default_factory=dict)
    dense_single: dict = field(default_factory=dict)


def _corpus_for(collection: str, account_id: int) -> list[tuple[int, str]]:
    """All (id, text) rows this account currently has in the given
    collection, read straight from SQLite, not Qdrant, so the BM25
    baseline is scored against a corpus definition independent of
    whatever happens to actually be indexed in Qdrant at eval time (a
    stale or partial index shouldn't make either system look artificially
    better or worse than the other). Ids match the exact space
    app/retrieval/search.py's hits come back in (offset ids for
    experience-linked skill evidence), so scoring against golden labels
    (which are recorded against real search results, see
    scripts/label_golden_set.py) lines up correctly.
    """
    db = get_db()
    try:
        if collection == "skill_evidence":
            repo_rows = (
                db.execute(
                    select(SkillEvidence)
                    .join(Repository, SkillEvidence.repo_id == Repository.id)
                    .where(Repository.account_id == account_id)
                )
                .scalars()
                .all()
            )
            exp_rows = (
                db.execute(
                    select(ExperienceSkillEvidence)
                    .join(Experience, ExperienceSkillEvidence.experience_id == Experience.id)
                    .where(Experience.account_id == account_id)
                )
                .scalars()
                .all()
            )
            return [(e.id, evidence_text(e)) for e in repo_rows] + [
                (experience_evidence_point_id(e.id), evidence_text(e)) for e in exp_rows
            ]
        if collection == "experience_points":
            rows = (
                db.execute(
                    select(ExperiencePoint)
                    .join(Experience, ExperiencePoint.experience_id == Experience.id)
                    .where(Experience.account_id == account_id)
                )
                .scalars()
                .all()
            )
            return [(p.id, p.text) for p in rows]
        raise ValueError(f"unknown collection {collection!r}")
    finally:
        db.close()


def _search_hits(pair: GoldenPair, account_id: int, mode: SearchMode) -> list[int]:
    """The ranked ids the app's search returns for a pair. A golden query
    is posting text in job_posting_text's shape, so its "Skills:" line
    becomes the query's skills, the same ones the app reads from an
    extracted posting (search.py's query_for_posting)."""
    query = replace(Query.from_posting_text(pair.query_text), posting_text=pair.posting_text)
    if pair.collection == "skill_evidence":
        hits = search_skill_evidence(query, account_id, top_k=_RECALL_K, mode=mode)
    else:
        hits = search_experience_points(query, account_id, top_k=_RECALL_K, mode=mode)
    return [h.id for h in hits]


def _cost_and_latency_since(account_id: int, since: dt.datetime) -> tuple[float, float, int]:
    """Cost/latency attributable to this eval run specifically: every
    LLMCall row for this account created since the run started (the
    groundedness pass is the only part of run_eval that spends anything;
    the dense/BM25 retrieval comparison spends nothing)."""
    db = get_db()
    try:
        rows = list(
            db.execute(
                select(LLMCall).where(
                    LLMCall.account_id == account_id, LLMCall.created_at >= since
                )
            ).scalars()
        )
        if not rows:
            return 0.0, 0.0, 0
        total_cost = sum(r.cost_usd for r in rows)
        avg_latency = sum(r.latency_ms for r in rows) / len(rows)
        return total_cost, avg_latency, len(rows)
    finally:
        db.close()


def run_eval(
    account_id: int,
    golden_path: Path | None = None,
    include_groundedness: bool = True,
) -> MetricsReport:
    run_started_at = dt.datetime.now(dt.UTC)
    settings = get_settings()
    path = golden_path or Path(settings.evals_golden_dir) / "golden_set.yaml"
    all_pairs = load_golden_set(path)
    pairs = [p for p in all_pairs if p.account_id == account_id]

    notes: list[str] = []
    partial_pairs: list[str] = []
    retrieval_points: list[PairScore] = []
    dense_points: list[PairScore] = []
    dense_single_points: list[PairScore] = []
    bm25_points: list[PairScore] = []
    precision_at_10_points: list[float] = []
    corpora: dict[str, Bm25Corpus] = {}

    for pair in pairs:
        relevant = set(pair.relevant_ids)
        if not relevant:
            notes.append(f"pair {pair.id!r} has no labeled relevant ids, skipped")
            continue
        if pair.collection not in ("skill_evidence", "experience_points"):
            notes.append(f"pair {pair.id!r} has unknown collection {pair.collection!r}, skipped")
            continue
        if pair.partial:
            # Excluded rather than scored: in a partly judged pair the
            # candidates nobody ruled on are indistinguishable from ones ruled
            # irrelevant, so precision is biased down by an unknown amount and
            # recall counts relevant evidence that was never offered a label.
            partial_pairs.append(pair.id)
            continue

        retrieval_hits = _search_hits(pair, account_id, "hybrid")
        dense_hits = _search_hits(pair, account_id, "dense")
        dense_single_hits = _search_hits(pair, account_id, "dense-single")
        if pair.collection not in corpora:
            corpora[pair.collection] = Bm25Corpus(_corpus_for(pair.collection, account_id))
        bm25_hits = corpora[pair.collection].top_k(pair.query_text, _RECALL_K)

        retrieval_points.append(score_pair(retrieval_hits, relevant))
        dense_points.append(score_pair(dense_hits, relevant))
        dense_single_points.append(score_pair(dense_single_hits, relevant))
        bm25_points.append(score_pair(bm25_hits, relevant))
        # Plain precision@10: precision over the full retrieved window a
        # downstream generation step would actually be handed (top-10,
        # not the ranking-quality top-5), scored against real hand-labeled
        # ground truth. Not the rank-weighted RAGAS "context precision".
        precision_at_10_points.append(precision_at_k(retrieval_hits, relevant, _RECALL_K))

    retrieval_result: SystemScore = mean_system_score(retrieval_points)
    dense_result: SystemScore = mean_system_score(dense_points)
    bm25_result: SystemScore = mean_system_score(bm25_points)
    scored = retrieval_result.pairs_scored

    retrieval_beats_bm25 = None
    if scored > 0:
        retrieval_beats_bm25 = (
            retrieval_result.precision_at_5 >= bm25_result.precision_at_5
            and retrieval_result.recall_at_10 >= bm25_result.recall_at_10
        )

    differences: dict[str, dict[str, dict]] = {}
    if scored > 1:
        references = (("retrieval - bm25", bm25_points), ("retrieval - dense", dense_points))
        for label, other in references:
            differences[label] = {}
            for metric in ("precision_at_5", "recall_at_10", "ndcg_at_10"):
                mean, interval = paired_difference(
                    [getattr(p, metric) for p in retrieval_points],
                    [getattr(p, metric) for p in other],
                )
                differences[label][metric] = {
                    "mean": mean,
                    "ci95": list(interval) if interval is not None else None,
                }

    if partial_pairs:
        notes.append(
            f"{len(partial_pairs)} partly judged pair(s) excluded "
            f"({', '.join(sorted(partial_pairs))}); finish labeling them with "
            "scripts/label_golden_set.py and they rejoin the scores"
        )

    if scored == 0:
        notes.append(
            "no golden pairs with labeled relevant ids for this account; run "
            "scripts/label_golden_set.py to build a real golden set before "
            "trusting these numbers"
        )
    elif scored < 50:
        notes.append(
            f"only {scored} labeled pairs scored (target is 50); "
            "numbers below this size are directional, not final"
        )

    precision_at_10 = (
        sum(precision_at_10_points) / len(precision_at_10_points)
        if precision_at_10_points
        else None
    )

    groundedness_score: float | None = None
    groundedness_checked = 0
    if include_groundedness:
        result = score_groundedness(account_id)
        groundedness_score = result.score
        groundedness_checked = result.checked
        if result.skipped_reason:
            notes.append(f"groundedness: {result.skipped_reason}")
    else:
        notes.append("groundedness check skipped (include_groundedness=False)")

    cost_usd, latency_ms_avg, llm_calls = _cost_and_latency_since(account_id, run_started_at)
    candidate_points = candidate_scores(account_id, pairs)

    return MetricsReport(
        generated_at=dt.datetime.now(dt.UTC).isoformat(),
        account_id=account_id,
        golden_set_size=len(pairs),
        pairs_scored=scored,
        retrieval=asdict(retrieval_result),
        dense=asdict(dense_result),
        dense_single=asdict(mean_system_score(dense_single_points)),
        bm25=asdict(bm25_result),
        retrieval_beats_bm25=retrieval_beats_bm25,
        precision_at_10=precision_at_10,
        groundedness=groundedness_score,
        groundedness_checked=groundedness_checked,
        cost_usd=cost_usd,
        latency_ms_avg=latency_ms_avg,
        llm_calls=llm_calls,
        notes=notes,
        differences=differences,
        per_pair={
            "retrieval": [asdict(p) for p in retrieval_points],
            "dense": [asdict(p) for p in dense_points],
            "dense_single": [asdict(p) for p in dense_single_points],
            "bm25": [asdict(p) for p in bm25_points],
            "candidate_skills": [asdict(p) for p in candidate_points["skills"]],
        },
        candidates={
            name: asdict(mean_system_score(points)) for name, points in candidate_points.items()
        },
    )


def write_report(report: MetricsReport, results_dir: Path | None = None) -> Path:
    settings = get_settings()
    directory = results_dir or Path(settings.evals_results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = directory / f"eval-account{report.account_id}-{stamp}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2))
    return out_path
