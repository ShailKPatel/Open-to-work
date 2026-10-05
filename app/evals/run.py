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
from dataclasses import asdict, dataclass
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
from app.evals.golden import GoldenPair, load_golden_set
from app.evals.groundedness import score_groundedness
from app.evals.metrics import SystemScore, mean_system_score, precision_at_k, recall_at_k
from app.retrieval.index import evidence_text, experience_evidence_point_id
from app.retrieval.search import search_experience_points, search_skill_evidence

logger = logging.getLogger(__name__)

_PRECISION_K = 5
_RECALL_K = 10


@dataclass
class MetricsReport:
    generated_at: str
    account_id: int
    golden_set_size: int
    pairs_scored: int
    dense: dict
    bm25: dict
    dense_beats_bm25: bool | None
    precision_at_10: float | None
    groundedness: float | None
    groundedness_checked: int
    cost_usd: float
    latency_ms_avg: float
    llm_calls: int
    notes: list[str]


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


def _dense_hits(pair: GoldenPair, account_id: int) -> list[int] | None:
    if pair.collection == "skill_evidence":
        return [h.id for h in search_skill_evidence(pair.query_text, account_id, top_k=_RECALL_K)]
    if pair.collection == "experience_points":
        return [
            h.id for h in search_experience_points(pair.query_text, account_id, top_k=_RECALL_K)
        ]
    return None


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
    dense_points: list[tuple[float, float]] = []
    bm25_points: list[tuple[float, float]] = []
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

        dense_hits = _dense_hits(pair, account_id) or []
        if pair.collection not in corpora:
            corpora[pair.collection] = Bm25Corpus(_corpus_for(pair.collection, account_id))
        bm25_hits = corpora[pair.collection].top_k(pair.query_text, _RECALL_K)

        dense_points.append(
            (
                precision_at_k(dense_hits, relevant, _PRECISION_K),
                recall_at_k(dense_hits, relevant, _RECALL_K),
            )
        )
        bm25_points.append(
            (
                precision_at_k(bm25_hits, relevant, _PRECISION_K),
                recall_at_k(bm25_hits, relevant, _RECALL_K),
            )
        )
        # Plain precision@10: precision over the full retrieved window a
        # downstream generation step would actually be handed (top-10,
        # not the ranking-quality top-5), scored against real hand-labeled
        # ground truth. Not the rank-weighted RAGAS "context precision".
        precision_at_10_points.append(precision_at_k(dense_hits, relevant, _RECALL_K))

    dense_result: SystemScore = mean_system_score(dense_points)
    bm25_result: SystemScore = mean_system_score(bm25_points)
    scored = dense_result.pairs_scored

    dense_beats_bm25 = None
    if scored > 0:
        dense_beats_bm25 = (
            dense_result.precision_at_5 >= bm25_result.precision_at_5
            and dense_result.recall_at_10 >= bm25_result.recall_at_10
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

    return MetricsReport(
        generated_at=dt.datetime.now(dt.UTC).isoformat(),
        account_id=account_id,
        golden_set_size=len(pairs),
        pairs_scored=scored,
        dense=asdict(dense_result),
        bm25=asdict(bm25_result),
        dense_beats_bm25=dense_beats_bm25,
        precision_at_10=precision_at_10,
        groundedness=groundedness_score,
        groundedness_checked=groundedness_checked,
        cost_usd=cost_usd,
        latency_ms_avg=latency_ms_avg,
        llm_calls=llm_calls,
        notes=notes,
    )


def write_report(report: MetricsReport, results_dir: Path | None = None) -> Path:
    settings = get_settings()
    directory = results_dir or Path(settings.evals_results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = directory / f"eval-account{report.account_id}-{stamp}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2))
    return out_path
