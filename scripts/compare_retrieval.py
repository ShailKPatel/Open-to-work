"""Compares retrieval strategies on the synthetic and real-text eval sets,
so a change to how the app searches is chosen by a number, not a guess.

Every strategy ranks the same documents for the same golden pairs: the
skill evidence rows (and, on the synthetic set, experience points) of each
profile, against each posting's query text. Documents are embedded with
the app's own model and scored by exact cosine similarity, which is what
Qdrant returns for collections this size, so no vector store is involved.
Runs inside app/evals/synthetic.py's isolated_environment, so the
embedding cache it fills is a throwaway one.

Strategies, all of which return a ranked list of document ids:
  dense            the full query against each document, as the app does now
  dense+instr      the same, with the BGE query instruction prefixed
  bm25             keyword baseline over the same document text
  hybrid           dense+instr and bm25 fused by reciprocal rank
  decomposed       one dense+instr query per skill the posting names, plus
                   the full query, fused by reciprocal rank
  decomposed+bm25  decomposed, with a bm25 run per sub-query as well
  X+rerank         X's top candidates re-scored by a cross-encoder

and two document texts: the current "Skill: X. Evidence: Y." and a
context-carrying one that adds the project or role the skill came from.

Usage: .venv/bin/python -m scripts.compare_retrieval [--real] [--rerankers NAME ...]
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.evals.bm25 import Bm25Corpus, tokenize
from app.evals.golden import GoldenPair
from app.evals.synthetic import (
    Job,
    Persona,
    evidence_rows,
    golden_pairs,
    isolated_environment,
    load_jobs,
    load_personas,
    point_ids,
)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_RRF_K = 60
_CANDIDATES = 50
_DEFAULT_RERANKERS = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base")


@dataclass
class Doc:
    id: int
    text: str


def _doc_texts(persona: Persona, variant: str) -> dict[str, list[Doc]]:
    repos = {r.key: r for r in persona.repos}
    roles = {r.key: r for r in persona.roles}
    evidence = []
    for row in evidence_rows(persona):
        if variant == "current":
            text = f"Skill: {row.skill}. Evidence: {row.evidence_type}."
        elif row.repo_key is not None:
            repo = repos[row.repo_key]
            text = f"Skill: {row.skill}. Project: {repo.name}. {repo.description}"
        else:
            role = roles[row.role_key or ""]
            text = f"Skill: {row.skill}. Role: {role.title} at {role.company}."
        evidence.append(Doc(row.point_id, text))
    points = [Doc(pid, point.text) for pid, _, point in point_ids(persona)]
    return {"skill_evidence": evidence, "experience_points": points}


def _rrf(rankings: list[list[int]]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class Ranker:
    """Ranks one profile's documents of one collection, caching every
    embedding so the strategies below differ only in how they combine."""

    def __init__(self, docs: list[Doc], embed: Callable[[list[str]], list[list[float]]]):
        self.docs = docs
        self.ids = [d.id for d in docs]
        self.embed = embed
        self.matrix = np.array(embed([d.text for d in docs])) if docs else np.zeros((0, 1))
        self.bm25 = Bm25Corpus([(d.id, d.text) for d in docs])
        self._bm25_scores: dict[str, list[int]] = {}

    def dense(self, query: str, instruction: bool = True) -> list[int]:
        if not self.docs:
            return []
        text = (QUERY_INSTRUCTION + query) if instruction else query
        scores = self.matrix @ np.array(self.embed([text])[0])
        return [self.ids[i] for i in np.argsort(-scores)]

    def keyword(self, query: str) -> list[int]:
        """BM25, dropping documents that share no term with the query: a
        zero score is no match, and keeping those rows would hand fusion an
        arbitrary tail."""
        if not self.docs or self.bm25._bm25 is None:
            return []
        scores = self.bm25._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [self.ids[i] for i in order if scores[i] > 0]


def _strategies(job: Job, query: str, ranker: Ranker) -> dict[str, list[int]]:
    subqueries = [s for s in job.skills]
    full_dense = ranker.dense(query)
    full_bm25 = ranker.keyword(query)
    decomposed_runs = [full_dense] + [ranker.dense(s) for s in subqueries]
    return {
        "dense": ranker.dense(query, instruction=False),
        "dense+instr": full_dense,
        "bm25": full_bm25,
        "hybrid": _rrf([full_dense, full_bm25]),
        "decomposed": _rrf(decomposed_runs),
        "decomposed+bm25": _rrf(
            decomposed_runs + [full_bm25] + [ranker.keyword(s) for s in subqueries]
        ),
    }


def _metrics(ranking: list[int], relevant: set[int]) -> dict[str, float]:
    top5, top10 = ranking[:5], ranking[:10]
    hits10 = sum(1 for d in top10 if d in relevant)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top10) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    first = next((i for i, d in enumerate(ranking) if d in relevant), None)
    return {
        "p@5": sum(1 for d in top5 if d in relevant) / 5,
        "r@10": hits10 / len(relevant),
        "capped r@10": hits10 / min(len(relevant), 10),
        "ndcg@10": dcg / ideal if ideal else 0.0,
        "mrr": 1 / (first + 1) if first is not None else 0.0,
    }


def _rerank(model: object, query: str, ranking: list[int], docs: dict[int, str]) -> list[int]:
    head = ranking[:_CANDIDATES]
    scores = model.predict([(query, docs[d]) for d in head])  # type: ignore[attr-defined]
    reranked = [d for _, d in sorted(zip(scores, head, strict=True), key=lambda x: -x[0])]
    return reranked + ranking[_CANDIDATES:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies.")
    parser.add_argument("--real", action="store_true", help="use the real-text set")
    parser.add_argument("--rerankers", nargs="*", default=list(_DEFAULT_RERANKERS))
    args = parser.parse_args()

    if args.real:
        from app.evals.real import build_jobs, build_portfolio

        personas, jobs = [build_portfolio()], build_jobs()
    else:
        personas, jobs = load_personas(), load_jobs()
    pairs = golden_pairs(personas, jobs)
    jobs_by_key = {j.key: j for j in jobs}

    from sentence_transformers import CrossEncoder

    rerankers = {name: CrossEncoder(name) for name in args.rerankers}

    with isolated_environment():
        from app.core.db import init_db
        from app.core.llm import embed

        init_db()
        results: dict[tuple[str, str], list[dict[str, float]]] = {}
        for variant in ("current", "context"):
            for persona in personas:
                collections = _doc_texts(persona, variant)
                rankers = {c: Ranker(docs, embed) for c, docs in collections.items()}
                texts = {c: {d.id: d.text for d in docs} for c, docs in collections.items()}
                mine: list[GoldenPair] = [p for p in pairs if p.account_id == persona.account_id]
                for pair in mine:
                    job = jobs_by_key[pair.id.split("@", 1)[0]]
                    ranker = rankers[pair.collection]
                    relevant = set(pair.relevant_ids)
                    runs = _strategies(job, pair.query_text, ranker)
                    for base in ("hybrid", "decomposed+bm25"):
                        for name, model in rerankers.items():
                            short = name.split("/")[-1]
                            runs[f"{base}+rerank[{short}]"] = _rerank(
                                model, pair.query_text, runs[base], texts[pair.collection]
                            )
                    for strategy, ranking in runs.items():
                        key = (f"{variant}/{pair.collection}", strategy)
                        results.setdefault(key, []).append(_metrics(ranking, relevant))

    names = ("p@5", "r@10", "capped r@10", "ndcg@10", "mrr")
    print(f"{'documents/collection':34}{'strategy':44}" + "".join(f"{n:>13}" for n in names))
    for (group, strategy), rows in sorted(results.items()):
        means = [sum(r[n] for r in rows) / len(rows) for n in names]
        print(f"{group:34}{strategy:44}" + "".join(f"{m:>13.3f}" for m in means))
    print(f"\n{len(pairs)} golden pairs; recall ceiling for r@10 is below 1 when a pair has")
    print("more than 10 relevant documents, which capped r@10 corrects for.")


if __name__ == "__main__":
    main()
