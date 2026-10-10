"""Compares embedding models for retrieval on the synthetic and real-text
eval sets, so the model in app/core/embeddings.py is chosen by a number.

Each model embeds the same skill evidence documents and is scored two
ways: on its own (one dense query per posting) and inside the hybrid search
the app ships (one sub-query per named skill plus the full query, dense and
BM25, fused by reciprocal rank; app/retrieval/search.py). The second is the
one that decides, since it is what runs. Each model gets the query and
document prefixes its authors specify, and the time to embed the profile is
reported because the app embeds on a laptop CPU.

Runs inside isolated_environment(), so the embedding cache it fills is a
throwaway one.

Usage: .venv/bin/python -m scripts.compare_embedding_models [--real]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from app.evals.synthetic import golden_pairs, isolated_environment, load_jobs, load_personas
from scripts.compare_retrieval import Doc, _doc_texts, _metrics, _rrf

# (model, query prefix, document prefix), each as its model card specifies.
MODELS: tuple[tuple[str, str, str], ...] = (
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    (
        "mixedbread-ai/mxbai-embed-large-v1",
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    ("intfloat/e5-base-v2", "query: ", "passage: "),
    ("thenlper/gte-base", "", ""),
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
)


class ModelRanker:
    """Exact cosine ranking of one profile's documents under one model."""

    def __init__(self, docs: list[Doc], model: str, query_prefix: str, doc_prefix: str):
        from app.core.embeddings import embed
        from app.evals.bm25 import Bm25Corpus

        self.ids = [d.id for d in docs]
        self.model, self.query_prefix = model, query_prefix
        self.embed = embed
        started = time.perf_counter()
        self.matrix = np.array(embed([doc_prefix + d.text for d in docs], model_name=model))
        self.embed_seconds = time.perf_counter() - started
        self.keyword = Bm25Corpus([(d.id, d.text) for d in docs])

    def dense(self, query: str) -> list[int]:
        vector = np.array(self.embed([self.query_prefix + query], model_name=self.model)[0])
        return [self.ids[i] for i in np.argsort(-(self.matrix @ vector))]

    def hybrid(self, query: str, skills: list[str]) -> list[int]:
        sub_queries = [query, *dict.fromkeys(skills)]
        runs = []
        for text in sub_queries:
            runs.append(self.dense(text))
            runs.append(self.keyword.top_k(text, 50))
        return _rrf(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding models for retrieval.")
    parser.add_argument("--real", action="store_true", help="use the real-text set")
    args = parser.parse_args()

    if args.real:
        from app.evals.real import build_jobs, build_portfolio

        personas, jobs = [build_portfolio()], build_jobs()
    else:
        personas, jobs = load_personas(), load_jobs()
    pairs = [p for p in golden_pairs(personas, jobs) if p.collection == "skill_evidence"]
    jobs_by_key = {j.key: j for j in jobs}

    names = ("p@5", "ndcg@10", "mrr")
    print(f"{'model':42}{'system':10}" + "".join(f"{n:>10}" for n in names) + f"{'embed s':>10}")
    with isolated_environment():
        from app.core.db import init_db

        init_db()
        for model, query_prefix, doc_prefix in MODELS:
            rows: dict[str, list[dict[str, float]]] = {"dense": [], "hybrid": []}
            seconds = 0.0
            for persona in personas:
                docs = _doc_texts(persona, "current")["skill_evidence"]
                ranker = ModelRanker(docs, model, query_prefix, doc_prefix)
                seconds += ranker.embed_seconds
                for pair in (p for p in pairs if p.account_id == persona.account_id):
                    job = jobs_by_key[pair.id.split("@", 1)[0]]
                    relevant = set(pair.relevant_ids)
                    rows["dense"].append(_metrics(ranker.dense(pair.query_text), relevant))
                    rows["hybrid"].append(
                        _metrics(ranker.hybrid(pair.query_text, job.skills), relevant)
                    )
            for system, results in rows.items():
                means = [sum(r[n] for r in results) / len(results) for n in names]
                print(
                    f"{model:42}{system:10}" + "".join(f"{m:>10.3f}" for m in means)
                    + f"{seconds:>10.1f}",
                    flush=True,
                )
    print(f"\n{len(pairs)} skill evidence pairs.")


if __name__ == "__main__":
    main()
