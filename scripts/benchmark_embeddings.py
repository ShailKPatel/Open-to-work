"""Compares embedding models on this account's own skill names, to decide
what app/core/embeddings.py should use and what text it should embed.

The skill map places a skill by its embedding, so a model that reads
characters instead of meaning puts ElasticNet next to EfficientNet. Two
things decide that: which model encodes the text, and how much context
the text carries. Both are measured here, as a grid, because a weaker
model with context often beats a stronger one without it.

Ground truth is hand-labeled in evals/golden/skill_taxonomy.yaml: a
domain per skill, plus look-alike pairs that must score low and
unrelated-looking pairs that must score high.

Metrics per (model, template):
  purity@5   share of a skill's 5 nearest neighbours in its own domain
  silhouette cosine silhouette of the domain labels (-1 to 1)
  trap       mean similarity of the look-alike pairs (lower is better)
  true       mean similarity of the same-domain pairs (higher is better)
  margin     true - trap, the headline number
  encode     seconds to encode the sample, after the model is loaded

Usage:
  .venv/bin/python -m scripts.benchmark_embeddings --account 1
  .venv/bin/python -m scripts.benchmark_embeddings --account 1 --sample 40
  .venv/bin/python -m scripts.benchmark_embeddings --account 1 \
      --models BAAI/bge-base-en-v1.5,thenlper/gte-base
  .venv/bin/python -m scripts.benchmark_embeddings --account 1 --projections

Models are downloaded to the usual Hugging Face cache on first run
(roughly 5 GB for the default set) and reused after that, so a rerun
with HF_HUB_OFFLINE=1 does no network work at all. Results are
written to evals/results/embeddings-<timestamp>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlalchemy import select

from app.core.db import (
    Experience,
    ExperienceSkillEvidence,
    Repository,
    Skill,
    SkillEvidence,
    get_db,
    init_db,
)

TAXONOMY_PATH = Path("evals/golden/skill_taxonomy.yaml")
RESULTS_DIR = Path("evals/results")


@dataclass
class ModelSpec:
    """One candidate. `query_prefix` is the string the model's own card
    says to put in front of a short standalone phrase; e5 scores far
    worse without it, bge-style models want it only on search queries,
    so it stays per-model rather than global.
    """

    name: str
    query_prefix: str = ""
    trust_remote_code: bool = False
    note: str = ""


DEFAULT_MODELS = [
    ModelSpec("BAAI/bge-base-en-v1.5", note="current default, 109M params, 768d"),
    ModelSpec("BAAI/bge-small-en-v1.5", note="33M params, 384d, 4x faster to load"),
    ModelSpec("sentence-transformers/all-MiniLM-L6-v2", note="22M params, 384d, speed floor"),
    ModelSpec("intfloat/e5-base-v2", query_prefix="query: ", note="110M params, 768d"),
    ModelSpec("thenlper/gte-base", note="110M params, 768d"),
    ModelSpec("mixedbread-ai/mxbai-embed-large-v1", note="335M params, 1024d"),
    # Trained on code or technical text rather than general prose, which
    # is the obvious guess for a list of software skills. None of them
    # won: they are built for code snippets, not for a name followed by
    # a sentence about where it was used.
    ModelSpec(
        "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        note="code search, 82M params, 768d",
    ),
    ModelSpec("Alibaba-NLP/gte-modernbert-base", note="149M params, 768d, slow to encode"),
    ModelSpec(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        note="137M params, 768d",
    ),
    # Does not load on transformers 5.x: its remote code imports
    # find_pruneable_heads_and_indices, which was removed. Left in so a
    # rerun on an older transformers still measures it.
    ModelSpec(
        "jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
        note="trained on code, 161M params, 768d",
    ),
]


# Each template turns a skill name into the string the model actually
# sees. "context" is the only one that gets to look at the database.
TEMPLATES = ("bare", "current", "typed", "context")


@dataclass
class SkillContext:
    """What the database knows about one skill name, flattened into the
    pieces a context template can spend its token budget on.
    """

    name: str
    languages: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    co_skills: list[str] = field(default_factory=list)


def load_taxonomy() -> dict[str, Any]:
    if not TAXONOMY_PATH.exists():
        raise SystemExit(f"missing ground truth: {TAXONOMY_PATH}")
    with TAXONOMY_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_contexts(account_id: int) -> dict[str, SkillContext]:
    """Reads every skill name for an account plus the evidence around it.

    Co-occurring skills are the strongest signal available for free: a
    name that shows up in the same repository as FastAPI and SQLAlchemy
    is a backend thing, whatever its characters look like.
    """
    db = get_db()
    try:
        rows = db.execute(
            select(
                SkillEvidence.skill,
                SkillEvidence.evidence_type,
                Repository.id,
                Repository.full_name,
                Repository.primary_language,
            )
            .join(Repository, Repository.id == SkillEvidence.repo_id)
            .where(Repository.account_id == account_id)
        ).all()

        exp_rows = db.execute(
            select(
                ExperienceSkillEvidence.skill,
                ExperienceSkillEvidence.evidence_type,
                Experience.id,
                Experience.title,
                Experience.company,
            )
            .join(Experience, Experience.id == ExperienceSkillEvidence.experience_id)
            .where(Experience.account_id == account_id)
        ).all()

        manual = (
            db.execute(select(Skill.name).where(Skill.account_id == account_id)).scalars().all()
        )
    finally:
        db.close()

    contexts: dict[str, SkillContext] = {}
    by_container: dict[str, list[str]] = {}

    def touch(name: str) -> SkillContext:
        key = name.strip().casefold()
        if key not in contexts:
            contexts[key] = SkillContext(name=name.strip())
        return contexts[key]

    for skill, ev_type, repo_id, full_name, language in rows:
        ctx = touch(skill)
        if language and language not in ctx.languages:
            ctx.languages.append(language)
        short = full_name.split("/")[-1]
        if short not in ctx.repos:
            ctx.repos.append(short)
        if ev_type not in ctx.evidence_types:
            ctx.evidence_types.append(ev_type)
        by_container.setdefault(f"repo:{repo_id}", []).append(skill.strip())

    for skill, ev_type, exp_id, title, company in exp_rows:
        ctx = touch(skill)
        label = f"{title} at {company}" if company else title
        if label not in ctx.repos:
            ctx.repos.append(label)
        if ev_type not in ctx.evidence_types:
            ctx.evidence_types.append(ev_type)
        by_container.setdefault(f"exp:{exp_id}", []).append(skill.strip())

    for name in manual:
        touch(name)

    for members in by_container.values():
        for skill in members:
            ctx = contexts[skill.casefold()]
            for other in members:
                if other.casefold() == skill.casefold():
                    continue
                if other not in ctx.co_skills:
                    ctx.co_skills.append(other)

    return contexts


def render(template: str, ctx: SkillContext) -> str:
    """The string handed to the encoder. Kept short on purpose: these
    models truncate at 512 tokens and lose focus long before that, and
    the map has to re-encode every name whenever a skill is added.
    """
    name = ctx.name
    if template == "bare":
        return name
    if template == "current":
        return f"Skill: {name}"
    if template == "typed":
        return f"{name}, a technology or practice used in software engineering"
    if template == "context":
        parts = [f"{name}, a technology or practice used in software engineering."]
        if ctx.languages:
            parts.append("Used in " + ", ".join(ctx.languages[:3]) + " projects.")
        if ctx.co_skills:
            parts.append("Used alongside " + ", ".join(ctx.co_skills[:8]) + ".")
        if ctx.evidence_types:
            readable = [t.replace("_", " ") for t in ctx.evidence_types[:3]]
            parts.append("Evidence: " + ", ".join(readable) + ".")
        return " ".join(parts)
    raise ValueError(f"unknown template: {template}")


def _encoder(spec: ModelSpec):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(spec.name, **kwargs)

    def encode(texts: list[str]) -> np.ndarray:
        prefixed = [spec.query_prefix + t for t in texts] if spec.query_prefix else texts
        return np.asarray(
            model.encode(
                prefixed, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
        )

    return model, encode


def purity_at_k(vectors: np.ndarray, labels: list[str], k: int) -> float:
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    neighbours = np.argsort(-sims, axis=1)[:, :k]
    hits = [
        sum(1 for j in neighbours[i] if labels[j] == labels[i]) / k
        for i in range(len(labels))
        # A domain with a single member has no correct neighbour to find,
        # so scoring it would only measure the taxonomy's shape.
        if labels.count(labels[i]) > 1
    ]
    return float(np.mean(hits)) if hits else 0.0


def silhouette(vectors: np.ndarray, labels: list[str]) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(vectors, labels, metric="cosine"))


def pair_similarity(
    pairs: list[list[str]],
    encode,
    cache: dict[str, np.ndarray],
    template: str,
    contexts: dict[str, SkillContext],
) -> float:
    """Scored with the same template as the grid cell it belongs to,
    otherwise the trap pairs would report one number per model and say
    nothing about whether context is what pulls ElasticNet away from
    EfficientNet.

    A pair may name a skill this profile does not have (Kubernetes,
    TensorFlow), which is the point: the pairs probe what the model
    knows, not what happens to be in the database. Those names have no
    context to render, so they fall back to the bare-name form of the
    same template.
    """
    needed = sorted({name for pair in pairs for name in pair if name not in cache})
    if needed:
        texts = [
            render(template, contexts.get(name.casefold(), SkillContext(name=name)))
            for name in needed
        ]
        fresh = encode(texts)
        for name, vector in zip(needed, fresh, strict=True):
            cache[name] = vector
    sims = [float(cache[a] @ cache[b]) for a, b in pairs]
    return float(np.mean(sims))


def run_grid(
    specs: list[ModelSpec],
    templates: list[str],
    contexts: dict[str, SkillContext],
    taxonomy: dict[str, Any],
    sample: list[str],
) -> list[dict[str, Any]]:
    labels = [taxonomy["categories"][name] for name in sample]
    results: list[dict[str, Any]] = []

    for spec in specs:
        print(f"\n### {spec.name}  ({spec.note})")
        load_start = time.perf_counter()
        try:
            model, encode = _encoder(spec)
        except Exception as exc:  # a model that will not load is a real result
            print(f"  load failed: {exc}")
            results.append({"model": spec.name, "error": str(exc)})
            continue
        load_seconds = time.perf_counter() - load_start
        dim = int(model.get_sentence_embedding_dimension())
        params = sum(p.numel() for p in model.parameters()) / 1e6
        for template in templates:
            # Cleared per template: the same name embeds differently
            # under each one.
            pair_cache: dict[str, np.ndarray] = {}
            texts = [render(template, contexts[name.casefold()]) for name in sample]
            encode_start = time.perf_counter()
            vectors = encode(texts)
            encode_seconds = time.perf_counter() - encode_start

            trap = pair_similarity(taxonomy["trap_pairs"], encode, pair_cache, template, contexts)
            true = pair_similarity(taxonomy["true_pairs"], encode, pair_cache, template, contexts)
            row = {
                "model": spec.name,
                "template": template,
                "dim": dim,
                "params_m": round(params, 1),
                "load_seconds": round(load_seconds, 2),
                "encode_seconds": round(encode_seconds, 2),
                "purity_at_5": round(purity_at_k(vectors, labels, 5), 4),
                "silhouette": round(silhouette(vectors, labels), 4),
                "trap_similarity": round(trap, 4),
                "true_similarity": round(true, 4),
                "margin": round(true - trap, 4),
            }
            results.append(row)
            print(
                f"  {template:8s} purity@5 {row['purity_at_5']:.3f}"
                f"  silhouette {row['silhouette']:+.3f}"
                f"  trap {row['trap_similarity']:.3f}"
                f"  true {row['true_similarity']:.3f}"
                f"  margin {row['margin']:+.3f}"
                f"  {row['encode_seconds']:.2f}s"
            )

        del model, encode
        import gc

        gc.collect()

    return results


def compare_projections(
    spec: ModelSpec, template: str, contexts: dict[str, SkillContext], sample: list[str]
) -> list[dict[str, Any]]:
    """The map's second failure mode. Even a perfect embedding is useless
    on screen if the 768-to-2 dimension step scrambles the neighbourhoods,
    so each projection is scored by how many of a skill's true nearest
    neighbours are still nearest on the flat map.
    """
    _, encode = _encoder(spec)
    texts = [render(template, contexts[name.casefold()]) for name in sample]
    vectors = encode(texts)

    k = 5
    high = vectors @ vectors.T
    np.fill_diagonal(high, -np.inf)
    high_neighbours = np.argsort(-high, axis=1)[:, :k]

    def preservation(coords: np.ndarray) -> float:
        d = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        low_neighbours = np.argsort(d, axis=1)[:, :k]
        return float(
            np.mean(
                [
                    len(set(high_neighbours[i]) & set(low_neighbours[i])) / k
                    for i in range(len(sample))
                ]
            )
        )

    out: list[dict[str, Any]] = []

    from sklearn.decomposition import PCA

    start = time.perf_counter()
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    out.append(
        {
            "projection": "PCA",
            "knn_preservation": round(preservation(pca_coords), 4),
            "seconds": round(time.perf_counter() - start, 2),
        }
    )

    from sklearn.manifold import TSNE

    start = time.perf_counter()
    perplexity = min(30, max(5, (len(sample) - 1) // 3))
    tsne_coords = TSNE(
        n_components=2, random_state=42, perplexity=perplexity, init="pca"
    ).fit_transform(vectors)
    out.append(
        {
            "projection": f"t-SNE (perplexity {perplexity})",
            "knn_preservation": round(preservation(tsne_coords), 4),
            "seconds": round(time.perf_counter() - start, 2),
        }
    )

    try:
        import umap

        start = time.perf_counter()
        umap_coords = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(sample) - 1),
            min_dist=0.25,
            metric="cosine",
            random_state=42,
        ).fit_transform(vectors)
        out.append(
            {
                "projection": "UMAP",
                "knn_preservation": round(preservation(umap_coords), 4),
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
    except ImportError:
        out.append({"projection": "UMAP", "error": "umap-learn not installed"})

    # A middle ground that needs no new dependency: reduce to a few dozen
    # dimensions with PCA first, then let MDS lay those out. Cheaper than
    # t-SNE and deterministic.
    from sklearn.manifold import MDS

    start = time.perf_counter()
    reduced = PCA(n_components=min(50, len(sample) - 1), random_state=42).fit_transform(vectors)
    mds = MDS(n_components=2, random_state=42, normalized_stress="auto")
    mds_coords = mds.fit_transform(reduced)
    out.append(
        {
            "projection": "PCA(50) + MDS",
            "knn_preservation": round(preservation(mds_coords), 4),
            "seconds": round(time.perf_counter() - start, 2),
        }
    )

    return out


def write_report(
    results: list[dict[str, Any]],
    projections: list[dict[str, Any]],
    sample: list[str],
    path_stem: Path,
) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "sample_size": len(sample),
        "sample": sample,
        "grid": results,
        "projections": projections,
    }
    path_stem.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    scored = [r for r in results if "error" not in r]
    scored.sort(key=lambda r: r["margin"], reverse=True)
    lines = [
        f"# Skill embedding benchmark ({len(sample)} skills)",
        "",
        f"Generated {payload['generated_at']}.",
        "",
        "| model | template | purity@5 | silhouette | trap | true | margin | dim | encode |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in scored:
        lines.append(
            f"| {r['model']} | {r['template']} | {r['purity_at_5']:.3f} | {r['silhouette']:+.3f} "
            f"| {r['trap_similarity']:.3f} | {r['true_similarity']:.3f} | {r['margin']:+.3f} "
            f"| {r['dim']} | {r['encode_seconds']:.2f}s |"
        )
    failed = [r for r in results if "error" in r]
    if failed:
        lines += ["", "## Did not load", ""]
        lines += [f"- {r['model']}: {r['error']}" for r in failed]
    if projections:
        lines += [
            "",
            "## 2D projection quality",
            "",
            "| projection | kNN preservation | seconds |",
            "| --- | --- | --- |",
        ]
        for p in projections:
            if "error" in p:
                lines.append(f"| {p['projection']} | {p['error']} | |")
            else:
                lines.append(
                    f"| {p['projection']} | {p['knn_preservation']:.3f} | {p['seconds']:.2f} |"
                )
    path_stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", type=int, required=True)
    parser.add_argument(
        "--sample", type=int, default=0, help="random subset size, 0 means every skill"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", type=str, default="", help="comma-separated override of the model list"
    )
    parser.add_argument("--templates", type=str, default=",".join(TEMPLATES))
    parser.add_argument(
        "--projections", action="store_true", help="also compare PCA, t-SNE, UMAP and MDS"
    )
    args = parser.parse_args()

    init_db()
    taxonomy = load_taxonomy()
    contexts = load_contexts(args.account)

    labeled = [name for name in taxonomy["categories"] if name.casefold() in contexts]
    known = {n.casefold() for n in taxonomy["categories"]}
    missing = [name for name in contexts if name not in known]
    if missing:
        print(
            f"{len(missing)} skills have no label in {TAXONOMY_PATH} "
            f"and are skipped: {sorted(missing)[:10]}"
        )
    if not labeled:
        raise SystemExit("no labeled skills for this account")

    sample = labeled
    if args.sample and args.sample < len(labeled):
        sample = random.Random(args.seed).sample(labeled, args.sample)
        sample.sort(key=str.casefold)

    specs = DEFAULT_MODELS
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        by_name = {s.name: s for s in DEFAULT_MODELS}
        specs = [by_name.get(name, ModelSpec(name)) for name in wanted]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]

    print(f"{len(sample)} labeled skills, {len(specs)} models, {len(templates)} templates")
    results = run_grid(specs, templates, contexts, taxonomy, sample)

    projections: list[dict[str, Any]] = []
    if args.projections:
        scored = [r for r in results if "error" not in r]
        best = max(scored, key=lambda r: r["margin"])
        by_name = {s.name: s for s in specs}
        print(f"\n### projections, using {best['model']} / {best['template']}")
        projections = compare_projections(
            by_name[best["model"]], best["template"], contexts, sample
        )
        for p in projections:
            if "error" in p:
                print(f"  {p['projection']:22s} {p['error']}")
            else:
                print(
                    f"  {p['projection']:22s} kNN preservation "
                    f"{p['knn_preservation']:.3f}  {p['seconds']:.2f}s"
                )

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = RESULTS_DIR / f"embeddings-{stamp}"
    write_report(results, projections, sample, stem)
    print(f"\nwrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
