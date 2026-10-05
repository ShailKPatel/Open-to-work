"""The 2D skill map: what text a skill is embedded as, how those vectors
are flattened onto a plane, and how the finished layout is cached.

Three decisions live here, all of them measured by
scripts/benchmark_embeddings.py against evals/golden/skill_taxonomy.yaml
rather than guessed:

Text. A bare name is mostly characters, and short strings embed by
shape: "ElasticNet" and "EfficientNet" land on top of each other, so do
"Netlify" and "NetworkX". Each name is therefore embedded together with
the evidence around it (the languages of the repositories it came from,
the skills it appears next to), which is what separates them.

Projection. Flattening several hundred dimensions to two always loses
most of the distances; the only question is which ones. PCA keeps the
directions of largest variance, which on this data kept 0.294 of each
skill's true nearest neighbours against 0.530 for t-SNE
(evals/results/embeddings-20260921T111424Z.md). t-SNE optimises for
keeping neighbours neighbours, which is exactly what a map is read for,
and 100-odd points take under two seconds.

Model. The map compares skills only to each other and never writes to
Qdrant, so it is free to use a different model from retrieval, and the
benchmark picked a smaller one than retrieval runs: on the same labeled
set, bge-small placed 0.396 of each skill's five nearest neighbours in
the right domain against bge-base's 0.346, while encoding in half the
time at half the width. Four models trained on code or technical text
were tried too (jina-embeddings-v2-base-code, st-codesearch,
gte-modernbert, nomic-embed-text) and none of them beat it; they are
built for code snippets, not for a name plus a sentence of context.
The embedding cache is keyed by model, so both live side by side.

Cache. The layout is a pure function of the skill set, so it is stored
whole against a fingerprint of that set and rebuilt only when the set
changes. Without this, opening the map paid for a model load.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import select

from app.core.db import (
    Experience,
    ExperienceSkillEvidence,
    Repository,
    SkillEvidence,
    SkillMapCache,
    get_db,
)
from app.core.embeddings import embed

logger = logging.getLogger(__name__)

# Bump when the text template, the projection or the cluster logic
# changes, so every stored layout is treated as stale without anyone
# having to clear a table by hand.
LAYOUT_VERSION = 2

# The map's own encoder, deliberately not app/core/embeddings.py's
# EMBEDDING_MODEL: see this module's docstring. Changing it invalidates
# every stored layout through the fingerprint, and costs one model
# download, but nothing in Qdrant has to be rebuilt.
MAP_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# How many neighbouring skills a name is described by. Past roughly this
# many the shared context starts to look the same for every skill in a
# large repository, which pulls unrelated names together again.
MAX_CONTEXT_SKILLS = 8
MAX_CONTEXT_LANGUAGES = 3

# Clusters are chosen from this range by silhouette rather than fixed at
# n // 3: a profile with 20 skills and one with 200 do not want the same
# number of groups, and a wrong k is visible as a group that spans half
# the map.
MIN_CLUSTERS = 3
MAX_CLUSTERS = 9


@dataclass
class SkillContext:
    """What the database knows about one skill name, in the pieces the
    text template spends its room on."""

    name: str
    languages: list[str] = field(default_factory=list)
    co_skills: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)


def skill_contexts(account_id: int) -> dict[str, SkillContext]:
    """Keyed by casefolded name, matching how GET /api/skills groups.

    Co-occurrence is read per container (a repository, an experience)
    rather than globally: skills that shipped in the same project are
    related in a way that skills merely present in the same profile are
    not.
    """
    db = get_db()
    try:
        repo_rows = db.execute(
            select(
                SkillEvidence.skill,
                SkillEvidence.evidence_type,
                SkillEvidence.repo_id,
                Repository.primary_language,
            )
            .join(Repository, Repository.id == SkillEvidence.repo_id)
            .where(Repository.account_id == account_id)
        ).all()
        exp_rows = db.execute(
            select(
                ExperienceSkillEvidence.skill,
                ExperienceSkillEvidence.evidence_type,
                ExperienceSkillEvidence.experience_id,
            )
            .join(Experience, Experience.id == ExperienceSkillEvidence.experience_id)
            .where(Experience.account_id == account_id)
        ).all()
    finally:
        db.close()

    contexts: dict[str, SkillContext] = {}
    grouped: dict[str, list[str]] = {}

    def touch(name: str) -> SkillContext:
        key = name.strip().casefold()
        if key not in contexts:
            contexts[key] = SkillContext(name=name.strip())
        return contexts[key]

    for skill, evidence_type, repo_id, language in repo_rows:
        ctx = touch(skill)
        if language and language not in ctx.languages:
            ctx.languages.append(language)
        if evidence_type not in ctx.evidence_types:
            ctx.evidence_types.append(evidence_type)
        grouped.setdefault(f"repo:{repo_id}", []).append(skill.strip())

    for skill, evidence_type, experience_id in exp_rows:
        ctx = touch(skill)
        if evidence_type not in ctx.evidence_types:
            ctx.evidence_types.append(evidence_type)
        grouped.setdefault(f"exp:{experience_id}", []).append(skill.strip())

    for members in grouped.values():
        for skill in members:
            ctx = contexts[skill.casefold()]
            for other in members:
                if other.casefold() != skill.casefold() and other not in ctx.co_skills:
                    ctx.co_skills.append(other)

    return contexts


def skill_text(name: str, ctx: SkillContext | None) -> str:
    """The string actually encoded. A skill with no evidence (added by
    hand, nothing linked yet) still gets the type sentence, which alone
    is worth more than the bare name.
    """
    parts = [f"{name}, a technology or practice used in software engineering."]
    if ctx is not None:
        if ctx.languages:
            languages = ", ".join(ctx.languages[:MAX_CONTEXT_LANGUAGES])
            parts.append(f"Used in {languages} projects.")
        if ctx.co_skills:
            parts.append("Used alongside " + ", ".join(ctx.co_skills[:MAX_CONTEXT_SKILLS]) + ".")
        if ctx.evidence_types:
            readable = [t.replace("_", " ") for t in ctx.evidence_types[:3]]
            parts.append("Evidence: " + ", ".join(readable) + ".")
    return " ".join(parts)


def layout_fingerprint(texts: list[str]) -> str:
    """Identifies a layout by exactly what went into it. The texts carry
    the skill names and their evidence context, so a new repository that
    changes what a skill sits next to invalidates the map too, not only
    adding or removing a name.
    """
    payload = json.dumps(
        {"version": LAYOUT_VERSION, "model": MAP_EMBEDDING_MODEL, "texts": texts},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_2d(vectors: np.ndarray) -> np.ndarray:
    """Vectors to screen coordinates, scaled to roughly [-400, 400].

    Fewer than four points cannot be laid out meaningfully by anything
    that fits a neighbourhood, so they are placed by hand.
    """
    n = len(vectors)
    if n == 1:
        return np.array([[0.0, 0.0]])
    if n == 2:
        return np.array([[-250.0, 0.0], [250.0, 0.0]])
    if n == 3:
        return np.array([[0.0, -260.0], [-260.0, 180.0], [260.0, 180.0]])

    from sklearn.manifold import TSNE

    # perplexity is roughly "how many neighbours count as local".
    # sklearn requires it strictly below the sample size, so a profile
    # with five skills cannot ask for thirty; the floor of 2 is the
    # smallest value t-SNE accepts at all.
    perplexity = max(2.0, min(30.0, (n - 1) / 3))
    perplexity = min(perplexity, float(n - 2))
    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        metric="cosine",
    ).fit_transform(vectors)

    spread = np.max(np.abs(coords), axis=0)
    spread = np.where(spread == 0, 1.0, spread)
    return (coords / spread) * 400.0


def cluster(coords: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, int]:
    """Groups are found on the projected coordinates, not the full
    vectors: the hulls drawn on screen have to agree with where the dots
    actually are, and clustering the full space put members of one group
    on opposite sides of the map.
    """
    n = len(coords)
    if n < MIN_CLUSTERS:
        return np.zeros(n, dtype=int), 1

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_labels = np.zeros(n, dtype=int)
    best_k = 1
    best_score = -1.0
    for k in range(MIN_CLUSTERS, min(MAX_CLUSTERS, n - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_labels, best_k, best_score = labels, k, score
    return best_labels, best_k


def cluster_label(members: list[str], coords: np.ndarray, member_indices: list[int]) -> str:
    """Names a group after its most central member, which reads as a
    heading ("PyTorch") instead of the arbitrary first three names the
    map used to print.
    """
    if not members:
        return "Group"
    points = coords[member_indices]
    centre = points.mean(axis=0)
    medoid = int(np.argmin(((points - centre) ** 2).sum(axis=1)))
    return members[medoid]


def build_layout(account_id: int, groups: list[Any]) -> dict[str, Any]:
    """`groups` are GET /api/skills' SkillGroup objects, passed in rather
    than re-read here so the map and the list can never disagree about
    what the account's skills are.
    """
    contexts = skill_contexts(account_id)
    texts = [skill_text(g.name, contexts.get(g.name.strip().casefold())) for g in groups]
    vectors = np.asarray(embed(texts, model_name=MAP_EMBEDDING_MODEL))

    coords = project_2d(vectors)
    labels, n_clusters = cluster(coords, vectors)

    members_by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(labels):
        members_by_cluster.setdefault(int(cid), []).append(idx)

    labels_by_cluster: dict[int, str] = {}
    clusters: list[dict[str, Any]] = []
    for cid in sorted(members_by_cluster):
        indices = members_by_cluster[cid]
        names = [groups[i].name for i in indices]
        label = cluster_label(names, coords, indices)
        labels_by_cluster[cid] = label
        clusters.append({"id": cid, "label": label, "count": len(indices)})

    nodes = [
        {
            "name": g.name,
            "x": round(float(coords[idx][0]), 2),
            "y": round(float(coords[idx][1]), 2),
            "cluster_id": int(labels[idx]),
            "cluster_label": labels_by_cluster[int(labels[idx])],
            "starred": g.starred,
            "manual_skill_id": g.manual_skill_id,
            "sources": [s.model_dump() for s in g.sources],
        }
        for idx, g in enumerate(groups)
    ]

    logger.info(
        "built skill map for account %s: %s skills, %s clusters", account_id, len(nodes), n_clusters
    )
    return {"clusters": clusters, "nodes": nodes}


def load_cached(account_id: int, fingerprint: str) -> dict[str, Any] | None:
    db = get_db()
    try:
        row = db.execute(
            select(SkillMapCache).where(SkillMapCache.account_id == account_id)
        ).scalar_one_or_none()
        if row is None or row.fingerprint != fingerprint:
            return None
        return dict(row.payload_json)
    finally:
        db.close()


def store_cached(account_id: int, fingerprint: str, payload: dict[str, Any]) -> None:
    """One row per account: an old layout for a skill set that no longer
    exists is never wanted back."""
    db = get_db()
    try:
        row = db.execute(
            select(SkillMapCache).where(SkillMapCache.account_id == account_id)
        ).scalar_one_or_none()
        if row is None:
            db.add(
                SkillMapCache(account_id=account_id, fingerprint=fingerprint, payload_json=payload)
            )
        else:
            row.fingerprint = fingerprint
            row.payload_json = payload
        db.commit()
    finally:
        db.close()
