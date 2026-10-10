"""An external retrieval check on SkillSpan (Zhang et al., NAACL 2022,
CC BY 4.0): StackOverflow job postings whose knowledge spans (technologies,
tools, languages) were marked by trained annotators, none of them on this
project.

A portfolio skill is relevant to a posting when an annotated knowledge span
names it. Rule v2 (the default, docs/RETRIEVAL_IMPROVEMENTS.md entry 6)
also reads the parts of a compound span ("core-java/spring/spring-boot")
and maps names through StackOverflow tag synonyms (so_tag_synonyms.json,
CC BY-SA 4.0); rule v1, the first one run, needed the whole span to equal
the skill name. Either way only skills a posting names count, and keyword
matching is favored, so the set is a check that nothing regressed on
independent labels, not a measure of the judgment-based relevance the
LinkedIn sets label.

All three splits are used (only the "tech" source): nothing here is
trained, so the splits carry no meaning for this project. The data is
downloaded into evals/skillspan/cache/ (gitignored) by
scripts/fetch_skillspan.py; queries are the app's own extraction of each
posting, saved by `scripts.run_llm_evals --only skillspan`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

SKILLSPAN_DIR = Path(__file__).resolve().parents[2] / "evals" / "skillspan"
CACHE_DIR = SKILLSPAN_DIR / "cache"
SPLITS = ("train", "dev", "test")
SYNONYMS_PATH = SKILLSPAN_DIR / "so_tag_synonyms.json"
LabelRule = Literal["v1", "v2"]

_PARTS = re.compile(r"\s*(?:[/,;()&|]|\band\b|\bor\b)\s*", re.IGNORECASE)
_IGNORED = re.compile(r"[\s\-._]+")
DATASET_URL = "https://huggingface.co/datasets/jjzha/skillspan/resolve/main/{split}.json"


def spans(tokens: list[str], tags: list[str]) -> list[str]:
    """The text of each BIO-tagged span, tokens joined by single spaces."""
    out: list[str] = []
    current: list[str] = []
    for token, tag in zip(tokens, tags, strict=True):
        if tag == "B":
            if current:
                out.append(" ".join(current))
            current = [token]
        elif tag == "I" and current:
            current.append(token)
        else:
            if current:
                out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def load_postings(cache_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every tech posting as {"text", "knowledge"}, keyed
    "skillspan-<split>-<idx>": its sentences one per line, and the set of
    its annotated knowledge spans, casefolded and trimmed."""
    directory = cache_dir or CACHE_DIR
    sentences: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = directory / f"{split}.json"
        if not path.exists():
            from app.evals.real import CacheMissingError

            raise CacheMissingError(f"{path} is missing; run python -m scripts.fetch_skillspan")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "tech":
                sentences.setdefault(f"skillspan-{split}-{row['idx']}", []).append(row)
    postings = {}
    for key, rows in sentences.items():
        knowledge = {
            span.casefold().strip()
            for row in rows
            for span in spans(row["tokens"], row["tags_knowledge"])
        }
        text = "\n".join(" ".join(row["tokens"]) for row in rows)
        postings[key] = {"text": text, "knowledge": knowledge}
    return postings


@lru_cache
def _synonyms() -> dict[str, str]:
    return dict(json.loads(SYNONYMS_PATH.read_text(encoding="utf-8"))["synonyms"])


def name_key(name: str) -> str:
    """Rule v2's comparison key: the name as a StackOverflow tag, mapped
    through the tag synonyms, with case, spaces, hyphens, dots and
    underscores ignored."""
    tag = re.sub(r"\s+", "-", name.strip().casefold())
    tag = _synonyms().get(tag, tag)
    return _IGNORED.sub("", tag)


def span_parts(span: str) -> list[str]:
    """The span itself and each part of it, split on separators."""
    return [span, *(p.strip() for p in _PARTS.split(span) if p and p.strip())]


def relevant_skills(
    knowledge: set[str], portfolio_skills: list[str], rule: LabelRule = "v2"
) -> list[str]:
    """The portfolio skills the annotated knowledge spans name. v1: a whole
    span equals the name, ignoring case. v2: any part of a span has the
    name's key; whole parts are compared, never substrings."""
    if rule == "v1":
        return sorted(s for s in portfolio_skills if s.casefold().strip() in knowledge)
    keys = {name_key(part) for span in knowledge for part in span_parts(span)}
    keys.discard("")
    return sorted(s for s in portfolio_skills if name_key(s) in keys)


def extraction_path(key: str) -> Path:
    return CACHE_DIR / "extractions" / f"{key}.json"


def build_extraction_jobs() -> list[Any]:
    """The postings as Jobs to extract, with nothing to score: SkillSpan
    has no form fields, only spans."""
    from app.evals.synthetic import Job

    return [
        Job(key=key, for_personas=[], covers=["skillspan"], text=p["text"], expected={})
        for key, p in sorted(load_postings().items())
    ]


def build_retrieval_jobs(rule: LabelRule = "v2") -> list[Any]:
    """The postings as retrieval queries against the composite portfolio,
    searched with the app's own extraction of each one."""
    from app.evals.real import CacheMissingError, build_portfolio
    from app.evals.synthetic import Job, evidence_rows

    portfolio = sorted({row.skill for row in evidence_rows(build_portfolio())})
    jobs = []
    for key, posting in sorted(load_postings().items()):
        path = extraction_path(key)
        if not path.exists():
            raise CacheMissingError(
                f"{key} has no saved extraction; run python -m scripts.run_llm_evals "
                "--only skillspan"
            )
        extracted = json.loads(path.read_text(encoding="utf-8"))
        relevant = relevant_skills(posting["knowledge"], portfolio, rule)
        jobs.append(
            Job(
                key=key,
                for_personas=["real-portfolio"] if relevant else [],
                covers=["skillspan"],
                text=posting["text"],
                expected={
                    "title": extracted.get("title", ""),
                    "company": extracted.get("company", ""),
                    "summary": extracted.get("role_summary", ""),
                    "skills": [s["skill"] for s in extracted.get("skills_required", [])],
                },
                relevant_skills=relevant,
            )
        )
    return jobs
